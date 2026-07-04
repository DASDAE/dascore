"""
Abstract index backend and the shared SQL implementation.

Backends persist the six-table schema and answer flat-relation queries.
All engine differences live in `dialect.py` plus a handful of hooks; the
write/query logic here is shared so the contract test suite exercises
identical semantics on every backend.
"""

from __future__ import annotations

import abc
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dascore as dc
from dascore.io.index.dialect import BaseDialect
from dascore.io.index.ingest import SourceRecord, attr_column_name
from dascore.io.index.query import Query, apply_residuals, build_query_sql
from dascore.io.index.schema import (
    COORDS,
    INDEX_VERSION,
    INDEXES,
    KIND_STORAGE,
    PATCHES,
    SOURCES,
    TABLES,
    WHAT_IS_THIS,
)

# Structural columns whose ns-integer storage maps to pandas time types.
_TIME_COLS = {"time_min": "datetime", "time_max": "datetime", "time_step": "timedelta"}


def adapt_params(params) -> list:
    """Convert numpy scalars (and NaN) to plain python for DB drivers."""
    out = []
    for p in params:
        if hasattr(p, "item"):  # numpy scalar
            p = p.item()
        if isinstance(p, float) and np.isnan(p):
            p = None
        out.append(p)
    return out


class AbstractIndexBackend(abc.ABC):
    """Interface every index backend must implement."""

    @abc.abstractmethod
    def write_sources(self, records: list[SourceRecord]) -> None:
        """Insert or replace sources (and dependents) transactionally."""

    @abc.abstractmethod
    def delete_sources(self, source_paths: list[str]) -> None:
        """Remove sources and all dependent rows."""

    @abc.abstractmethod
    def query(self, query: Query) -> pd.DataFrame:
        """Return the flat patch-row relation matching a query."""

    @abc.abstractmethod
    def get_sources(self) -> pd.DataFrame:
        """Return the sources table."""

    @abc.abstractmethod
    def get_metadata(self) -> dict:
        """Return index-level metadata."""

    @abc.abstractmethod
    def attr_names(self) -> set[str]:
        """Return original attr names known to the index."""

    @abc.abstractmethod
    def coord_names(self) -> set[str]:
        """Return coord names known to the index."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources."""


class SQLIndexBackend(AbstractIndexBackend):
    """Shared implementation for SQL-speaking backends."""

    dialect: BaseDialect

    def __init__(self):
        self._ensure_schema()

    # --- hooks each engine provides ---------------------------------

    @abc.abstractmethod
    def _execute(self, sql: str, params=()) -> None:
        """Execute one statement."""

    @abc.abstractmethod
    def _executemany(self, sql: str, seq_of_params) -> None:
        """Execute one statement for many parameter tuples."""

    @abc.abstractmethod
    def _fetch_df(self, sql: str, params=()) -> pd.DataFrame:
        """Execute a SELECT and return a dataframe."""

    @abc.abstractmethod
    def _begin(self) -> None:
        """Start a transaction."""

    @abc.abstractmethod
    def _commit(self) -> None:
        """Commit the open transaction."""

    @abc.abstractmethod
    def _rollback(self) -> None:
        """Roll back the open transaction."""

    # --- schema ------------------------------------------------------

    def _ensure_schema(self) -> None:
        for name, columns in TABLES.items():
            self._execute(self.dialect.create_table(name, columns))
        for index_name, table, column in INDEXES:
            self._execute(
                f"CREATE INDEX IF NOT EXISTS {index_name} " f"ON {table} ({column})"
            )
        meta = self._fetch_df("SELECT * FROM meta_data")
        if meta.empty:
            self._execute(
                "INSERT INTO meta_data VALUES (?, ?, ?, ?)",
                (WHAT_IS_THIS, INDEX_VERSION, dc.__version__, time.time_ns()),
            )

    def _attr_meta(self) -> pd.DataFrame:
        return self._fetch_df("SELECT * FROM attr_meta")

    def _next_id(self, table: str, column: str) -> int:
        df = self._fetch_df(f"SELECT max({column}) AS m FROM {table}")
        value = df["m"].iloc[0]
        return 1 if pd.isnull(value) else int(value) + 1

    def _ensure_attr_columns(self, records: list[SourceRecord]) -> None:
        """Lazily add typed attr columns and register them in attr_meta."""
        known = {
            (row.attr_name, row.value_kind) for row in self._attr_meta().itertuples()
        }
        needed: dict[tuple[str, str], str | None] = {}
        for record in records:
            for patch in record.patches:
                for name, typed in patch.attrs.items():
                    key = (name, typed.kind)
                    if key not in known and key not in needed:
                        needed[key] = typed.units
        for (name, kind), units in needed.items():
            column = attr_column_name(name, kind)
            self._execute(self.dialect.add_column("attrs", column, KIND_STORAGE[kind]))
            self._execute(
                "INSERT INTO attr_meta VALUES (?, ?, ?, ?)",
                (name, kind, column, units),
            )

    def _bulk_insert(self, table: str, columns: tuple, rows: list) -> None:
        """Insert many rows; engines override for faster bulk paths."""
        if not rows:
            return
        quoted = ", ".join(self.dialect.quote(c) for c in columns)
        marks = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO {self.dialect.quote(table)} ({quoted}) VALUES ({marks})"
        self._executemany(sql, rows)

    # --- writes ------------------------------------------------------

    def write_sources(self, records: list[SourceRecord]) -> None:
        """
        Insert or replace sources and all dependent rows, atomically.

        Rows are batched per table (attrs grouped by column signature) so
        columnar engines aren't punished by row-at-a-time inserts.
        """
        self._begin()
        try:
            self._delete_by_paths([r.source_path for r in records])
            self._ensure_attr_columns(records)
            source_id = self._next_id("sources", "source_id")
            patch_id = self._next_id("patches", "patch_id")
            now = time.time_ns()
            source_rows, patch_rows, coord_rows = [], [], []
            attr_groups: dict[tuple[str, ...], list] = {}
            for record in records:
                source_rows.append(
                    (
                        source_id,
                        record.base_uri,
                        record.source_path,
                        record.source_format,
                        record.format_version,
                        record.mtime_ns,
                        record.size_bytes,
                        now,
                    )
                )
                for patch in record.patches:
                    patch_rows.append(
                        (
                            patch_id,
                            source_id,
                            patch.source_patch_id,
                            patch.n_dims,
                            patch.dims,
                            patch.shape,
                            patch.sample_count_total,
                            patch.time_min,
                            patch.time_max,
                            patch.time_step,
                            patch.distance_min,
                            patch.distance_max,
                            patch.distance_step,
                        )
                    )
                    columns = tuple(
                        attr_column_name(name, tv.kind)
                        for name, tv in patch.attrs.items()
                    )
                    attr_groups.setdefault(columns, []).append(
                        [patch_id, *(tv.value for tv in patch.attrs.values())]
                    )
                    coord_rows.extend(
                        (
                            patch_id,
                            c.coord_name,
                            c.value_kind,
                            c.dtype,
                            c.coord_dims,
                            c.length,
                            c.units,
                            c.min_num,
                            c.max_num,
                            c.step_num,
                            c.min_ns,
                            c.max_ns,
                            c.step_ns,
                            c.min_str,
                            c.max_str,
                            c.is_monotonic,
                            c.is_relative,
                            c.coord_hash,
                        )
                        for c in patch.coords
                    )
                    patch_id += 1
                source_id += 1
            self._bulk_insert("sources", tuple(SOURCES), source_rows)
            self._bulk_insert("patches", tuple(PATCHES), patch_rows)
            for columns, rows in attr_groups.items():
                self._bulk_insert("attrs", ("patch_id", *columns), rows)
            self._bulk_insert("coords", tuple(COORDS), list(coord_rows))
            self._execute("UPDATE meta_data SET last_indexed_ns = ?", (now,))
        except Exception:
            self._rollback()
            raise
        self._commit()

    def _delete_by_paths(self, source_paths: list[str]) -> None:
        if not source_paths:
            return
        marks = ", ".join("?" for _ in source_paths)
        ids = self._fetch_df(
            f"SELECT source_id FROM sources WHERE source_path IN ({marks})",
            source_paths,
        )["source_id"].tolist()
        if not ids:
            return
        id_marks = ", ".join("?" for _ in ids)
        for sql in (
            f"DELETE FROM coords WHERE patch_id IN "
            f"(SELECT patch_id FROM patches WHERE source_id IN ({id_marks}))",
            f"DELETE FROM attrs WHERE patch_id IN "
            f"(SELECT patch_id FROM patches WHERE source_id IN ({id_marks}))",
            f"DELETE FROM patches WHERE source_id IN ({id_marks})",
            f"DELETE FROM sources WHERE source_id IN ({id_marks})",
        ):
            self._execute(sql, ids)

    def delete_sources(self, source_paths: list[str]) -> None:
        """Remove sources and all dependent rows."""
        self._begin()
        try:
            self._delete_by_paths(source_paths)
        except Exception:
            self._rollback()
            raise
        self._commit()

    # --- queries -----------------------------------------------------

    def query(self, query: Query | None = None) -> pd.DataFrame:
        """Return the flat patch-row relation for a query."""
        query = query if query is not None else Query()
        attr_meta = self._attr_meta()
        sql, params, residuals = build_query_sql(query, self.dialect, attr_meta)
        df = self._fetch_df(sql, params)
        df = self._flatten(df, attr_meta)
        if residuals:
            df = apply_residuals(df, residuals)
        return df.reset_index(drop=True)

    def _flatten(self, df: pd.DataFrame, attr_meta: pd.DataFrame) -> pd.DataFrame:
        """Post-process raw SQL output into the flat-relation contract."""
        out = df.copy()
        # structural time columns: ns ints -> numpy time types
        for col, flavor in _TIME_COLS.items():
            if col in out:
                as_int = out[col].astype("float64")  # NaN-safe intermediate
                if flavor == "datetime":
                    out[col] = pd.to_datetime(as_int, unit="ns")
                else:
                    out[col] = pd.to_timedelta(as_int, unit="ns")
        # typed attr columns -> original names (coalesce multi-kind attrs)
        for name in attr_meta["attr_name"].unique():
            rows = attr_meta[attr_meta["attr_name"] == name]
            kinds = set(rows["value_kind"])
            series = None
            for row in rows.itertuples():
                if row.column_name not in out:
                    continue
                col = out[row.column_name]
                if row.value_kind == "time":
                    col = pd.to_datetime(col.astype("float64"), unit="ns")
                elif row.value_kind == "dur":
                    col = pd.to_timedelta(col.astype("float64"), unit="ns")
                elif row.value_kind == "bool":
                    col = col.astype("boolean")
                series = col if series is None else series.where(series.notna(), col)
                out = out.drop(columns=[row.column_name])
            if series is not None:
                if kinds == {"str"}:
                    # flat-contract convention: missing strings are ""
                    series = series.fillna("")
                out[name] = series
        # flat-contract names for source columns
        renames = {
            "source_path": "path",
            "source_format": "file_format",
            "format_version": "file_version",
        }
        out = out.rename(columns=renames)
        if "base_uri" in out:
            has_base = out["base_uri"].notna()
            out.loc[has_base, "path"] = (
                out.loc[has_base, "base_uri"].str.rstrip("/")
                + "/"
                + out.loc[has_base, "path"]
            )
            out = out.drop(columns=["base_uri"])
        return out.drop(columns=["source_id"], errors="ignore")

    # --- introspection -----------------------------------------------

    def get_sources(self) -> pd.DataFrame:
        """Return the sources table."""
        return self._fetch_df("SELECT * FROM sources")

    def get_metadata(self) -> dict:
        """Return index-level metadata."""
        return self._fetch_df("SELECT * FROM meta_data").iloc[0].to_dict()

    def attr_names(self) -> set[str]:
        """Return original attr names known to the index."""
        return set(self._attr_meta()["attr_name"])

    def coord_names(self) -> set[str]:
        """Return coord names known to the index."""
        return set(
            self._fetch_df("SELECT DISTINCT coord_name FROM coords")["coord_name"]
        )


def resolve_query(
    backend: AbstractIndexBackend, _attrs=None, _coords=None, **kwargs
) -> Query:
    """
    Resolve bare kwargs into a Query: attrs first, then coords.

    Implements section 1 of the selector spec; raises on unknown names or
    names supplied in more than one namespace.
    """
    from dascore.io.index.query import InvalidSpoolQueryError

    attrs = dict(_attrs or {})
    coords = dict(_coords or {})
    known_attrs = backend.attr_names()
    known_coords = backend.coord_names()
    for name, value in kwargs.items():
        if name in attrs or name in coords:
            msg = f"{name!r} given as both a bare kwarg and in _attrs/_coords."
            raise InvalidSpoolQueryError(msg)
        if name in known_attrs:
            attrs[name] = value
        elif name in known_coords:
            coords[name] = value
        else:
            msg = (
                f"{name!r} is neither an attribute nor a coordinate of any "
                f"patch in this spool."
            )
            raise InvalidSpoolQueryError(msg)
    for name in attrs:
        if name not in known_attrs:
            raise InvalidSpoolQueryError(f"Unknown attribute {name!r}.")
    for name in coords:
        if name not in known_coords:
            raise InvalidSpoolQueryError(f"Unknown coordinate {name!r}.")
    return Query(attrs=attrs, coords=coords)


def get_backend(path: str | Path, kind: str = "duckdb") -> AbstractIndexBackend:
    """Create an index backend of the given kind at path."""
    if kind == "duckdb":
        from dascore.io.index.duck import DuckDBBackend

        return DuckDBBackend(path)
    if kind == "sqlite":
        from dascore.io.index.lite import SQLiteBackend

        return SQLiteBackend(path)
    if kind == "parquet":
        from dascore.io.index.parq import ParquetBackend

        return ParquetBackend(path)
    msg = f"Unknown index backend {kind!r}."
    raise ValueError(msg)
