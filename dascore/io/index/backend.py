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
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd

import dascore as dc
from dascore.io.index.dialect import BaseDialect
from dascore.io.index.ingest import SourceRecord, attr_column_name
from dascore.io.index.query import Query, apply_residuals, build_query_sql
from dascore.io.index.schema import (
    COORD_DEFS,
    INDEX_VERSION,
    INDEXES,
    KIND_STORAGE,
    PATCH_COORDS,
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

    def _ensure_attr_columns(
        self, records: list[SourceRecord]
    ) -> dict[tuple[str, str], str]:
        """
        Lazily add typed attr columns; return the (name, kind) -> column map.

        attr_meta is the single source of truth for column names: distinct
        attr names can sanitize to the same identifier ("Shot Number" vs
        "shot_number"), so collisions get a deterministic numeric suffix.
        """
        mapping = {
            (row.attr_name, row.value_kind): row.column_name
            for row in self._attr_meta().itertuples()
        }
        taken = set(mapping.values())
        needed: dict[tuple[str, str], str | None] = {}
        for record in records:
            for patch in record.patches:
                for name, typed in patch.attrs.items():
                    key = (name, typed.kind)
                    if key not in mapping and key not in needed:
                        needed[key] = typed.units
        for (name, kind), units in needed.items():
            column = base = attr_column_name(name, kind)
            suffix = 2
            while column in taken:
                column = f"{base}_{suffix}"
                suffix += 1
            taken.add(column)
            self._execute(self.dialect.add_column("attrs", column, KIND_STORAGE[kind]))
            self._execute(
                "INSERT INTO attr_meta VALUES (?, ?, ?, ?)",
                (name, kind, column, units),
            )
            mapping[(name, kind)] = column
        return mapping

    def _ensure_coord_defs(self, defs_needed: dict) -> dict[str, int]:
        """
        Ensure unique coord definitions exist; return def_key -> id.

        Coord summaries are deduplicated across patches: identical values
        (by fingerprint, or by summary content when no fingerprint is
        available) share one coord_defs row. This is what will later let
        chunk/merge recognize shared coordinates by id equality.
        """
        keys = list(defs_needed)
        mapping: dict[str, int] = {}
        batch = self._in_clause_batch
        for start in range(0, len(keys), batch):
            chunk = keys[start : start + batch]
            marks = ", ".join("?" for _ in chunk)
            found = self._fetch_df(
                f"SELECT def_key, coord_def_id FROM coord_defs "
                f"WHERE def_key IN ({marks})",
                chunk,
            )
            mapping.update(
                zip(found["def_key"], (int(x) for x in found["coord_def_id"]))
            )
        new_keys = [k for k in keys if k not in mapping]
        next_id = self._next_id("coord_defs", "coord_def_id")
        def_rows = []
        for key in new_keys:
            c = defs_needed[key]
            def_rows.append(
                (
                    next_id,
                    key,
                    c.coord_hash,
                    c.value_kind,
                    c.dtype,
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
                )
            )
            mapping[key] = next_id
            next_id += 1
        self._bulk_insert("coord_defs", tuple(COORD_DEFS), def_rows)
        return mapping

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
            column_map = self._ensure_attr_columns(records)
            source_id = self._next_id("sources", "source_id")
            patch_id = self._next_id("patches", "patch_id")
            now = time.time_ns()
            source_rows, patch_rows, link_rows = [], [], []
            defs_needed: dict[str, object] = {}
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
                        column_map[(name, tv.kind)] for name, tv in patch.attrs.items()
                    )
                    attr_groups.setdefault(columns, []).append(
                        [patch_id, *(tv.value for tv in patch.attrs.values())]
                    )
                    for c in patch.coords:
                        key = c.def_key
                        defs_needed.setdefault(key, c)
                        link_rows.append((patch_id, c.coord_name, c.coord_dims, key))
                    patch_id += 1
                source_id += 1
            self._bulk_insert("sources", tuple(SOURCES), source_rows)
            self._bulk_insert("patches", tuple(PATCHES), patch_rows)
            for columns, rows in attr_groups.items():
                self._bulk_insert("attrs", ("patch_id", *columns), rows)
            def_ids = self._ensure_coord_defs(defs_needed)
            self._bulk_insert(
                "patch_coords",
                tuple(PATCH_COORDS),
                [(pid, name, dims, def_ids[key]) for pid, name, dims, key in link_rows],
            )
            self._execute("UPDATE meta_data SET last_indexed_ns = ?", (now,))
        except Exception:
            # A failed rollback must not mask the original error.
            with suppress(Exception):
                self._rollback()
            raise
        self._commit()

    # Batch size for IN (...) parameter lists; SQLite caps bound
    # variables (32766 by default) so large replacements must chunk.
    _in_clause_batch = 5000

    def _delete_by_paths(self, source_paths: list[str]) -> None:
        if not source_paths:
            return
        batch = self._in_clause_batch
        ids: list = []
        for start in range(0, len(source_paths), batch):
            chunk = source_paths[start : start + batch]
            marks = ", ".join("?" for _ in chunk)
            found = self._fetch_df(
                f"SELECT source_id FROM sources WHERE source_path IN ({marks})",
                chunk,
            )["source_id"].tolist()
            ids.extend(found)
        for start in range(0, len(ids), batch):
            chunk = ids[start : start + batch]
            id_marks = ", ".join("?" for _ in chunk)
            # coord_defs rows may orphan; harmless, a rebuild compacts them
            for sql in (
                f"DELETE FROM patch_coords WHERE patch_id IN "
                f"(SELECT patch_id FROM patches WHERE source_id IN ({id_marks}))",
                f"DELETE FROM attrs WHERE patch_id IN "
                f"(SELECT patch_id FROM patches WHERE source_id IN ({id_marks}))",
                f"DELETE FROM patches WHERE source_id IN ({id_marks})",
                f"DELETE FROM sources WHERE source_id IN ({id_marks})",
            ):
                self._execute(sql, chunk)

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
        df = self._pivot_coords(df)
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
                if len(rows) > 1:
                    # multi-kind attrs coalesce in object space; typed
                    # extension arrays refuse cross-dtype fills
                    col = col.astype(object).where(col.notna(), np.nan)
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

    def _pivot_coords(self, out: pd.DataFrame) -> pd.DataFrame:
        """
        Add per-coord envelope columns to the flat relation.

        Emits {name}_min/{name}_max/{name}_step for every coord in the
        result beyond the time/distance envelopes already cached on
        patches (memory-spool parity: chunking on any dim needs these),
        plus a private _{name}_def_key column for every coord — the
        globally-stable coordinate identity future chunk/merge grouping
        uses (private so it does not yet participate in merge
        compatibility comparisons).
        """
        if out.empty or "patch_id" not in out.columns:
            return out
        ids = out["patch_id"].tolist()
        frames = []
        batch = self._in_clause_batch
        for start in range(0, len(ids), batch):
            chunk = ids[start : start + batch]
            marks = ", ".join("?" for _ in chunk)
            frames.append(
                self._fetch_df(
                    "SELECT pc.patch_id, pc.coord_name, cd.def_key, "
                    "cd.value_kind, cd.is_relative, cd.min_num, cd.max_num, "
                    "cd.step_num, cd.min_ns, cd.max_ns, cd.step_ns, "
                    "cd.min_str, cd.max_str "
                    "FROM patch_coords pc "
                    "JOIN coord_defs cd ON cd.coord_def_id = pc.coord_def_id "
                    f"WHERE pc.patch_id IN ({marks})",
                    chunk,
                )
            )
        coords = pd.concat(frames, ignore_index=True)
        if coords.empty:
            return out
        for name, group in coords.groupby("coord_name"):
            mins, maxs, steps, keys = {}, {}, {}, {}
            for row in group.itertuples():
                keys[row.patch_id] = row.def_key
                if row.value_kind == "num":
                    mn, mx = row.min_num, row.max_num
                    st = row.step_num
                elif row.value_kind == "time":
                    conv = (
                        pd.to_timedelta
                        if pd.notnull(row.is_relative) and row.is_relative
                        else pd.to_datetime
                    )
                    mn = conv(int(row.min_ns), unit="ns")
                    mx = conv(int(row.max_ns), unit="ns")
                    st = (
                        pd.to_timedelta(int(row.step_ns), unit="ns")
                        if pd.notnull(row.step_ns)
                        else None
                    )
                else:
                    mn, mx, st = row.min_str, row.max_str, None
                mins[row.patch_id], maxs[row.patch_id] = mn, mx
                steps[row.patch_id] = st
            out[f"_{name}_def_key"] = out["patch_id"].map(keys)
            # time/distance envelopes already live on patches
            if name in ("time", "distance"):
                continue
            out[f"{name}_min"] = out["patch_id"].map(mins)
            out[f"{name}_max"] = out["patch_id"].map(maxs)
            out[f"{name}_step"] = out["patch_id"].map(steps)
        return out

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
        df = self._fetch_df("SELECT DISTINCT coord_name FROM patch_coords")
        return set(df["coord_name"])


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
