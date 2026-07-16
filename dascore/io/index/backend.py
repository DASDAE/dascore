"""
Index backend interface and SQLite SQL implementation.

The backend persists the seven-table schema and answers flat-relation queries.
Storage hooks remain separate from the write/query logic so the index contract
has a clear boundary.
"""

from __future__ import annotations

import abc
import time
import warnings
from contextlib import contextmanager, suppress
from pathlib import Path

import numpy as np
import pandas as pd

import dascore as dc
from dascore.exceptions import (
    InvalidIndexError,
    InvalidIndexVersionError,
    ParameterError,
    UnitError,
)
from dascore.io.index.dialect import BaseDialect
from dascore.io.index.ingest import SourceRecord, attr_column_name
from dascore.io.index.query import (
    Query,
    _as_query_list,
    apply_residuals,
    build_sql,
    normalize_range_forms,
)
from dascore.io.index.schema import (
    COORD_DEFS,
    INDEX_VERSION,
    INDEXES,
    KIND_STORAGE,
    PATCH_COORDS,
    PATCHES,
    SOURCES,
    TABLE_CONSTRAINTS,
    TABLES,
    WHAT_IS_THIS,
)
from dascore.units import convert_units

# Structural columns whose ns-integer storage maps to pandas time types.
_TIME_COLS = {"time_min": "datetime", "time_max": "datetime", "time_step": "timedelta"}


def _ns_to_time(series: pd.Series, flavor: str) -> pd.Series:
    """
    Convert nullable ns-integer columns to datetime64/timedelta64 exactly.

    Never goes through float64: ns epochs exceed float64's 2**53 integer
    range, and the resulting ~100 ns corruption breaks merge boundary
    arithmetic downstream. Float input means precision was already lost
    upstream (a fetch path rounding nullable integers through float64),
    so it is rejected rather than silently converted.
    """
    if series.dtype.kind == "f":
        msg = (
            f"ns column {series.name!r} arrived as {series.dtype}; values "
            "above 2**53 ns are already corrupted. Fetch nullable integer "
            "columns exactly (e.g. pandas nullable Int64)."
        )
        raise TypeError(msg)
    mask = series.isna()
    values = np.zeros(len(series), dtype="int64")
    if (~mask).any():
        values[~mask.to_numpy()] = series[~mask].astype("int64").to_numpy()
    dtype = "datetime64[ns]" if flavor == "datetime" else "timedelta64[ns]"
    out = pd.Series(values.view(dtype), index=series.index)
    if mask.any():
        out[mask] = pd.NaT
    return out


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
    def delete_sources(self, source_paths: list[str], base_uri: str = "") -> None:
        """Remove sources (identified by base_uri + path) and dependents."""

    @abc.abstractmethod
    def query(self, query: Query) -> pd.DataFrame:
        """Return the flat patch-row relation matching a query."""

    @abc.abstractmethod
    def count(self, query: Query) -> int:
        """Return how many patches match a query, without projecting rows."""

    @abc.abstractmethod
    def export_records(self, patch_ids=None) -> list:
        """Reconstruct source records (optionally for a subset of patches)."""

    @abc.abstractmethod
    def get_sources(self) -> pd.DataFrame:
        """Return the sources table."""

    @abc.abstractmethod
    def source_stats(self) -> pd.DataFrame:
        """Return only (source_path, mtime_ns, size_bytes) for change checks."""

    @abc.abstractmethod
    def get_metadata(self) -> dict:
        """Return index-level metadata."""

    @abc.abstractmethod
    def mark_initial_update_done(self) -> None:
        """Persist that the directory index completed its first update."""

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
        """
        Execute a SELECT and return a dataframe.

        Contract: nullable integer columns must round-trip exactly (use a
        nullable integer dtype, never float64) — ns epochs exceed
        float64's 2**53 integer range.
        """

    @abc.abstractmethod
    def _begin(self) -> None:
        """Start a transaction."""

    @abc.abstractmethod
    def _commit(self) -> None:
        """Commit the open transaction."""

    @abc.abstractmethod
    def _rollback(self) -> None:
        """Roll back the open transaction."""

    @abc.abstractmethod
    def _existing_tables(self) -> set[str]:
        """Return persisted user table names."""

    @abc.abstractmethod
    def _table_columns(self, table: str) -> set[str]:
        """Return persisted columns for one table."""

    @contextmanager
    def _transaction(self):
        """
        Run the wrapped body inside one transaction.

        Commits on normal (or early-return) exit; on any error rolls back
        without letting a failed rollback mask the original exception.
        """
        self._begin()
        try:
            yield
            self._commit()
        except Exception:
            with suppress(Exception):
                self._rollback()
            raise

    # --- schema ------------------------------------------------------

    def _ensure_schema(self) -> None:
        tables = self._existing_tables()
        if tables:
            self._validate_schema(tables)
            return
        with self._transaction():
            # Another connection may have initialized the file while this
            # writer waited for BEGIN IMMEDIATE. Re-check under the lock.
            tables = self._existing_tables()
            if tables:
                self._validate_schema(tables)
                return
            for name, columns in TABLES.items():
                self._execute(
                    self.dialect.create_table(
                        name, columns, TABLE_CONSTRAINTS.get(name, ())
                    )
                )
            for index_name, table, column in INDEXES:
                self._execute(
                    f"CREATE INDEX IF NOT EXISTS {index_name} " f"ON {table} ({column})"
                )
            self._execute(
                "INSERT INTO meta_data VALUES (?, ?, ?, ?)",
                (WHAT_IS_THIS, INDEX_VERSION, dc.__version__, 0),
            )

    def _validate_schema(self, tables: set[str]) -> None:
        """Validate an existing index before issuing any DDL or mutation."""
        required = set(TABLES)
        missing = required - tables
        if missing:
            msg = (
                "Existing spool index is incomplete; missing tables "
                f"{sorted(missing)}. Delete it and rebuild the index."
            )
            raise InvalidIndexError(msg)
        meta = self._fetch_df("SELECT * FROM meta_data")
        if len(meta) != 1 or meta["what_is_this"].iloc[0] != WHAT_IS_THIS:
            msg = "File is not a valid DASCore spool index; delete it and rebuild."
            raise InvalidIndexError(msg)
        version = int(meta["index_version"].iloc[0])
        if version != INDEX_VERSION:
            msg = (
                f"Spool index version {version} is incompatible with supported "
                f"version {INDEX_VERSION}; delete it and rebuild."
            )
            raise InvalidIndexVersionError(msg)
        for table, expected in TABLES.items():
            actual = self._table_columns(table)
            if not set(expected) <= actual:
                absent = sorted(set(expected) - actual)
                msg = (
                    f"Spool index table {table!r} is missing columns {absent}; "
                    "delete it and rebuild."
                )
                raise InvalidIndexError(msg)
        attr_columns = self._table_columns("attrs")
        meta_columns = set(self._attr_meta().get("column_name", ()))
        if not meta_columns <= attr_columns:
            absent = sorted(meta_columns - attr_columns)
            msg = (
                f"Spool index attrs table is missing dynamic columns {absent}; "
                "delete it and rebuild."
            )
            raise InvalidIndexError(msg)

    def _attr_meta(self) -> pd.DataFrame:
        return self._fetch_df("SELECT * FROM attr_meta")

    def _coord_meta(self, names=None) -> pd.DataFrame:
        """
        Return distinct coordinate names, kinds, and canonical units,
        optionally restricted to the given coord names.
        """
        sql = (
            "SELECT DISTINCT pc.coord_name, cd.value_kind, cd.units, "
            "cd.is_relative FROM patch_coords pc "
            "JOIN coord_defs cd ON cd.coord_def_id = pc.coord_def_id"
        )
        params: list = []
        if names is not None:
            params = sorted(names)
            sql += f" WHERE pc.coord_name IN ({self._placeholders(len(params))})"
        return self._fetch_df(sql, params)

    def _next_id(self, table: str, column: str) -> int:
        df = self._fetch_df(f"SELECT max({column}) AS m FROM {table}")
        value = df["m"].iloc[0]
        return 1 if pd.isnull(value) else int(value) + 1

    @staticmethod
    def _units_compatible(to_units: str, from_units: str) -> bool:
        """True when one unit converts to the other (same dimensionality)."""
        try:
            convert_units(1.0, to_units=to_units, from_units=from_units)
        except UnitError:
            return False
        return True

    def _ensure_attr_columns(
        self, records: list[SourceRecord]
    ) -> tuple[dict[tuple[str, str], str], set[tuple[str, str, str]]]:
        """
        Lazily add typed attr columns; return the (name, kind) -> column
        map and a set of (name, kind, units) values to skip.

        attr_meta is the single source of truth for column names: distinct
        attr names can sanitize to the same identifier ("Shot Number" vs
        "shot_number"), so collisions get a deterministic numeric suffix.

        One attr name occasionally carries dimensionally incompatible
        units across files (e.g. a "resolution" in meters here, seconds
        there). A single canonical unit cannot describe both, so the
        incompatible values are skipped (with a warning) rather than
        failing the whole index update.
        """
        meta = self._attr_meta()
        mapping = {
            (row.attr_name, row.value_kind): row.column_name
            for row in meta.itertuples()
        }
        stored_units = {
            (row.attr_name, row.value_kind): (
                None if pd.isnull(row.units) else row.units
            )
            for row in meta.itertuples()
        }
        taken = set(mapping.values())
        observed: dict[tuple[str, str], set[str | None]] = {}
        for record in records:
            for patch in record.patches:
                for name, typed in patch.attrs.items():
                    key = (name, typed.kind)
                    observed.setdefault(key, set()).add(typed.units)
        needed: dict[tuple[str, str], str | None] = {}
        skip_units: set[tuple[str, str, str]] = set()
        for key, units_seen in observed.items():
            canonical = stored_units.get(key) if key in mapping else None
            for unit in sorted(x for x in units_seen if x is not None):
                if canonical is None:
                    canonical = unit
                elif not self._units_compatible(canonical, unit):
                    skip_units.add((*key, unit))
                    msg = (
                        f"Attr {key[0]!r} has units {unit!r} incompatible "
                        f"with the indexed units {canonical!r}; skipping "
                        "these values in the index."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
            if key not in mapping:
                needed[key] = canonical
            elif stored_units.get(key) is None and canonical is not None:
                self._execute(
                    "UPDATE attr_meta SET units = ? "
                    "WHERE attr_name = ? AND value_kind = ?",
                    (canonical, *key),
                )
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
        return mapping, skip_units

    def _ensure_coord_defs(self, defs_needed: dict) -> dict[str, int]:
        """
        Ensure unique coord definitions exist; return def_key -> id.

        Coord summaries are deduplicated across patches: identical values
        (by fingerprint, or by summary content when no fingerprint is
        available) share one coord_defs row. Only fingerprint-backed rows
        are exposed as exact coordinate identity to merge planning.
        """
        keys = list(defs_needed)
        mapping: dict[str, int] = {}
        for chunk, marks in self._iter_in_batches(keys):
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
        marks = self._placeholders(len(columns))
        sql = f"INSERT INTO {self.dialect.quote(table)} ({quoted}) VALUES ({marks})"
        self._executemany(sql, rows)

    # --- writes ------------------------------------------------------

    def mark_initial_update_done(self) -> None:
        """Persist successful completion of a directory index's first update."""
        with self._transaction():
            self._execute(
                "UPDATE meta_data SET last_indexed_ns = ?",
                (time.time_ns(),),
            )

    def write_sources(self, records: list[SourceRecord]) -> None:
        """
        Insert or replace sources and all dependent rows, atomically.

        Rows are batched per table (attrs grouped by column signature) so
        columnar engines aren't punished by row-at-a-time inserts.
        """
        with self._transaction():
            by_base: dict[str, list[str]] = {}
            for record in records:
                by_base.setdefault(record.base_uri or "", []).append(record.source_path)
            for base_uri, paths in by_base.items():
                self._delete_by_paths(paths, base_uri=base_uri)
            column_map, skip_units = self._ensure_attr_columns(records)
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
                        record.base_uri or "",
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
                    attrs = patch.attrs
                    if skip_units:
                        attrs = {
                            name: tv
                            for name, tv in attrs.items()
                            if (name, tv.kind, tv.units) not in skip_units
                        }
                    columns = tuple(
                        column_map[(name, tv.kind)] for name, tv in attrs.items()
                    )
                    attr_groups.setdefault(columns, []).append(
                        [patch_id, *(tv.value for tv in attrs.values())]
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

    # Batch size for IN (...) parameter lists; SQLite caps bound
    # variables (32766 by default) so large replacements must chunk.
    _in_clause_batch = 5000

    @staticmethod
    def _placeholders(count: int) -> str:
        """Return a comma-separated run of ``count`` ``?`` bind markers."""
        return ", ".join("?" for _ in range(count))

    def _iter_in_batches(self, items):
        """
        Yield ``(chunk, marks)`` for an ``IN (...)`` list.

        Splitting on ``_in_clause_batch`` keeps each statement under
        SQLite's bound-variable cap; ``marks`` is the placeholder run for
        the chunk.
        """
        batch = self._in_clause_batch
        for start in range(0, len(items), batch):
            chunk = items[start : start + batch]
            yield chunk, self._placeholders(len(chunk))

    def _delete_by_paths(self, source_paths: list[str], base_uri: str = "") -> None:
        """
        Delete sources by (base_uri, source_path) identity.

        The schema declares sources -> patches -> attrs/patch_coords with
        ON DELETE CASCADE and the connection enables foreign keys, so
        deleting the sources removes every dependent row. coord_defs are
        intentionally left (they may orphan; a rebuild compacts them).
        """
        if not source_paths:
            return
        for chunk, marks in self._iter_in_batches(source_paths):
            self._execute(
                f"DELETE FROM sources WHERE source_path IN ({marks}) AND base_uri = ?",
                [*chunk, base_uri],
            )

    def delete_sources(self, source_paths: list[str], base_uri: str = "") -> None:
        """Remove sources (identified by base_uri + path) and dependents."""
        with self._transaction():
            self._delete_by_paths(source_paths, base_uri=base_uri)

    # --- queries -----------------------------------------------------

    def _query_context(self, query):
        """
        Normalize a query (or several) and fetch the metadata SQL needs.

        Returns ``(queries, attr_meta, coord_meta)``; coord metadata is
        only consulted for coord predicates, so the (whole-relation
        DISTINCT) scan is skipped for attr-only/empty queries.
        """
        queries = _as_query_list(query if query is not None else Query())
        attr_meta = self._attr_meta()
        coord_names = {name for q in queries for name in q.coords}
        coord_meta = self._coord_meta(coord_names) if coord_names else pd.DataFrame()
        return queries, attr_meta, coord_meta

    def query(self, query=None) -> pd.DataFrame:
        """Return the flat patch-row relation for a query (or several)."""
        queries, attr_meta, coord_meta = self._query_context(query)
        sql, params, residuals = build_sql(queries, self.dialect, attr_meta, coord_meta)
        df = self._fetch_df(sql, params)
        df = self._flatten(df, attr_meta)
        df = self._pivot_coords(df)
        if residuals:
            df = apply_residuals(df, residuals)
        return df.reset_index(drop=True)

    def count(self, query=None) -> int:
        """Count matching patches without projecting or pivoting rows."""
        queries, attr_meta, coord_meta = self._query_context(query)
        sql, params, residuals = build_sql(
            queries, self.dialect, attr_meta, coord_meta, count=True
        )
        if not residuals:
            return int(self._fetch_df(sql, params)["n"].iloc[0])
        # A regex residual must inspect string values, so a database count
        # cannot resolve it; the full relation already applies the residual.
        return len(self.query(queries))

    def _fetch_in(self, base_sql: str, column: str, ids: list) -> pd.DataFrame:
        """Fetch ``{base_sql} WHERE {column} IN ids``, batching large sets."""
        if not ids:
            return self._fetch_df(f"{base_sql} WHERE 0")
        frames = []
        for chunk, marks in self._iter_in_batches(ids):
            frames.append(
                self._fetch_df(f"{base_sql} WHERE {column} IN ({marks})", chunk)
            )
        return pd.concat(frames, ignore_index=True)

    def export_records(self, patch_ids=None) -> list:
        """
        Reconstruct source records, filtering by patch id in SQL.

        With patch_ids given, only those patches (and the sources, attrs,
        coordinate links, and coordinate definitions they reference) are
        fetched — O(selected membership), not O(total archive). The frames
        are assembled into the backend-independent transfer format.
        """
        from dascore.io.index.ingest import assemble_source_records

        if patch_ids is None:
            sources = self._fetch_df("SELECT * FROM sources")
            patches = self._fetch_df("SELECT * FROM patches")
            attrs = self._fetch_df("SELECT * FROM attrs")
            links = self._fetch_df("SELECT * FROM patch_coords")
            defs = self._fetch_df("SELECT * FROM coord_defs")
        else:
            ids = [int(x) for x in patch_ids]
            patches = self._fetch_in("SELECT * FROM patches", "patch_id", ids)
            if patches.empty:
                return []
            source_ids = [int(x) for x in patches["source_id"].unique()]
            sources = self._fetch_in("SELECT * FROM sources", "source_id", source_ids)
            attrs = self._fetch_in("SELECT * FROM attrs", "patch_id", ids)
            links = self._fetch_in("SELECT * FROM patch_coords", "patch_id", ids)
            def_ids = (
                [int(x) for x in links["coord_def_id"].unique()]
                if not links.empty
                else []
            )
            defs = self._fetch_in("SELECT * FROM coord_defs", "coord_def_id", def_ids)
        return assemble_source_records(
            sources, patches, attrs, links, defs, self._attr_meta()
        )

    def _flatten(self, df: pd.DataFrame, attr_meta: pd.DataFrame) -> pd.DataFrame:
        """Post-process raw SQL output into the flat-relation contract."""
        out = df.copy()
        # structural time columns: ns ints -> numpy time types (exactly)
        for col, flavor in _TIME_COLS.items():
            if col in out:
                out[col] = _ns_to_time(out[col], flavor)
        # numeric envelopes: engines return object columns when all-NULL;
        # downstream sorting needs float64 with NaN, never object None.
        for col in ("distance_min", "distance_max", "distance_step"):
            if col in out:
                out[col] = pd.to_numeric(out[col])
        # typed attr columns -> original names (coalesce multi-kind attrs).
        # Group the metadata once and drop every typed column in a single
        # pass: per-name refiltering plus one-drop-per-column was ~O(A^2)
        # metadata scans and frame copies for A dynamic attrs.
        # Every name that could collide with a structural or envelope
        # column (RESERVED_ATTR_COLUMNS and *_min/_max/_step) is refused at
        # ingest, so a dynamic attr name never shadows an existing column
        # here.
        cols_to_drop: list[str] = []
        new_columns: dict[str, pd.Series] = {}
        for name, rows in attr_meta.groupby("attr_name", sort=False):
            kinds = set(rows["value_kind"])
            multi_kind = len(rows) > 1
            series = None
            for row in rows.itertuples():
                if row.column_name not in out:
                    continue
                col = out[row.column_name]
                if row.value_kind == "time":
                    col = _ns_to_time(col, "datetime")
                elif row.value_kind == "dur":
                    col = _ns_to_time(col, "timedelta")
                elif row.value_kind == "bool":
                    col = col.astype("boolean")
                if multi_kind:
                    # multi-kind attrs coalesce in object space; typed
                    # extension arrays refuse cross-dtype fills
                    col = col.astype(object).where(col.notna(), np.nan)
                series = col if series is None else series.where(series.notna(), col)
                cols_to_drop.append(row.column_name)
            if series is not None:
                if kinds == {"str"}:
                    # flat-contract convention: missing strings are ""
                    series = series.fillna("")
                new_columns[name] = series
        if cols_to_drop:
            out = out.drop(columns=cols_to_drop)
        for name, series in new_columns.items():
            out[name] = series
        # flat-contract names for source columns
        renames = {
            "source_path": "path",
            "source_format": "file_format",
            "format_version": "file_version",
        }
        out = out.rename(columns=renames)
        if "base_uri" in out:
            has_base = out["base_uri"].notna() & (out["base_uri"] != "")
            out.loc[has_base, "path"] = (
                out.loc[has_base, "base_uri"].str.rstrip("/")
                + "/"
                + out.loc[has_base, "path"]
            )
            out = out.drop(columns=["base_uri"])
        return out.drop(columns=["source_id"], errors="ignore")

    @staticmethod
    def _add_envelope_objects(coords: pd.DataFrame) -> pd.DataFrame:
        """
        Add per-row envelope object columns (_env_min/_env_max/_env_step)
        and the merge-identity _key column to the coord-link relation.

        Conversions run on whole columns: per-row scalar pd.to_datetime
        calls cost ~40us each and dominated large realizations.
        """
        kind = coords["value_kind"].to_numpy()
        num_mask = kind == "num"
        time_mask = kind == "time"
        str_mask = ~(num_mask | time_mask)
        # NULL means not relative; via to_numeric so object/float/int
        # columns all coerce without pandas downcasting warnings.
        relative = (
            pd.to_numeric(coords["is_relative"], errors="coerce")
            .to_numpy(dtype="float64", na_value=0.0)
            .astype(bool)
        )

        def _time_objects(ns_series: pd.Series, flavor: str) -> np.ndarray:
            """Exact int-ns -> Timestamp/Timedelta objects (None for null)."""
            series = _ns_to_time(ns_series, flavor)
            return series.astype(object).where(series.notna(), None).to_numpy()

        fields = (
            ("_env_min", "min_num", "min_ns", "min_str"),
            ("_env_max", "max_num", "max_ns", "max_str"),
            ("_env_step", "step_num", "step_ns", None),
        )
        for out_col, num_col, ns_col, str_col in fields:
            values = np.empty(len(coords), dtype=object)
            if num_mask.any():
                values[num_mask] = coords[num_col].to_numpy(dtype=object)[num_mask]
            if str_col is not None and str_mask.any():
                values[str_mask] = coords[str_col].to_numpy(dtype=object)[str_mask]
            # absolute times are datetimes, relative ones timedeltas; steps
            # are timedeltas either way.
            time_flavors = (
                ((time_mask & ~relative), "datetime"),
                ((time_mask & relative), "timedelta"),
            )
            if str_col is None:
                time_flavors = ((time_mask, "timedelta"),)
            for mask, flavor in time_flavors:
                if mask.any():
                    values[mask] = _time_objects(coords[ns_col][mask], flavor)
            coords[out_col] = values
        # Summary-only definitions are useful for indexing/dedup but cannot
        # prove coordinate value identity for merge grouping.
        coords["_key"] = coords["def_key"].where(coords["fingerprint"].notna(), None)
        return coords

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
        link_sql = (
            "SELECT pc.patch_id, pc.coord_name, cd.def_key, cd.fingerprint, "
            "cd.value_kind, cd.is_relative, cd.min_num, cd.max_num, "
            "cd.step_num, cd.min_ns, cd.max_ns, cd.step_ns, "
            "cd.min_str, cd.max_str "
            "FROM patch_coords pc "
            "JOIN coord_defs cd ON cd.coord_def_id = pc.coord_def_id"
        )
        n_patches = self._fetch_df("SELECT count(*) AS n FROM patches")["n"].iloc[0]
        if len(ids) * 4 >= n_patches:
            # Most patches selected: one scan plus a pandas filter beats
            # many batched IN queries and their frame concatenation.
            coords = self._fetch_df(link_sql)
            coords = coords[coords["patch_id"].isin(set(ids))].reset_index(drop=True)
        else:
            coords = self._fetch_in(link_sql, "pc.patch_id", ids)
        if coords.empty:
            return out
        coords = self._add_envelope_objects(coords)
        for name, group in coords.groupby("coord_name"):
            pids = group["patch_id"]
            # last row wins for duplicate patch ids, like the mapping loop
            # this replaces.
            keys = dict(zip(pids, group["_key"]))
            mins = dict(zip(pids, group["_env_min"]))
            maxs = dict(zip(pids, group["_env_max"]))
            steps = dict(zip(pids, group["_env_step"]))
            out[f"_{name}_def_key"] = out["patch_id"].map(keys)
            kinds = set(group["value_kind"])
            # time/distance envelopes already live on patches...
            if name in ("time", "distance"):
                col = f"{name}_min"
                # ...but relative-time patches leave them NULL by design;
                # when the whole result is relative, serve timedelta
                # envelopes so chunking on relative time works (#553).
                if col in out.columns and out[col].isnull().all() and mins:
                    out[f"{name}_min"] = out["patch_id"].map(mins)
                    out[f"{name}_max"] = out["patch_id"].map(maxs)
                    out[f"{name}_step"] = out["patch_id"].map(steps)
                continue
            out[f"{name}_min"] = out["patch_id"].map(mins)
            out[f"{name}_max"] = out["patch_id"].map(maxs)
            out[f"{name}_step"] = out["patch_id"].map(steps)
            if kinds == {"num"}:  # object-None -> float NaN for sorting
                for suffix in ("_min", "_max", "_step"):
                    out[f"{name}{suffix}"] = pd.to_numeric(out[f"{name}{suffix}"])
        return out

    # --- introspection -----------------------------------------------

    def get_sources(self) -> pd.DataFrame:
        """Return the sources table."""
        return self._fetch_df("SELECT * FROM sources")

    def source_stats(self) -> pd.DataFrame:
        """Return only the columns incremental change detection needs."""
        return self._fetch_df("SELECT source_path, mtime_ns, size_bytes FROM sources")

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

    def _drop_noops(mapping: dict) -> dict:
        """Bare None/... selectors are no-ops, matching Patch.select."""
        return {k: v for k, v in mapping.items() if v is not None and v is not Ellipsis}

    def _shape_coord_selector(name: str, value):
        """
        Normalize one coordinate selector to the spec's accepted shapes.

        Coordinates select by range — a (start, stop) tuple or slice
        with None/... open ends (a 2-element list is the legacy range
        form) — or by a patch-local boolean mask. Scalars and value
        membership have no exact patch-level meaning and raise here,
        eagerly, rather than failing when a patch is materialized.
        """
        if value is None or value is Ellipsis:
            return value
        if isinstance(value, np.ndarray):
            if value.dtype == np.bool_:
                return value
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(x, bool | np.bool_) for x in value)
        ):
            return np.asarray(value, dtype=bool)
        elif isinstance(value, tuple | list):
            # range-like: a (start, stop) pair (2-element list is the
            # legacy range form). Wrong arity is a malformed range.
            if len(value) != 2:
                msg = f"Coordinate range for {name!r} must be a length 2 sequence."
                raise ParameterError(msg)
            return tuple(value)
        msg = (
            f"Coordinate {name!r} accepts range selectors (a (start, stop) "
            "tuple or slice, None/... for open ends) or boolean masks; "
            f"scalar and membership values are not supported. Got {value!r}."
        )
        raise InvalidSpoolQueryError(msg)

    # accept the same open/slice range forms patch-level select does
    attrs = {k: normalize_range_forms(v) for k, v in (_attrs or {}).items()}
    coords = {k: normalize_range_forms(v) for k, v in (_coords or {}).items()}
    duplicates = set(attrs) & set(coords)
    if duplicates:
        names = ", ".join(repr(x) for x in sorted(duplicates))
        raise InvalidSpoolQueryError(f"{names} given in both _attrs and _coords.")
    known_attrs = backend.attr_names()
    known_coords = backend.coord_names()
    for name, value in kwargs.items():
        if name in attrs or name in coords:
            msg = f"{name!r} given as both a bare kwarg and in _attrs/_coords."
            raise InvalidSpoolQueryError(msg)
        value = normalize_range_forms(value)
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
            raise InvalidSpoolQueryError(f"{name!r} is not an attribute of this spool.")
    for name in coords:
        if name not in known_coords:
            raise InvalidSpoolQueryError(f"{name!r} is not a coordinate of this spool.")
    coords = {k: _shape_coord_selector(k, v) for k, v in coords.items()}
    return Query(attrs=_drop_noops(attrs), coords=_drop_noops(coords))


def get_backend(path: str | Path) -> AbstractIndexBackend:
    """Create the SQLite spool-index backend at path."""
    from dascore.io.index.lite import SQLiteBackend

    return SQLiteBackend(path)
