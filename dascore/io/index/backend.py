"""
The SQLite index backend.

Persists the seven-table schema and answers flat-relation queries. One
engine, one class: the connection handling, the SQL, and the schema
management are all SQLite's, and `get_backend` is the seam a second
engine would reopen.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import warnings
import weakref
from contextlib import contextmanager, suppress
from pathlib import Path

import numpy as np
import pandas as pd

import dascore as dc
from dascore.exceptions import (
    InvalidIndexError,
    InvalidIndexVersionError,
    UnitError,
)
from dascore.io.index.ingest import (
    SourceRecord,
    assemble_source_records,
    attr_column_name,
    dump_path_attrs,
    hive_typed_attrs,
)
from dascore.io.index.query import (
    Query,
    _as_query_list,
    _normalize_unit,
    apply_residuals,
    build_sql,
)
from dascore.io.index.schema import (
    INDEX_VERSION,
    INDEXES,
    KIND_STORAGE,
    TABLE_CONSTRAINTS,
    TABLES,
    WHAT_IS_THIS,
    CoordDefRow,
    PatchCoordRow,
    PatchRow,
    SourceRow,
    add_column_sql,
    create_table_sql,
    quote,
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


# A serialized SQLite build (threadsafety == 3) lets one connection be
# used and closed from any thread, so cross-thread garbage collection of
# a backend is safe. On rarer non-serialized builds the connection is
# thread-bound and must stay check_same_thread.
_SQLITE_SERIALIZED = sqlite3.threadsafety == 3


def _safe_close(con: sqlite3.Connection) -> None:
    """
    Close a connection, tolerating cross-thread finalization.

    On a serialized build closing works from any thread. On a
    thread-bound build a finalizer firing on another thread would raise
    ProgrammingError; suppress it (the underlying handle is freed at
    interpreter teardown) rather than emit an unraisable exception.
    """
    with suppress(sqlite3.ProgrammingError):
        con.close()


def _adapt(params):
    """Convert numpy/py types sqlite3 can't bind natively."""
    return [int(p) if isinstance(p, bool) else p for p in adapt_params(params)]


def _classic_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert nullable extension columns back to classic numpy dtypes.

    Fetching with dtype_backend="numpy_nullable" is what keeps nullable
    INTEGER columns exact (the default assembly rounds >2**53 ns values
    through float64), but downstream spool code expects classic dtypes.
    Only int columns that actually hold NULLs stay nullable (Int64) —
    the exactness they exist for; consumers handle them via isna().
    """
    for name in df.columns:
        col = df[name]
        dtype = col.dtype
        if not isinstance(dtype, pd.api.extensions.ExtensionDtype):
            continue
        if dtype.kind == "i":
            if not col.isna().any():
                df[name] = col.to_numpy(dtype="int64")
        elif dtype.kind == "f":
            df[name] = col.to_numpy(dtype="float64", na_value=np.nan)
        else:  # string/boolean/... -> classic object with None for missing
            df[name] = col.to_numpy(dtype=object, na_value=None)
    return df


class SQLiteIndexBackend:
    """
    The index backend: persists the schema and answers flat-relation queries.

    Everything above this layer (catalog, indexer) talks to this class.
    Its state is one SQLite connection, and it is safe to share across
    threads: statement execution is serialized on a reentrant lock, since
    a pandas fetch spans many cursor calls and interleaving them corrupts
    result frames.
    """

    def __init__(self, path: str | Path):
        self._path = str(path)
        # On a serialized build, drop the thread affinity so the shared
        # backend can be used (and finalized) from worker threads, e.g.
        # a thread-pool Spool.map over one catalog.
        self._con = sqlite3.connect(
            self._path, check_same_thread=not _SQLITE_SERIALIZED
        )
        # autocommit off; we manage transactions explicitly.
        self._con.isolation_level = None
        self._con.execute("PRAGMA foreign_keys = ON")
        self._con.execute("PRAGMA busy_timeout = 30000")
        # Catalog views share this backend object, so tying connection
        # cleanup to *its* collection is safe (close() stays idempotent
        # for explicit use). The finalizer tolerates cross-thread firing.
        self._finalizer = weakref.finalize(self, _safe_close, self._con)
        self._lock = threading.RLock()
        # collision name-sets already warned about (see _apply_attr_columns)
        self._warned_attr_clobber: set[frozenset] = set()
        try:
            self._ensure_schema()
        except Exception:
            self._con.close()
            raise

    def __getstate__(self) -> dict:
        """
        Pickle by database path; the file is the durable state.

        This makes file-backed spools usable with process pools: the
        receiving process reopens its own connection. In-memory backends
        have no file to reopen; their owners (catalogs) serialize their
        contents separately and never pickle the backend itself.
        """
        if self._path == ":memory:":
            msg = (
                "In-memory index backends cannot be pickled; pickle their "
                "owning catalog/spool instead."
            )
            raise TypeError(msg)
        return {"_path": self._path}

    def __setstate__(self, state: dict) -> None:
        """Reconnect to the database file."""
        self.__init__(state["_path"])

    # --- statements ---------------------------------------------------

    def _execute(self, sql: str, params=()) -> None:
        with self._lock:
            self._con.execute(sql, _adapt(params))

    def _executemany(self, sql: str, seq_of_params) -> None:
        # sqlite3.executemany consumes an iterator, so adapt lazily rather
        # than materializing a second copy of each already-built batch.
        with self._lock:
            self._con.executemany(sql, (_adapt(p) for p in seq_of_params))

    def _fetch_df(self, sql: str, params=()) -> pd.DataFrame:
        """
        Execute a SELECT and return a dataframe.

        numpy_nullable assembly keeps nullable INTEGER columns exact; the
        default path rounds them through float64, corrupting ns epochs
        (>2**53). A dtype= hint does NOT prevent that: pandas builds
        float64 first and casts after.
        """
        with self._lock:
            df = pd.read_sql_query(
                sql, self._con, params=_adapt(params), dtype_backend="numpy_nullable"
            )
        return _classic_dtypes(df)

    def _begin(self) -> None:
        self._con.execute("BEGIN IMMEDIATE")

    def _commit(self) -> None:
        self._con.execute("COMMIT")

    def _rollback(self) -> None:
        self._con.execute("ROLLBACK")

    def _existing_tables(self) -> set[str]:
        rows = self._con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {row[0] for row in rows}

    def _table_columns(self, table: str) -> set[str]:
        return {
            row[1] for row in self._con.execute(f"PRAGMA table_info({quote(table)})")
        }

    def close(self) -> None:
        """Close the database connection."""
        # detach the GC finalizer; closing twice is harmless but tidy.
        self._finalizer.detach()
        _safe_close(self._con)

    @contextmanager
    def _transaction(self):
        """
        Run the wrapped body inside one transaction.

        Commits on normal (or early-return) exit; on any error rolls back
        without letting a failed rollback mask the original exception.
        The statement lock is held for the whole transaction, so readers
        sharing the connection can never observe a half-applied write;
        being reentrant, it keeps the statement helpers inside the body
        working unchanged.
        """
        with self._lock:
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
                    create_table_sql(name, columns, TABLE_CONSTRAINTS.get(name, ()))
                )
            for index_name, table, column in INDEXES:
                self._execute(
                    f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({column})"
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

    def coord_meta(self, names=None) -> pd.DataFrame:
        """
        Return distinct coordinate names, kinds, and canonical units,
        optionally restricted to the given coord names.

        Public because the catalog asks: deciding which selectors are
        numeric (and so unit-convertible) needs the stored kinds.
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
        """Lazily add typed attr columns for the records' patch attrs."""
        attr_dicts = [p.attrs for record in records for p in record.patches]
        return self._ensure_attr_columns_for(attr_dicts)

    def _ensure_attr_columns_for(
        self, attr_dicts
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
        keys = list(zip(meta["attr_name"], meta["value_kind"], strict=True))
        mapping = dict(zip(keys, meta["column_name"], strict=True))
        stored_units = {
            key: (None if pd.isnull(units) else units)
            for key, units in zip(keys, meta["units"], strict=True)
        }
        taken = set(mapping.values())
        observed: dict[tuple[str, str], set[str | None]] = {}
        for attrs in attr_dicts:
            for name, typed in attrs.items():
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
            self._execute(add_column_sql("attrs", column, KIND_STORAGE[kind]))
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
                    c.is_relative,
                )
            )
            mapping[key] = next_id
            next_id += 1
        self._bulk_insert("coord_defs", CoordDefRow._fields, def_rows)
        return mapping

    def _bulk_insert(self, table: str, columns: tuple, rows: list) -> None:
        """Insert many rows in one statement."""
        if not rows:
            return
        quoted = ", ".join(quote(c) for c in columns)
        marks = self._placeholders(len(columns))
        sql = f"INSERT INTO {quote(table)} ({quoted}) VALUES ({marks})"
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

        Rows are batched per table (attrs grouped by column signature)
        rather than inserted one at a time.
        """
        with self._transaction():
            by_base: dict[str, list[str]] = {}
            for record in records:
                by_base.setdefault(record.base_uri or "", []).append(record.source_path)
            # Ordering contract: a replaced source keeps its ordinal
            # (first-occurrence position, dict-merge semantics) while new
            # sources append after every existing position. Read both
            # before the delete below discards them.
            kept_ordinals = self._existing_ordinals(by_base)
            max_df = self._fetch_df("SELECT max(ordinal) AS m FROM sources")
            max_ordinal = max_df["m"].iloc[0]
            next_ordinal = 0 if pd.isnull(max_ordinal) else int(max_ordinal) + 1
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
                identity = (record.base_uri or "", record.source_path)
                ordinal = kept_ordinals.get(identity)
                if ordinal is None:
                    ordinal = next_ordinal
                    next_ordinal += 1
                source_rows.append(
                    (
                        source_id,
                        record.base_uri or "",
                        record.source_path,
                        record.source_format,
                        record.format_version,
                        record.mtime_ns,
                        record.size_bytes,
                        dump_path_attrs(record.path_attrs),
                        now,
                        ordinal,
                    )
                )
                for patch in record.patches:
                    patch_rows.append(
                        (
                            patch_id,
                            source_id,
                            patch.source_patch_key,
                            patch.dims,
                            patch.dtype,
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
            self._bulk_insert("sources", SourceRow._fields, source_rows)
            self._bulk_insert("patches", PatchRow._fields, patch_rows)
            for columns, rows in attr_groups.items():
                self._bulk_insert("attrs", ("patch_id", *columns), rows)
            def_ids = self._ensure_coord_defs(defs_needed)
            self._bulk_insert(
                "patch_coords",
                PatchCoordRow._fields,
                [(pid, name, dims, def_ids[key]) for pid, name, dims, key in link_rows],
            )
            # meta_data.last_indexed_ns is the initial-update-complete
            # marker; only mark_initial_update_done (after renumbering
            # succeeds) may set it, or an interruption here would defeat
            # the reopen recovery path. Per-source timestamps already
            # live on the sources rows.

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

    def _existing_ordinals(self, by_base: dict[str, list[str]]) -> dict:
        """Map (base_uri, source_path) -> ordinal for already-stored sources."""
        out: dict[tuple[str, str], int] = {}
        for base_uri, paths in by_base.items():
            for chunk, marks in self._iter_in_batches(paths):
                df = self._fetch_df(
                    f"SELECT source_path, ordinal FROM sources "
                    f"WHERE source_path IN ({marks}) AND base_uri = ?",
                    [*chunk, base_uri],
                )
                for path, ordinal in zip(df["source_path"], df["ordinal"], strict=True):
                    if not pd.isnull(ordinal):
                        out[(base_uri, path)] = int(ordinal)
        return out

    def renumber_ordinals_by_time(self) -> None:
        """
        Renumber source ordinals into time order.

        The directory syncer owns its catalog's presentation order and
        calls this after each sync so file archives keep their
        conventional time-ordered iteration; sources are ordered by the
        earliest patch time (sources without patches last), path as the
        deterministic tiebreak.
        """
        with self._transaction():
            # the WHERE clause skips rows whose ordinal is already
            # correct, so a one-file sync of a large archive rewrites
            # one row instead of churning the whole table through WAL
            self._execute(
                "WITH ranked AS ("
                " SELECT s2.source_id AS sid, ROW_NUMBER() OVER ("
                "  ORDER BY t.min_time IS NULL, t.min_time, s2.source_path"
                " ) - 1 AS rn"
                " FROM sources s2 LEFT JOIN ("
                "  SELECT source_id, MIN(time_min) AS min_time"
                "  FROM patches GROUP BY source_id"
                " ) t ON t.source_id = s2.source_id"
                ")"
                "UPDATE sources SET ordinal = ("
                " SELECT rn FROM ranked WHERE sid = sources.source_id"
                ") WHERE ordinal IS NOT ("
                " SELECT rn FROM ranked WHERE sid = sources.source_id"
                ")"
            )

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

    def move_sources(
        self,
        moves: dict[str, str],
        new_path_attrs: dict[str, dict[str, str]] | None = None,
        base_uri: str = "",
    ) -> None:
        """
        Rewrite source paths in place (filesystem renames), atomically.

        Updates each source's stored path and its hive path attrs without
        touching patch/coord rows, so a directory rename never re-reads
        file contents. ``moves`` maps old -> new source_path;
        ``new_path_attrs`` maps old source_path -> the new path's hive
        attrs. The caller (the directory syncer) guarantees the new hive
        keys are a superset of the old ones — a removed key would need
        the file's own attr value back, which only a rescan can supply.
        """
        path_attrs = new_path_attrs or {}
        attrs_by_old = {
            old: hive_typed_attrs(attrs) for old, attrs in path_attrs.items() if attrs
        }
        with self._transaction():
            column_map, _ = self._ensure_attr_columns_for(attrs_by_old.values())
            # attr name -> [(kind, column)] once; per-move dataframe
            # filtering dominated large renames.
            kinds_by_name: dict[str, list[tuple[str, str]]] = {}
            meta = self._attr_meta()
            for name, kind, column in zip(
                meta["attr_name"], meta["value_kind"], meta["column_name"], strict=True
            ):
                kinds_by_name.setdefault(name, []).append((kind, column))
            now = time.time_ns()
            ids: dict[str, int] = {}
            for chunk, marks in self._iter_in_batches(list(moves)):
                df = self._fetch_df(
                    f"SELECT source_id, source_path FROM sources "
                    f"WHERE source_path IN ({marks}) AND base_uri = ?",
                    [*chunk, base_uri],
                )
                ids.update(zip(df["source_path"], (int(x) for x in df["source_id"])))
            source_rows = [
                (
                    new,
                    dump_path_attrs(path_attrs.get(old)),
                    now,
                    ids[old],
                )
                for old, new in moves.items()
                if old in ids
            ]
            self._executemany(
                "UPDATE sources SET source_path = ?, path_attrs = ?, "
                "last_indexed_ns = ? WHERE source_id = ?",
                source_rows,
            )
            # hive wins: set the str column and clear any other-kind columns
            # of the same attr name, matching what a fresh ingest of the
            # merged attrs would have produced. A directory rename gives
            # every moved source the same assignments, so group by the
            # assignment signature and update each group's patches at once.
            groups: dict[tuple, list[int]] = {}
            for old in moves:
                attrs = attrs_by_old.get(old)
                if not attrs or old not in ids:
                    continue
                sig = []
                for name, typed in attrs.items():
                    sig.append((column_map[(name, typed.kind)], typed.value))
                    sig.extend(
                        (column, None)
                        for kind, column in kinds_by_name.get(name, ())
                        if kind != typed.kind
                    )
                groups.setdefault(tuple(sorted(sig)), []).append(ids[old])
            for sig, source_ids in groups.items():
                assignments = ", ".join(f"{quote(col)} = ?" for col, _ in sig)
                values = [value for _, value in sig]
                for chunk, marks in self._iter_in_batches(source_ids):
                    self._execute(
                        f"UPDATE attrs SET {assignments} WHERE patch_id IN "
                        f"(SELECT patch_id FROM patches "
                        f"WHERE source_id IN ({marks}))",
                        (*values, *chunk),
                    )

    # --- queries -----------------------------------------------------

    def _query_context(self, query, order_by=None):
        """
        Normalize a query (or several) and fetch the metadata SQL needs.

        Returns ``(queries, attr_meta, coord_meta)``; coord metadata is
        only consulted for coord predicates and coord ordering, so the
        (whole-relation DISTINCT) scan is skipped for attr-only/empty
        queries.
        """
        queries = _as_query_list(query if query is not None else Query())
        attr_meta = self._attr_meta()
        coord_names = {name for q in queries for name in q.coords}
        if order_by is not None and order_by[0] == "coord":
            coord_names.add(order_by[1])
        coord_meta = self.coord_meta(coord_names) if coord_names else pd.DataFrame()
        return queries, attr_meta, coord_meta

    def query(self, query=None, order_by=None, patch_ids=None) -> pd.DataFrame:
        """Return the flat patch-row relation for a query (or several)."""
        queries, attr_meta, coord_meta = self._query_context(query, order_by=order_by)
        sql, params, residuals = build_sql(
            queries,
            attr_meta,
            coord_meta,
            order_by=order_by,
            patch_ids=patch_ids,
        )
        df = self._fetch_df(sql, params)
        df, attr_columns = self._flatten(df, attr_meta)
        df = self._pivot_coords(df)
        df = self._apply_attr_columns(df, attr_columns)
        if residuals:
            # Residuals only ever come from attr predicates, so they must
            # evaluate against the attr values even when a collision kept
            # the attr column out of the flat frame.
            df = apply_residuals(df, residuals, attr_columns)
        return df.reset_index(drop=True)

    def query_ids(self, query=None, order_by=None, patch_ids=None) -> list[int]:
        """Return matching patch ids in presentation order (ids only)."""
        queries, attr_meta, coord_meta = self._query_context(query, order_by=order_by)
        sql, params, residuals = build_sql(
            queries,
            attr_meta,
            coord_meta,
            order_by=order_by,
            patch_ids=patch_ids,
            ids_only=True,
        )
        if residuals:
            # regex residuals need string values; realize the relation
            df = self.query(queries, order_by=order_by, patch_ids=patch_ids)
            return [int(x) for x in df["patch_id"]]
        return [int(x) for x in self._fetch_df(sql, params)["patch_id"]]

    def count(self, query=None, patch_ids=None) -> int:
        """Count matching patches without projecting or pivoting rows."""
        queries, attr_meta, coord_meta = self._query_context(query)
        sql, params, residuals = build_sql(
            queries,
            attr_meta,
            coord_meta,
            count=True,
            patch_ids=patch_ids,
        )
        if not residuals:
            return int(self._fetch_df(sql, params)["n"].iloc[0])
        # A regex residual must inspect string values, so a database count
        # cannot resolve it; the full relation already applies the residual.
        return len(self.query(queries, patch_ids=patch_ids))

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

    def _flatten(
        self, df: pd.DataFrame, attr_meta: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        """
        Post-process raw SQL output into the flat-relation contract.

        Returns the frame plus the dynamic attr columns keyed by original
        attr name. The attr columns are applied by `_apply_attr_columns`
        only after `_pivot_coords` has added the per-coord envelope columns,
        so genuine name collisions can be detected against the full frame.
        """
        out = df.copy()
        # structural time columns: ns ints -> numpy time types (exactly)
        for col, flavor in _TIME_COLS.items():
            if col in out:
                out[col] = _ns_to_time(out[col], flavor)
        # numeric envelopes: an all-NULL column arrives as object;
        # downstream sorting needs float64 with NaN, never object None.
        for col in ("distance_min", "distance_max", "distance_step"):
            if col in out:
                out[col] = pd.to_numeric(out[col])
        # typed attr columns -> original names (coalesce multi-kind attrs).
        # Group the metadata once and drop every typed column in a single
        # pass: per-name refiltering plus one-drop-per-column was ~O(A^2)
        # metadata scans and frame copies for A dynamic attrs.
        # Structural names (RESERVED_ATTR_COLUMNS) are refused at ingest,
        # but an attr may legitimately share a name with a coordinate
        # envelope column ({coord}_min/max/step) when another patch in the
        # catalog has that coord; the coord column wins the flat name and
        # the attr stays queryable through the _attrs namespace.
        cols_to_drop: list[str] = []
        new_columns: dict[str, pd.Series] = {}
        for name, rows in attr_meta.groupby("attr_name", sort=False):
            kinds = set(rows["value_kind"])
            multi_kind = len(rows) > 1
            series = None
            for column, kind in zip(
                rows["column_name"], rows["value_kind"], strict=True
            ):
                if column not in out:
                    continue
                col = out[column]
                if kind == "time":
                    col = _ns_to_time(col, "datetime")
                elif kind == "dur":
                    col = _ns_to_time(col, "timedelta")
                elif kind == "bool":
                    col = col.astype("boolean")
                if multi_kind:
                    # multi-kind attrs coalesce in object space; typed
                    # extension arrays refuse cross-dtype fills
                    col = col.astype(object).where(col.notna(), np.nan)
                series = col if series is None else series.where(series.notna(), col)
                cols_to_drop.append(column)
            if series is not None:
                if kinds == {"str"}:
                    # flat-contract convention: missing strings are ""
                    series = series.fillna("")
                new_columns[str(name)] = series
        if cols_to_drop:
            out = out.drop(columns=cols_to_drop)
        # flat-contract names for source columns; path_attrs goes private
        # so chunk merge-compat grouping (non-private columns) ignores it
        renames = {
            "format_version": "source_version",
            "path_attrs": "_path_attrs",
        }
        out = out.rename(columns=renames)
        if "base_uri" in out:
            has_base = out["base_uri"].notna() & (out["base_uri"] != "")
            out.loc[has_base, "source_path"] = (
                out.loc[has_base, "base_uri"].str.rstrip("/")
                + "/"
                + out.loc[has_base, "source_path"]
            )
            out = out.drop(columns=["base_uri"])
        return out.drop(columns=["source_id"], errors="ignore"), new_columns

    def _apply_attr_columns(
        self, out: pd.DataFrame, new_columns: dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Add dynamic attr columns to the flat frame, coord envelopes winning.

        Runs after `_pivot_coords` so every coordinate envelope column
        exists; an attr whose name equals one is a genuine collision — it is
        omitted from the flat view (still queryable via the _attrs
        namespace) with a warning. A private `_{name}_units` column claims
        the public `{name}_units` spelling too, since presentation renames
        it there: the coordinate's unit must never read as attr data.
        """

        def _collides(name: str) -> bool:
            return name in out.columns or (
                name.endswith("_units") and f"_{name}" in out.columns
            )

        clobbered = frozenset(name for name in new_columns if _collides(name))
        # The flat frame is also materialized internally (patch naming,
        # chunking); warn once per backend per colliding name set so users
        # learn about the shadowing without a warning on every access.
        if clobbered and clobbered not in self._warned_attr_clobber:
            self._warned_attr_clobber.add(clobbered)
            names = ", ".join(sorted(clobbered))
            msg = (
                f"Attr(s) {names} collide with coordinate envelope columns "
                "and are omitted from the flat contents; query them via the "
                "_attrs namespace."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        for name, series in new_columns.items():
            if _collides(name):
                continue
            out[name] = series
        return out

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
            # Assigning the bare array lets pandas re-infer a dtype; a result
            # whose numeric coords all have null envelopes holds only
            # Timestamp/Timedelta and None, which would infer a temporal dtype
            # and silently turn those numeric nulls into NaT.
            coords[out_col] = pd.Series(values, index=coords.index, dtype=object)
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
            "cd.value_kind, cd.is_relative, cd.units, cd.min_num, cd.max_num, "
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
            units = dict(zip(pids, group["units"]))
            out[f"_{name}_def_key"] = out["patch_id"].map(keys)
            # the ORIGINAL unit spelling, matching the native envelope
            # values; chunk partitioning normalizes compatible spellings
            # to one unit per dimensionality before using this
            out[f"_{name}_units"] = out["patch_id"].map(units)
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

    def attr_units_map(self, kind: str = "num") -> dict[str, str | None]:
        """
        Return the canonical units the index stores each attr of one kind in.

        Units belong to an (attr name, value kind) pair, and only numbers
        carry any, so the default asks about the numeric kind; an attr the
        index holds in that kind without units maps to None, and one it
        does not hold in that kind is absent. One read of the metadata.
        """
        rows = self._attr_meta()
        rows = rows[rows["value_kind"] == kind]
        return {
            str(name): _normalize_unit(unit)
            for name, unit in zip(rows["attr_name"], rows["units"], strict=True)
        }

    def attr_units(self, name: str) -> dict[str, str | None]:
        """Return the canonical units the index stores one attr's kinds in."""
        rows = self._attr_meta()
        rows = rows[rows["attr_name"] == name]
        return {
            str(kind): _normalize_unit(unit)
            for kind, unit in zip(rows["value_kind"], rows["units"], strict=True)
        }

    def attr_stated_ids(self, name: str, patch_ids=None) -> set[int]:
        """
        Return the ids of the patches which state a value for one attr.

        An attr may be stored in more than one typed column across an
        archive, and a patch states it when any of them holds a value.
        This answers without projecting the relation, so a caller can
        tell whether it needs to look anywhere else for the rest.
        """
        rows = self._attr_meta()
        columns = rows[rows["attr_name"] == name]["column_name"].unique()
        if not len(columns):
            return set()
        stated = " OR ".join(f"{quote(x)} IS NOT NULL" for x in columns)
        sql = f"SELECT patch_id FROM attrs WHERE ({stated})"
        params: list = []
        if patch_ids is not None:
            sql += " AND patch_id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps([int(x) for x in patch_ids]))
        return {int(x) for x in self._fetch_df(sql, params)["patch_id"]}

    def coord_dims_map(self) -> dict[str, str]:
        """Return each coord name's dims string (first observed wins)."""
        df = self._fetch_df("SELECT DISTINCT coord_name, coord_dims FROM patch_coords")
        out: dict[str, str] = {}
        for name, dims in zip(df["coord_name"], df["coord_dims"]):
            out.setdefault(str(name), str(dims))
        return out


def get_backend(path: str | Path) -> SQLiteIndexBackend:
    """
    Create the spool-index backend at path.

    The seam a second engine would reopen: everything above the backend
    asks for one through this rather than naming a class.
    """
    return SQLiteIndexBackend(path)
