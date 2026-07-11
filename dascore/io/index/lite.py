"""SQLite index backend (stdlib sqlite3, STRICT tables)."""

from __future__ import annotations

import sqlite3
import weakref
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd

from dascore.io.index.backend import SQLIndexBackend, adapt_params
from dascore.io.index.dialect import SQLiteDialect

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


class SQLiteBackend(SQLIndexBackend):
    """Index backend storing tables in a single SQLite file."""

    dialect = SQLiteDialect()

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
        try:
            super().__init__()
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

    def _execute(self, sql: str, params=()) -> None:
        self._con.execute(sql, _adapt(params))

    def _executemany(self, sql: str, seq_of_params) -> None:
        # sqlite3.executemany consumes an iterator, so adapt lazily rather
        # than materializing a second copy of each already-built batch.
        self._con.executemany(sql, (_adapt(p) for p in seq_of_params))

    def _fetch_df(self, sql: str, params=()) -> pd.DataFrame:
        # numpy_nullable assembly keeps nullable INTEGER columns exact;
        # the default path rounds them through float64, corrupting ns
        # epochs (>2**53). A dtype= hint does NOT prevent that: pandas
        # builds float64 first and casts after.
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
        sql = f"PRAGMA table_info({self.dialect.quote(table)})"
        return {row[1] for row in self._con.execute(sql).fetchall()}

    def close(self) -> None:
        """Close the database connection."""
        # detach the GC finalizer; closing twice is harmless but tidy.
        self._finalizer.detach()
        _safe_close(self._con)
