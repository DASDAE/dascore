"""SQLite index backend (stdlib sqlite3, STRICT tables)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from dascore.io.index.backend import SQLIndexBackend, adapt_params
from dascore.io.index.dialect import SQLiteDialect


def _adapt(params):
    """Convert numpy/py types sqlite3 can't bind natively."""
    return [int(p) if isinstance(p, bool) else p for p in adapt_params(params)]


class SQLiteBackend(SQLIndexBackend):
    """Index backend storing tables in a single SQLite file."""

    dialect = SQLiteDialect()

    def __init__(self, path: str | Path):
        self._con = sqlite3.connect(str(path))
        # autocommit off; we manage transactions explicitly.
        self._con.isolation_level = None
        super().__init__()

    def _execute(self, sql: str, params=()) -> None:
        self._con.execute(sql, _adapt(params))

    def _executemany(self, sql: str, seq_of_params) -> None:
        self._con.executemany(sql, [_adapt(p) for p in seq_of_params])

    def _fetch_df(self, sql: str, params=()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self._con, params=_adapt(params))

    def _begin(self) -> None:
        self._con.execute("BEGIN")

    def _commit(self) -> None:
        self._con.execute("COMMIT")

    def _rollback(self) -> None:
        self._con.execute("ROLLBACK")

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()
