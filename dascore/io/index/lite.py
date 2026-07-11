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
        self._con.execute("PRAGMA foreign_keys = ON")
        self._con.execute("PRAGMA busy_timeout = 30000")
        try:
            super().__init__()
        except Exception:
            self._con.close()
            raise

    def _execute(self, sql: str, params=()) -> None:
        self._con.execute(sql, _adapt(params))

    def _executemany(self, sql: str, seq_of_params) -> None:
        self._con.executemany(sql, [_adapt(p) for p in seq_of_params])

    def _fetch_df(self, sql: str, params=()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self._con, params=_adapt(params))

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
        self._con.close()
