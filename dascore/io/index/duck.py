"""DuckDB index backend."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dascore.io.index.backend import SQLIndexBackend, adapt_params
from dascore.io.index.dialect import DuckDBDialect


def duck_bulk_insert(con, dialect, table: str, columns: tuple, rows: list) -> None:
    """
    Bulk-insert rows through a registered dataframe.

    DuckDB's executemany binds row-at-a-time in Python (about 60x slower
    than SQLite for ingest); routing bulk rows through its dataframe
    scanner keeps ingest columnar.
    """
    if not rows:
        return
    df = pd.DataFrame(
        [adapt_params(r) for r in rows], columns=list(columns), dtype=object
    )
    con.register("_bulk_rows", df)
    try:
        quoted = ", ".join(dialect.quote(c) for c in columns)
        con.execute(
            f"INSERT INTO {dialect.quote(table)} ({quoted}) " "SELECT * FROM _bulk_rows"
        )
    finally:
        con.unregister("_bulk_rows")


class DuckDBBackend(SQLIndexBackend):
    """Index backend storing tables in a single DuckDB file."""

    dialect = DuckDBDialect()

    def __init__(self, path: str | Path, read_only: bool = False):
        import duckdb

        self._con = duckdb.connect(str(path), read_only=read_only)
        super().__init__()

    def _execute(self, sql: str, params=()) -> None:
        self._con.execute(sql, adapt_params(params))

    def _executemany(self, sql: str, seq_of_params) -> None:
        rows = [adapt_params(p) for p in seq_of_params]
        if rows:
            self._con.executemany(sql, rows)

    def _fetch_df(self, sql: str, params=()) -> pd.DataFrame:
        return self._con.execute(sql, adapt_params(params)).df()

    def _bulk_insert(self, table: str, columns: tuple, rows: list) -> None:
        duck_bulk_insert(self._con, self.dialect, table, columns, rows)

    def _begin(self) -> None:
        self._con.execute("BEGIN TRANSACTION")

    def _commit(self) -> None:
        self._con.execute("COMMIT")

    def _rollback(self) -> None:
        self._con.execute("ROLLBACK")

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()
