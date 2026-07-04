"""
Parquet-manifest index backend.

Tables live as immutable parquet files plus a small manifest; readers
never need locks (a half-written update is invisible until the manifest
swap). This prototype materializes the tables in an in-memory DuckDB for
querying/mutation and dumps changed tables to new parquet files on
commit, replacing the manifest atomically.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pandas as pd

from dascore.io.index.backend import SQLIndexBackend, adapt_params
from dascore.io.index.dialect import DuckDBDialect
from dascore.io.index.schema import TABLES

_MANIFEST = "manifest.json"


class ParquetBackend(SQLIndexBackend):
    """Index backend storing tables as parquet files + manifest."""

    dialect = DuckDBDialect()

    def __init__(self, path: str | Path):
        import duckdb

        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(":memory:")
        self._manifest = self._read_manifest()
        for table, filename in self._manifest.get("tables", {}).items():
            file_path = str(self._dir / filename).replace("'", "''")
            self._con.execute(
                f"CREATE TABLE {self.dialect.quote(table)} AS "
                f"SELECT * FROM read_parquet('{file_path}')"
            )
        super().__init__()

    def _read_manifest(self) -> dict:
        manifest_path = self._dir / _MANIFEST
        if manifest_path.exists():
            with manifest_path.open() as fi:
                return json.load(fi)
        return {"tables": {}}

    # --- SQL hooks (all against the in-memory duckdb) -----------------

    def _execute(self, sql: str, params=()) -> None:
        self._con.execute(sql, adapt_params(params))

    def _executemany(self, sql: str, seq_of_params) -> None:
        rows = [adapt_params(p) for p in seq_of_params]
        if rows:
            self._con.executemany(sql, rows)

    def _fetch_df(self, sql: str, params=()) -> pd.DataFrame:
        return self._con.execute(sql, adapt_params(params)).df()

    def _bulk_insert(self, table: str, columns: tuple, rows: list) -> None:
        from dascore.io.index.duck import duck_bulk_insert

        duck_bulk_insert(self._con, self.dialect, table, columns, rows)

    def _begin(self) -> None:
        self._con.execute("BEGIN TRANSACTION")

    def _commit(self) -> None:
        self._con.execute("COMMIT")
        self._persist()

    def _rollback(self) -> None:
        self._con.execute("ROLLBACK")

    # --- persistence ---------------------------------------------------

    def _persist(self) -> None:
        """Write all tables to new parquet files and swap the manifest."""
        new_tables = {}
        for table in TABLES:
            filename = f"{table}-{uuid.uuid4().hex[:12]}.parquet"
            target = str(self._dir / filename).replace("'", "''")
            self._con.execute(
                f"COPY {self.dialect.quote(table)} TO '{target}' (FORMAT PARQUET)"
            )
            new_tables[table] = filename
        old = self._manifest.get("tables", {})
        self._manifest = {"tables": new_tables}
        tmp = self._dir / (_MANIFEST + ".tmp")
        with tmp.open("w") as fi:
            json.dump(self._manifest, fi)
        os.replace(tmp, self._dir / _MANIFEST)
        # best-effort cleanup of superseded files (readers using the old
        # manifest may still hold them open; deletion failing is fine).
        for filename in old.values():
            try:
                (self._dir / filename).unlink(missing_ok=True)
            except OSError:
                pass

    def _ensure_schema(self) -> None:
        super()._ensure_schema()
        if not (self._dir / _MANIFEST).exists():
            self._persist()

    def close(self) -> None:
        """Close the in-memory database."""
        self._con.close()
