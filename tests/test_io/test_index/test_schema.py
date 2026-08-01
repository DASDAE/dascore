"""Schema and initialization tests for the SQLite spool index."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import get_args, get_type_hints

import pytest

from dascore.exceptions import InvalidIndexError, InvalidIndexVersionError
from dascore.io.index import get_backend
from dascore.io.index.schema import INDEX_VERSION, TABLE_ROWS, TABLES

# The python type each logical storage type surfaces as.
_STORAGE_TYPES = {"int64": int, "float64": float, "str": str, "bool": bool}


class TestRowViews:
    """The row views index code reads must match the stored columns."""

    @pytest.mark.parametrize("table", sorted(TABLE_ROWS))
    def test_fields_match_columns(self, table):
        """Each row view declares exactly its table's columns, in order."""
        assert TABLE_ROWS[table]._fields == tuple(TABLES[table])

    @pytest.mark.parametrize("table", sorted(TABLE_ROWS))
    def test_field_types_match_storage(self, table):
        """Each field's type is its column's storage type, nullable or not."""
        hints = get_type_hints(TABLE_ROWS[table])
        for column, storage in TABLES[table].items():
            declared = set(get_args(hints[column])) or {hints[column]}
            assert declared - {type(None)} == {_STORAGE_TYPES[storage]}


class TestSchemaValidation:
    """Existing files are validated without repair or implicit migration."""

    def test_unrelated_database_rejected(self, tmp_path):
        """A SQLite database belonging to another application is rejected."""
        path = tmp_path / "other.sqlite3"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE other_app (value TEXT)")
        con.close()
        with pytest.raises(InvalidIndexError, match="missing tables"):
            get_backend(path)

    def test_old_version_rejected(self, tmp_path):
        """Prototype schemas require an explicit delete and rebuild."""
        path = tmp_path / "old.sqlite3"
        backend = get_backend(path)
        backend._execute("UPDATE meta_data SET index_version = ?", (INDEX_VERSION - 1,))
        backend.close()
        with pytest.raises(InvalidIndexVersionError, match="delete it and rebuild"):
            get_backend(path)

    def test_schema_has_foreign_keys_and_constraints(self, tmp_path):
        """SQLite enforces source/patch ownership and cascades."""
        backend = get_backend(tmp_path / "index.sqlite3")
        assert backend._con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert backend._con.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000
        tables = backend._existing_tables()
        assert set(TABLES) <= tables
        with pytest.raises(sqlite3.IntegrityError):
            backend._execute(
                "INSERT INTO patches (patch_id, source_id, source_patch_id) "
                "VALUES (1, 999, '0')"
            )
        backend.close()


class TestConcurrentInitialization:
    """Only one writer initializes a new index file."""

    @pytest.mark.concurrency
    def test_concurrent_open(self, tmp_path):
        """Connections racing to create one index all open successfully."""
        path = tmp_path / "shared.sqlite3"
        barrier = Barrier(4)

        def open_index(_):
            barrier.wait()
            backend = get_backend(path)
            metadata = backend.get_metadata()
            backend.close()
            return metadata["index_version"]

        with ThreadPoolExecutor(max_workers=4) as pool:
            versions = list(pool.map(open_index, range(4)))
        assert versions == [INDEX_VERSION] * 4
        backend = get_backend(path)
        assert len(backend._fetch_df("SELECT * FROM meta_data")) == 1
        backend.close()
