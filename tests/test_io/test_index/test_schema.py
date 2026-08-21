"""Schema and initialization tests for the SQLite spool index."""

from __future__ import annotations

import contextlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from dascore.exceptions import InvalidIndexError, InvalidIndexVersionError
from dascore.io.index.backend import get_backend
from dascore.io.index.schema import INDEX_VERSION, TABLES, TYPE_MAP


class TestSchemaDeclaration:
    """The row classes are the schema; check they reach SQLite intact."""

    def test_stored_columns_match_declaration(self, tmp_path):
        """A created index has each table's declared columns and types."""
        backend = get_backend(tmp_path / "index.sqlite3")
        for table, columns in TABLES.items():
            info = backend._con.execute(f'PRAGMA table_info("{table}")').fetchall()
            stored = {row[1]: row[2] for row in info}
            expected = {name: TYPE_MAP[logical] for name, logical in columns.items()}
            assert stored == expected
        backend.close()


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
        # dims is supplied so the row fails on the foreign key rather than
        # on the NOT NULL check, which would pass with no FK declared.
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            backend._execute(
                "INSERT INTO patches (patch_id, source_id, source_patch_key, dims) "
                "VALUES (1, 999, '0', 'time')"
            )
        backend.close()

    def test_patch_dims_are_never_null(self, tmp_path):
        """The column states an invariant every ingest path already keeps."""
        backend = get_backend(tmp_path / "index.sqlite3")
        backend._execute(
            "INSERT INTO sources (source_id, base_uri, source_path, source_format, "
            "format_version, mtime_ns, size_bytes, path_attrs, last_indexed_ns, "
            "ordinal) VALUES (1, '', 'p', 'DASDAE', '1', 0, 0, NULL, 0, 0)"
        )
        with pytest.raises(sqlite3.IntegrityError, match="dims"):
            backend._execute(
                "INSERT INTO patches (patch_id, source_id, source_patch_key, dims) "
                "VALUES (1, 1, '0', NULL)"
            )
        backend.close()


class TestConcurrentInitialization:
    """Only one writer initializes a new index file."""

    @pytest.mark.concurrency
    def test_concurrent_open(self, tmp_path):
        """Connections racing to create one index all open successfully."""
        path = tmp_path / "shared.sqlite3"
        # Hold the write lock on the empty file. Every opener then gets past
        # its "no tables yet" read and piles up waiting to create them, so
        # exactly one wins and the rest take the re-check branch. Left to
        # chance, a loaded machine runs the four openers one after another
        # and that branch is never reached.
        gate = sqlite3.connect(path, isolation_level=None)
        gate.execute("BEGIN IMMEDIATE")
        ready = Barrier(5)

        def open_index(_):
            # The same read the backend makes first: proof this thread
            # reached the gate while the database was still schema-less.
            with contextlib.closing(sqlite3.connect(path)) as con:
                assert not con.execute("SELECT name FROM sqlite_master").fetchall()
            ready.wait(timeout=60)
            backend = get_backend(path)
            metadata = backend.get_metadata()
            backend.close()
            return metadata["index_version"]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(open_index, num) for num in range(4)]
            ready.wait(timeout=60)
            # The openers only have a connect and a BEGIN left to run; give
            # them that before the lock they are queueing for is released.
            time.sleep(0.05)
            gate.execute("ROLLBACK")
            gate.close()
            versions = [future.result() for future in futures]
        assert versions == [INDEX_VERSION] * 4
        backend = get_backend(path)
        assert len(backend._fetch_df("SELECT * FROM meta_data")) == 1
        backend.close()
