"""Schema and initialization tests for the SQLite spool index."""

from __future__ import annotations

import contextlib
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import h5py
import numpy as np
import pytest

import dascore as dc
import dascore.io.silixah5.utils as silixa_utils
from dascore.exceptions import InvalidIndexError, InvalidIndexVersionError
from dascore.io.index.backend import get_backend
from dascore.io.index.schema import INDEX_VERSION, TABLES, TYPE_MAP
from dascore.utils.downloader import fetch


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


@pytest.fixture(scope="module")
def dasdae_spool(tmp_path_factory):
    """A directory of a one-patch and a two-patch DASDAE file."""
    path = tmp_path_factory.mktemp("stored_shapes")
    dc.write(dc.get_example_patch(), path / "one.h5", "DASDAE")
    dc.write(dc.get_example_spool("random_das", length=2), path / "two.h5", "DASDAE")
    return dc.spool(path).update(progress=None)


@pytest.fixture(scope="module")
def silixa_spool(tmp_path_factory):
    """Two consecutive gzip Silixa files, so a merge reads members."""
    path = tmp_path_factory.mktemp("stored_shapes_silixa")
    for num, stamp in enumerate(("26.158", "26.358")):
        out = path / f"{num}.hdf5"
        shutil.copy(fetch("silixa_h5_1.hdf5"), out)
        with h5py.File(out, "a") as h5:
            attrs, data = dict(h5["Acoustic"].attrs), h5["Acoustic"][()]
            del h5["Acoustic"]
            dataset = h5.create_dataset("Acoustic", data=data, compression="gzip")
            attrs["GPSTimeStamp"] = f"22/09/2023 18:29:{stamp} (UTC)"
            dataset.attrs.update(attrs)
    return dc.spool(path).update(progress=None)


def _stored_keys(spool) -> dict[str, list[str]]:
    """The source_patch_key the contents state for each source, by file name."""
    out = {}
    frame = spool.get_contents()[["source_path", "source_patch_key"]]
    for path, key in frame.itertuples(index=False):
        out.setdefault(str(path).replace("\\", "/").rsplit("/", 1)[-1], []).append(key)
    return out


class TestStoredShapes:
    """The index stores nothing it can derive; pin the shapes it saves."""

    def test_coord_data_id_from_def_key(self, dasdae_spool):
        """
        coord_defs has no data_id column; the id still round-trips.

        An engine test (rule E) of the index's tables and export records.
        """
        backend = dasdae_spool._catalog.backend
        info = backend._con.execute("PRAGMA table_info(coord_defs)")
        assert "data_id" not in {row[1] for row in info}
        expected = dc.get_example_patch().get_coord("time").data_id
        records = backend.export_records()
        ids = {c.data_id for r in records for p in r.patches for c in p.coords}
        assert expected in ids

    def test_only_several_patches_need_keys(self, dasdae_spool):
        """A one-patch source stores no key; a two-patch source keeps both."""
        keys = _stored_keys(dasdae_spool)
        assert keys["one.h5"] == [""]
        assert len(keys["two.h5"]) == 2 and all(keys["two.h5"])

    def test_dataset_path_key_kept(self, silixa_spool):
        """
        An HDF5 dataset path is kept, so a merge reads without rescans.

        A performance guard (rule C): a rescan reads the same patch, so the
        attribute reads are counted.
        """
        assert list(_stored_keys(silixa_spool).values()) == [["/Acoustic"]] * 2
        calls = []
        original = silixa_utils._get_attr_dict

        def counted(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(silixa_utils, "_get_attr_dict", counted)
            data = np.asarray(silixa_spool.chunk(time=None)[0].data)
        assert data.shape[0] == 200
        assert not calls

    def test_patch_coords_without_rowid(self, tmp_path):
        """patch_coords is clustered on its key, so the key is stored once."""
        backend = get_backend(tmp_path / "index.sqlite3")
        (sql,) = backend._con.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'patch_coords'"
        ).fetchone()
        assert "WITHOUT ROWID" in sql
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

    def test_missing_table_rejected(self, tmp_path):
        """A current index missing one of its tables asks to be rebuilt."""
        path = tmp_path / "index.sqlite3"
        get_backend(path).close()
        con = sqlite3.connect(path)
        con.execute("DROP TABLE coord_variants")
        con.commit()
        con.close()
        with pytest.raises(InvalidIndexError, match=r"missing tables \['coord_var"):
            get_backend(path)

    def test_missing_trigger_rejected(self, tmp_path):
        """An index which lost a counting trigger would miscount; it is refused."""
        path = tmp_path / "index.sqlite3"
        get_backend(path).close()
        con = sqlite3.connect(path)
        con.execute("DROP TRIGGER count_patch_added")
        con.commit()
        con.close()
        with pytest.raises(InvalidIndexError, match="missing triggers"):
            get_backend(path)

    def test_old_version_rejected(self, tmp_path):
        """Prototype schemas require an explicit delete and rebuild."""
        path = tmp_path / "old.sqlite3"
        backend = get_backend(path)
        backend._execute("UPDATE meta_data SET index_version = ?", (INDEX_VERSION - 1,))
        backend.close()
        with pytest.raises(InvalidIndexVersionError, match="delete it and rebuild"):
            get_backend(path)

    def test_version_41_rejected(self, tmp_path):
        """A version 41 index states float grids by their ends; it is rebuilt."""
        path = tmp_path / "old.sqlite3"
        backend = get_backend(path)
        backend._execute("UPDATE meta_data SET index_version = 41")
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
                "INSERT INTO patches (patch_row, source_row, source_patch_key, dims) "
                "VALUES (1, 999, '0', 'time')"
            )
        backend.close()

    def test_patch_dims_are_never_null(self, tmp_path):
        """The column states an invariant every ingest path already keeps."""
        backend = get_backend(tmp_path / "index.sqlite3")
        backend._execute(
            "INSERT INTO sources (source_row, base_uri, source_path, source_format, "
            "format_version, mtime_ns, size_bytes, path_attrs, last_indexed_ns, "
            "ordinal) VALUES (1, '', 'p', 'DASDAE', '1', 0, 0, NULL, 0, 0)"
        )
        with pytest.raises(sqlite3.IntegrityError, match="dims"):
            backend._execute(
                "INSERT INTO patches (patch_row, source_row, source_patch_key, dims) "
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
