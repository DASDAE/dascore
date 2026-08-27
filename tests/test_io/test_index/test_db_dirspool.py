"""
Integration tests: directory spools running on the database index.

Exercises the full path — directory walk, scan, ingest, query, patch
loading, and chunk against real files.
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.spool import Spool
from dascore.examples import spool_to_directory


@pytest.fixture(scope="class")
def spool_directory(tmp_path_factory):
    """A directory of DASDAE files from the random example spool."""
    spool = dc.get_example_spool("random_das")
    return spool_to_directory(spool, path=tmp_path_factory.mktemp("db_spool"))


@pytest.fixture()
def db_spool(spool_directory):
    """A directory spool using its SQLite index."""
    spool = Spool.from_directory(spool_directory)
    out = spool.update(progress=None)
    yield out
    out.indexer.close()


class TestDBDirectorySpools:
    """Directory spools wired to the database index."""

    def test_length(self, db_spool):
        """One entry per patch in the source spool."""
        assert len(db_spool) == 3

    def test_contents_columns(self, db_spool):
        """The contents df carries what the spool machinery needs."""
        df = db_spool.get_contents()
        for col in ("source_path", "time_min", "time_max", "time_step", "dims"):
            assert col in df.columns

    def test_load_patches(self, db_spool):
        """Every indexed patch is loadable and matches the source data."""
        source = dc.get_example_spool("random_das")
        source_starts = {patch.get_coord("time").min() for patch in source}
        loaded_starts = set()
        for patch in db_spool:
            assert patch.data.size > 0
            loaded_starts.add(patch.get_coord("time").min())
        assert loaded_starts == source_starts

    def test_select_time(self, db_spool):
        """Time select narrows the spool."""
        df = db_spool.get_contents()
        t0 = df["time_min"].min().to_datetime64()
        sub = db_spool.select(time=(t0, t0 + np.timedelta64(2, "s")))
        assert len(sub) >= 1
        patch = sub[0]
        assert patch.get_coord("time").max() <= t0 + np.timedelta64(2, "s")

    def test_chunk_merge(self, db_spool):
        """Contiguous patches merge with chunk(time=None)."""
        merged = db_spool.chunk(time=None)
        assert len(merged) == 1
        patch = merged[0]
        assert patch.data.ndim == 2

    def test_update_is_incremental(self, db_spool):
        """A second update with no changes rescans nothing."""
        indexer = db_spool.indexer
        before = indexer._backend.get_sources()["last_indexed_ns"].max()
        db_spool.update(progress=None)
        after = indexer._backend.get_sources()["last_indexed_ns"].max()
        assert before == after


class TestUpdateLifecycle:
    """New, modified, and deleted files are tracked per source."""

    @pytest.fixture()
    def fresh(self, tmp_path):
        """A modifiable spool directory and database spool."""
        spool = dc.get_example_spool("random_das")
        path = spool_to_directory(spool, path=tmp_path / "data")
        out = Spool.from_directory(path).update(progress=None)
        yield path, out
        out.indexer.close()

    def test_new_file_found(self, fresh):
        """A file added after indexing appears on the next update."""
        path, spool = fresh
        patch = dc.get_example_patch()
        patch.io.write(path / "new_file.hdf5", "dasdae")
        updated = spool.update(progress=None)
        assert len(updated) == 4

    def test_deleted_file_removed(self, fresh):
        """A deleted file's rows are dropped on the next update."""
        path, spool = fresh
        target = next(iter(path.glob("*.hdf5")))
        target.unlink()
        updated = spool.update(progress=None)
        assert len(updated) == 2

    def test_no_change_update_uses_narrow_projection(self, fresh, monkeypatch):
        """A no-op update reads only the stat columns, not the wide table."""
        _path, spool = fresh
        backend = spool.indexer._backend

        def _boom(self):
            raise AssertionError("wide get_sources() during no-change update")

        # A no-change update must not fetch the full sources table.
        monkeypatch.setattr(type(backend), "get_sources", _boom)
        reupdated = spool.update(progress=None)
        assert len(reupdated) == 3
