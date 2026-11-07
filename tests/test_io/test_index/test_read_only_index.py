"""Tests for spools opened from an index, including read-only and remote ones."""

from __future__ import annotations

import pickle
import shutil
import sqlite3

import pytest
from upath import UPath

import dascore as dc
from dascore.config import config_context
from dascore.constants import INDEX_NAME
from dascore.exceptions import (
    InvalidIndexVersionError,
    InvalidSpoolError,
    ParameterError,
)
from dascore.io.index.schema import INDEX_VERSION
from dascore.utils.misc import suppress_warnings
from dascore.utils.remote_io import clear_remote_file_cache


def _assert_same_spool(spool1, spool2):
    """Both spools hold equal patches in the same order."""
    assert len(spool1) == len(spool2)
    for patch1, patch2 in zip(spool1, spool2, strict=True):
        assert patch1.equals(patch2)


@pytest.fixture()
def data_dir(tmp_path):
    """An indexed directory of example files."""
    path = tmp_path / "data"
    dc.examples.spool_to_directory(dc.get_example_spool(), path)
    dc.spool(path).update()
    return path


class TestIndexArgument:
    """The index keyword of dc.spool on local paths."""

    def test_directory_with_index(self, data_dir, tmp_path):
        """A local directory with an index elsewhere indexes it there."""
        index = tmp_path / "elsewhere.sqlite3"
        spool = dc.spool(data_dir, index=index).update()
        assert spool.indexer.index_path == index
        assert index.exists()
        _assert_same_spool(spool, dc.spool(data_dir))

    def test_index_path_deprecated(self, data_dir, tmp_path):
        """index_path still works as index does, with a deprecation warning."""
        index = tmp_path / "elsewhere.sqlite3"
        with pytest.warns(DeprecationWarning, match="index_path"):
            spool = dc.spool(data_dir, index_path=index).update()
        assert spool.indexer.index_path == index
        _assert_same_spool(spool, dc.spool(data_dir, index=index))
        with pytest.raises(ParameterError, match="not both"):
            dc.spool(data_dir, index=index, index_path=index)

    def test_index_alone_reads_its_directory(self, data_dir):
        """An index alone resolves its rows against its own directory."""
        spool = dc.spool(index=data_dir / INDEX_NAME)
        _assert_same_spool(spool, dc.spool(data_dir))

    def test_relocated_index(self, data_dir, tmp_path):
        """An index copied with its data reads the copy, not the original."""
        moved = tmp_path / "moved"
        shutil.copytree(data_dir, moved)
        expected = list(dc.spool(data_dir))
        shutil.rmtree(data_dir)
        _assert_same_spool(dc.spool(index=moved / INDEX_NAME), expected)

    def test_read_only_never_scans(self, data_dir):
        """A read-only spool neither scans new files nor writes its index."""
        index = data_dir / INDEX_NAME
        before = index.read_bytes()
        dc.write(dc.get_example_patch(), data_dir / "new.h5", "DASDAE")
        spool = dc.spool(index=index)
        assert len(spool) == len(dc.get_example_spool())
        with pytest.raises(InvalidSpoolError, match="read-only"):
            spool.update()
        assert index.read_bytes() == before

    def test_pickled_spool_stays_read_only(self, data_dir):
        """A pickled read-only spool reopens the same index read-only."""
        spool = dc.spool(index=data_dir / INDEX_NAME)[1:]
        out = pickle.loads(pickle.dumps(spool))
        _assert_same_spool(out, spool)
        assert out._catalog.read_only

    def test_other_snap_raises(self, data_dir):
        """Opening an index under another snap raises rather than rescanning."""
        with pytest.raises(ParameterError, match="snap"):
            dc.spool(index=data_dir / INDEX_NAME, snap=False)
        with config_context(max_shift=0.25):
            with pytest.raises(ParameterError, match="max_shift"):
                dc.spool(index=data_dir / INDEX_NAME)

    @pytest.mark.parametrize("version", [1, INDEX_VERSION + 1])
    def test_other_index_version_raises(self, data_dir, version):
        """An older or newer index version raises, naming both versions."""
        index = data_dir / INDEX_NAME
        con = sqlite3.connect(index)
        con.execute(f"UPDATE meta_data SET index_version = {version}")
        con.commit()
        con.close()
        match = rf"version {version} .*version {INDEX_VERSION}"
        with pytest.raises(InvalidIndexVersionError, match=match):
            dc.spool(index=index)

    def test_bad_arguments_raise(self, data_dir, tmp_path):
        """No source, a non-path source, a file root, or no index raise."""
        index = data_dir / INDEX_NAME
        with pytest.raises(ParameterError, match="data source"):
            dc.spool()
        with pytest.raises(ParameterError, match="data root"):
            dc.spool(dc.get_example_patch(), index=index)
        with pytest.raises(InvalidSpoolError, match="not a directory"):
            dc.spool(next(data_dir.glob("*.hdf5")), index=index)
        with pytest.raises(FileNotFoundError):
            dc.spool(index=tmp_path / "missing.sqlite3")


@pytest.mark.network
@pytest.mark.timeout(30)
class TestRemoteIndex:
    """Spools over an index or data served over HTTP are read-only."""

    @pytest.fixture(autouse=True)
    def isolated_remote_cache(self, tmp_path):
        """Use one isolated remote cache, and keep its warnings quiet."""
        msg = "Downloading remote file .* to local cache at .*"
        cache = tmp_path / "remote_cache"
        with (
            config_context(remote_cache_dir=cache, warn_on_gc_pause=False),
            suppress_warnings(message=msg),
        ):
            clear_remote_file_cache()
            yield
            clear_remote_file_cache()

    @pytest.fixture(scope="class")
    def served(self, http_test_data_root, http_range_das_path):
        """An indexed directory, its local path and its URL."""
        local = http_test_data_root / "das" / "read_only_index"
        if not local.exists():
            dc.examples.spool_to_directory(dc.get_example_spool(), local)
            dc.spool(local).update()
        return local, http_range_das_path / "read_only_index"

    def test_remote_root_local_index(self, served):
        """A local index reads its patches from a remote root."""
        local, url = served
        spool = dc.spool(url, index=local / INDEX_NAME)
        _assert_same_spool(spool, dc.spool(local))

    def test_remote_index(self, served):
        """A remote index reads from its URL's parent and cannot update."""
        local, url = served
        spool = dc.spool(index=url / INDEX_NAME)
        _assert_same_spool(spool, dc.spool(local))
        with pytest.raises(InvalidSpoolError, match="read-only"):
            spool.update()

    def test_combined_remote_spool(self, served):
        """A remote spool combined with another still reads its rows by URL."""
        local, url = served
        extra = dc.spool([dc.get_example_patch()])
        combined = dc.spool(index=url / INDEX_NAME) + extra
        _assert_same_spool(combined, dc.spool(local) + extra)

    def test_root_keeps_index_storage_options(self, served):
        """The default root keeps the index's storage options, uncached by name."""
        local, url = served
        headers = {"Authorization": "Bearer secret-token"}
        index = UPath(str(url / INDEX_NAME), headers=headers)
        spool = dc.spool(index=index)
        assert spool._catalog.resolver._root.storage_options["headers"] == headers
        assert "secret-token" not in spool._catalog.backend._path
        assert "secret-token" not in repr(spool)
        _assert_same_spool(spool, dc.spool(local))
