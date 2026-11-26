"""Tests for spools opened from an index, including read-only and remote ones."""

from __future__ import annotations

import pickle
import shutil
import sqlite3
from http import HTTPStatus

import pytest
from upath import UPath

import dascore as dc
from dascore.config import config_context
from dascore.constants import INDEX_NAME
from dascore.exceptions import (
    InvalidIndexError,
    InvalidIndexVersionError,
    InvalidSpoolError,
    ParameterError,
)
from dascore.io.index.schema import INDEX_VERSION
from dascore.utils.misc import suppress_warnings
from dascore.utils.remote_io import clear_remote_file_cache
from tests.test_io.conftest import _RangeHTTPRequestHandler, _serve_das_tree

TOKEN = "Bearer secret-token"


class _TokenHTTPRequestHandler(_RangeHTTPRequestHandler):
    """A range server which refuses the index tree without the test's token."""

    def send_head(self):
        """Serve the index tree only to requests carrying the token."""
        guarded = self.path.startswith("/das/read_only_index")
        if guarded and self.headers.get("Authorization") != TOKEN:
            self.send_error(HTTPStatus.UNAUTHORIZED, "No token")
            return None
        return super().send_head()


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


@pytest.mark.wasm_skip
class TestIndexArgument:
    """The index keyword of dc.spool on local paths."""

    def test_index_path_deprecated(self, data_dir, tmp_path):
        """index_path still works as index does, with a deprecation warning."""
        index = tmp_path / "elsewhere.sqlite3"
        with pytest.warns(DeprecationWarning, match="index_path"):
            spool = dc.spool(data_dir, index_path=index).update()
        assert spool.indexer.index_path == index
        _assert_same_spool(spool, dc.spool(data_dir, index=index))
        # a file:// URI names the same local index
        assert dc.spool(data_dir, index=index.as_uri()).indexer.index_path == index
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

    def test_remote_root(self, data_dir):
        """A remote root reads the index's rows, also once combined or chunked."""
        root = UPath(f"memory://read_only_root/{data_dir.parent.name}")
        for path in data_dir.glob("*.hdf5"):
            (root / path.name).write_bytes(path.read_bytes())
        spool = dc.spool(root, index=data_dir / INDEX_NAME)
        expected = dc.spool(data_dir)
        extra = dc.spool([dc.get_example_patch()])
        _assert_same_spool(spool, expected)
        _assert_same_spool(spool + extra, expected + extra)
        _assert_same_spool(spool.chunk(time=None), expected.chunk(time=None))

    def test_read_only_never_scans(self, data_dir):
        """
        A read-only spool neither scans new files nor writes its index.

        The last check asks the index connection itself (rule E): no write
        through the spool reaches it to be refused.
        """
        index = data_dir / INDEX_NAME
        before = index.read_bytes()
        dc.write(dc.get_example_patch(), data_dir / "new.h5", "DASDAE")
        spool = dc.spool(index=index)
        assert len(spool) == len(dc.get_example_spool())
        with pytest.raises(InvalidSpoolError, match="read-only"):
            spool.update()
        assert index.read_bytes() == before
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            spool._catalog.backend._con.execute("CREATE TABLE t (a)")

    def test_relative_index(self, data_dir, monkeypatch):
        """A relative index keeps reading its files after the cwd changes."""
        monkeypatch.chdir(data_dir.parent)
        spool = dc.spool(index=f"{data_dir.name}/{INDEX_NAME}")
        monkeypatch.chdir(data_dir)
        _assert_same_spool(spool, dc.spool(data_dir))

    def test_unfinished_index_raises(self, tmp_path):
        """An index whose first update never finished raises, not reads empty."""
        path = tmp_path / "data"
        dc.examples.spool_to_directory(dc.get_example_spool(), path)
        dc.spool(path)  # creates the index without scanning
        with pytest.raises(InvalidIndexError, match="first update"):
            dc.spool(index=path / INDEX_NAME)

    def test_pickled_spool_stays_read_only(self, data_dir):
        """A pickled read-only spool reopens the same index read-only."""
        spool = dc.spool(index=data_dir / INDEX_NAME)[1:]
        out = pickle.loads(pickle.dumps(spool))
        _assert_same_spool(out, spool)
        with pytest.raises(InvalidSpoolError, match="read-only"):
            out.update()

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

    def test_changed_remote_index_is_fetched_again(self, data_dir, tmp_path):
        """A remote index which changed is downloaded again when reopened."""
        index = UPath("memory://read_only_index/changed.sqlite3")
        index.write_bytes((data_dir / INDEX_NAME).read_bytes())
        msg = "Downloading remote file"
        with (
            config_context(remote_cache_dir=tmp_path / "cache"),
            suppress_warnings(message=msg),
        ):
            assert len(dc.spool(data_dir, index=index)) == 3
            dc.write(dc.get_example_patch(), data_dir / "new.h5", "DASDAE")
            index.write_bytes(
                dc.spool(data_dir).update().indexer.index_path.read_bytes()
            )
            assert len(dc.spool(data_dir, index=index)) == 4

    def test_bad_arguments_raise(self, data_dir, tmp_path):
        """No source, a non-path source, a file root, or no index raise."""
        index = data_dir / INDEX_NAME
        with pytest.raises(ParameterError, match="data source"):
            dc.spool()
        with pytest.raises(ParameterError, match="data root"):
            dc.spool(dc.get_example_patch(), index=index)
        with pytest.raises(InvalidSpoolError, match="not a directory"):
            dc.spool(next(data_dir.glob("*.hdf5")), index=index)
        with pytest.raises(InvalidSpoolError, match="not a directory"):
            dc.spool(tmp_path / "missing", index="https://host/index.sqlite3")
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

    @pytest.fixture()
    def token_url(self, served, http_test_data_root):
        """The served index's URL on a server which demands a token."""
        with _serve_das_tree(
            _TokenHTTPRequestHandler, http_test_data_root, "token server"
        ) as path:
            yield path / "read_only_index"

    def test_storage_options_reach_every_read(self, served, token_url):
        """
        A remote index's storage options carry to its downloads and reads,
        so a server refusing requests without its token serves them all.
        """
        local, _ = served
        headers = {"Authorization": TOKEN}
        with pytest.raises(FileNotFoundError):  # the server refuses bare reads
            dc.spool(index=token_url / INDEX_NAME)
        spool = dc.spool(index=UPath(str(token_url / INDEX_NAME), headers=headers))
        extra = dc.spool([dc.get_example_patch()])
        expected = dc.spool(local)
        _assert_same_spool(spool, expected)
        _assert_same_spool((spool + extra)[:-1], expected)
        _assert_same_spool(spool.chunk(time=None), expected.chunk(time=None))
        assert "secret-token" not in repr(spool)
