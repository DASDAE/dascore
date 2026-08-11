"""Tests for indexing local file systems."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from contextlib import suppress
from pathlib import Path

import pandas as pd
import pytest
from upath import UPath

from dascore.config import config_context
from dascore.exceptions import InvalidSpoolError
from dascore.io.index import indexer as indexer_mod
from dascore.io.index.indexer import (
    DBDirectoryIndexer,
    _get_mapped_index_path,
    _map_entry_path,
    _path_digest,
    _set_mapped_index_path,
)
from dascore.io.index.query import InvalidSpoolQueryError
from dascore.utils.patch import get_patch_names


@pytest.fixture(scope="class")
def basic_indexer(two_patch_directory):
    """Return an indexer on the basic spool directory."""
    indexer = DBDirectoryIndexer(two_patch_directory).update(progress=None)
    yield indexer
    indexer.close()


@pytest.fixture(scope="class")
def diverse_indexer(diverse_spool_directory):
    """Return an indexer on the diverse spool directory."""
    indexer = DBDirectoryIndexer(diverse_spool_directory).update(progress=None)
    yield indexer
    indexer.close()


@pytest.fixture(scope="class")
def diverse_df(diverse_indexer):
    """Return the contents of the diverse indexer."""
    return diverse_indexer()


@pytest.fixture()
def empty_index(tmp_path_factory):
    """Create an index around an empty directory."""
    path = tmp_path_factory.mktemp("index_created_test")
    indexer = DBDirectoryIndexer(path).update(progress=None)
    yield indexer
    indexer.close()


class TestFindIndex:
    """Tests for finding the index."""

    @pytest.fixture()
    def unwritable_directory(self, tmp_path_factory):
        """Return an un-writable directory."""
        if "windows" in platform.system().lower():
            pytest.skip("Cant run this test on windows")
        path = tmp_path_factory.mktemp("read_only_data_file")
        os.chmod(path, 0o444)
        yield path
        os.chmod(path, 0o755)

    def test_directory_cant_write(self, unwritable_directory):
        """Ensure correct path is found when a read-only directory is used."""
        dir_index = DBDirectoryIndexer(unwritable_directory)
        index_path = dir_index.index_path
        assert dir_index.index_map_dir == index_path.parent

    def test_read_only_index_name_is_stable(self, unwritable_directory, tmp_path):
        """The read-only fallback index name must not depend on hash().

        hash() of a str/Path is randomized per process, so a hash-derived
        name would differ every session and orphan the prior index. The
        name is a stable digest of the directory path.
        """
        map_dir = tmp_path / "path_map"
        with config_context(directory_index_map_dir=map_dir):
            first = DBDirectoryIndexer(unwritable_directory).index_path
        digest = _path_digest(unwritable_directory)
        assert first.name == f"_dascore_index_{digest}.sqlite3"

    def test_specify_index_path(self, tmp_path_factory):
        """Ensure specifying a Path works and is remembered."""
        data_path = tmp_path_factory.mktemp("data_dir")
        index_path = tmp_path_factory.mktemp("index_dir") / "index.sqlite"
        dir_index = DBDirectoryIndexer(data_path, index_path=index_path)
        assert dir_index.index_path == index_path
        # loading the same data dir should now remember where this is.
        dir_index2 = DBDirectoryIndexer(data_path)
        assert dir_index2.index_path == index_path

    def test_writeable_dir_index_not_there(self, tmp_path_factory):
        """Tests for when there is a writeable directory."""
        path = tmp_path_factory.mktemp("normal_indexer_test")
        dir_indexer = DBDirectoryIndexer(path)
        assert dir_indexer.index_path.parent == path

    def test_writable_dir_index_exists(self, tmp_path_factory):
        """A test case where the index does exist."""
        path = tmp_path_factory.mktemp("normal_indexer_test")
        first = DBDirectoryIndexer(path)
        second = DBDirectoryIndexer(path)
        assert first.index_path == second.index_path
        assert first.index_path.exists()

    def test_corrupt_cache(self, tmp_path):
        """Ensure a corrupt map entry doesn't crash indexing. See #508."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        map_dir = tmp_path / "path_map"
        entry = _map_entry_path(data_dir, map_dir)
        entry.parent.mkdir(parents=True)
        entry.write_text("{'bad': 'json'")
        with config_context(directory_index_map_dir=map_dir):
            indexer = DBDirectoryIndexer(data_dir)
        # The corrupt entry reads as a miss, so the writable data dir keeps
        # its in-directory index rather than crashing.
        assert indexer.index_path.parent == data_dir

    def test_remote_directory_not_supported(self):
        """Remote directory indexing should fail fast."""
        path = UPath("memory://dascore/indexer")
        (path / "file.txt").write_text("x")
        with pytest.raises(InvalidSpoolError, match="local filesystem"):
            DBDirectoryIndexer(path)

    def test_local_upath_normalized_to_path(self, tmp_path):
        """Local UPath inputs should normalize to pathlib.Path internally."""
        out = DBDirectoryIndexer(UPath(tmp_path))
        # A local UPath is itself a Path subclass, so isinstance is not
        # enough to show it was normalized.
        assert type(out.path) is type(Path(tmp_path))
        assert out.path == Path(tmp_path).absolute()

    def test_index_map_dir_comes_from_config(self, tmp_path):
        """Index map dir should be sourced from runtime configuration."""
        index_map_dir = tmp_path / "path_map"
        with config_context(directory_index_map_dir=index_map_dir):
            out = DBDirectoryIndexer(tmp_path)
            assert out.index_map_dir == index_map_dir


class TestIndexMap:
    """Tests for the per-directory index-location entries."""

    def test_digest_falls_back_for_non_fspath(self):
        """A directory os.fsencode rejects still digests (future URL support)."""

        class _Remote:
            """Stand-in for a remote UPath whose fspath is unavailable."""

            def __fspath__(self):
                raise NotImplementedError

            def __str__(self):
                return "memory://data/dir"

        expected = hashlib.sha256(b"memory://data/dir").hexdigest()
        assert _path_digest(_Remote()) == expected

    def test_roundtrip(self, tmp_path):
        """A recorded index path is read back for the same directory."""
        map_dir = tmp_path / "path_map"
        index_path = tmp_path / "idx.sqlite3"
        _set_mapped_index_path(tmp_path / "data", index_path, map_dir)
        assert _get_mapped_index_path(tmp_path / "data", map_dir) == index_path

    def test_missing_entry_is_none(self, tmp_path):
        """An unmapped directory reads back as None (a cache miss)."""
        assert _get_mapped_index_path(tmp_path / "nope", tmp_path / "path_map") is None

    def test_distinct_dirs_dont_collide(self, tmp_path):
        """Separate directories use separate entry files (no lost writes)."""
        map_dir = tmp_path / "path_map"
        _set_mapped_index_path(tmp_path / "a", "index-a", map_dir)
        _set_mapped_index_path(tmp_path / "b", "index-b", map_dir)
        assert _get_mapped_index_path(tmp_path / "a", map_dir) == Path("index-a")
        assert _get_mapped_index_path(tmp_path / "b", map_dir) == Path("index-b")

    def test_corrupt_entry_is_miss(self, tmp_path):
        """A corrupt entry reads as a miss (and is not deleted). See #508."""
        map_dir = tmp_path / "path_map"
        entry = _map_entry_path(tmp_path / "a", map_dir)
        entry.parent.mkdir(parents=True)
        entry.write_text("{not json")
        assert _get_mapped_index_path(tmp_path / "a", map_dir) is None
        # Reads never delete the entry; a later write self-heals it.
        assert entry.exists()

    def test_bad_payload_shapes_are_miss(self, tmp_path):
        """Non-string or empty index paths read as a miss, not an error."""
        map_dir = tmp_path / "path_map"
        entry = _map_entry_path(tmp_path / "a", map_dir)
        entry.parent.mkdir(parents=True)
        entry.write_text(
            json.dumps({"directory": str(tmp_path / "a"), "index_path": []})
        )
        assert _get_mapped_index_path(tmp_path / "a", map_dir) is None

    def test_digest_collision_is_miss(self, tmp_path):
        """An entry whose stored directory differs reads as a miss."""
        map_dir = tmp_path / "path_map"
        entry = _map_entry_path(tmp_path / "a", map_dir)
        entry.parent.mkdir(parents=True)
        # Same file, but recorded for a different directory.
        entry.write_text(json.dumps({"directory": "other", "index_path": "x"}))
        assert _get_mapped_index_path(tmp_path / "a", map_dir) is None

    def test_update_is_atomic_and_leaves_no_temp(self, tmp_path):
        """Writes swap a temp file into place, leaving no debris."""
        map_dir = tmp_path / "path_map"
        _set_mapped_index_path(tmp_path / "a", "1", map_dir)
        _set_mapped_index_path(tmp_path / "a", "2", map_dir)
        assert _get_mapped_index_path(tmp_path / "a", map_dir) == Path("2")
        entry = _map_entry_path(tmp_path / "a", map_dir)
        assert [p.name for p in map_dir.iterdir()] == [entry.name]

    def test_failed_swap_cleans_up_temp(self, tmp_path, monkeypatch):
        """A failure during the atomic swap unlinks the temp file and re-raises."""
        map_dir = tmp_path / "path_map"

        def boom(*args, **kwargs):
            raise RuntimeError("swap failed")

        monkeypatch.setattr(indexer_mod.os, "replace", boom)
        with pytest.raises(RuntimeError, match="swap failed"):
            indexer_mod._set_mapped_index_path(tmp_path / "a", "1", map_dir)
        # No temp debris and no half-written entry left behind.
        assert list(map_dir.iterdir()) == []


class TestWalkResilience:
    """Tests for the filesystem walk tolerating concurrent changes."""

    def test_walk_skips_file_removed_mid_scan(self, tmp_path, monkeypatch):
        """A file vanishing between the walk and its stat is skipped, not fatal."""
        good = tmp_path / "good.h5"
        good.write_bytes(b"")
        # Never created: models a file deleted between the walk yielding it and
        # _walk's stat() call, so its real stat() raises FileNotFoundError.
        vanished = tmp_path / "vanished.h5"
        indexer = DBDirectoryIndexer(tmp_path)

        def fake_iter(*args, **kwargs):
            # Deterministically feed _walk both candidates, independent of the
            # real filesystem, so the stat guard is exercised on the vanished one.
            yield good
            yield vanished

        monkeypatch.setattr(indexer_mod, "_iter_filesystem", fake_iter)
        try:
            walked = indexer._walk()
        finally:
            indexer.close()
        names = {Path(entry[-1]).name for entry in walked.values()}
        assert "good.h5" in names
        assert "vanished.h5" not in names


class TestBasics:
    """Basic tests for indexer."""

    def test_str_repr(self, basic_indexer):
        """Ensure a useful (not the default) str/repr is implemented."""
        out = str(basic_indexer)
        assert "object at" not in out

    def test_metadata(self, basic_indexer):
        """The index records its schema version and identity."""
        meta = basic_indexer._backend.get_metadata()
        assert meta["what_is_this"] == "dascore_spool_index"
        assert meta["index_version"] >= 1


class TestGetContents:
    """Test cases for getting contents of indexer as dataframes."""

    def test_get_contents(self, basic_indexer, two_patch_directory):
        """Ensure contents are returned."""
        out = basic_indexer()
        files = list(Path(two_patch_directory).rglob("*.hdf5"))
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(files)
        names_df = {x.split("/")[-1] for x in out["source_path"]}
        names_files = {x.name for x in files}
        assert names_df == names_files

    def test_filter_time_after(self, diverse_df, diverse_indexer):
        """Half-open time range keeps every file overlapping it."""
        max_starttime = diverse_df["time_min"].max()
        expected = diverse_df[diverse_df["time_max"] >= max_starttime]
        out = diverse_indexer(time=(max_starttime, None))
        assert len(out) == len(expected)

    def test_filter_time_before(self, diverse_df, diverse_indexer):
        """Half-open time range keeps every file overlapping it."""
        min_endtime = diverse_df["time_max"].min()
        expected = diverse_df[diverse_df["time_min"] <= min_endtime]
        out = diverse_indexer(time=(None, min_endtime))
        assert len(out) == len(expected)

    def test_filter_tag_exact(self, diverse_df, diverse_indexer):
        """Ensure contents can be filtered on an attr."""
        # empty strings mean "attr missing" and are not queryable (spec),
        # so an empty result would satisfy the check below for free.
        exact_name = next(x for x in diverse_df["tag"].unique() if x)
        new_df = diverse_indexer(tag=exact_name)
        assert len(new_df)
        assert (new_df["tag"] == exact_name).all()

    def test_filter_isin(self, diverse_df, diverse_indexer):
        """Ensure contents can be filtered with a collection."""
        # empty strings mean "attr missing" and are not queryable (spec).
        tags = [x for x in diverse_df["tag"].unique() if x]
        new_df = diverse_indexer(tag=tags[:2])
        assert set(new_df["tag"]) <= set(tags[:2])
        assert len(new_df)

    def test_empty_index(self, empty_index):
        """An empty index should return an empty dataframe."""
        df = empty_index()
        assert df.empty


class TestUpdate:
    """Tests for updating the index."""

    @pytest.fixture(scope="class")
    def spool_directory_with_non_das_file(self, two_patch_directory, tmp_path_factory):
        """Create a directory with some das files and some non-das files."""
        new = tmp_path_factory.mktemp("unreadable_test") / "sub"
        shutil.copytree(two_patch_directory, new)
        with suppress(FileNotFoundError):
            for index in Path(new).glob(".dascore_index*"):
                index.unlink()
        with open(new / "not_das.open", "w") as fi:
            fi.write("cant be das, can it?")
        return new

    def test_add_one_patch(self, empty_index, random_patch):
        """Ensure a new patch added to the directory shows up."""
        path = empty_index.path / get_patch_names(random_patch).iloc[0]
        random_patch.io.write(path, file_format="dasdae")
        new_index = empty_index.update(progress=None)
        contents = new_index()
        assert len(contents) == 1

    def test_index_with_bad_file(self, spool_directory_with_non_das_file):
        """Ensure if one file is not readable index continues."""
        indexer = DBDirectoryIndexer(spool_directory_with_non_das_file)
        updated = indexer.update(progress=None)
        assert isinstance(updated, DBDirectoryIndexer)
        assert len(updated()) == 2

    def test_removed_file_dropped(self, two_patch_directory, tmp_path_factory):
        """A deleted file's rows disappear on the next update."""
        new = tmp_path_factory.mktemp("removed_file_test") / "sub"
        shutil.copytree(two_patch_directory, new)
        for index in Path(new).glob(".dascore_index*"):
            index.unlink()
        indexer = DBDirectoryIndexer(new).update(progress=None)
        assert len(indexer()) == 2
        next(iter(Path(new).glob("*.hdf5"))).unlink()
        assert len(indexer.update(progress=None)()) == 1

    def test_noop_update_rescans_nothing(self, basic_indexer):
        """Unchanged sources are not rescanned."""
        before = basic_indexer._backend.get_sources()["last_indexed_ns"].max()
        basic_indexer.update(progress=None)
        after = basic_indexer._backend.get_sources()["last_indexed_ns"].max()
        assert before == after

    def test_update_with_specific_paths(self, basic_indexer):
        """Updating with specific paths restricts the rescan."""
        files = sorted(basic_indexer.path.rglob("*.hdf5"))
        assert len(files) >= 2

        def _indexed_times():
            sources = basic_indexer._backend.get_sources().set_index("source_path")
            return sources["last_indexed_ns"].to_dict()

        before = _indexed_times()
        for path in files[:2]:
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        first, second = (basic_indexer._rel(path) for path in files[:2])
        basic_indexer.update(paths=[files[0].name], progress=None)
        after_relative = _indexed_times()
        assert after_relative[first] > before[first]
        assert after_relative[second] == before[second]

        basic_indexer.update(paths=[str(files[1])], progress=None)
        after_absolute = _indexed_times()
        assert after_absolute[first] == after_relative[first]
        assert after_absolute[second] > after_relative[second]


class TestNameResolution:
    """Unknown names raise per the selector spec."""

    def test_unknown_name_raises(self, basic_indexer):
        """Names in neither namespace error clearly (#435)."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            basic_indexer(bad_dimension=(1, 2))
