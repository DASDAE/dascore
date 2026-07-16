"""Tests for indexing local file systems."""

from __future__ import annotations

import os
import platform
import shutil
from contextlib import suppress
from pathlib import Path

import pandas as pd
import pytest
from upath import UPath

from dascore.config import set_config
from dascore.exceptions import InvalidSpoolError
from dascore.io.index.backend import resolve_query
from dascore.io.index.indexer import DBDirectoryIndexer
from dascore.io.index.schema import SPOOL_HIDDEN_COLUMNS
from dascore.utils.patch import get_patch_names


def index_contents(indexer, **kwargs) -> pd.DataFrame:
    """
    Return an indexer's flat relation, the way a catalog queries it.

    Bare kwargs resolve attrs-first then coords, per the selector
    semantics spec.
    """
    indexer.ensure_updated()
    query = resolve_query(indexer._backend, **kwargs)
    df = indexer._backend.query(query)
    df = df.drop(columns=list(SPOOL_HIDDEN_COLUMNS), errors="ignore")
    return df.rename(columns={"patch_id": "_patch_id"})


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
    return index_contents(diverse_indexer)


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

    @pytest.fixture()
    def directory_indexer_bad_cache(self, tmp_path_factory):
        """Create a bad index_map file."""
        path = tmp_path_factory.mktemp("corrupt_cache_test")
        cache_path = path / "corrupt_cache.json"
        with cache_path.open("wt") as fi:
            fi.write("{'bad': 'json'")
        return cache_path

    def test_directory_cant_write(self, unwritable_directory):
        """Ensure correct path is found when a read-only directory is used."""
        dir_index = DBDirectoryIndexer(unwritable_directory)
        index_path = dir_index.index_path
        index_map_path = dir_index.index_map_path
        assert index_map_path.parent == index_path.parent

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

    def test_corrupt_cache(self, directory_indexer_bad_cache, tmp_path_factory):
        """Ensure a corrupted cache doesn't crash indexing. See #508."""
        path = tmp_path_factory.mktemp("corrupt_cache_test")
        assert directory_indexer_bad_cache.exists()
        with set_config(directory_index_map_path=directory_indexer_bad_cache):
            DBDirectoryIndexer(path)
        assert not directory_indexer_bad_cache.exists()

    def test_remote_directory_not_supported(self):
        """Remote directory indexing should fail fast."""
        path = UPath("memory://dascore/indexer")
        (path / "file.txt").write_text("x")
        with pytest.raises(InvalidSpoolError, match="local filesystem"):
            DBDirectoryIndexer(path)

    def test_local_upath_normalized_to_path(self, tmp_path):
        """Local UPath inputs should normalize to pathlib.Path internally."""
        out = DBDirectoryIndexer(UPath(tmp_path))
        assert isinstance(out.path, Path)
        assert out.path == Path(tmp_path).absolute()

    def test_index_map_path_comes_from_config(self, tmp_path):
        """Index map paths should be sourced from runtime configuration."""
        index_map_path = tmp_path / "cache_paths.json"
        with set_config(directory_index_map_path=index_map_path):
            out = DBDirectoryIndexer(tmp_path)
            assert out.index_map_path == index_map_path


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
        out = index_contents(basic_indexer)
        files = list(Path(two_patch_directory).rglob("*.hdf5"))
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(files)
        names_df = {x.split("/")[-1] for x in out["path"]}
        names_files = {x.name for x in files}
        assert names_df == names_files

    def test_filter_time_after(self, diverse_df, diverse_indexer):
        """Half-open time range keeps every file overlapping it."""
        max_starttime = diverse_df["time_min"].max()
        expected = diverse_df[diverse_df["time_max"] >= max_starttime]
        out = index_contents(diverse_indexer, time=(max_starttime, None))
        assert len(out) == len(expected)

    def test_filter_time_before(self, diverse_df, diverse_indexer):
        """Half-open time range keeps every file overlapping it."""
        min_endtime = diverse_df["time_max"].min()
        expected = diverse_df[diverse_df["time_min"] <= min_endtime]
        out = index_contents(diverse_indexer, time=(None, min_endtime))
        assert len(out) == len(expected)

    def test_filter_station_exact(self, diverse_df, diverse_indexer):
        """Ensure contents can be filtered on an attr."""
        exact_name = diverse_df["station"].unique()[0]
        new_df = index_contents(diverse_indexer, station=exact_name)
        assert (new_df["station"] == exact_name).all()

    def test_filter_isin(self, diverse_df, diverse_indexer):
        """Ensure contents can be filtered with a collection."""
        # empty strings mean "attr missing" and are not queryable (spec).
        stations = [x for x in diverse_df["station"].unique() if x]
        new_df = index_contents(diverse_indexer, station=stations[:2])
        assert set(new_df["station"]) <= set(stations[:2])
        assert len(new_df)

    def test_empty_index(self, empty_index):
        """An empty index should return an empty dataframe."""
        df = index_contents(empty_index)
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
        contents = index_contents(new_index)
        assert len(contents) == 1

    def test_index_with_bad_file(self, spool_directory_with_non_das_file):
        """Ensure if one file is not readable index continues."""
        indexer = DBDirectoryIndexer(spool_directory_with_non_das_file)
        updated = indexer.update(progress=None)
        assert isinstance(updated, DBDirectoryIndexer)
        assert len(index_contents(updated)) == 2

    def test_removed_file_dropped(self, two_patch_directory, tmp_path_factory):
        """A deleted file's rows disappear on the next update."""
        new = tmp_path_factory.mktemp("removed_file_test") / "sub"
        shutil.copytree(two_patch_directory, new)
        for index in Path(new).glob(".dascore_index*"):
            index.unlink()
        indexer = DBDirectoryIndexer(new).update(progress=None)
        assert len(index_contents(indexer)) == 2
        next(iter(Path(new).glob("*.hdf5"))).unlink()
        assert len(index_contents(indexer.update(progress=None))) == 1

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
        from dascore.io.index.query import InvalidSpoolQueryError

        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            index_contents(basic_indexer, bad_dimension=(1, 2))
