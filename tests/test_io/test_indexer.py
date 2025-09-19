"""Tests for indexing local file systems."""

from __future__ import annotations

import os
import platform
import shutil
from contextlib import suppress
from pathlib import Path

import pandas as pd
import pytest
from packaging.version import parse as get_version

import dascore as dc
from dascore.examples import spool_to_directory
from dascore.io.indexer import DirectoryIndexer
from dascore.utils.patch import get_patch_names


@pytest.fixture(scope="class")
def basic_indexer(two_patch_directory):
    """Return and indexer on the basic spool directory."""
    return DirectoryIndexer(two_patch_directory)


@pytest.fixture(scope="class")
def adjacent_indexer(adjacent_spool_directory):
    """Return and indexer on the basic spool directory."""
    return DirectoryIndexer(adjacent_spool_directory).update()


@pytest.fixture(scope="class")
def diverse_indexer(diverse_spool_directory):
    """Return and indexer on the basic spool directory."""
    return DirectoryIndexer(diverse_spool_directory).update()


@pytest.fixture(scope="class")
def diverse_df(diverse_indexer):
    """Return the contents of the diverse indexer."""
    return diverse_indexer()


@pytest.fixture()
def diverse_df_reset_cache(diverse_indexer):
    """Return the indexer with a reset cache."""
    return DirectoryIndexer(diverse_indexer.path)


@pytest.fixture(params=[diverse_indexer, diverse_df_reset_cache])
def diverse_ind(request):
    """Aggregate the diverse indexers."""
    return request.getfixturevalue(request.param.__name__)


@pytest.fixture()
def empty_index(tmp_path_factory):
    """Create an index around an empty directory."""
    path = tmp_path_factory.mktemp("index_created_test")
    return DirectoryIndexer(path).update()


class TestFindIndex:
    """Tests for finding the index."""

    @pytest.fixture()
    def unwritable_directory(self, tmp_path_factory):
        """Return an un-writable directory."""
        # currently this doesn't work on windows so we need to skip any test
        # that depend on this fixture if running on windows
        if "windows" in platform.system().lower():
            pytest.skip("Cant run this test on windows")
        path = tmp_path_factory.mktemp("read_only_data_file")
        os.chmod(path, 0o444)
        return path

    @pytest.fixture()
    def directory_indexer_bad_cache(self, tmp_path_factory):
        """Create a subclass of indexer which has a bd index_map file."""
        path = tmp_path_factory.mktemp("corrupt_cache_test")
        cache_path = path / "corrupt_cache.json"

        with cache_path.open("wt") as fi:
            fi.write("{'bad': 'json'")

        class SubIndexer(DirectoryIndexer):
            index_map_path = cache_path

        return SubIndexer

    def test_directory_cant_write(self, unwritable_directory):
        """Ensure correct path is found when a read-only directory is used."""
        dir_index = DirectoryIndexer(unwritable_directory)
        index_path = dir_index.index_path
        index_map_path = dir_index.index_map_path
        assert index_map_path.parent == index_path.parent

    def test_specify_index_path(self, tmp_path_factory):
        """Ensure specifying a Path works."""
        data_path = tmp_path_factory.mktemp("data_dir")
        index_path = tmp_path_factory.mktemp("index_dir") / "index.h5"
        dir_index = DirectoryIndexer(data_path, index_path=index_path)
        assert dir_index.index_path == index_path
        # loading a new data dir should now remember where this is.
        dir_index2 = DirectoryIndexer(data_path)
        assert dir_index2.index_path == index_path

    def test_writeable_dir_index_not_there(self, tmp_path_factory):
        """Tests for when there is writeable directory."""
        path = tmp_path_factory.mktemp("normal_indexer_test")
        dir_indexer = DirectoryIndexer(path)
        assert dir_indexer.index_path.parent == path

    def test_writable_dir_index_exists(self, tmp_path_factory):
        """A test case where the index does exist."""
        path = tmp_path_factory.mktemp("normal_indexer_test")
        index_path = path / DirectoryIndexer._index_name
        index_path.open("w").close()
        dir_indexer = DirectoryIndexer(path)
        assert dir_indexer.index_path == index_path

    def test_corrupt_cache(
        self,
        directory_indexer_bad_cache,
        tmp_path_factory,
    ):
        """Ensure a corrupted cache doesnt crash indexing. See #508."""
        path = tmp_path_factory.mktemp("corrupt_cache_test")
        # Test passes if this doesn't raise should not raise.
        assert directory_indexer_bad_cache.index_map_path.exists()
        directory_indexer_bad_cache(path)
        assert not directory_indexer_bad_cache.index_map_path.exists()


class TestBasics:
    """Basic tests for indexer."""

    def test_str_repr(self, basic_indexer):
        """Ensure a useful (not the default) str/repr is implemented."""
        out = str(basic_indexer)
        assert "object at" not in out

    def test_version(self, basic_indexer):
        """Ensure the version written to file is correct."""
        updated = basic_indexer.update()
        index_version = updated._index_table._index_version
        assert index_version == dc.__last_version__
        assert get_version(index_version) > get_version("0.0.1")


class TestGetContents:
    """Test cases for getting contents of indexer as dataframes."""

    def test_get_contents(self, basic_indexer, two_patch_directory):
        """Ensure contents are returned."""
        out = basic_indexer()
        files = list(Path(two_patch_directory).rglob("*.hdf5"))
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(files)
        names_df = {x.split("/")[-1] for x in out["path"]}
        names_files = {x.name for x in files}
        assert names_df == names_files

    def test_filter_large_starttime(self, diverse_df, diverse_ind):
        """Ensure the index can be filtered by end time."""
        max_starttime = diverse_df["time_min"].max()
        filtered = diverse_df[diverse_df["time_min"] >= max_starttime]
        out = diverse_ind(time_min=max_starttime)
        assert len(out) == len(filtered)

    def test_filter_small_starttime(self, diverse_df, diverse_ind):
        """Ensure the index can be filtered by start time."""
        min_endtime = diverse_df["time_max"].min()
        filtered = diverse_df[diverse_df["time_max"] <= min_endtime]
        out = diverse_ind(time_max=min_endtime)
        assert len(out) == len(filtered)

    def test_filter_station_exact(self, diverse_df, diverse_ind):
        """Ensure contents can be filtered on time."""
        # tests for filtering with exact station name
        exact_name = diverse_df["station"].unique()[0]
        new_df = diverse_ind(station=exact_name)
        assert (new_df["station"] == exact_name).all()

    def test_filter_isin(self, diverse_df, diverse_ind):
        """Ensure contents can be filtered on time."""
        # tests for filtering with exact station name
        exact_name = diverse_df["station"].unique()[0]
        new_df = diverse_ind(station=exact_name)
        assert (new_df["station"] == exact_name).all()

    def test_emtpy_index(self, empty_index):
        """An empty index should return an empty dataframe."""
        df = empty_index()
        assert df.empty


class TestUpdate:
    """Tests for updating index."""

    def make_simple_index_with_version(self, path, version):
        """Helper function to make a simple index with desired version."""
        patch = dc.get_example_patch()
        spool_to_directory([patch], path)
        # this ensure the version is set to fake version
        old_version = dc.__last_version__
        # for some reason monkeypatch fixture wasnt setting version back
        # so I had to manually set and revert dascore version.
        setattr(dc, "__last_version__", version)
        spool = dc.spool(path).update()
        setattr(dc, "__last_version__", old_version)
        # ensure version monkey patch worked.
        meta = spool.indexer.get_index_metadata()
        assert meta["index_version"] == version
        return path

    @pytest.fixture(scope="class")
    def spool_directory_with_non_das_file(self, two_patch_directory, tmp_path_factory):
        """Create a directory with some das files and some non-das files."""
        new = tmp_path_factory.mktemp("unreadable_test") / "sub"
        shutil.copytree(two_patch_directory, new)
        indexer = DirectoryIndexer(new)
        # remove index if it exists
        with suppress(FileNotFoundError):
            indexer.index_path.unlink()
        # add a non das file
        with open(new / "not_das.open", "w") as fi:
            fi.write("cant be das, can it?")
        return new

    @pytest.fixture()
    def index_old_version(self, monkeypatch, tmp_path_factory):
        """Create an index which has an old, incompatible version."""
        # cant use random_patch fixture due to scope-mismatch w/ monkeypatch.
        path = tmp_path_factory.mktemp("index_old_version ")
        self.make_simple_index_with_version(path, "0.0.1")
        return path

    @pytest.fixture()
    def index_new_version(self, monkeypatch, tmp_path_factory):
        """Create an index which has an old, incompatible version."""
        # cant use random_patch fixture due to scope-mismatch w/ monkeypatch.
        path = tmp_path_factory.mktemp("index_new_version ")
        # a ridiculously high version
        fake_version = "1000.0.1"
        assert get_version(fake_version) > get_version(dc.__last_version__)
        self.make_simple_index_with_version(path, fake_version)
        return path

    def test_add_one_patch(self, empty_index, random_patch):
        """Ensure a new patch added to the directory shows up."""
        path = empty_index.path / get_patch_names(random_patch).iloc[0]
        random_patch.io.write(path, file_format="dasdae")
        new_index = empty_index.update()
        contents = new_index()
        assert len(contents) == 1

    def test_index_with_bad_file(self, spool_directory_with_non_das_file):
        """Ensure if one file is not readable index continues."""
        indexer = DirectoryIndexer(spool_directory_with_non_das_file)
        # if this doesn't fail the test passes
        updated = indexer.update()
        assert isinstance(updated, DirectoryIndexer)

    def test_old_index_recreated(self, index_old_version):
        """Ensure the old index is recreated when update is called."""
        msg = "Recreating the index now."
        with pytest.warns(UserWarning, match=msg):
            dc.spool(index_old_version).update()

    def test_new_version_warnings(self, index_new_version):
        """Ensure an index file with a newer version of dascore issues a warning."""
        msg = "The index was created with a newer version of dascore"
        dc.spool(index_new_version)
        with pytest.warns(UserWarning, match=msg):
            dc.spool(index_new_version).update()

    def test_update_with_specific_paths(self, basic_indexer):
        """Test updating with specific file paths to cover _get_paths method."""
        # Get files in the directory
        files = list(basic_indexer.path.rglob("*.hdf5"))
        assert len(files) > 0, "Need at least one file for testing"

        # Test update with specific paths (relative paths)
        relative_paths = [f.name for f in files[:1]]  # Use just first file
        updated = basic_indexer.update(paths=relative_paths)
        contents = updated()

        # Should have at least the file we specified
        assert len(contents) >= 1

        # Test update with absolute paths
        absolute_paths = [str(f) for f in files[:1]]
        updated2 = basic_indexer.update(paths=absolute_paths)
        contents2 = updated2()
        assert len(contents2) >= 1
