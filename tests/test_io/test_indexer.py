"""
Tests for indexing local file systems.
"""
from pathlib import Path

import pandas as pd
import pytest

from dascore.io.indexer import DirectoryIndexer
from dascore.utils.patch import get_default_patch_name


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
    """Return the contents of the diverse indexer"""
    return diverse_indexer()


@pytest.fixture()
def diverse_df_reset_cache(diverse_indexer):
    """Return the indexer with a reset cache"""
    return DirectoryIndexer(diverse_indexer.path)


@pytest.fixture(params=[diverse_indexer, diverse_df_reset_cache])
def diverse_ind(request):
    """Aggregate the diverse indexers"""
    return request.getfixturevalue(request.param.__name__)


@pytest.fixture()
def empty_index(tmp_path_factory):
    """Create an index around an empty directory."""
    path = tmp_path_factory.mktemp("index_created_test")
    return DirectoryIndexer(path).update()


class TestBasics:
    """Basic tests for indexer"""

    def test_str_repr(self, basic_indexer):
        """Ensure a useful (not the default) str/repr is implemented"""
        out = str(basic_indexer)
        assert "object at" not in out


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

    def test_add_one_patch(self, empty_index, random_patch):
        """Ensure a new patch added to the directory shows up."""
        path = empty_index.path / get_default_patch_name(random_patch)
        random_patch.io.write(path, file_format="dasdae")
        new_index = empty_index.update()
        contents = new_index()
        assert len(contents) == 1
