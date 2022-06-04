"""
Tests for indexing local file systems.
"""
from pathlib import Path

import pandas as pd
import pytest

from dascore.io.indexer import HDFIndexer


@pytest.fixture(scope="class")
def basic_indexer(two_patch_directory):
    """Return and indexer on the basic spool directory."""
    return HDFIndexer(two_patch_directory)


@pytest.fixture(scope="class")
def adjacent_indexer(adjacent_spool_directory):
    """Return and indexer on the basic spool directory."""
    return HDFIndexer(adjacent_spool_directory).update()


@pytest.fixture(scope="class")
def diverse_indexer(diverse_spool_directory):
    """Return and indexer on the basic spool directory."""
    return HDFIndexer(diverse_spool_directory).update()


@pytest.fixture(scope='class')
def diverse_df(diverse_indexer):
    """Return the contents of the diverse indexer"""
    return diverse_indexer()


@pytest.fixture()
def diverse_df_reset_cache(diverse_indexer):
    """Return the indexer with a reset cache"""
    return HDFIndexer(diverse_indexer.path)


@pytest.fixture(params=[diverse_indexer, diverse_df_reset_cache])
def diverse_ind(request):
    """Aggregate the diverse indexers"""
    return request.getfixturevalue(request.param.__name__)


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
        max_starttime = diverse_df['time_min'].max()
        filtered = diverse_df[diverse_df['time_min'] >= max_starttime]
        out = diverse_ind(time_min=max_starttime)
        assert len(out) == len(filtered)

    def test_filter_small_starttime(self, diverse_df, diverse_ind):
        """Ensure the index can be filtered by start time."""
        min_endtime = diverse_df['time_max'].min()
        filtered = diverse_df[diverse_df['time_max'] <= min_endtime]
        out = diverse_ind(time_max=min_endtime)
        assert len(out) == len(filtered)

    def test_filter_station_exact(self, diverse_df, diverse_ind):
        """Ensure contents can be filtered on time."""
        # tests for filtering with exact station name
        exact_name = diverse_df['station'].unique()[0]
        new_df = diverse_ind(station=exact_name)
        assert (new_df['station'] == exact_name).all()


class TestPutData:
    """Test case for putting data into indexer."""
