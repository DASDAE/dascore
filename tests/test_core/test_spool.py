"""
Test for stream functions.
"""
import pandas as pd
import pytest

import dascore
from dascore.utils.time import to_timedelta64


class TestSpoolBasics:
    """Tests for the basics of the spool."""

    def test_not_default_str(self, random_spool):
        """Ensure the default str is not used on the spool."""
        out = str(random_spool)
        assert "object at" not in out


class TestSpoolIterablity:
    """Tests for indexing/iterating Spools"""

    def test_len(self, random_spool):
        """Ensure the stream has a length"""
        assert len(random_spool) == len(list(random_spool))

    def test_index(self, random_spool):
        """Ensure the stream can be indexed."""
        assert isinstance(random_spool[0], dascore.Patch)

    def test_list_o_patches(self, random_spool):
        """Ensure random_string can be iterated"""
        for pa in random_spool:
            assert isinstance(pa, dascore.Patch)
        patch_list = list(random_spool)
        for pa in patch_list:
            assert isinstance(pa, dascore.Patch)

    def test_index_error(self, random_spool):
        """Ensure an IndexError is raised when indexing beyond spool."""
        spool_len = len(random_spool)
        with pytest.raises(IndexError, match="out of bounds"):
            _ = random_spool[spool_len]

    def test_index_returns_corresponding_patch(self, random_spool):
        """Ensure the index returns the correct patch"""
        spool_list = list(random_spool)
        for num, (patch1, patch2) in enumerate(zip(spool_list, random_spool)):
            patch3 = random_spool[num]
            assert patch1 == patch2 == patch3


class TestGetContents:
    """Ensure the contents of the spool can be returned via dataframe."""

    def test_no_filter(self, random_spool):
        """Ensure the entirety of the contents are returned."""
        df = random_spool.get_contents()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(random_spool)

    def test_filter(self, random_spool):
        """Ensure the dataframe can be filtered."""
        full_df = random_spool.get_contents()
        new_max = full_df["time_min"].max() - to_timedelta64(1)
        sub = random_spool.select(time=(None, new_max)).get_contents()
        assert len(sub) == (len(full_df) - 1)
        assert (sub["time_min"] < new_max).all()


class TestChunk:
    """
    Tests for merging/chunking patches.
    """

    def test_merge_chunk_adjacent_no_overlap(self, adjacent_spool_no_overlap):
        """
        Ensure chunking works on simple case of contiguous data w/ no overlap.
        """
        new = adjacent_spool_no_overlap.chunk(time=None)
        out_list = list(new)
        assert len(new) == len(out_list) == 1

    def test_adjacent_merge_no_overlap(self, adjacent_spool_no_overlap):
        """Test that the adjacent patches get merged."""
        st = adjacent_spool_no_overlap
        st_len = len(st)
        merged_st = st.merge()
        merged_len = len(merged_st)
        assert merged_len < st_len
        assert merged_len == 1

    def test_chunk_doesnt_modify_original(self, random_spool):
        """Chunking shouldn't modify original spool"""
        first = random_spool.get_contents().copy()
        _ = random_spool.chunk(time=2)
        second = random_spool.get_contents().copy()
        assert first.equals(second)

    def test_patches_match_df_contents(self, random_spool):
        """Ensure the patch content matches the dataframe."""
        new = random_spool.chunk(time=2)
        # get contents of chunked spool
        chunk_df = new.get_contents()
        new_patches = list(new)
        new_patch = dascore.MemorySpool(new_patches)
        # get content of spool created from patches in chunked spool.
        new_content = new_patch.get_contents()
        # these should be (nearly) identical.
        common = set(chunk_df.columns) & set(new_content.columns)
        cols = sorted(common - {"history"})  # no need to compare history
        assert chunk_df[cols].equals(new_content[cols])
