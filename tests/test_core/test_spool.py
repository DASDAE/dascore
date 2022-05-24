"""
Test for stream functions.
"""
import pandas as pd

import dascore
from dascore.utils.time import to_timedelta64


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
        sub = random_spool.get_contents(time=(None, new_max))
        assert len(sub) == (len(full_df) - 1)
        assert (sub["time_min"] < new_max).all()


class TestMerge:
    """
    Tests for merging patches together.
    Note: This doesn't need to be tested very well since the work here is done
    by utils.patch.merge
    """

    def test_adjacent_merge_no_overlap(self, adjacent_stream_no_overlap):
        """Test that the adjacent patches get merged."""
        st = adjacent_stream_no_overlap
        st_len = len(st)
        merged_st = st.merge()
        merged_len = len(merged_st)
        assert merged_len < st_len
        assert merged_len == 1
