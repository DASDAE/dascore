"""
Test for stream functions.
"""
import pytest

import fios
from fios.utils.time import to_timedelta64


@pytest.fixture()
def adjacent_stream_no_overlap(random_patch):
    """
    Create a stream with several patches within one time sample but not
    overlapping.
    """
    pa1 = random_patch
    t2 = random_patch.attrs['time_max']
    dt = random_patch.attrs['dt']

    pa2 = random_patch.update_attrs(time_min=t2 + dt)
    t3 = pa1.attrs['time_max']

    pa3 = pa2.update_attrs(time_min=t3 + dt)

    return fios.Stream([pa2, pa1, pa3])


class TestStreamIterableness:
    """Tests for indexing/iterating Streams"""

    def test_len(self, random_stream):
        """Ensure the stream has a length"""
        assert len(random_stream) == 1

    def test_index(self, random_stream):
        """Ensure the stream can be indexed."""
        assert isinstance(random_stream[0], fios.Patch)

    def test_list_o_patches(self, random_stream):
        """Ensure random_string can be iterated"""
        for pa in random_stream:
            assert isinstance(pa, fios.Patch)
        patch_list = list(random_stream)
        for pa in patch_list:
            assert isinstance(pa, fios.Patch)


class TestMerge:
    """Tests for merging patches together."""

    def test_adjacent_merge_no_overlap(self, adjacent_stream_no_overlap):
        """Test that the adjacent patches get merged."""
        st = adjacent_stream_no_overlap
        st_len = len(st)
        merged_st = st.merge()
        merged_len = len(merged_st)
        assert merged_len < st_len
        assert merged_len == 1
