"""
Test for stream functions.
"""

import dascore


class TestStreamIterableness:
    """Tests for indexing/iterating Streams"""

    def test_len(self, random_stream):
        """Ensure the stream has a length"""
        assert len(random_stream) == 1

    def test_index(self, random_stream):
        """Ensure the stream can be indexed."""
        assert isinstance(random_stream[0], dascore.Patch)

    def test_list_o_patches(self, random_stream):
        """Ensure random_string can be iterated"""
        for pa in random_stream:
            assert isinstance(pa, dascore.Patch)
        patch_list = list(random_stream)
        for pa in patch_list:
            assert isinstance(pa, dascore.Patch)


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
