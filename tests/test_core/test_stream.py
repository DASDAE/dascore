"""
Test for stream functions.
"""
import pytest

import fios
from fios.utils.time import to_timedelta64


@pytest.fixture()
def adjacent_stream(random_patch):
    """Create a stream with several patches close together in time."""
    t1 = random_patch.attrs["time_min"]


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

    @pytest.fixture()
    def adjacent_stream(self, random_patch):
        """A stream with two patches adjacent in time."""
        pa1 = random_patch
        dt = to_timedelta64(pa1.attrs["dt"])

        # pa2 = pa1.update_attrs(starttime=)

    def test_adjacent_merge(self, adjacent_stream):
        """Test that the adjacent patches get merged."""
