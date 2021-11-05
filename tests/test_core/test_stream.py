"""
Test for stream functions.
"""
import pytest

import fios
from fios.utils.time import to_timedelta64


@pytest.fixture()
def adjacent_stream(random_patch):
    """Create a stream with several patches close together in time."""


class TestIndex:
    """Tests for indexing Stream"""


class TestMerge:
    """Tests for merging patches together."""

    @pytest.fixture()
    def adjacent_stream(self, random_patch):
        """A stream with two patches adjacent in time."""
        pa1 = random_patch
        dt = to_timedelta64(pa1.attrs["dt"])
        breakpoint()

        # pa2 = pa1.update_attrs(starttime=)

    def test_adjacent_merge(self, adjacent_stream):
        """Test that the adjacent patches get merged."""
