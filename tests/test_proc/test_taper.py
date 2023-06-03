"""
Tests for taper processing function.
"""

import pytest

import dascore as dc
from dascore.proc.taper import TAPER_FUNCTIONS, taper


@pytest.fixture(scope="session", params=sorted(TAPER_FUNCTIONS))
def time_tapered_patch(request, random_patch):
    """Return a tapered trace."""
    out = taper(random_patch, time=0.05, type=request.param)
    return out


class TestTaperBasics:
    """Ensure each taper runs."""

    def test_each_taper(self, time_tapered_patch, random_patch):
        """Ensure each taper type runs."""
        assert isinstance(time_tapered_patch, dc.Patch)
        assert time_tapered_patch.shape == random_patch.shape

    def test_ends_near_zero(self, time_tapered_patch):
        """Ensure the ends of the patch are near zero."""
