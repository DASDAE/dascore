"""Tests for planning explicit windows over the coordinates patches hold."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc


@pytest.fixture(scope="module")
def patches():
    """Patches with `lat` along distance, then along time (high at its ends)."""
    patch = dc.get_example_patch()
    distance, time = patch.shape
    along_distance = patch.update_coords(lat=("distance", np.linspace(0, 1, distance)))
    later = patch.get_coord("time").values + np.timedelta64(100, "s")
    along_time = patch.update_coords(time=later).update_coords(
        lat=("time", np.abs(np.linspace(-1, 1, time)))
    )
    return along_distance, along_time


class TestRiderOnOtherDimensions:
    """A coordinate riding one dimension in a patch and another in the next."""

    @pytest.mark.parametrize("method", ["select", "cut"])
    @pytest.mark.parametrize("first", [0, 1])
    def test_window_where_rider_range_kept_nothing(self, patches, method, first):
        """
        A window over the middle of the time the `lat` range removed keeps
        nothing, whichever patch the spool holds first.
        """
        start = patches[1].get_coord("time").min()
        window = np.array(
            [[start + np.timedelta64(3, "s"), start + np.timedelta64(5, "s")]]
        )
        spool = dc.spool([patches[first], patches[1 - first]]).select(lat=(0.6, 1.0))
        view = getattr(spool, method)(time=window)
        assert len(view) == 0 and list(view) == []

    @pytest.mark.parametrize("first", [0, 1])
    def test_window_where_rider_range_kept_samples(self, patches, first):
        """A window over samples the `lat` range kept loads them, as stated."""
        start = patches[1].get_coord("time").min()
        window = (start, start + np.timedelta64(1, "s"))
        spool = dc.spool([patches[first], patches[1 - first]]).select(lat=(0.6, 1.0))
        (patch,) = spool.select(time=np.array([window]))
        expected = patches[1].select(lat=(0.6, 1.0)).select(time=window)
        assert patch.get_coord("time") == expected.get_coord("time")
