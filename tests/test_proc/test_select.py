"""
Tests for selecting data from traces.
"""

import numpy as np

from dascore.utils.time import to_timedelta64


class TestSelect:
    """Tests for selecting data from Trace."""

    def test_select_by_distance(self, random_patch):
        """
        Ensure distance can be used to filter trace.
        """
        dmin, dmax = 100, 200
        pa = random_patch.select(distance=(dmin, dmax))
        assert pa.data.shape < random_patch.data.shape
        # the attrs should have updated as well
        assert pa.attrs["distance_min"] >= 100
        assert pa.attrs["distance_max"] <= 200

    def test_select_by_absolute_time(self, random_patch):
        """
        Ensure the data can be sub-selected using absolute time.
        """
        shape = random_patch.data.shape
        t1 = random_patch.attrs["time_min"] + np.timedelta64(1, "s")
        t2 = t1 + np.timedelta64(3, "s")

        pa1 = random_patch.select(time=(None, t1))
        assert pa1.attrs["time_max"] <= t1
        assert pa1.data.shape < shape

        pa2 = random_patch.select(time=(t1, None))
        assert pa2.attrs["time_min"] >= t1
        assert pa2.data.shape < shape

        tr3 = random_patch.select(time=(t1, t2))
        assert tr3.attrs["time_min"] >= t1
        assert tr3.attrs["time_max"] <= t2
        assert tr3.data.shape < shape

    def test_select_by_positive_float(self, random_patch):
        """Floats in time dim should usable to reference start of the trace."""
        shape = random_patch.data.shape
        t1 = random_patch.attrs["time_min"]
        pa1 = random_patch.select(time=(1, None))
        expected_start = t1 + to_timedelta64(1)
        assert pa1.attrs["time_min"] <= expected_start
        assert pa1.data.shape < shape

    def test_select_by_negative_float(self, random_patch):
        """Ensure negative floats reference end of trace."""
        shape = random_patch.data.shape
        pa1 = random_patch.select(time=(None, -2))
        expected_end = random_patch.attrs["time_max"] - to_timedelta64(2)
        assert pa1.attrs["time_max"] >= expected_end
        assert pa1.data.shape < shape

    def test_select_distance_leaves_time_attr_unchanged(self, random_patch):
        """Ensure selecting on distance doesn't change time"""
        dist = random_patch.coords["distance"]
        dist_max, dist_mean = np.max(dist), np.mean(dist)
        out = random_patch.select(distance=(dist_mean, dist_max - 1))
        assert out.attrs["time_max"] == out.coords["time"].max()
