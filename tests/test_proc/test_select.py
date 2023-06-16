"""
Tests for selecting data from patches.
"""

import numpy as np

import dascore as dc


class TestSelect:
    """Tests for selecting data from patch."""

    def test_select_by_distance(self, random_patch):
        """
        Ensure distance can be used to filter patch.
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

    def test_select_out_of_bounds_time(self, random_patch):
        """Selecting out of coordinate range should leave patch unchanged."""
        # this equates to a timestamp of 1 (eg 1 sec after 1970)
        pa1 = random_patch.select(time=(1, None))
        assert pa1 == random_patch
        # it should also work with proper datetimes.
        t1 = random_patch.attrs["time_min"] - dc.to_timedelta64(1)
        pa2 = random_patch.select(time=(t1, None))
        assert pa2 == random_patch

    def test_select_distance_leaves_time_attr_unchanged(self, random_patch):
        """Ensure selecting on distance doesn't change time"""
        dist = random_patch.coords["distance"]
        dist_max, dist_mean = np.max(dist), np.mean(dist)
        out = random_patch.select(distance=(dist_mean, dist_max - 1))
        assert out.attrs["time_max"] == out.coords["time"].max()

    def test_select_emptify_array(self, random_patch):
        """If select range excludes data range patch should be emptied."""
        out = random_patch.select(distance=(-100, -10))
        assert len(out.shape) == len(random_patch.shape)
        deleted_axis = out.dims.index("distance")
        assert out.shape[deleted_axis] == 0
        assert np.size(out.data) == 0


class TestSelectHistory:
    """Test behavior of history added by select."""

    def test_select_outside_bounds(self, random_patch):
        """Selecting outside the bounds should do nothing."""
        attrs = random_patch.attrs
        dt = dc.to_timedelta64(1)
        time = (attrs["time_min"] - dt, attrs["time_max"] + dt)
        dist = (attrs["distance_min"] - 1, attrs["distance_max"] + 1)
        new = random_patch.select(time=time, distance=dist)
        # if no select performed everything should be identical.
        assert new.equals(random_patch, only_required_attrs=False)
