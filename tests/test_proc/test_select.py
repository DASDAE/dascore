"""Tests for selecting data from patches."""
from __future__ import annotations

import numpy as np
import pytest

import dascore as dc


class TestSelect:
    """Tests for selecting data from patch."""

    def test_select_by_distance(self, random_patch):
        """Ensure distance can be used to filter patch."""
        dmin, dmax = 100, 200
        pa = random_patch.select(distance=(dmin, dmax))
        assert pa.data.shape < random_patch.data.shape
        # the attrs should have updated as well
        assert pa.attrs["distance_min"] >= 100
        assert pa.attrs["distance_max"] <= 200

    def test_select_by_absolute_time(self, random_patch):
        """Ensure the data can be sub-selected using absolute time."""
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
        """Ensure selecting on distance doesn't change time."""
        dist = random_patch.coords.get_array("distance")
        dist_max, dist_mean = np.max(dist), np.mean(dist)
        out = random_patch.select(distance=(dist_mean, dist_max - 1))
        assert out.attrs["time_max"] == out.coords.max("time")

    def test_select_emptify_array(self, random_patch):
        """If select range excludes data range patch should be emptied."""
        out = random_patch.select(distance=(-100, -10))
        assert len(out.shape) == len(random_patch.shape)
        deleted_axis = out.dims.index("distance")
        assert out.shape[deleted_axis] == 0
        assert np.size(out.data) == 0

    def test_select_relative_start_end(self, random_patch):
        """Ensure relative select works on start to end."""
        patch1 = random_patch.select(time=(1, -1), relative=True)
        t1 = random_patch.attrs.time_min + dc.to_timedelta64(1)
        t2 = random_patch.attrs.time_max - dc.to_timedelta64(1)
        patch2 = random_patch.select(time=(t1, t2))
        assert patch1 == patch2

    def test_select_relative_end_end(self, random_patch):
        """Ensure relative works for end to end."""
        patch1 = random_patch.select(time=(-3, -1), relative=True)
        t1 = random_patch.attrs.time_max - dc.to_timedelta64(1)
        t2 = random_patch.attrs.time_max - dc.to_timedelta64(3)
        patch2 = random_patch.select(time=(t1, t2))
        assert patch1 == patch2

    def test_select_relative_start_start(self, random_patch):
        """Ensure relative start ot start."""
        patch1 = random_patch.select(time=(1, 3), relative=True)
        t1 = random_patch.attrs.time_min + dc.to_timedelta64(1)
        t2 = random_patch.attrs.time_min + dc.to_timedelta64(3)
        patch2 = random_patch.select(time=(t1, t2))
        assert patch1 == patch2

    def test_select_relative_start_open(self, random_patch):
        """Ensure relative start to open end."""
        patch1 = random_patch.select(time=(1, None), relative=True)
        t1 = random_patch.attrs.time_min + dc.to_timedelta64(1)
        patch2 = random_patch.select(time=(t1, None))
        assert patch1 == patch2

    def test_select_relative_end_open(self, random_patch):
        """Ensure relative start to open end."""
        patch1 = random_patch.select(time=(-1, None), relative=True)
        t1 = random_patch.attrs.time_max - dc.to_timedelta64(1)
        patch2 = random_patch.select(time=(t1, None))
        assert patch1 == patch2

    def test_time_slice_samples(self, random_patch):
        """Ensure a simple time slice works."""
        pa1 = random_patch.select(time=(1, 5), samples=True)
        pa2 = random_patch.select(time=slice(1, 5), samples=True)
        assert pa1 == pa2

    def test_non_slice_samples(self, random_patch):
        """Ensure a non-slice doesnt change patch."""
        pa1 = random_patch.select(distance=(..., ...), samples=True)
        pa2 = random_patch.select(distance=(None, ...), samples=True)
        pa3 = random_patch.select(distance=slice(None, None), samples=True)
        pa4 = random_patch.select(distance=...)
        assert pa1 == pa2 == pa3 == pa4

    def test_iselect_deprecated(self, random_patch):
        """Ensure Patch.iselect raises deprecation error."""
        msg = "iselect is deprecated"
        with pytest.warns(DeprecationWarning, match=msg):
            _ = random_patch.iselect(time=(10, -10))


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
