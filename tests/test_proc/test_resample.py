"""
Tests for decimation
"""
import numpy as np
import pytest

from dascore.exceptions import ParameterError
from dascore.utils.patch import get_start_stop_step


class TestInterpolate:
    """Tests for interpolating data along an axis in patch."""

    def test_interp_upsample_distance(self, random_patch):
        """Ensure interpolation between distance works."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        axis = random_patch.dims.index("distance")
        new_sampling = step / 2.2
        new_coord = np.arange(start, stop, new_sampling)
        out = random_patch.interpolate(distance=new_coord)
        assert out.data.shape[axis] == len(new_coord)
        assert np.all(out.coords["distance"] == new_coord)

    def test_interp_down_sample_distance(self, random_patch):
        """Ensure interp can be used to downsample data."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        axis = random_patch.dims.index("distance")
        new_sampling = step / 0.1
        new_coord = np.arange(start, stop, new_sampling)
        out = random_patch.interpolate(distance=new_coord)
        assert out.data.shape[axis] == len(new_coord)
        assert np.all(out.coords["distance"] == new_coord)

    def test_uneven_sampling_rates(self, random_patch):
        """Ensure uneven sampling raises an exception."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        new = [start, start + step, start + 4 * step, stop]
        with pytest.raises(ParameterError):
            random_patch.interpolate(distance=new)

    def test_upsample_time(self, random_patch):
        """Ensure time can be upsampled."""
        start, stop, step = get_start_stop_step(random_patch, "time")
        axis = random_patch.dims.index("time")
        new = np.arange(start, stop, step / 2)
        out = random_patch.interpolate(time=new)
        assert out.attrs["time_min"] == np.min(new)
        assert out.attrs["time_max"] == np.max(new)
        assert out.attrs["d_time"] == np.mean(np.diff(new))
        assert out.data.shape[axis] == len(new)

    def test_endtime_updated(self, random_patch):
        """Ensure the endtime/starttime in coords and dims are consistent."""
        dist = [0, 42, 84, 126, 168, 210, 252, 294]
        out = random_patch.interpolate(distance=dist)
        coord = out.coords["distance"]
        assert out.attrs["distance_max"] == coord.max()
        assert out.attrs["distance_min"] == coord.min()
        assert out.attrs["d_distance"] == np.median(np.diff(coord))


class TestDecimate:
    """Ensure Patch can be decimated."""

    def test_decimate_no_lowpass(self, random_patch):
        """Simple decimation"""
        p1 = random_patch
        old_time = p1.coords["time"]
        old_dt = old_time[1:] - old_time[:-1]
        # apply decimation,
        pa2 = random_patch.decimate(time=2)
        new_time = pa2.coords["time"]
        new_dt = new_time[1:] - new_time[:-1]
        # ensure distance between time samples and shapes have changed
        len_ratio = np.round(len(old_dt) / len(new_dt))
        assert np.isclose(len_ratio, 2.0)
        dt_ratio = np.round(new_dt[0] / old_dt[0])
        assert np.isclose(dt_ratio, 2.0)

    def test_update_time_max(self, random_patch):
        """Ensure the time_max is updated after decimation."""
        out = random_patch.decimate(time=10)
        assert out.attrs["time_max"] == out.coords["time"].max()

    def test_update_delta_dim(self, random_patch):
        """
        Since decimate changes the spacing of dimension this should be updated.
        """
        dt1 = random_patch.attrs["d_time"]
        out = random_patch.decimate(time=10)
        assert out.attrs["d_time"] == dt1 * 10


class TestResample:
    """
    Tests for resampling along a given dimension.
    """

    def test_downsample_time(self, random_patch):
        """Test decreasing the temporal sampling rate."""
        start, stop, step = get_start_stop_step(random_patch, "time")
        patch = random_patch
        axis = patch.dims.index("time")
        new_dt = 2 * step
        new = patch.resample(time=new_dt)
        assert new_dt == new.attrs["d_time"]
        assert np.all(np.diff(new.coords["time"]) == new_dt)
        # ensure only the time dimension has changed.
        shape1, shape2 = random_patch.data.shape, new.data.shape
        for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
            if ax == axis:  # Only resampled axis should have changed len
                assert len1 > len2
            else:
                assert len1 == len2

    def test_upsample_time(self, random_patch):
        """Test increasing the temporal sampling rate"""
        current_dt = random_patch.attrs["d_time"]
        axis = random_patch.dims.index("time")
        new_dt = current_dt / 2
        new = random_patch.resample(time=new_dt)
        assert new_dt == new.attrs["d_time"]
        assert np.all(np.diff(new.coords["time"]) == new_dt)
        shape1, shape2 = random_patch.data.shape, new.data.shape
        for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
            if ax == axis:  # Only resampled axis should have changed len
                assert len1 < len2
            else:
                assert len1 == len2

    def test_upsample_time_float(self, random_patch):
        """Test int as time sampling rate."""
        current_dt = random_patch.attrs["d_time"]
        axis = random_patch.dims.index("time")
        new_dt = current_dt / 2
        new = random_patch.resample(time=new_dt / np.timedelta64(1, "s"))
        assert new_dt == new.attrs["d_time"]
        assert np.all(np.diff(new.coords["time"]) == new_dt)
        shape1, shape2 = random_patch.data.shape, new.data.shape
        for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
            if ax == axis:  # Only resampled axis should have changed len
                assert len1 < len2
            else:
                assert len1 == len2

    def test_resample_distance(self, random_patch):
        """Ensure distance dimension is also resample-able."""
        current_dx = random_patch.attrs["d_distance"]
        new_dx = current_dx / 2
        new = random_patch.resample(distance=new_dx)
        axis = random_patch.dims.index("distance")
        assert new_dx == new.attrs["d_distance"]
        assert np.allclose(np.diff(new.coords["distance"]), new_dx)
        shape1, shape2 = random_patch.data.shape, new.data.shape
        for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
            if ax == axis:  # Only resampled axis should have changed len
                assert len1 < len2
            else:
                assert len1 == len2

    def test_odd_sampling_rate(self, random_patch):
        """Tests for resampling to a non-int sampling rate."""
        new_step = 1.232132323222
        out = random_patch.resample(distance=new_step)
        assert out.attrs["distance_max"] <= random_patch.attrs["distance_max"]
        assert np.allclose(out.attrs["d_distance"], new_step)

    def test_slightly_above_current_rate(self, random_patch):
        """Tests for resampling slightly above current rate."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        new_step = step + 0.0000001
        out = random_patch.resample(distance=new_step)
        assert out.attrs["distance_max"] <= random_patch.attrs["distance_max"]
        assert np.allclose(out.attrs["d_distance"], new_step)

    def test_slightly_under_current_rate(self, random_patch):
        """Tests for resampling slightly under current rate."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        new_step = step - 0.0000001
        out = random_patch.resample(distance=new_step)
        assert out.attrs["distance_max"] <= random_patch.attrs["distance_max"]
        assert np.allclose(out.attrs["d_distance"], new_step)

    def test_odd_time(self, random_patch):
        """Tests resampling to odd time interval."""
        dt = np.timedelta64(1234567, "ns")
        out = random_patch.resample(time=dt)
        assert out.attrs["d_time"] == dt
        assert out.attrs

    def test_huge_resample(self, random_patch):
        """Tests for greatly increasing the sampling_period."""
        out = random_patch.resample(distance=42)
        assert len(out.coords["distance"] == 42)


class TestIResample:
    """Tests for resampling based on numbers of sample per dimension."""

    def test_iresample_time(self, random_patch):
        """Tests iresample in time dim."""
        time_samples = 40
        out = random_patch.iresample(time=time_samples)
        assert len(out.coords["time"]) == time_samples

    def test_iresample_distance(self, random_patch):
        """Test for resampling distance to set len"""
        dist = 42
        out = random_patch.iresample(distance=dist)
        assert len(out.coords["distance"]) == dist
