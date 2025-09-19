"""Tests for decimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.compat import random_state
from dascore.exceptions import FilterValueError
from dascore.units import Hz, m, s
from dascore.utils.patch import get_start_stop_step


class TestInterpolate:
    """Tests for interpolating data along an axis in patch."""

    def test_interp_upsample_distance(self, random_patch):
        """Ensure interpolation between distance works."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        axis = random_patch.get_axis("distance")
        new_sampling = step / 2.2
        new_coord = np.arange(start, stop, new_sampling)
        out = random_patch.interpolate(distance=new_coord)
        assert out.data.shape[axis] == len(new_coord)
        assert np.allclose(out.coords.get_array("distance"), new_coord)

    def test_interp_down_sample_distance(self, random_patch):
        """Ensure interp can be used to downsample data."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        axis = random_patch.get_axis("distance")
        new_sampling = step / 0.1
        new_coord = np.arange(start, stop, new_sampling)
        out = random_patch.interpolate(distance=new_coord)
        assert out.data.shape[axis] == len(new_coord)
        assert np.allclose(out.coords.get_array("distance"), new_coord)

    def test_uneven_sampling_rates(self, random_patch):
        """Uneven sampling should now work fine."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        new_samps = np.array([start, start + step, start + 4 * step, stop])
        out = random_patch.interpolate(distance=new_samps)
        assert not out.coords.coord_map["distance"].evenly_sampled

    def test_upsample_time(self, random_patch):
        """Ensure time can be upsampled."""
        start, stop, step = get_start_stop_step(random_patch, "time")
        axis = random_patch.get_axis("time")
        new = np.arange(start, stop, step / 2)
        out = random_patch.interpolate(time=new)
        assert out.attrs["time_min"] == np.min(new)
        assert out.attrs["time_max"] == np.max(new)
        assert out.attrs["time_step"] == np.mean(np.diff(new))
        assert out.data.shape[axis] == len(new)

    def test_endtime_updated(self, random_patch):
        """Ensure the endtime/starttime in coords and dims are consistent."""
        dist = [0, 42, 84, 126, 168, 210, 252, 294]
        out = random_patch.interpolate(distance=dist)
        coord = out.coords.get_array("distance")
        assert out.attrs["distance_max"] == coord.max()
        assert out.attrs["distance_min"] == coord.min()
        assert out.attrs["distance_step"] == np.median(np.diff(coord))

    def test_snap_like(self, wacky_dim_patch):
        """Ensure interpolate can be used to snapping coords."""
        patch = wacky_dim_patch.interpolate(time=None)
        time = patch.coords.coord_map["time"]
        assert time.evenly_sampled and time.sorted


class TestDecimate:
    """Ensure Patch can be decimated."""

    def test_decimate_no_lowpass(self, random_patch):
        """Simple decimation."""
        p1 = random_patch
        old_time = p1.coords.get_array("time")
        old_dt = old_time[1:] - old_time[:-1]
        # apply decimation,
        pa2 = random_patch.decimate(time=2)
        new_time = pa2.coords.get_array("time")
        new_dt = new_time[1:] - new_time[:-1]
        # ensure distance between time samples and shapes have changed
        len_ratio = np.round(len(old_dt) / len(new_dt))
        assert np.isclose(len_ratio, 2.0)
        dt_ratio = np.round(new_dt[0] / old_dt[0])
        assert np.isclose(dt_ratio, 2.0)

    def test_update_time_max(self, random_patch):
        """Ensure the time_max is updated after decimation."""
        out = random_patch.decimate(time=10)
        assert out.attrs["time_max"] == out.coords.get_array("time").max()

    def test_update_delta_dim(self, random_patch):
        """Since decimate changes the spacing of dimension this should be updated."""
        dt1 = random_patch.attrs.time_step
        out = random_patch.decimate(time=10)
        assert out.attrs["time_step"] == dt1 * 10

    def test_float_32_stability(self, random_patch):
        """
        Ensure float32 works for decimation.

        See scipy#15072.
        """
        ar = random_state.random((10_000, 2)).astype("float32")
        dt = dc.to_timedelta64(0.001)
        t1 = dc.to_datetime64("2020-01-01")
        coords = {
            "distance": np.array([1, 2]),
            "time": np.arange(0, ar.shape[0]) * dt + t1,
        }
        dims = ("time", "distance")
        attrs = {"time_step": dt, "time_min": t1}
        patch = dc.Patch(data=ar, coords=coords, dims=dims, attrs=attrs)
        # ensure all modes of decimation don't produce NaN values.
        decimated_iir = patch.decimate(time=10, filter_type="iir")
        assert not np.any(pd.isnull(decimated_iir.data))

        decimated_fir = patch.decimate(time=10, filter_type="fir")
        assert not np.any(pd.isnull(decimated_fir.data))

        decimated_none = patch.decimate(time=10, filter_type=None)
        assert not np.any(pd.isnull(decimated_none.data))

    def test_decimate_small_dimension(self, random_patch):
        """Ensure decimation raises helpful error on small dimensions."""
        small_patch = random_patch.select(distance=(0, 10), samples=True)
        match = "Scipy decimation failed."
        with pytest.raises(FilterValueError, match=match):
            small_patch.decimate(distance=2)


class TestResample:
    """Tests for resampling along a given dimension."""

    def test_downsample_time(self, random_patch):
        """Test decreasing the temporal sampling rate."""
        start, stop, step = get_start_stop_step(random_patch, "time")
        patch = random_patch
        axis = patch.get_axis("time")
        new_dt = 2 * step
        new = patch.resample(time=new_dt)
        assert new_dt == new.attrs["time_step"]
        assert np.all(np.diff(new.coords.get_array("time")) == new_dt)
        # ensure only the time dimension has changed.
        shape1, shape2 = random_patch.data.shape, new.data.shape
        for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
            if ax == axis:  # Only resampled axis should have changed len
                assert len1 > len2
            else:
                assert len1 == len2

    def test_upsample_time(self, random_patch):
        """Test increasing the temporal sampling rate."""
        current_dt = random_patch.attrs["time_step"]
        axis = random_patch.get_axis("time")
        new_dt = current_dt / 2
        new = random_patch.resample(time=new_dt)
        assert new_dt == new.attrs["time_step"]
        assert np.all(np.diff(new.coords.get_array("time")) == new_dt)
        shape1, shape2 = random_patch.data.shape, new.data.shape
        for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
            if ax == axis:  # Only resampled axis should have changed len
                assert len1 < len2
            else:
                assert len1 == len2

    def test_upsample_time_float(self, random_patch):
        """Test int as time sampling rate."""
        current_dt = random_patch.attrs["time_step"]
        axis = random_patch.get_axis("time")
        new_dt = current_dt / 2
        new = random_patch.resample(time=new_dt / np.timedelta64(1, "s"))
        assert new_dt == new.attrs["time_step"]
        assert np.all(np.diff(new.coords.get_array("time")) == new_dt)
        shape1, shape2 = random_patch.data.shape, new.data.shape
        for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
            if ax == axis:  # Only resampled axis should have changed len
                assert len1 < len2
            else:
                assert len1 == len2

    def test_resample_distance(self, random_patch):
        """Ensure distance dimension is also resample-able."""
        current_dx = random_patch.attrs["distance_step"]
        new_dx = current_dx / 2
        new = random_patch.resample(distance=new_dx)
        axis = random_patch.get_axis("distance")
        assert new_dx == new.attrs["distance_step"]
        assert np.allclose(np.diff(new.coords.get_array("distance")), new_dx)
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
        assert np.allclose(out.attrs["distance_step"], new_step)

    def test_slightly_above_current_rate(self, random_patch):
        """Tests for resampling slightly above current rate."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        new_step = step + 0.0000001
        out = random_patch.resample(distance=new_step)
        assert out.attrs["distance_max"] <= random_patch.attrs["distance_max"]
        assert np.allclose(out.attrs["distance_step"], new_step)

    def test_slightly_under_current_rate(self, random_patch):
        """Tests for resampling slightly under current rate."""
        start, stop, step = get_start_stop_step(random_patch, "distance")
        new_step = step - 0.0000001
        out = random_patch.resample(distance=new_step)
        assert out.attrs["distance_max"] <= random_patch.attrs["distance_max"]
        assert np.allclose(out.attrs["distance_step"], new_step)

    def test_odd_time(self, random_patch):
        """Tests resampling to odd time interval."""
        dt = np.timedelta64(1234567, "ns")
        out = random_patch.resample(time=dt)
        new_dt = out.attrs["time_step"]
        assert np.isclose(float(new_dt), float(dt))
        assert out.attrs

    def test_huge_resample(self, random_patch):
        """Tests for greatly increasing the sampling_period."""
        out = random_patch.resample(distance=42, samples=True)
        assert len(out.coords.get_array("distance")) == 42

    def test_resample_with_units_hz(self, random_patch):
        """Ensure resample works with units."""
        new1 = random_patch.resample(time=50 * Hz)
        new2 = random_patch.resample(time=1 / 50)
        new3 = random_patch.resample(time=1 / 50 * s)
        assert new1 == new2 == new3

    def test_resample_distance_with_units(self, random_patch):
        """Ensure distance can be resampled with units as well."""
        new1 = random_patch.resample(distance=5 * m)
        new2 = random_patch.resample(distance=1 / 5 * (1 / m))
        new3 = random_patch.resample(distance=5)
        assert new1 == new2 == new3

    def test_resample_docs(self, random_patch):
        """Ensure docstring examples runs."""
        patch = random_patch
        time = patch.coords.get_array("time")
        ts = patch.attrs.time_step
        new_time = np.arange(time.min(), time.max(), 0.5 * ts)
        uptime = patch.interpolate(time=new_time)
        assert isinstance(uptime, dc.Patch)
        # interpolate unevenly sampled dim to evenly sampled
        patch = dc.get_example_patch("wacky_dim_coords_patch")
        patch_time_even = patch.interpolate(time=None)
        assert isinstance(patch_time_even, dc.Patch)

    def test_iresample_time(self, random_patch):
        """Tests iresample in time dim."""
        time_samples = 40
        out = random_patch.resample(time=time_samples, samples=True)
        assert len(out.coords.get_array("time")) == time_samples

    def test_iresample_distance(self, random_patch):
        """Test for resampling distance to set len."""
        dist = 42
        out = random_patch.resample(distance=dist, samples=True)
        assert len(out.coords.get_array("distance")) == dist

    def test_iresample_deprecated(self, random_patch):
        """Ensure iresample issues deprecation warning."""
        with pytest.warns(DeprecationWarning):
            random_patch.iresample(distance=42)

    def test_resample_fft(self, random_patch):
        """Tests for resample rft axis. See #272."""
        out = random_patch.dft("time", real="time").resample(ft_time=1)
        assert isinstance(out, dc.Patch)
