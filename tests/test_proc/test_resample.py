"""Tests for decimation."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.compat import random_state
from dascore.exceptions import FilterValueError, ParameterError
from dascore.units import Hz, m, s
from dascore.utils.patch import get_start_stop_step

resample_mod = importlib.import_module("dascore.proc.resample")


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
        assert out.coords["time"].min() == np.min(new)
        assert out.coords["time"].max() == np.max(new)
        assert out.coords["time"].step == np.mean(np.diff(new))
        assert out.data.shape[axis] == len(new)

    def test_endtime_updated(self, random_patch):
        """Ensure the endtime/starttime in coords and dims are consistent."""
        dist = [0, 42, 84, 126, 168, 210, 252, 294]
        out = random_patch.interpolate(distance=dist)
        coord = out.coords.get_array("distance")
        assert out.coords["distance"].max() == coord.max()
        assert out.coords["distance"].min() == coord.min()
        assert out.coords["distance"].step == np.median(np.diff(coord))

    def test_snap_like(self, wacky_dim_patch):
        """Ensure interpolate can be used to snapping coords."""
        patch = wacky_dim_patch.interpolate(time=None)
        time = patch.coords.coord_map["time"]
        assert time.evenly_sampled and time.sorted

    def test_associated_coords_interpolated(self, random_patch_many_coords):
        """A numeric coord on the dim is interpolated with it. See #1041."""
        patch = random_patch_many_coords
        start, stop, step = get_start_stop_step(patch, "distance")
        new_coord = np.arange(start, stop, step / 2)
        out = patch.interpolate(distance=new_coord)
        assert out.coords.dim_map["lat"] == ("distance",)
        assert out.get_coord("lat").units == patch.get_coord("lat").units
        assert len(out.get_array("lat")) == len(new_coord)
        # The samples which did not move keep the values they had.
        kept = out.get_array("lat")[::2]
        assert np.allclose(kept, patch.get_array("lat")[: len(kept)])

    @pytest.mark.parametrize("factor", (0.5, 1.0))
    def test_uninterpolatable_coords_dropped(self, random_patch, factor):
        """What cannot be resampled is dropped, however many samples remain.

        At factor 1.0 the coordinate is the same length as before, which
        is exactly when a stale one would go unnoticed.
        """
        shape = random_patch.coord_shapes["distance"]
        stamps = np.arange(shape[0]).astype("datetime64[s]")
        patch = random_patch.update_coords(
            label=("distance", np.full(shape, "a")),
            flag=("distance", np.ones(shape, dtype=bool)),
            stamp=("distance", stamps),
            # numpy counts a duration as a number; it is still a time.
            lag=("distance", np.arange(shape[0]).astype("timedelta64[ns]")),
        )
        start, stop, step = get_start_stop_step(patch, "distance")
        new_coord = np.arange(start, stop, step * factor) + step / 4
        out = patch.interpolate(distance=new_coord)
        assert {"label", "flag", "stamp", "lag"}.isdisjoint(out.coords.coord_map)

    def test_multidimensional_coords_interpolated(self, random_patch_many_coords):
        """A coordinate spanning both dimensions rides the one being set."""
        patch = random_patch_many_coords
        start, stop, step = get_start_stop_step(patch, "distance")
        new_coord = np.arange(start, stop, step / 2)
        out = patch.interpolate(distance=new_coord)
        assert out.coords.dim_map["quality"] == ("distance", "time")
        assert out.get_array("quality").shape == out.shape


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
        assert out.coords["time"].max() == out.coords.get_array("time").max()

    def test_update_delta_dim(self, random_patch):
        """Since decimate changes the spacing of dimension this should be updated."""
        dt1 = random_patch.coords["time"].step
        out = random_patch.decimate(time=10)
        assert out.coords["time"].step == dt1 * 10

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
        attrs = {}
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
        match = "dimensions with few elements"
        with pytest.raises(FilterValueError, match=match):
            small_patch.decimate(distance=2)

    def test_scipy_decimation_gets_patch(self, random_patch, monkeypatch):
        """The scipy decimation helper should receive the patch, not its data."""
        calls = []

        def decimate_spy(patch, factor, ftype, axis):
            assert isinstance(patch, dc.Patch)
            calls.append((patch, factor, ftype, axis))
            slicer = [slice(None)] * patch.ndim
            slicer[axis] = slice(None, None, int(factor))
            return patch.data[tuple(slicer)]

        monkeypatch.setattr(resample_mod, "_apply_scipy_decimation", decimate_spy)

        out = random_patch.decimate(time=2, filter_type="iir")

        patch, factor, _, axis = calls[0]
        assert patch is random_patch
        assert out.shape[axis] == random_patch.shape[axis] // factor

    @pytest.mark.parametrize("filter_type", ("iir", None))
    def test_associated_coords_decimated(self, random_patch_many_coords, filter_type):
        """Coords on the decimated dim are subsampled with it. See #1041."""
        patch = random_patch_many_coords
        out = patch.decimate(distance=2, filter_type=filter_type)
        assert np.allclose(out.get_array("lat"), patch.get_array("lat")[::2])
        assert np.allclose(out.get_array("quality"), patch.get_array("quality")[::2])
        assert np.allclose(out.get_array("time2"), patch.get_array("time2"))
        assert out.coords.dim_map == patch.coords.dim_map


class TestResample:
    """Tests for resampling along a given dimension."""

    @pytest.mark.parametrize("samples", (False, True))
    def test_associated_coords_resampled(self, random_patch_many_coords, samples):
        """Numeric coordinates on the resampled dimension follow it. See #1090."""
        patch = random_patch_many_coords
        distance = patch.get_coord("distance")
        value = len(distance) * 2 if samples else distance.step * 1.232132323222

        out = patch.resample(distance=value, samples=samples)
        old_distance = patch.get_array("distance")
        new_distance = out.get_array("distance")

        def _linear_extrapolate(values):
            expected = np.interp(new_distance, old_distance, values)
            above = new_distance > old_distance[-1]
            slope = (values[-1] - values[-2]) / (old_distance[-1] - old_distance[-2])
            expected[above] = values[-1] + slope * (
                new_distance[above] - old_distance[-1]
            )
            return expected

        assert out.coords.dim_map == patch.coords.dim_map
        expected_lat = _linear_extrapolate(patch.get_array("lat"))
        assert np.allclose(out.get_array("lat"), expected_lat)
        expected_quality = np.apply_along_axis(
            _linear_extrapolate,
            patch.get_axis("distance"),
            patch.get_array("quality"),
        )
        assert np.allclose(out.get_array("quality"), expected_quality)
        assert np.allclose(out.get_array("time2"), patch.get_array("time2"))

    def test_datetime_dim_associated_coord(self):
        """Large epoch nanoseconds retain fine interpolation precision."""
        size = 8
        time = np.datetime64("2026-01-01", "ns") + np.arange(size) * np.timedelta64(
            100, "ns"
        )
        patch = dc.Patch(
            data=np.arange(size, dtype=float),
            coords={"time": time, "clock": ("time", np.arange(size, dtype=float))},
            dims=("time",),
        )

        out = patch.resample(time=16, samples=True)

        assert np.allclose(out.get_array("clock"), np.arange(16) / 2)

        samples = time[0] + np.arange(11) * np.timedelta64(70, "ns")
        out = patch.interpolate(time=samples)
        assert np.allclose(out.data, out.get_array("clock"))

    def test_associated_coord_name_like_dim_attr(self):
        """A legal coordinate name is not parsed as dimension metadata."""
        distance = np.arange(8, dtype=float)
        patch = dc.Patch(
            data=distance,
            coords={"distance": distance, "distance_min": ("distance", distance * 10)},
            dims=("distance",),
        )

        out = patch.resample(distance=16, samples=True)

        assert np.allclose(out.get_array("distance_min"), np.arange(16) * 5)

        out = patch.resample(distance=1.02)
        assert np.allclose(
            out.get_array("distance_min"), out.get_array("distance") * 10
        )

    def test_exact_resample_ignores_interp_kind_for_coords(self):
        """Fourier-only resampling does not apply the fallback interpolation kind."""
        values = np.arange(3, dtype=float)
        patch = dc.Patch(
            data=values,
            coords={"x": values, "aux": ("x", values)},
            dims=("x",),
        )

        out = patch.resample(x=6, samples=True, interp_kind="cubic")

        assert np.allclose(out.get_array("aux"), np.arange(6) / 2)

    def test_missing_period_raises(self, random_patch):
        """A null sampling period is rejected rather than producing NaN."""
        match = "requires a sampling period"
        with pytest.raises(ParameterError, match=match):
            random_patch.resample(time=None)

    def test_downsample_time(self, random_patch):
        """Test decreasing the temporal sampling rate."""
        _, _, step = get_start_stop_step(random_patch, "time")
        patch = random_patch
        axis = patch.get_axis("time")
        new_dt = 2 * step
        new = patch.resample(time=new_dt)
        assert new_dt == new.get_coord("time").step
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
        current_dt = random_patch.get_coord("time").step
        axis = random_patch.get_axis("time")
        new_dt = current_dt / 2
        new = random_patch.resample(time=new_dt)
        assert new_dt == new.get_coord("time").step
        assert np.all(np.diff(new.coords.get_array("time")) == new_dt)
        shape1, shape2 = random_patch.data.shape, new.data.shape
        for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
            if ax == axis:  # Only resampled axis should have changed len
                assert len1 < len2
            else:
                assert len1 == len2

    def test_upsample_time_float(self, random_patch):
        """Test int as time sampling rate."""
        current_dt = random_patch.get_coord("time").step
        axis = random_patch.get_axis("time")
        new_dt = current_dt / 2
        new = random_patch.resample(time=new_dt / np.timedelta64(1, "s"))
        assert new_dt == new.get_coord("time").step
        assert np.all(np.diff(new.coords.get_array("time")) == new_dt)
        shape1, shape2 = random_patch.data.shape, new.data.shape
        for ax, (len1, len2) in enumerate(zip(shape1, shape2)):
            if ax == axis:  # Only resampled axis should have changed len
                assert len1 < len2
            else:
                assert len1 == len2

    def test_resample_distance(self, random_patch):
        """Ensure distance dimension is also resample-able."""
        current_dx = random_patch.get_coord("distance").step
        new_dx = current_dx / 2
        new = random_patch.resample(distance=new_dx)
        axis = random_patch.get_axis("distance")
        assert new_dx == new.get_coord("distance").step
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
        assert (
            out.get_coord("distance").max() <= random_patch.get_coord("distance").max()
        )
        assert np.allclose(out.get_coord("distance").step, new_step)

    def test_slightly_above_current_rate(self, random_patch):
        """Tests for resampling slightly above current rate."""
        _, _, step = get_start_stop_step(random_patch, "distance")
        new_step = step + 0.0000001
        out = random_patch.resample(distance=new_step)
        assert (
            out.get_coord("distance").max() <= random_patch.get_coord("distance").max()
        )
        assert np.allclose(out.get_coord("distance").step, new_step)

    def test_slightly_under_current_rate(self, random_patch):
        """Tests for resampling slightly under current rate."""
        _, _, step = get_start_stop_step(random_patch, "distance")
        new_step = step - 0.0000001
        out = random_patch.resample(distance=new_step)
        assert (
            out.get_coord("distance").max() <= random_patch.get_coord("distance").max()
        )
        assert np.allclose(out.get_coord("distance").step, new_step)

    def test_odd_time(self, random_patch):
        """Tests resampling to odd time interval."""
        dt = np.timedelta64(1234567, "ns")
        out = random_patch.resample(time=dt)
        new_dt = out.get_coord("time").step
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
        ts = patch.get_coord("time").step
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

    def test_resample_fft(self, random_patch):
        """Tests for resample rft axis. See #272."""
        out = random_patch.dft("time", real="time").resample(ft_time=1)
        assert isinstance(out, dc.Patch)
