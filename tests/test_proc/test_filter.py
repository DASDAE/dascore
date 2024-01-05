"""Tests for filters."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import (
    CoordDataError,
    FilterValueError,
    PatchDimError,
    UnitError,
)


class TestPassFilterChecks:
    """Test that bad filter checks raise appropriate errors."""

    def test_no_kwargs_raises(self, random_patch):
        """Ensure ValueError is raised with no dim parameters."""
        with pytest.raises(FilterValueError):
            _ = random_patch.pass_filter()

    def test_wrong_kwarg_length_raises(self, random_patch):
        """Only len two sequence is allowed."""
        with pytest.raises(FilterValueError, match="length two sequence"):
            _ = random_patch.pass_filter(time=[1])
        with pytest.raises(FilterValueError, match="length two sequence"):
            _ = random_patch.pass_filter(time=[1, 3, 3])

    def test_all_null_kwarg_raises(self, random_patch):
        """There must be one Non-null kwarg."""
        with pytest.raises(FilterValueError, match="at least one filter"):
            _ = random_patch.pass_filter(time=[None, np.NAN])

    def test_unordered_params(self, random_patch):
        """Ensure a low parameter greater than a high parameter raises."""
        with pytest.raises(FilterValueError):
            _ = random_patch.pass_filter(time=[10, 1])

    def test_bad_low_param(self, random_patch):
        """Ensure a low parameter above nyquist raises."""
        with pytest.raises(FilterValueError, match="possible filter bounds"):
            _ = random_patch.pass_filter(time=[1000, None])

    def test_bad_high_param(self, random_patch):
        """Ensure a high parameter above nyquist raises."""
        with pytest.raises(FilterValueError, match="possible filter bounds"):
            _ = random_patch.pass_filter(distance=[None, 10])

    def test_filt_param_less_than_0(self, random_patch):
        """Ensure a high parameter above nyquist raises."""
        with pytest.raises(FilterValueError, match="possible filter bounds"):
            _ = random_patch.pass_filter(distance=[None, -10])

    def test_bad_units_error(self, random_patch):
        """Ensure incompatible units raise an Error."""
        m = dc.units.get_quantity("m")
        with pytest.raises(UnitError):
            random_patch.pass_filter(time=(1 * m, 10 * m))

    def test_units_dont_match(self, random_patch):
        """Tests for Units matching."""
        sec = dc.get_quantity("s")
        m = dc.get_quantity("m")
        filt = (1 * m, 10 * sec)
        match = "Units must match"
        with pytest.raises(UnitError, match=match):
            random_patch.pass_filter(time=filt)

    def test_high_time_raises(self, random_patch):
        """Ensure too high freq band in time axis raises."""
        nyquest = 0.5 / (random_patch.attrs.time_step / dc.to_timedelta64(1))
        hz = dc.get_quantity("Hz")
        filt = (1 * hz, nyquest * 1.1 * hz)
        match = "possible filter bounds are"
        with pytest.raises(FilterValueError, match=match):
            random_patch.pass_filter(time=filt)


class TestPassFilter:
    """Simple tests to make sure filter logic runs."""

    def test_time_bandpass_runs(self, random_patch):
        """Ensure filtering along time_axis works."""
        out = random_patch.pass_filter(time=(10, 100))
        assert isinstance(out, dc.Patch)
        assert not np.any(pd.isnull(out.data))

    def test_time_lowpass_runs(self, random_patch):
        """Ensure stopfilter on time works."""
        out = random_patch.pass_filter(time=(None, 100.22))
        assert isinstance(out, dc.Patch)
        assert not np.any(pd.isnull(out.data))

    def test_time_highpass_runs(self, random_patch):
        """Ensure a highpass on time runs."""
        out = random_patch.pass_filter(time=(10.2, None))
        assert isinstance(out, dc.Patch)
        assert not np.any(pd.isnull(out.data))

    def test_apply_distance_filter(self, random_patch):
        """Apply bandpass along distance dimension."""
        out = random_patch.pass_filter(distance=(0.1, 0.2))
        assert isinstance(out, dc.Patch)
        assert not np.any(pd.isnull(out.data))

    def test_uneven_sampling_raises(self, wacky_dim_patch):
        """A nice error message should be raised if the samples arent even."""
        match = "is not evenly sampled"
        with pytest.raises(CoordDataError, match=match):
            wacky_dim_patch.pass_filter(time=(10, 100))

    def test_specify_units_simple(self, random_patch):
        """Ensure units can be specified in patch arguments."""
        hz = dc.units.get_quantity("Hz")
        out1 = random_patch.pass_filter(time=(1 * hz, 10 * hz))
        out2 = random_patch.pass_filter(time=(1, 10))
        assert out1.equals(out2)

    def test_specify_units_inverse(self, random_patch):
        """Passing 1/Hz (eg seconds) should also work."""
        s = dc.units.get_quantity("s")
        out1 = random_patch.pass_filter(time=(1 * s, 10 * s))
        out2 = random_patch.pass_filter(time=(0.1, 1))
        assert out1.equals(out2)

    def test_misc_units(self, random_patch):
        """Catchall for other unit tests."""
        patch = random_patch
        m, ft = dc.units.get_unit("m"), dc.units.get_unit("ft")
        hz = dc.units.get_unit("Hz")
        # Filter from 1 Hz to 10 Hz in time dimension
        patch.pass_filter(time=(1 * hz, 10 * hz))
        # Filter wavelengths 50m to 100m
        patch.pass_filter(distance=(50 * m, 100 * m))
        # filter wavelengths less than 200 ft
        patch.pass_filter(distance=(200 * ft, None))

    def test_one_unit_raises(self, random_patch):
        """When only one unit is specified it should raise."""
        match = "Both inputs must be "
        s = dc.units.get_quantity("1")
        with pytest.raises(UnitError, match=match):
            random_patch.pass_filter(time=(1 * s, 10 * s))

    def test_ellipses(self, random_patch):
        """Ellipses should work for filtering."""
        p = random_patch
        p.pass_filter(time=(..., 20))
        assert p.pass_filter(time=(None, 20)) == p.pass_filter(time=(..., 20))
        assert p.pass_filter(time=(10, None)) == p.pass_filter(time=(10, ...))

    def test_non_zero_phase(self, random_patch):
        """Ensure non-zero-phase filter logic runs."""
        out = random_patch.pass_filter(time=(..., 20), zerophase=False)
        assert isinstance(out, dc.Patch)


class TestSobelFilter:
    """Simple tests to make sure Sobel filter runs."""

    def test_invalid_mode(self, random_patch):
        """Ensure ValueError is raised with an invalid mode."""
        with pytest.raises(FilterValueError):
            _ = random_patch.sobel_filter(dim="time", mode="test", cval=0.0)

    def test_invalid_mode_type(self, random_patch):
        """Ensure ValueError is raised with an invalid mode type."""
        with pytest.raises(FilterValueError):
            _ = random_patch.sobel_filter(dim="time", mode=0.0)

    def test_invalid_dim(self, random_patch):
        """Ensure ValueError is raised with an invalid dim type."""
        with pytest.raises(FilterValueError):
            _ = random_patch.sobel_filter(dim=-1)

    def test_invalid_axis(self, random_patch):
        """Ensure ValueError is raised with an invalid axis value."""
        with pytest.raises(FilterValueError):
            _ = random_patch.sobel_filter(dim=None, mode="constant", cval=0.0)

    def test_invalid_cval(self, random_patch):
        """Ensure ValueError is raised with an invalid cval value."""
        with pytest.raises(FilterValueError):
            _ = random_patch.sobel_filter(dim="distance", mode="constant", cval=None)

    def test_sobel_runs(self, random_patch):
        """Ensure Sobel filter works with default params."""
        out = random_patch.sobel_filter(dim="time")
        assert isinstance(out, dc.Patch)
        assert not np.any(pd.isnull(out.data))


class TestMedianFilter:
    """Simple tests on median filter."""

    def test_median_no_kwargs_raises(self, random_patch):
        """Apply default values."""
        msg = "You must specify one or more dimension in keyword args."
        with pytest.raises(PatchDimError, match=msg):
            random_patch.median_filter()

    def test_median_filter_time(self, random_patch):
        """Test median filter in time dimension."""
        out = random_patch.median_filter(time=0.5)
        assert isinstance(out, dc.Patch)
        assert not np.any(pd.isnull(out.data))

    def test_median_filter_time_distance(self, random_patch):
        """Apply default values."""
        out = random_patch.median_filter(time=0.05, distance=2)
        assert isinstance(out, dc.Patch)
        assert not np.any(pd.isnull(out.data))

    def test_median_filter_ones(self, random_patch):
        """Apply default values."""
        out = random_patch.median_filter(time=1, distance=1, samples=True)
        assert out == random_patch


class TestSavgolFilter:
    """Simple tests on Savgol filter."""

    def test_savgol_no_kwargs_raises(self, random_patch):
        """Apply default values."""
        msg = "You must specify one or more"
        with pytest.raises(PatchDimError, match=msg):
            random_patch.savgol_filter(polyorder=2)

    def test_savgol_filter_time(self, random_patch):
        """Test savgol filter in time dimension."""
        out = random_patch.savgol_filter(polyorder=2, time=5)
        assert isinstance(out, dc.Patch)
        assert not np.any(pd.isnull(out.data))

    def test_savgol_filter_ones(self, random_patch):
        """Apply default values."""
        out = random_patch.savgol_filter(polyorder=2, distance=5, samples=True)
        assert out != random_patch

    def test_savgol_smoothing(self, random_patch):
        """Test smoothing in one dimension."""
        new_array = np.array(random_patch.data)
        midpoint = new_array.shape[0] // 2
        new_array[:midpoint] = 1
        new_array[midpoint:] = 0
        new_patch = random_patch.new(data=new_array)
        dim = new_patch.dims[0]
        out = new_patch.savgol_filter(polyorder=2, samples=True, **{dim: 5})
        assert np.allclose(out.data[:5], 1)
        assert np.allclose(out.data[-5:], 0)
        middle = out.data[midpoint - 10 : midpoint + 10]
        assert np.any((middle < 1) & (middle > 0))

    def test_savgol_filter_multiple_dims(self, event_patch_2):
        """Ensure multiple dimensions can be filtered."""
        out = event_patch_2.savgol_filter(distance=10, time=0.001, polyorder=4)
        assert out.shape == event_patch_2.shape
        assert not np.allclose(out.data, event_patch_2.data)


class TestGaussianFilter:
    """Test the Guassian Filter."""

    def test_filter_time(self, event_patch_2):
        """Test for simple filter along time axis."""
        out = event_patch_2.gaussian_filter(time=0.001)
        assert isinstance(out, dc.Patch)
        assert out.shape == event_patch_2.shape

    def test_filter_distance(self, event_patch_2):
        """Ensure filter can be applied along distance axis with samples."""
        out = event_patch_2.gaussian_filter(distance=5, samples=True)
        assert isinstance(out, dc.Patch)
        assert out.shape == event_patch_2.shape

    def test_filter_time_distance(self, event_patch_2):
        """Ensure both time and distance can be filtered."""
        out = event_patch_2.gaussian_filter(time=5, distance=5, samples=True)
        assert isinstance(out, dc.Patch)
        assert out.shape == event_patch_2.shape
