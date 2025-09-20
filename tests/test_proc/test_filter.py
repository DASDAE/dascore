"""Tests for filters."""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import (
    CoordDataError,
    FilterValueError,
    ParameterError,
    UnitError,
)
from dascore.units import Hz, convert_units, get_unit, m
from dascore.utils.misc import broadcast_for_index


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
            _ = random_patch.pass_filter(time=[None, np.nan])

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
        msg = "You must"
        with pytest.raises(ParameterError, match=msg):
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


class TestNotchFilter:
    """Tests for the notch filter."""

    def test_notch_no_kwargs_raises(self, random_patch):
        """Test that no dimension raises an appropriate error."""
        msg = "You must"
        with pytest.raises(ParameterError, match=msg):
            random_patch.notch_filter(q=30)

    def test_notch_filter_time(self, random_patch):
        """Test the notch filter along the time dimension."""
        filtered_patch = random_patch.notch_filter(time=60, q=30)
        assert isinstance(filtered_patch, dc.Patch)
        assert not np.any(np.isnan(filtered_patch.data))

    def test_notch_filter_distance(self, random_patch):
        """Test the notch filter along the distance dimension."""
        filtered_patch = random_patch.notch_filter(distance=0.2, q=20)
        assert isinstance(filtered_patch, dc.Patch)
        assert not np.any(np.isnan(filtered_patch.data))

    def test_notch_filter_time_distance(self, random_patch):
        """Test the notch filter along the time and distance dimension."""
        filtered_patch = random_patch.notch_filter(distance=0.25, time=12, q=40)
        assert isinstance(filtered_patch, dc.Patch)
        assert not np.any(np.isnan(filtered_patch.data))

    def test_notch_filter_high_frequency_error(self, random_patch):
        """Test notch filter raises error for frequency beyond Nyquist."""
        sr = dc.utils.patch.get_dim_sampling_rate(random_patch, "time")
        nyquist = 0.5 * sr
        too_high_freq = nyquist + 1
        msg = f"possible filter values are in [0, {nyquist}] you passed {too_high_freq}"
        with pytest.raises(FilterValueError, match=re.escape(msg)):
            random_patch.notch_filter(time=too_high_freq, q=30)

    def test_notch_filter_time_units(self, random_patch):
        """Test notch filter with time dimension and frequency in Hz."""
        filtered_patch = random_patch.notch_filter(time=60 * Hz, q=40)
        assert isinstance(filtered_patch, dc.Patch)
        assert not np.any(np.isnan(filtered_patch.data))

    def test_notch_filter_distance_units(self, random_patch):
        """Test notch filter with distance dimension in meters."""
        filtered_patch = random_patch.notch_filter(distance=0.4 * 1 / m, q=25)
        assert isinstance(filtered_patch, dc.Patch)
        assert not np.any(np.isnan(filtered_patch.data))


class TestSavgolFilter:
    """Simple tests on Savgol filter."""

    def test_savgol_no_kwargs_raises(self, random_patch):
        """Ensure no kwargs raises."""
        msg = "You must"
        with pytest.raises(ParameterError, match=msg):
            random_patch.savgol_filter(polyorder=2)

    def test_savgol_filter_time(self, random_patch):
        """Test savgol filter in time dimension."""
        out = random_patch.savgol_filter(polyorder=2, time=5)
        assert isinstance(out, dc.Patch)
        assert not np.any(pd.isnull(out.data))

    def test_savgol_samples(self, random_patch):
        """Test using samples parameter."""
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
        """Test for simple filter along the time axis."""
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


class TestSlopeFilter:
    """Test suite for slope filter."""

    def get_slope_array(self, patch, dims=("ft_time", "ft_distance")):
        """Get an array of slopes values for the patch ."""
        dim1, dim2 = dims
        dims = patch.dims
        ndims = patch.ndim
        coord1 = patch.get_array(dim1)
        coord2 = patch.get_array(dim2) + sys.float_info.epsilon
        # Need to add appropriate blank dims to keep overall shape of patch.
        ax1, ax2 = dims.index(dim1), dims.index(dim2)
        shape_1 = broadcast_for_index(ndims, ax1, value=slice(None), fill=None)
        shape_2 = broadcast_for_index(ndims, ax2, value=slice(None), fill=None)
        # Then just allow broadcasting to do its magic
        return coord1[shape_1] / coord2[shape_2]

    @pytest.fixture
    def example_patch(self, event_patch_1):
        """Return the example patch ready for slope filter."""
        out = (
            event_patch_1.taper(time=0.05)
            .pass_filter(time=(1, 500))
            .set_units(time="s", distance="m")
        )
        return out

    def test_basic(self, example_patch):
        """Ensure the basic filter works."""
        filt = [2e3, 2.2e3, 8e3, 2e4]
        filtered_patch = example_patch.slope_filter(filt=filt)
        assert isinstance(filtered_patch, dc.Patch)
        assert filtered_patch.shape == example_patch.shape
        assert not np.array_equal(filtered_patch.data, example_patch.data)

    def test_attenuated_slopes(self, event_patch_1):
        """Ensure attenuated slopes are much lower in absolute values."""
        # For some reason when padding isn't performed the attenation can
        # be slightly off, need to look into thos.
        example_patch = event_patch_1.pad(time="fft", distance="fft")
        filt = [2e3, 2.2e3, 8e3, 2e4]
        filtered_patch = example_patch.slope_filter(filt=filt)
        dft_unfiltered = example_patch.dft(("time", "distance"))
        dft_filtered = filtered_patch.dft(("time", "distance"))
        dft_ft_filtered = dft_filtered.slope_filter(filt=filt)
        # Get slope values
        slope = np.abs(self.get_slope_array(dft_filtered))
        # Get values that should be zeroed by filter.
        in_attenuated_range = (slope <= filt[0]) | (slope >= filt[3])
        # Ensure the patch filtered in ft domain is nearly 0
        assert np.allclose(dft_ft_filtered.data[in_attenuated_range], 0)
        assert np.allclose(dft_filtered.data[in_attenuated_range], 0)
        # compare filtered vs unfiltered absolute values
        unfilt = np.abs(dft_unfiltered.data[in_attenuated_range].flatten())
        filt = np.abs(dft_filtered.data[in_attenuated_range].flatten())
        assert np.all(filt < unfilt)

    def test_directional_filter(self, example_patch):
        """Ensure directional logic runs."""
        filtered_patch = example_patch.slope_filter(
            filt=[2e3, 2.2e3, 8e3, 2e4], directional=True
        )

        assert isinstance(filtered_patch, dc.Patch)

    def test_notch_filter(self, example_patch):
        """Ensure notching can be performed with slope filter."""
        filtered_patch = example_patch.slope_filter(
            filt=[2e3, 2.2e3, 8e3, 2e4], notch=True
        )
        assert isinstance(filtered_patch, dc.Patch)

    def test_different_params_not_equal(self, example_patch):
        """Ensure filter is sensitive to different parameters."""
        filtered_patch1 = example_patch.slope_filter(filt=[1e3, 1.5e3, 5e3, 1e4])
        filtered_patch2 = example_patch.slope_filter(filt=[2e3, 2.2e3, 8e3, 2e4])
        assert not np.array_equal(filtered_patch1.data, filtered_patch2.data)

    def test_fk_input_patch(self, example_patch):
        """Ensure an input fk patch is returned in the fk domain."""
        fk_patch = example_patch.dft(dim=("time", "distance"), real="time")
        out = fk_patch.slope_filter(filt=[1e3, 1.5e3, 5e3, 1e4])
        # Ensure the output is still in the fk domain
        dims = set(out.dims)
        assert "ft_time" in dims
        assert "ft_distance" in dims

    def test_other_dims(self, example_patch):
        """Ensure dims other than time and distance work."""
        patch = example_patch.rename_coords(time="money")
        out = patch.slope_filter(
            filt=[2e3, 2.2e3, 8e3, 2e4],
            dims=("money", "distance"),
        )
        assert out.dims == patch.dims

    def test_bad_filt_raises(self, example_patch):
        """Bad filter params should raise Parameter error."""
        msg = "filt must be a sorted length 4 sequence"
        filt = [1e3, 1.5e3, 5e3]
        with pytest.raises(ParameterError, match=msg):
            example_patch.slope_filter(filt=filt)

    def test_bad_dims(self, example_patch):
        """Ensure passing dimensions not in patch raises."""
        patch = example_patch.rename_coords(time="money")
        msg = "are missing from patch"
        filt = [1e3, 1.5e3, 5e3, 10e3]
        with pytest.raises(ParameterError, match=msg):
            patch.slope_filter(filt=filt, dims=("time", "distance"))

    def test_units_raise_no_unit_coords(self, example_patch):
        """Ensure A UnitError is raised if one of the coords doesn't have units."""
        patch = example_patch.set_units(distance="")
        filt = np.array([1e3, 1.5e3, 5e3, 10e3]) * get_unit("m/s")
        with pytest.raises(UnitError):
            patch.slope_filter(filt=filt)

    def test_units(self, example_patch):
        """Ensure units can be specified on filt."""
        filt1 = np.array([1e3, 1.5e3, 5e3, 10e3])
        filt2 = convert_units(filt1, "ft/s", "m/s")
        # All these should provide the same filter.
        out1 = example_patch.slope_filter(filt=filt1)
        out2 = example_patch.slope_filter(filt=filt1 * get_unit("m / s"))
        out3 = example_patch.slope_filter(filt=filt2 * get_unit("ft / s"))

        assert np.allclose(out1.data, out2.data)
        assert np.allclose(out2.data, out3.data)

    def test_inverted_units(self, example_patch):
        """Ensure units are automatically inverted (eg slowness should work)"""
        filt = np.array([1e3, 1.5e3, 5e3, 10e3])
        slowness = np.sort(1 / filt * get_unit("s/m"))
        out1 = example_patch.slope_filter(filt=slowness)
        out2 = example_patch.slope_filter(filt=filt * get_unit("m/s"))
        assert np.allclose(out1.data, out2.data)

    def test_units_list(self, example_patch):
        """Ensure units as a list still work (see #463)."""
        speed = 5_000 * dc.get_quantity("m/s")
        filt = [speed * 0.90, speed * 0.95, speed * 1.05, speed * 1.1]
        # The test passes if this line doesn't raise an error.
        out = example_patch.slope_filter(filt)
        assert isinstance(out, dc.Patch)
