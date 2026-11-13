"""Tests for detrending functions."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.exceptions import ParameterError


class TestDetrend:
    """Tests for detrending data."""

    def test_detrend(self, random_patch):
        """Ensure detrending removes mean."""
        new = random_patch.new(data=random_patch.data + 10)
        # perform detrend, ensure all mean values are close to zero
        det = new.detrend(dim="time", type="linear")
        means = np.mean(det.data, axis=det.get_axis("time"))
        assert np.allclose(means, 0)

    def test_bad_dim_raises(self, random_patch):
        """A dim not in the patch should raise ParameterError."""
        with pytest.raises(ParameterError, match="not in patch dim"):
            random_patch.detrend(dim="not_a_dim")

    def test_bad_type_raises(self, random_patch):
        """Only the fit types scipy supports are accepted."""
        with pytest.raises(ValidationError, match="type"):
            random_patch.detrend(dim="time", type="quadratic")


class TestDetrendNaN:
    """NaN samples are left out of the fit and stay NaN."""

    @pytest.fixture()
    def gap_patch(self):
        """The example patch with a NaN gap in time and an all-NaN trace."""
        patch = dc.get_example_patch()
        data = patch.data.copy()
        data[:, 100:110] = np.nan
        data[0] = np.nan
        return patch.update(data=data)

    @pytest.mark.parametrize("type", ["linear", "constant"])
    def test_fit_skips_gap(self, gap_patch, type):
        """Finite samples match a fit to the finite samples; all-NaN stays NaN."""
        out = gap_patch.detrend("time", type=type).data
        trace = gap_patch.data[1]
        index = np.arange(len(trace))
        valid = ~np.isnan(trace)
        coefs = np.polyfit(index[valid], trace[valid], 1 if type == "linear" else 0)
        expected = trace - np.polyval(coefs, index)
        np.testing.assert_allclose(out[1], expected, equal_nan=True)
        assert np.isnan(out[:, 100:110]).all()
        assert np.isfinite(out[1:, 110:]).all()
        assert np.isnan(out[0]).all()

    @pytest.mark.parametrize("type", ["linear", "constant"])
    def test_narrow_float(self, type):
        """Float16 sums do not overflow; a constant trace detrends to zero."""
        data = np.full((2, 1000), 100, dtype=np.float16)
        data[0, 5] = np.nan
        coords = {"x": np.arange(2), "time": np.arange(1000)}
        patch = dc.Patch(data=data, coords=coords, dims=("x", "time"))
        out = patch.detrend("time", type=type).data
        assert out.dtype == np.float16
        np.testing.assert_allclose(out[1], 0, atol=0.1)
