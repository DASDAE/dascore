"""Tests for kurtosis transform."""

from __future__ import annotations

import numpy as np
import pytest


class TestKurtosis:
    """Tests for kurtosis transform."""

    def test_runs_recursive(self, random_patch):
        """Ensure recursive kurtosis runs and returns a patch."""
        out = random_patch.kurtosis(winlen=0.01, recursive=True)

        assert out.dims == random_patch.dims
        assert out.data.shape == random_patch.data.shape
        assert out.attrs.data_type == "Kurtosis"
        assert out.attrs.data_units is None

    def test_runs_windowed(self, random_patch):
        """Ensure windowed kurtosis runs and returns a patch."""
        out = random_patch.kurtosis(winlen=0.01, recursive=False)

        assert out.dims == random_patch.dims
        assert out.data.shape == random_patch.data.shape
        assert out.attrs.data_type == "Kurtosis"
        assert out.attrs.data_units is None

    def test_patch_method_runs(self, random_patch):
        """Ensure kurtosis is registered as a patch method."""
        out = random_patch.kurtosis(winlen=0.01)

        assert out.dims == random_patch.dims
        assert out.data.shape == random_patch.data.shape
        assert out.attrs.data_type == "Kurtosis"

    def test_windowed_constant_data_returns_nan(self, random_patch):
        """Ensure zero-variance windows produce NaN values."""
        patch = random_patch.update(data=np.ones_like(random_patch.data))

        out = patch.kurtosis(winlen=0.01, recursive=False)

        assert np.all(np.isnan(out.data))

    def test_windowed_impulse_has_high_kurtosis(self, random_patch):
        """Ensure an impulsive signal increases windowed kurtosis."""
        data = random_patch.data.copy()
        time_axis = random_patch.dims.index("time")
        index = [slice(None)] * data.ndim
        index[time_axis] = data.shape[time_axis] // 2
        data[tuple(index)] = 10.0

        patch = random_patch.update(data=data)
        out = patch.kurtosis(winlen=0.01, recursive=False)

        assert np.nanmax(out.data) > 1.0

    def test_recursive_output_is_finite_for_random_data(self, random_patch):
        """Ensure recursive kurtosis produces finite values for random data."""
        out = random_patch.kurtosis(winlen=0.01, recursive=True)

        assert np.isfinite(out.data).any()

    def test_recursive_and_windowed_are_different_algorithms(self, random_patch):
        """Ensure recursive and windowed modes do not accidentally alias."""
        rec = random_patch.kurtosis(winlen=0.01, recursive=True)
        win = random_patch.kurtosis(winlen=0.01, recursive=False)

        assert not np.allclose(rec.data, win.data, equal_nan=True)

    def test_output_preserves_original_dim_order(self, random_patch):
        """Ensure output is transposed back to the original dimension order."""
        patch_t = random_patch.transpose("time", ...)

        out_t = patch_t.kurtosis(winlen=0.01, dim="time")
        out_d = random_patch.kurtosis(winlen=0.01, dim="time")

        assert out_t.dims == patch_t.dims
        assert out_d.dims == random_patch.dims

    def test_non_time_dim(self, random_patch):
        """Ensure kurtosis can run over a non-time dimension."""
        dim = "distance"
        step = random_patch.get_coord(dim).step
        winlen = 3 * step

        out = random_patch.kurtosis(winlen=winlen, dim=dim, recursive=False)

        assert out.dims == random_patch.dims
        assert out.data.shape == random_patch.data.shape
        assert out.attrs.data_type == "Kurtosis"

    def test_negative_winlen_raises(self, random_patch):
        """Ensure negative window length raises."""
        with pytest.raises(ValueError, match="winlen must be positive"):
            random_patch.kurtosis(winlen=-1, recursive=False)

    def test_zero_winlen_raises(self, random_patch):
        """Ensure zero window length raises."""
        with pytest.raises(ValueError, match="winlen must be positive"):
            random_patch.kurtosis(winlen=0, recursive=False)

    def test_too_small_winlen_raises(self, random_patch):
        """Ensure windows shorter than two samples raise."""
        step = random_patch.get_coord("time").step / np.timedelta64(1, "s")

        with pytest.raises(ValueError, match="winlen is too small"):
            random_patch.kurtosis(winlen=step / 2, recursive=False)

    def test_recursive_requires_positive_winlen(self, random_patch):
        """Ensure recursive mode rejects invalid winlen."""
        with pytest.raises(ValueError, match="winlen must be positive"):
            random_patch.kurtosis(winlen=0, recursive=True)

    def test_attrs_are_set(self, random_patch):
        """Ensure output attrs are set consistently."""
        out = random_patch.kurtosis(winlen=0.01)

        assert out.attrs.data_type == "Kurtosis"
        assert out.attrs.data_units is None
