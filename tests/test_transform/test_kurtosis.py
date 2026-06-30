"""Tests for kurtosis transform."""

from __future__ import annotations

import numpy as np
import pytest

from dascore.exceptions import ParameterError
from dascore.transform.kurtosis import (
    _moving_sum,
    _recursive_kurtosis,
    _validate_window,
    _windowed_kurtosis,
)


class TestKurtosisHelpers:
    """Tests for kurtosis helper functions."""

    def test_validate_window_returns_sample_count(self):
        """Ensure window length converts to sample count."""
        assert _validate_window(winlen=0.02, step=0.01) == 2

    def test_validate_window_rejects_non_positive_winlen(self):
        """Ensure non-positive window lengths raise."""
        with pytest.raises(ValueError, match="winlen must be positive"):
            _validate_window(winlen=0, step=0.01)

    def test_validate_window_rejects_too_short_window(self):
        """Ensure windows shorter than two samples raise."""
        with pytest.raises(ValueError, match="too small"):
            _validate_window(winlen=0.005, step=0.01)

    def test_moving_sum_values(self):
        """Ensure moving sum uses clipped centered windows."""
        data = np.arange(5, dtype=float).reshape(5, 1)

        out, counts = _moving_sum.func(data, nwin=3)

        assert np.allclose(out[:, 0], [1, 3, 6, 9, 7])
        assert np.allclose(counts, [2, 3, 3, 3, 2])

    def test_windowed_kurtosis_constant_data_returns_nan(self):
        """Ensure zero-variance data returns NaNs."""
        data = np.ones((10, 3), dtype=float)

        out = _windowed_kurtosis.func(data, nwin=3)

        assert np.isnan(out).all()

    def test_windowed_kurtosis_non_constant_data(self):
        """Ensure positive-variance windows return finite values."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(10, 3))

        out = _windowed_kurtosis.func(data, nwin=3)

        assert out.shape == data.shape
        assert np.isfinite(out).any()

    def test_recursive_kurtosis_preserves_shape(self):
        """Ensure recursive helper preserves input shape."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(20, 3))
        varx = np.var(data, axis=0)

        out = _recursive_kurtosis.func(data, step=0.01, winlen=0.05, varx=varx)

        assert out.shape == data.shape

    def test_recursive_kurtosis_constant_data_returns_nan(self):
        """Ensure recursive helper handles zero-variance data."""
        data = np.zeros((20, 3), dtype=float)

        out = _recursive_kurtosis.func(
            data,
            step=0.01,
            winlen=0.05,
            varx=np.zeros(data.shape[1]),
        )

        assert np.isnan(out).all()


class TestKurtosis:
    """Tests for kurtosis patch function."""

    def test_windowed_runs(self, random_patch):
        """Ensure windowed kurtosis runs."""
        out = random_patch.kurtosis(winlen=0.01, recursive=False)

        assert out.dims == random_patch.dims
        assert out.data.shape == random_patch.data.shape
        assert out.attrs.data_type == "kurtosis"
        assert out.attrs.data_units is None

    def test_recursive_runs(self, random_patch):
        """Ensure recursive kurtosis runs."""
        out = random_patch.kurtosis(winlen=0.01, recursive=True)

        assert out.dims == random_patch.dims
        assert out.data.shape == random_patch.data.shape
        assert out.attrs.data_type == "kurtosis"
        assert out.attrs.data_units is None

    def test_restores_original_dimension_order(self, random_patch):
        """Ensure internal transpose does not affect output dims."""
        patch = random_patch.transpose("distance", "time")

        out = patch.kurtosis(winlen=0.01, dim="time", recursive=False)

        assert out.dims == patch.dims
        assert out.data.shape == patch.data.shape

    def test_runs_on_distance_dimension(self, random_patch):
        """Ensure kurtosis can operate along distance."""
        step = random_patch.get_coord("distance").step

        out = random_patch.kurtosis(
            winlen=3 * step,
            dim="distance",
            recursive=False,
        )

        assert out.dims == random_patch.dims
        assert out.data.shape == random_patch.data.shape

    def test_runs_on_reversed_coordinate(self, random_patch):
        """Descending coordinates should use their absolute sample spacing."""
        time = random_patch.get_coord("time").data[::-1]
        patch = random_patch.update_coords(time=time)

        out = patch.kurtosis(winlen=0.01, dim="time")

        assert out.dims == patch.dims
        assert out.data.shape == patch.data.shape

    def test_uneven_coordinate_raises(self, random_patch):
        """Uneven coordinates should raise a clear ParameterError."""
        time = random_patch.get_coord("time").data.copy()
        time[2:] += np.timedelta64(1, "s")
        patch = random_patch.update_coords(time=time)

        with pytest.raises(ParameterError, match="not evenly sampled"):
            patch.kurtosis(winlen=0.01, dim="time")

    def test_invalid_winlen_raises(self, random_patch):
        """Ensure invalid winlen raises."""
        with pytest.raises(ValueError, match="winlen must be positive"):
            random_patch.kurtosis(winlen=0)

    def test_too_short_winlen_raises(self, random_patch):
        """Ensure too-short winlen raises."""
        step = random_patch.get_coord("time").step

        if isinstance(step, np.timedelta64):
            step = step / np.timedelta64(1, "s")

        with pytest.raises(ValueError, match="too small"):
            random_patch.kurtosis(winlen=step / 2)

    def test_constant_data_windowed_returns_nan(self, random_patch):
        """Ensure windowed kurtosis handles constant data."""
        patch = random_patch.update(data=np.ones_like(random_patch.data, dtype=float))

        out = patch.kurtosis(winlen=0.01, recursive=False)

        assert np.isnan(out.data).all()

    def test_constant_data_recursive_does_not_inf(self, random_patch):
        """Ensure recursive kurtosis handles constant data."""
        patch = random_patch.update(data=np.ones_like(random_patch.data, dtype=float))

        out = patch.kurtosis(winlen=0.01, recursive=True)

        assert not np.isinf(out.data).any()
