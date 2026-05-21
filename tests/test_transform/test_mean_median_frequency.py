"""Tests for mean and median frequency transforms."""

from __future__ import annotations

import numpy as np
import pytest

from dascore import units


class TestMeanFrequency:
    """Tests for mean_frequency patch function."""

    def test_runs(self, random_patch):
        """Ensure mean_frequency runs as a patch method."""
        out = random_patch.mean_frequency(winlen=0.01, step=0.01)

        assert out.attrs.data_type == "Mean Frequency"
        assert out.attrs.data_units == units.Hz
        assert "time" in out.dims
        assert "distance" in out.dims

    def test_fft_method_runs(self, random_patch):
        """Ensure FFT method runs."""
        out = random_patch.mean_frequency(winlen=0.01, step=0.01, method="fft")

        assert out.attrs.data_type == "Mean Frequency"
        assert out.attrs.data_units == units.Hz
        assert np.isfinite(out.data).any()

    def test_step_defaults_to_winlen(self, random_patch):
        """Ensure step defaults to winlen."""
        out = random_patch.mean_frequency(winlen=0.01)
        expected = random_patch.mean_frequency(winlen=0.01, step=0.01)

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_frequency_bounds(self, random_patch):
        """Ensure frequency bounds are accepted."""
        out = random_patch.mean_frequency(
            winlen=0.01,
            step=0.01,
            fmin=10,
            fmax=100,
        )

        assert out.attrs.data_type == "Mean Frequency"
        assert out.attrs.data_units == units.Hz

    def test_invalid_frequency_bounds_raise(self, random_patch):
        """Ensure invalid frequency bounds raise."""
        with pytest.raises(AssertionError, match="must be smaller than fmax"):
            random_patch.mean_frequency(
                winlen=0.01,
                step=0.01,
                fmin=100,
                fmax=10,
            )

    def test_collapse_mode_runs(self, random_patch):
        """Ensure winlen=None collapses over the full time range."""
        out = random_patch.mean_frequency()

        assert out.attrs.data_type == "Mean Frequency"
        assert out.attrs.data_units == units.Hz
        assert out.data.shape != random_patch.data.shape


class TestMedianFrequency:
    """Tests for median_frequency patch function."""

    def test_runs(self, random_patch):
        """Ensure median_frequency runs as a patch method."""
        out = random_patch.median_frequency(winlen=0.01, step=0.01)

        assert out.attrs.data_type == "Median Frequency"
        assert out.attrs.data_units == units.Hz
        assert "time" in out.dims
        assert "distance" in out.dims

    def test_fft_method_runs(self, random_patch):
        """Ensure FFT method runs."""
        out = random_patch.median_frequency(winlen=0.01, step=0.01, method="fft")

        assert out.attrs.data_type == "Median Frequency"
        assert out.attrs.data_units == units.Hz
        assert np.isfinite(out.data).any()

    def test_step_defaults_to_winlen(self, random_patch):
        """Ensure step defaults to winlen."""
        out = random_patch.median_frequency(winlen=0.01)
        expected = random_patch.median_frequency(winlen=0.01, step=0.01)

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_frequency_bounds(self, random_patch):
        """Ensure frequency bounds are accepted."""
        out = random_patch.median_frequency(
            winlen=0.01,
            step=0.01,
            fmin=10,
            fmax=100,
        )

        assert out.attrs.data_type == "Median Frequency"
        assert out.attrs.data_units == units.Hz

    def test_invalid_frequency_bounds_raise(self, random_patch):
        """Ensure invalid frequency bounds raise."""
        with pytest.raises(AssertionError, match="must be smaller than fmax"):
            random_patch.median_frequency(
                winlen=0.01,
                step=0.01,
                fmin=100,
                fmax=10,
            )

    def test_collapse_mode_runs(self, random_patch):
        """Ensure winlen=None collapses over the full time range."""
        out = random_patch.median_frequency()

        assert out.attrs.data_type == "Median Frequency"
        assert out.attrs.data_units == units.Hz
        assert out.data.shape != random_patch.data.shape
