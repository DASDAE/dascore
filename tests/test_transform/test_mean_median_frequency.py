"""Tests for mean and median frequency transforms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import get_window, welch

from dascore import units
from dascore.transform.mean_median_frequency import _fft_psd, _get_psd_in_window, _welch


class TestWelch:
    """Tests for the specialized Welch PSD helper."""

    def test_output_shapes(self):
        """Ensure frequency and PSD arrays have expected shapes."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(4, 3, 128))
        win = get_window("hann", 64)

        freq, pxx = _welch(data, win=win, fs=100.0, nperseg=64)

        assert freq.shape == (33,)
        assert pxx.shape == (4, 3, 33)

    def test_short_input_reduces_nperseg(self):
        """Ensure nperseg is reduced when the signal is shorter."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(4, 3, 32))
        win = get_window("hann", 64)

        freq, pxx = _welch(data, win=win, fs=100.0, nperseg=64)

        assert freq.shape == (17,)
        assert pxx.shape == (4, 3, 17)

    def test_matches_scipy_welch_even_nperseg(self):
        """Ensure helper matches scipy.signal.welch for even nperseg."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(4, 3, 128))
        nperseg = 64
        fs = 100.0
        win = get_window("hann", nperseg)

        freq, pxx = _welch(data, win=win, fs=fs, nperseg=nperseg)
        expected_freq, expected_pxx = welch(
            data,
            fs=fs,
            window=win,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
            return_onesided=True,
            axis=-1,
        )

        assert np.allclose(freq, expected_freq)
        assert np.allclose(pxx, expected_pxx)

    def test_matches_scipy_welch_odd_nperseg(self):
        """Ensure helper matches scipy.signal.welch for odd nperseg."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(4, 3, 129))
        nperseg = 63
        fs = 100.0
        win = get_window("hann", nperseg)

        freq, pxx = _welch(data, win=win, fs=fs, nperseg=nperseg)
        expected_freq, expected_pxx = welch(
            data,
            fs=fs,
            window=win,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
            return_onesided=True,
            axis=-1,
        )

        assert np.allclose(freq, expected_freq)
        assert np.allclose(pxx, expected_pxx)

    def test_constant_signal_has_zero_non_dc_power(self):
        """Ensure detrending removes constant signal power."""
        data = np.ones((2, 3, 128))
        nperseg = 64
        win = get_window("hann", nperseg)

        _, pxx = _welch(data, win=win, fs=100.0, nperseg=nperseg)

        assert np.allclose(pxx, 0.0)

    def test_frequency_axis_uses_sampling_frequency(self):
        """Ensure frequency spacing is fs / nperseg."""
        data = np.ones((2, 3, 128))
        nperseg = 64
        fs = 200.0
        win = get_window("hann", nperseg)

        freq, _ = _welch(data, win=win, fs=fs, nperseg=nperseg)

        assert np.allclose(np.diff(freq), fs / nperseg)
        assert freq[0] == 0
        assert freq[-1] == fs / 2


class TestMeanFrequency:
    """Tests for mean_frequency patch function."""

    def test_runs(self, random_patch):
        """Ensure mean_frequency runs as a patch method."""
        out = random_patch.mean_frequency(winlen=0.3, step=0.01)

        assert out.attrs.data_type == "Mean Frequency"
        assert out.attrs.data_units == units.Hz
        assert "time" in out.dims
        assert "distance" in out.dims

    def test_fft_method_runs(self, random_patch):
        """Ensure FFT method runs."""
        out = random_patch.mean_frequency(winlen=0.3, step=0.01, method="fft")

        assert out.attrs.data_type == "Mean Frequency"
        assert out.attrs.data_units == units.Hz
        assert np.isfinite(out.data).any()

    def test_frequency_bounds(self, random_patch):
        """Ensure frequency bounds are accepted."""
        out = random_patch.mean_frequency(
            winlen=0.3,
            step=0.01,
            fmin=10,
            fmax=100,
        )

        assert out.attrs.data_type == "Mean Frequency"
        assert out.attrs.data_units == units.Hz

    def test_invalid_frequency_bounds_raise(self, random_patch):
        """Ensure invalid frequency bounds raise."""
        with pytest.raises(ValueError, match="must be smaller than fmax"):
            random_patch.mean_frequency(
                winlen=0.3,
                step=0.01,
                fmin=100,
                fmax=10,
            )

    def test_transposes_output_when_distance_axis_changes(self, random_patch):
        """Ensure output is transposed back when rolling/apply changes axis order."""
        patch = random_patch.transpose("time", "distance")

        out = patch.mean_frequency(
            winlen=0.01,
            step=0.01,
        )

        assert out.get_axis("distance") == patch.get_axis("distance")
        assert out.dims == patch.dims

    def test_mean_frequency_step_defaults_to_time_step(self, random_patch):
        """Cover step=None branch in mean_frequency."""
        dt = random_patch.get_coord("time").step
        if isinstance(dt, np.timedelta64):
            dt = dt / np.timedelta64(1, "s")

        out = random_patch.mean_frequency(winlen=0.01, step=None, method="fft")
        expected = random_patch.mean_frequency(winlen=0.01, step=dt, method="fft")

        assert np.allclose(out.data, expected.data, equal_nan=True)


class TestMedianFrequency:
    """Tests for median_frequency patch function."""

    def test_runs(self, random_patch):
        """Ensure median_frequency runs as a patch method."""
        out = random_patch.median_frequency(winlen=0.3, step=0.01)

        assert out.attrs.data_type == "Median Frequency"
        assert out.attrs.data_units == units.Hz
        assert "time" in out.dims
        assert "distance" in out.dims

    def test_fft_method_runs(self, random_patch):
        """Ensure FFT method runs."""
        out = random_patch.median_frequency(winlen=0.3, step=0.01, method="fft")

        assert out.attrs.data_type == "Median Frequency"
        assert out.attrs.data_units == units.Hz
        assert np.isfinite(out.data).any()

    def test_frequency_bounds(self, random_patch):
        """Ensure frequency bounds are accepted."""
        out = random_patch.median_frequency(
            winlen=0.3,
            step=0.01,
            fmin=10,
            fmax=100,
        )

        assert out.attrs.data_type == "Median Frequency"
        assert out.attrs.data_units == units.Hz

    def test_invalid_frequency_bounds_raise(self, random_patch):
        """Ensure invalid frequency bounds raise."""
        with pytest.raises(ValueError, match="must be smaller than fmax"):
            random_patch.median_frequency(
                winlen=0.3,
                step=0.01,
                fmin=100,
                fmax=10,
            )

    def test_transposes_output_when_distance_axis_changes(self, random_patch):
        """Ensure output is transposed back when rolling/apply changes axis order."""
        patch = random_patch.transpose("time", "distance")

        out = patch.median_frequency(
            winlen=0.01,
            step=0.01,
        )

        assert out.get_axis("distance") == patch.get_axis("distance")
        assert out.dims == patch.dims

    def test_median_frequency_step_defaults_to_time_step(self, random_patch):
        """Cover step=None branch in median_frequency."""
        dt = random_patch.get_coord("time").step
        if isinstance(dt, np.timedelta64):
            dt = dt / np.timedelta64(1, "s")

        out = random_patch.median_frequency(winlen=0.01, step=None, method="fft")
        expected = random_patch.median_frequency(winlen=0.01, step=dt, method="fft")

        assert np.allclose(out.data, expected.data, equal_nan=True)


class TestFFTPSD:
    """Tests for simple FFT PSD helper."""

    def test_even_length_doubles_only_interior_bins(self):
        """Ensure even-length FFT excludes Nyquist from doubling."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(1, 1, 128))

        _, pxx = _fft_psd(data, fs=100.0)

        n = data.shape[-1]
        x = data - np.mean(data, axis=-1, keepdims=True)
        xft = np.fft.rfft(x, axis=-1)

        raw = (np.abs(xft) ** 2) / (100.0 * n)

        # DC unchanged
        assert np.allclose(pxx[..., 0], raw[..., 0])

        # Nyquist unchanged
        assert np.allclose(pxx[..., -1], raw[..., -1])

        # interior bins doubled
        assert np.allclose(pxx[..., 1:-1], raw[..., 1:-1] * 2)

    def test_fft_psd_odd_length_doubles_all_non_dc_bins(self):
        """Cover the odd-length branch in _fft_psd."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(2, 3, 127))
        fs = 100.0

        _, pxx = _fft_psd(data, fs=fs)

        n = data.shape[-1]
        demeaned = data - np.mean(data, axis=-1, keepdims=True)
        raw = np.abs(np.fft.rfft(demeaned, axis=-1)) ** 2 / (fs * n)

        assert np.allclose(pxx[..., 0], raw[..., 0])
        assert np.allclose(pxx[..., 1:], raw[..., 1:] * 2.0)

    def test_invalid_frequency_bounds_raise(self):
        """Cover fmin >= fmax validation in _get_psd_in_window."""
        data = np.ones((2, 3, 128))

        with pytest.raises(ValueError, match=r"must be smaller than fmax"):
            _get_psd_in_window(data, method="fft", dt=0.01, fmin=20, fmax=10)
