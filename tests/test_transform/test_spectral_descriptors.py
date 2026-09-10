"""Tests for spectral descriptor transforms."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc


@pytest.fixture(scope="class")
def sine_patch():
    """Return a simple sine-wave patch."""
    return dc.get_example_patch(
        "sin_wav", sample_rate=100, duration=3, frequency=2
    ).set_units(time="s")


@pytest.fixture(scope="class")
def sine_dft(sine_patch):
    """Return a full DFT of the sine patch."""
    return sine_patch.dft("time", pad=False)


class TestSpectralAPI:
    """Tests for the spectral descriptor public API."""

    @pytest.mark.parametrize(
        "name", ["spectral_peak_frequency", "spectral_peak_amplitude"]
    )
    def test_public_exports(self, sine_dft, name):
        """Patch methods and transform exports produce the same result."""
        expected = getattr(dc.transform, name)(sine_dft, fmin=1, fmax=3)
        out = getattr(sine_dft, name)(fmin=1, fmax=3)
        assert out.equals(expected)


class TestSpectralPeaks:
    """Tests for peak selection within a frequency band."""

    @pytest.mark.parametrize(
        "limits, frequency, amplitude",
        [({}, 2.0, 5.0), ({"fmin": 3, "fmax": 4}, 3.0, 4.0)],
    )
    def test_peak_selection(self, limits, frequency, amplitude):
        """Select the largest peak in the band and the first bin on ties."""
        patch = dc.Patch(
            data=np.array([[1.0, 5.0, 4.0, 4.0, 2.0]]),
            coords={"distance": np.array([0.0]), "ft_time": np.arange(1.0, 6.0)},
            dims=("distance", "ft_time"),
        )
        out_frequency = patch.spectral_peak_frequency(
            spectral_format="amplitude", **limits
        )
        out_amplitude = patch.spectral_peak_amplitude(
            spectral_format="amplitude", **limits
        )
        assert np.allclose(out_frequency.data, frequency)
        assert np.allclose(out_amplitude.data, amplitude)


class TestSpectralCentroid:
    """Tests for spectral centroid."""

    def test_dft_input(self, sine_dft):
        """Spectral centroid accepts DFT input."""
        out = sine_dft.spectral_centroid(fmin=1, fmax=3)

        assert out.dims == ("distance",)
        assert np.allclose(out.data, 2.0, rtol=0.01)

    def test_stft_input(self, sine_patch):
        """Spectral centroid accepts STFT input."""
        spec = sine_patch.stft(time=100, overlap=0, samples=True)
        out = spec.spectral_centroid(fmin=1, fmax=3)

        assert out.dims == ("distance", "time")
        assert np.all(np.isfinite(out.data))

    def test_spectral_representations(self, sine_patch):
        """DFT output representations are handled from metadata."""
        fft = sine_patch.dft("time", real=True, pad=False, output="FFT")
        amplitude = sine_patch.dft("time", real=True, pad=False, output="AS")
        power = sine_patch.dft("time", real=True, pad=False, output="PS")
        density = sine_patch.dft("time", real=True, pad=False, output="PSD")

        expected = fft.spectral_centroid(fmin=1, fmax=3)

        assert amplitude.spectral_centroid(fmin=1, fmax=3).equals(expected, close=True)
        assert power.spectral_centroid(fmin=1, fmax=3).equals(expected, close=True)
        assert density.spectral_centroid(fmin=1, fmax=3).equals(expected, close=True)

    def test_real_spectra_need_known_type(self, sine_dft):
        """Real-valued spectra without metadata require an explicit format."""
        unknown = sine_dft.abs().update_attrs(data_type=None)

        with pytest.raises(ValueError, match="requires complex"):
            unknown.spectral_centroid()

        out = unknown.spectral_centroid(spectral_format="amplitude", fmin=1, fmax=3)

        assert np.allclose(out.data, 2.0, rtol=0.01)


class TestFrequencySelection:
    """Tests for frequency-axis and negative-frequency handling."""

    def test_negative_frequencies_raise(self, sine_dft):
        """Negative frequency bins can be rejected."""
        with pytest.raises(ValueError, match="negative frequencies"):
            sine_dft.spectral_centroid(negative_frequencies="raise")

    def test_negative_frequencies_auto_drops_symmetric(self, sine_dft):
        """Auto mode drops negative bins for symmetric spectra."""
        out = sine_dft.spectral_peak_frequency()

        assert np.all(out.data >= 0)
        assert np.allclose(out.data, 2.0, rtol=0.01)

    def test_negative_frequencies_auto_rejects_nonsymmetric(self, sine_dft):
        """Auto mode rejects nonsymmetric negative-frequency spectra."""
        data = sine_dft.data.copy()
        freq_axis = sine_dft.get_axis("ft_time")
        neg_index = sine_dft.get_array("ft_time") < 0
        data[np.compress(neg_index, np.arange(data.shape[freq_axis])), :] *= 2
        patch = sine_dft.update(data=data)

        with pytest.raises(ValueError, match="non-symmetric power"):
            patch.spectral_centroid()

        out = patch.spectral_centroid(negative_frequencies="drop", fmin=1, fmax=3)

        assert np.allclose(out.data, 2.0, rtol=0.01)

    def test_multiple_ft_dims_require_dim(self, sine_patch):
        """Ambiguous Fourier axes require an explicit dimension."""
        patch = sine_patch.dft(("time", "distance"), pad=False)

        with pytest.raises(ValueError, match="Multiple Fourier dimensions"):
            patch.spectral_centroid()

        out = patch.spectral_centroid(dim="time", fmin=1, fmax=3)

        assert out.dims == ("ft_distance",)


class TestOtherDescriptors:
    """Smoke tests for remaining spectral descriptors."""

    def test_descriptors_accept_dft_input(self, sine_dft):
        """All descriptors accept Fourier-domain input."""
        funcs = (
            sine_dft.median_frequency,
            sine_dft.spectral_peak_frequency,
            sine_dft.spectral_peak_amplitude,
            sine_dft.spectral_entropy,
            sine_dft.spectral_kurtosis,
            sine_dft.spectral_flatness,
        )

        for func in funcs:
            out = func(fmin=1, fmax=3)
            assert out.dims == ("distance",)
            assert np.all(np.isfinite(out.data) | np.isnan(out.data))

    def test_time_domain_input_raises(self, sine_patch):
        """Descriptors reject non-Fourier input."""
        with pytest.raises(ValueError, match="Fourier-domain input"):
            sine_patch.spectral_centroid()

    def test_db_spectra_raise(self, sine_patch):
        """Decibel spectra are rejected."""
        spec = sine_patch.dft("time", real=True, output="PS", db=True)

        with pytest.raises(ValueError, match="Decibel-scaled"):
            spec.spectral_centroid()

    def test_peak_amplitude(self, sine_patch):
        """Peak amplitude agrees across amplitude and power spectra."""
        amplitude = sine_patch.dft("time", real=True, pad=False, output="AS")
        power = sine_patch.dft("time", real=True, pad=False, output="PS")

        expected = amplitude.spectral_peak_amplitude(fmin=1, fmax=3)
        out = power.spectral_peak_amplitude(fmin=1, fmax=3)

        assert out.equals(expected, close=True)
