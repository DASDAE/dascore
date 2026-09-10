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

    def test_zero_power_returns_nan_frequency(self):
        """Zero-power spectra have no defined peak frequency."""
        patch = dc.Patch(
            data=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]),
            coords={"distance": np.array([0.0, 10.0]), "ft_time": np.arange(3.0)},
            dims=("distance", "ft_time"),
        )

        out = patch.spectral_peak_frequency(spectral_format="amplitude")

        assert np.isnan(out.data[0])
        assert out.data[1] == 1.0


class TestOutputCoordinates:
    """Tests for coordinate metadata after spectral reduction."""

    @pytest.mark.parametrize(
        "name",
        [
            "median_frequency",
            "spectral_centroid",
            "spectral_peak_frequency",
            "spectral_peak_amplitude",
            "spectral_entropy",
            "spectral_kurtosis",
            "spectral_flatness",
        ],
    )
    @pytest.mark.parametrize("frequency_first", [False, True])
    def test_preserves_coordinates(self, name, frequency_first):
        """Keep unrelated coordinates and units; drop frequency dependents."""
        patch = dc.Patch(
            data=np.ones((2, 5)),
            coords={
                "distance": np.array([0.0, 10.0]),
                "ft_time": np.arange(1.0, 6.0),
                "elevation": ("distance", np.array([100.0, 101.0])),
                "reference": ((), np.array([42.0])),
                "frequency_label": ("ft_time", np.arange(5.0)),
                "calibration": (("distance", "ft_time"), np.ones((2, 5))),
            },
            dims=("distance", "ft_time"),
        ).set_units(distance="m", elevation="m", reference="m", ft_time="Hz")
        if frequency_first:
            patch = patch.transpose("ft_time", "distance")

        out = getattr(patch, name)(spectral_format="amplitude")

        assert out.dims == ("distance",)
        assert out.shape == (2,)
        assert set(out.coords.coord_map) == {"distance", "elevation", "reference"}
        for coord in out.coords.coord_map:
            assert out.get_coord(coord) == patch.get_coord(coord)
            assert out.coords.dim_map[coord] == patch.coords.dim_map[coord]


class TestSpectralValidation:
    """Tests for invalid input and spectral format inference."""

    @pytest.fixture
    def spectrum(self):
        """Return a spectrum without transform metadata."""
        return dc.Patch(
            data=np.array([[1.0, 4.0, 1.0]]),
            coords={"distance": np.array([0.0]), "ft_time": np.arange(1.0, 4.0)},
            dims=("distance", "ft_time"),
        )

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"dim": "missing"}, "Fourier dimension.*was not found"),
            ({"spectral_format": "invalid"}, "Unknown spectral_format"),
            ({"negative_frequencies": "invalid"}, "negative_frequencies must be"),
            ({"fmin": 100}, "exclude all Fourier frequency bins"),
            ({"fmin": 3, "fmax": 1}, "exclude all Fourier frequency bins"),
        ],
    )
    def test_invalid_options(self, sine_dft, kwargs, message):
        """Invalid dimensions, options, and empty bands raise clear errors."""
        with pytest.raises(ValueError, match=message):
            sine_dft.spectral_centroid(**kwargs)

    @pytest.mark.parametrize("spectral_format", ["amplitude", "power", "density"])
    def test_complex_real_format(self, spectrum, spectral_format):
        """Real spectral representations reject complex data."""
        patch = spectrum.update(data=spectrum.data.astype(complex))
        with pytest.raises(ValueError, match="requires real-valued spectral data"):
            patch.spectral_centroid(spectral_format=spectral_format)

    @pytest.mark.parametrize("spectral_format", ["amplitude", "power", "density"])
    def test_negative_values(self, spectrum, spectral_format):
        """Amplitude, power, and density cannot contain negative values."""
        patch = spectrum.update(data=-spectrum.data)
        with pytest.raises(ValueError, match="must be non-negative"):
            patch.spectral_centroid(spectral_format=spectral_format)

    def test_unknown_real_format(self, spectrum):
        """Real spectra without identifying metadata require a format."""
        with pytest.raises(ValueError, match="Could not infer spectral"):
            spectrum.spectral_centroid()

    @pytest.mark.parametrize(
        "data_type", ["amplitude spectrum", "power spectrum", "spectral density"]
    )
    def test_data_type_inference(self, spectrum, data_type):
        """Data-type metadata identifies otherwise unmarked spectra."""
        out = spectrum.update_attrs(data_type=data_type).spectral_centroid()
        assert np.allclose(out.data, 2.0)

    def test_complex_inference(self, spectrum):
        """Unmarked complex spectra are interpreted as Fourier coefficients."""
        patch = spectrum.update(data=spectrum.data.astype(complex))
        out = patch.spectral_centroid()
        assert np.allclose(out.data, 2.0)


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

    def test_entropy_single_frequency(self, sine_dft):
        """Normalized entropy of one frequency bin is zero."""
        freq = sine_dft.get_array("ft_time")
        frequency = freq[freq > 0][0]
        out = sine_dft.spectral_entropy(fmin=frequency, fmax=frequency)
        assert np.allclose(out.data, 0.0)

    def test_peak_amplitude(self, sine_patch):
        """Peak amplitude agrees across amplitude and power spectra."""
        amplitude = sine_patch.dft("time", real=True, pad=False, output="AS")
        power = sine_patch.dft("time", real=True, pad=False, output="PS")

        expected = amplitude.spectral_peak_amplitude(fmin=1, fmax=3)
        out = power.spectral_peak_amplitude(fmin=1, fmax=3)

        assert out.equals(expected, close=True)

    @pytest.mark.parametrize("output", ["FFT", "AS", "PS", "PSD"])
    def test_peak_amplitude_units(self, sine_patch, output):
        """Peak amplitude units match the returned linear amplitude."""
        patch = sine_patch.update_attrs(data_units="m/s").dft(
            "time", real=True, pad=False, output=output
        )

        out = patch.spectral_peak_amplitude(fmin=1, fmax=3)
        input_units = dc.get_quantity(patch.attrs.data_units)
        expected = input_units
        if output in {"PS", "PSD"}:
            expected = input_units**0.5

        assert out.attrs.data_units == expected
