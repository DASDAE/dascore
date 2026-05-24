"""Tests for spectra transform."""

from __future__ import annotations

import numpy as np
import pytest

import dascore.units as ureg


class TestSpectra:
    """Tests for spectra transform."""

    def test_runs_time_transform(self, random_patch):
        """Ensure spectra runs along the time dimension."""
        out = random_patch.spectra(dim="time", kind="PSD", db=False)

        assert out.attrs.data_type == "Power Spectral Density"
        assert "ft_time" in out.dims

    def test_runs_distance_transform(self, random_patch):
        """Ensure spectra runs along the distance dimension."""
        out = random_patch.spectra(dim="distance", kind="PSD", db=False)

        assert out.attrs.data_type == "Power Spectral Density"
        assert "ft_distance" in out.dims

    def test_amplitude_spectrum_matches_expected(self, random_patch):
        """Ensure AS matches abs of DFT."""
        out = random_patch.spectra(dim="time", kind="AS", db=False)

        expected = random_patch.dft(dim="time", real=True, pad=True).abs()

        assert np.allclose(out.data, expected.data)
        assert out.attrs.data_type == "Amplitude Spectrum"

    def test_power_spectrum_matches_expected(self, random_patch):
        """Ensure PS matches squared amplitude spectrum."""
        out = random_patch.spectra(dim="time", kind="PS", db=False)

        spec = random_patch.dft(dim="time", real=True, pad=True).abs()
        expected = spec * spec

        assert np.allclose(out.data, expected.data)
        assert out.attrs.data_type == "Power Spectrum"

    def test_psd_matches_expected(self, random_patch):
        """Ensure PSD matches normalized power spectrum."""
        out = random_patch.spectra(dim="time", kind="PSD", db=False)

        spec = random_patch.dft(dim="time", real=True, pad=True).abs()

        n = random_patch.data.shape[random_patch.get_axis("time")]

        fsamp = 1 / (spec.get_coord("ft_time").step * spec.get_coord("ft_time").units)

        expected = spec * spec / (n * fsamp)

        assert np.allclose(out.data, expected.data)
        assert out.attrs.data_type == "Power Spectral Density"

    def test_amplitude_spectrum_db_matches_expected(self, random_patch):
        """Ensure AS db conversion matches expected."""
        out = random_patch.spectra(dim="time", kind="AS", db=True)

        spec = random_patch.dft(dim="time", real=True, pad=True).abs()
        spec += np.finfo(spec.data.dtype).eps

        expected = 20 * spec.log10()

        assert np.allclose(out.data, expected.data)
        assert out.attrs.data_units == ureg.dB

    def test_power_spectrum_db_matches_expected(self, random_patch):
        """Ensure PS db conversion matches expected."""
        out = random_patch.spectra(dim="time", kind="PS", db=True)

        spec = random_patch.dft(dim="time", real=True, pad=True).abs()
        spec += np.finfo(spec.data.dtype).eps

        expected = 10 * (spec * spec).log10()

        assert np.allclose(out.data, expected.data)
        assert out.attrs.data_units == ureg.dB

    def test_psd_db_matches_expected(self, random_patch):
        """Ensure PSD db conversion matches expected."""
        out = random_patch.spectra(dim="time", kind="PSD", db=True)

        spec = random_patch.dft(dim="time", real=True, pad=True).abs()
        spec += np.finfo(spec.data.dtype).eps

        n = random_patch.data.shape[random_patch.get_axis("time")]

        fsamp = 1 / (spec.get_coord("ft_time").step * spec.get_coord("ft_time").units)

        expected = 10 * (spec * spec / (n * fsamp)).log10()

        assert np.allclose(out.data, expected.data)
        assert out.attrs.data_units == ureg.dB

    def test_db_sets_units(self, random_patch):
        """Ensure db=True sets dB units."""
        out = random_patch.spectra(dim="time", kind="PSD", db=True)

        assert out.attrs.data_units == ureg.dB

    def test_non_db_does_not_set_db_units(self, random_patch):
        """Ensure db=False does not set dB units."""
        out = random_patch.spectra(dim="time", kind="PSD", db=False)

        assert out.attrs.data_units != ureg.dB

    def test_invalid_kind_raises(self, random_patch):
        """Ensure invalid spectrum kind raises."""
        with pytest.raises(ValueError):
            random_patch.spectra(dim="time", kind="BAD_KIND")

    def test_frequency_dimension_replaces_input_dimension(self, random_patch):
        """Ensure transformed dimension becomes frequency dimension."""
        out = random_patch.spectra(dim="time", kind="PSD")

        assert "time" not in out.dims
        assert "ft_time" in out.dims

    def test_shape_matches_expected_rfft_output(self, random_patch):
        """Ensure output shape matches rFFT expectation."""
        out = random_patch.spectra(dim="time", kind="PSD")

        n_time = random_patch.data.shape[random_patch.get_axis("time")]
        expected_n_freq = n_time // 2 + 1

        assert out.shape[out.get_axis("ft_time")] == expected_n_freq
