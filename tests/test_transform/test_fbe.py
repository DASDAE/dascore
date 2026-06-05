"""Tests for Frequency Band Energy transform."""

from __future__ import annotations

import numpy as np
import pytest
from numpy import timedelta64

import dascore.units as ureg


class TestFBE:
    """Tests for Frequency-Band Energy transform."""

    def test_runs_time_filter(self, random_patch):
        """Ensure FBE runs along the time dimension."""
        out = random_patch.fbe(time=(10, 100), window=0.01, step=0.01, db=False)

        assert out.dims == random_patch.dims
        assert out.attrs.data_type == "frequency_band_energy"

    def test_runs_distance_filter(self, random_patch):
        """Ensure FBE runs along the distance dimension."""
        out = random_patch.fbe(distance=(0.01, 0.05), window=5, step=1, db=False)

        assert out.dims == random_patch.dims
        assert out.attrs.data_type == "frequency_band_energy"

    def test_db_false_matches_expected_rms(self, random_patch):
        """Ensure db=False returns filtered rolling RMS."""
        kwargs = {"time": (10, 100)}

        out = random_patch.fbe(**kwargs, window=0.01, step=0.01, db=False)

        filtered = random_patch.pass_filter(**kwargs)
        expected = (filtered**2).rolling(time=0.01, step=0.01).mean() ** 0.5

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_db_true_matches_expected_db(self, random_patch):
        """Ensure db=True converts filtered rolling RMS to dB."""
        kwargs = {"time": (10, 100)}

        out = random_patch.fbe(**kwargs, window=0.01, step=0.01, db=True)

        filtered = random_patch.pass_filter(**kwargs)
        rms = (filtered**2).rolling(time=0.01, step=0.01).mean() ** 0.5
        expected = 10 * rms.log10()

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_attrs_when_not_db(self, random_patch):
        """Ensure non-db output metadata are set."""
        out = random_patch.fbe(time=(10, 100), window=0.01, step=0.01, db=False)

        assert out.attrs.data_type == "frequency_band_energy"

    def test_attrs_when_db(self, random_patch):
        """Ensure db output metadata are set."""
        out = random_patch.fbe(time=(10, 100), window=0.01, step=0.01, db=True)

        assert out.attrs.data_type == "frequency_band_energy"
        assert out.attrs.data_units == ureg.dB

    def test_step_defaults_to_inverse_sampling_rate(self, random_patch):
        """Ensure step defaults to the coordinate sampling interval."""
        sample_rate = 1.0 / (random_patch.get_coord("time").step / timedelta64(1, "s"))
        step = 1.0 / sample_rate

        out = random_patch.fbe(time=(10, 100), window=0.01, step=None, db=False)
        filtered = random_patch.pass_filter(time=(10, 100))
        expected = (filtered**2).rolling(time=0.01, step=step).mean() ** 0.5

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_open_ended_lowpass_filter(self, random_patch):
        """Ensure open-ended lowpass filters are accepted."""
        out = random_patch.fbe(time=(None, 100), window=0.01, step=0.01, db=False)

        assert out.attrs.data_type == "frequency_band_energy"

    def test_open_ended_highpass_filter(self, random_patch):
        """Ensure open-ended highpass filters are accepted."""
        out = random_patch.fbe(time=(10, None), window=0.01, step=0.01, db=False)

        assert out.attrs.data_type == "frequency_band_energy"

    def test_invalid_frequency_range_raises(self, random_patch):
        """Ensure invalid filter ranges raise."""
        with pytest.raises(ValueError):
            random_patch.fbe(time=(100, 10), window=0.01, step=0.01)

    def test_missing_filter_kwargs_raise(self, random_patch):
        """Ensure a filter dimension must be supplied."""
        with pytest.raises(ValueError):
            random_patch.fbe(window=0.01, step=0.01)

    def test_multiple_filter_kwargs_raise(self, random_patch):
        """Ensure only one filter dimension can be supplied."""
        with pytest.raises(ValueError):
            random_patch.fbe(
                time=(10, 100),
                distance=(0.01, 0.05),
                window=0.01,
                step=0.01,
            )
