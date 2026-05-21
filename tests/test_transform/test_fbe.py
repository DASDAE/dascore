"""Tests for Frequency Band Energy transform."""

from __future__ import annotations

import numpy as np
import pytest

import dascore.units as ureg


class TestFBE:
    """Tests for frequency-band energy transform."""

    def test_runs(self, random_patch):
        """Ensure fbe runs and returns a patch."""
        out = random_patch.fbe(time=0.01, step=0.01, db=False)

        assert out.dims == random_patch.dims
        assert out.attrs.data_type == "Frequency-Band Energy"

    def test_patch_method_runs(self, random_patch):
        """Ensure fbe is registered as a patch method."""
        out = random_patch.fbe(time=0.01, step=0.01, db=False)

        assert out.attrs.data_type == "Frequency-Band Energy"

    def test_attrs_when_not_db(self, random_patch):
        """Ensure non-db output metadata are set."""
        out = random_patch.fbe(time=0.01, step=0.01, db=False)

        assert out.attrs.data_type == "Frequency-Band Energy"

    def test_attrs_when_db(self, random_patch):
        """Ensure db output metadata are set."""
        out = random_patch.fbe(time=0.01, step=0.01, db=True)

        assert out.attrs.data_type == "Frequency-Band Energy"
        assert out.attrs.data_units == ureg.dB

    def test_db_false_matches_expected_rms(self, random_patch):
        """Ensure db=False returns rolling RMS energy."""
        out = random_patch.fbe(time=0.01, step=0.01, db=False)
        expected = (random_patch**2).rolling(time=0.01, step=0.01).mean() ** 0.5

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_db_true_matches_expected_db(self, random_patch):
        """Ensure db=True converts rolling RMS energy to dB."""
        out = random_patch.fbe(time=0.01, step=0.01, db=True)
        rms = (random_patch**2).rolling(time=0.01, step=0.01).mean() ** 0.5
        expected = 10 * rms.log10()

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_filter_corners_are_applied(self, random_patch):
        """Ensure frequency corners trigger pass filtering."""
        corners = (10, 20)

        out = random_patch.fbe(
            freqmin=corners[0], freqmax=corners[1], time=0.01, step=0.01, db=False
        )
        filtered = random_patch.pass_filter(time=corners)
        expected = (filtered**2).rolling(time=0.01, step=0.01).mean() ** 0.5

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_defaults_and_invalid_frequency_bounds(self, random_patch):
        """Ensure defaults are derived correctly and invalid bounds raise."""
        step = random_patch.get_coord("time").step

        # defaults: step from coord, time = 20 * step
        out = random_patch.fbe(
            step=None,
            time=None,
            freqmin=None,
            freqmax=None,
            db=False,
        )

        expected = (random_patch**2).rolling(time=20 * step, step=step).mean() ** 0.5

        assert np.allclose(out.data, expected.data, equal_nan=True)

        # invalid bounds
        with pytest.raises(ValueError, match="freqmax must be larger than freqmin"):
            random_patch.fbe(
                freqmin=20,
                freqmax=10,
                time=0.01,
                step=0.01,
            )
