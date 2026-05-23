"""Tests for STA/LTA transform."""

from __future__ import annotations

import numpy as np
import pytest


class TestStaLta:
    """Tests for short-term average / long-term average ratio."""

    def test_runs_time_dimension(self, random_patch):
        """Ensure stalta runs along the time dimension."""
        out = random_patch.stalta(time=(0.01, 0.05))

        assert out.dims == random_patch.dims
        assert out.data.shape == random_patch.data.shape
        assert out.attrs.data_type == "STALTA"
        assert out.attrs.data_units is None

    def test_runs_distance_dimension(self, random_patch):
        """Ensure stalta runs along the distance dimension."""
        out = random_patch.stalta(distance=(5, 25))

        assert out.dims == random_patch.dims
        assert out.data.shape == random_patch.data.shape
        assert out.attrs.data_type == "STALTA"
        assert out.attrs.data_units is None

    def test_matches_expected_time_ratio(self, random_patch):
        """Ensure stalta returns rolling STA divided by rolling LTA."""
        out = random_patch.stalta(time=(0.01, 0.05))

        sta = random_patch.rolling(time=0.01).mean()
        lta = random_patch.rolling(time=0.05).mean()
        expected = sta / lta

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_matches_expected_distance_ratio(self, random_patch):
        """Ensure stalta uses the requested dimension for rolling windows."""
        out = random_patch.stalta(distance=(5, 25))

        sta = random_patch.rolling(distance=5).mean()
        lta = random_patch.rolling(distance=25).mean()
        expected = sta / lta

        assert np.allclose(out.data, expected.data, equal_nan=True)

    def test_attrs_are_set(self, random_patch):
        """Ensure output metadata are set."""
        out = random_patch.stalta(time=(0.01, 0.05))

        assert out.attrs.data_type == "STALTA"
        assert out.attrs.data_units is None

    def test_missing_dimension_kwargs_raise(self, random_patch):
        """Ensure a dimension/window kwarg is required."""
        with pytest.raises(ValueError):
            random_patch.stalta()

    def test_multiple_dimension_kwargs_raise(self, random_patch):
        """Ensure only one dimension/window kwarg is accepted."""
        with pytest.raises(ValueError):
            random_patch.stalta(time=(0.01, 0.05), distance=(5, 25))

    def test_bad_window_tuple_raises(self, random_patch):
        """Ensure invalid window kwargs are rejected."""
        with pytest.raises(ValueError):
            random_patch.stalta(time=(0.01, 0.05, 0.1))
