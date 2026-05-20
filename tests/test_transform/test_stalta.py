"""Tests for STA/LTA transform."""

from __future__ import annotations

import numpy as np


class TestStaLta:
    """Tests for short-term average / long-term average ratio."""

    def test_runs(self, random_patch):
        """Ensure stalta runs and returns a patch with the same shape."""
        out = random_patch.stalta(sta=0.01, lta=0.05)

        assert out.data.shape == random_patch.data.shape
        assert out.dims == random_patch.dims

    def test_patch_method_runs(self, random_patch):
        """Ensure stalta is registered as a patch method."""
        out = random_patch.stalta(sta=0.01, lta=0.05)

        assert out.data.shape == random_patch.data.shape
        assert out.attrs.data_type == "STALTA"

    def test_non_time_dim(self, random_patch):
        """Ensure dim controls the rolling dimension."""
        out = random_patch.stalta(sta=1, lta=5, dim="distance")
        expected = (
            random_patch.abs().rolling(distance=1).mean()
            / random_patch.abs().rolling(distance=5).mean()
        )

        assert np.allclose(out.data, expected.data, equal_nan=True)
