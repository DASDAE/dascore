"""Tests for detrending functions."""

from __future__ import annotations

import numpy as np


class TestDetrend:
    """Tests for detrending data."""

    def test_detrend(self, random_patch):
        """Ensure detrending removes mean."""
        new = random_patch.new(data=random_patch.data + 10)
        # perform detrend, ensure all mean values are close to zero
        det = new.detrend(dim="time", type="linear")
        means = np.mean(det.data, axis=det.get_axis("time"))
        assert np.allclose(means, 0)
