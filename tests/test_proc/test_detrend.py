"""Tests for detrending functions."""

from __future__ import annotations

import numpy as np
import pytest

from dascore.exceptions import ParameterError


class TestDetrend:
    """Tests for detrending data."""

    def test_detrend(self, random_patch):
        """Ensure detrending removes mean."""
        new = random_patch.new(data=random_patch.data + 10)
        # perform detrend, ensure all mean values are close to zero
        det = new.detrend(dim="time", type="linear")
        means = np.mean(det.data, axis=det.get_axis("time"))
        assert np.allclose(means, 0)

    def test_bad_dim_raises(self, random_patch):
        """A dim not in the patch should raise ParameterError."""
        with pytest.raises(ParameterError, match="not in patch dim"):
            random_patch.detrend(dim="not_a_dim")
