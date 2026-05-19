"""
Tests for stalta function.
"""

from __future__ import annotations

import numpy as np


class TestStalta:
    """Tests for the Hilbert transform function."""

    def test_stalta_basic(self, random_patch):
        """Test basic Hilbert transform functionality."""
        result = random_patch.hilbert(dim="time")

        # Result should be complex
        assert np.iscomplexobj(result.data)

        # Real part should equal original data
        assert np.allclose(result.data.real, random_patch.data)

        # Shape should be preserved
        assert result.shape == random_patch.shape

        # Coordinates should be preserved
        assert result.coords.equals(random_patch.coords)
