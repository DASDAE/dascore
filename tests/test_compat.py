"""Tests for compatibility module."""

from __future__ import annotations

import numpy as np

from dascore.compat import array_at_least, is_array


class TestArrayAtLeast:
    """Tests for array_at_least helper."""

    def test_scalar_to_1d(self):
        """Scalar input should promote to 1D."""
        out = array_at_least(3.14, 1)
        assert out.shape == (1,)

    def test_1d_to_2d(self):
        """1D input should promote to 2D."""
        out = array_at_least(np.arange(3), 2)
        assert out.shape == (1, 3)

    def test_2d_to_3d(self):
        """2D input should promote to 3D."""
        data = np.arange(6).reshape(2, 3)
        out = array_at_least(data, 3)
        assert out.shape == (1, 2, 3)

    def test_no_shrink(self):
        """Arrays with ndim >= target should remain unchanged in shape."""
        data = np.ones((2, 2))
        out = array_at_least(data, 1)
        assert out.shape == data.shape


class TestIsArray:
    """Tests for is_array helper."""

    def test_numpy_array_is_array(self):
        """A numpy array should be considered an array."""
        ar = np.ones(10)
        assert is_array(ar)
