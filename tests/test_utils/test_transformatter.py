"""Tests for transformatter."""

from __future__ import annotations

import pytest

from dascore.utils.transformatter import FourierTransformatter


@pytest.fixture()
def ft_reformatter():
    """Simple fourier transform formatter."""
    return FourierTransformatter()


class TestFTDimensionRename:
    """Tests for renaming dimensions using FT transformer."""

    dims = ("distance", "time")

    def test_forward_rename_one_index(self, ft_reformatter):
        """Ensure the name can be reassigned."""
        out = ft_reformatter.rename_dims(self.dims, 1)
        assert out == ("distance", "ft_time")

    def test_forward_rename_all_index(self, ft_reformatter):
        """Ensure all indices can be renamed."""
        out = ft_reformatter.rename_dims(self.dims)
        assert out == tuple(f"{ft_reformatter.forward_prefix}{x}" for x in self.dims)

    def test_forward_undo_inverse(self, ft_reformatter):
        """Ensure the inverse is correctly undone."""
        dims = tuple([f"{ft_reformatter.inverse_prefix}{x}" for x in self.dims])
        out = ft_reformatter.rename_dims(dims)
        assert out == self.dims

    def test_inverse_rename_one_index(self, ft_reformatter):
        """Ensure the name can be reassigned."""
        out = ft_reformatter.rename_dims(self.dims, 1, forward=False)
        assert out == ("distance", "ift_time")

    def test_undo_forward_index(self, ft_reformatter):
        """Ensure forward prefex is undone by inverse."""
        dims = tuple(f"{ft_reformatter.forward_prefix}{x}" for x in self.dims)
        out = ft_reformatter.rename_dims(dims, forward=False)
        assert out == self.dims

    def test_double_forward(self, ft_reformatter):
        """Prefixes should stack."""
        pre = ft_reformatter.forward_prefix
        dims1 = ft_reformatter.rename_dims(self.dims)
        dims2 = ft_reformatter.rename_dims(dims1)
        assert dims2 == tuple(f"{pre}{pre}{x}" for x in self.dims)
