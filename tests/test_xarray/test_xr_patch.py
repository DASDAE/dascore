"""Tests for converting patches to and from xarray objects."""

from __future__ import annotations

import pytest

import dascore as dc
from dascore.xarray import xarray_to_patch


class TestXarray:
    """Tests for xarray conversions."""

    @pytest.fixture
    def data_array_from_patch(self, random_patch):
        """Get a data array from a patch."""
        pytest.importorskip("xarray")
        return random_patch.io.to_xarray()

    def test_convert_to_xarray(self, data_array_from_patch):
        """Tests for converting to xarray object."""
        import xarray as xr  # noqa: PLC0415

        assert isinstance(data_array_from_patch, xr.DataArray)

    def test_convert_from_xarray(self, data_array_from_patch):
        """Ensure xarray data arrays can be converted back."""
        out = xarray_to_patch(data_array_from_patch)
        assert isinstance(out, dc.Patch)

    def test_round_trip(self, random_patch, data_array_from_patch):
        """Converting to xarray should be lossless."""
        out = xarray_to_patch(data_array_from_patch)
        assert out == random_patch

    def test_convert_non_coord(self, random_patch):
        """Ensure a patch with non-coord can still be converted."""
        xr = pytest.importorskip("xarray")
        patch = random_patch.sum("time")
        dar = patch.io.to_xarray()
        assert isinstance(dar, xr.DataArray)
        # Ensure it round-trips
        patch2 = xarray_to_patch(dar)
        assert isinstance(patch2, dc.Patch)


class TestPublishedPaths:
    """The conversions stay importable from where the docs published them."""

    def test_utils_io_reexports(self):
        """dascore.utils.io keeps the names it published before the move."""
        from dascore.utils import io as utils_io  # noqa: PLC0415
        from dascore.xarray import patch_to_xarray, xarray_to_patch  # noqa: PLC0415

        assert utils_io.patch_to_xarray is patch_to_xarray
        assert utils_io.xarray_to_patch is xarray_to_patch
