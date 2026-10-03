"""Tests for converting patches to and from xarray objects."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.xarray import patch_to_xarray, xarray_to_patch


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

    def test_fractional_rate_round_trip(self, random_patch):
        """A 1024 Hz time coordinate survives an eager xarray round trip."""
        pytest.importorskip("xarray")
        time = dc.get_coord(
            start=np.datetime64("2020-01-01", "ns"),
            step=(1, 1024),
            shape=(random_patch.coord_shapes["time"][0],),
        )
        patch = random_patch.update_coords(time=time)
        out = xarray_to_patch(patch_to_xarray(patch))
        assert out.get_coord("time") == time

    def test_convert_non_coord(self, random_patch):
        """Ensure a patch with non-coord can still be converted."""
        xr = pytest.importorskip("xarray")
        patch = random_patch.sum("time")
        dar = patch.io.to_xarray()
        assert isinstance(dar, xr.DataArray)
        # Ensure it round-trips
        patch2 = xarray_to_patch(dar)
        assert isinstance(patch2, dc.Patch)


class TestCoordinateConversion:
    """xarray coordinates become patch coordinates holding the same labels."""

    @pytest.fixture
    def xr(self):
        """The optional xarray module."""
        return pytest.importorskip("xarray")

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_scalar_coordinate(self, xr, lazy, dim):
        """A coordinate left scalar by a selection stays one scalar."""
        patch = dc.get_example_patch()
        array = patch_to_xarray(patch, lazy_coords=lazy).isel({dim: 3})
        out = xarray_to_patch(array)
        coord = out.get_coord(dim)
        assert coord.shape == ()
        assert coord.values == patch.get_array(dim)[3]
        assert xarray_to_patch(array.dc.abs()).get_coord(dim) == coord

    def test_scalar_integer_without_units(self, xr):
        """A scalar integer is one label, never the length of a partial coord."""
        array = xr.DataArray(np.arange(3), dims="x", coords={"x": [5, 6, 7]})
        coord = xarray_to_patch(array.isel(x=1)).get_coord("x")
        assert coord.shape == () and coord.values == 6

    def test_labels_are_not_moved(self, xr):
        """Nearly even labels keep their exact values."""
        labels = np.array([0.0, 1.0, 2.00000001, 3.0])
        array = xr.DataArray(np.arange(4), dims="x", coords={"x": labels})
        np.testing.assert_array_equal(xarray_to_patch(array).get_array("x"), labels)
        array["x"].attrs["units"] = "m"
        np.testing.assert_array_equal(xarray_to_patch(array).get_array("x"), labels)

    @pytest.mark.parametrize("step", [0.1, 1.0209, 3])
    def test_even_labels_stay_even(self, xr, step):
        """Evenly sampled labels still come back as an evenly sampled coord."""
        coord = dc.get_coord(start=0, step=step, shape=(50,))
        patch = dc.Patch(data=np.arange(50), dims=("x",), coords={"x": coord})
        out = xarray_to_patch(patch_to_xarray(patch)).get_coord("x")
        assert out.evenly_sampled
        np.testing.assert_array_equal(out.values, coord.values)


class TestPublishedPaths:
    """The conversions stay importable from where the docs published them."""

    def test_utils_io_reexports(self):
        """dascore.utils.io keeps the names it published before the move."""
        from dascore.utils import io as utils_io  # noqa: PLC0415
        from dascore.xarray import patch_to_xarray, xarray_to_patch  # noqa: PLC0415

        assert utils_io.patch_to_xarray is patch_to_xarray
        assert utils_io.xarray_to_patch is xarray_to_patch
