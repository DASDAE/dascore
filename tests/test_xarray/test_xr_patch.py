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

    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_scalar_coordinate(self, xr, dim):
        """A coordinate left scalar by a selection stays one scalar."""
        patch = dc.get_example_patch()
        array = patch_to_xarray(patch).isel({dim: 3})
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

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_float_axis_is_evenly_sampled(self, xr, dtype):
        """An ordinary float axis converts to an evenly sampled coord."""
        labels = np.linspace(0, 1, 1001, dtype=dtype)
        array = xr.DataArray(np.zeros(1001), dims="x", coords={"x": labels})
        assert xarray_to_patch(array).get_coord("x").evenly_sampled
        assert array.dc.pass_filter(x=(None, 0.1)).shape == array.shape


class TestPublishedPaths:
    """The conversions stay importable from where the docs published them."""

    def test_utils_io_reexports(self):
        """dascore.utils.io keeps the names it published before the move."""
        from dascore.utils import io as utils_io  # noqa: PLC0415
        from dascore.xarray import patch_to_xarray, xarray_to_patch  # noqa: PLC0415

        assert utils_io.patch_to_xarray is patch_to_xarray
        assert utils_io.xarray_to_patch is xarray_to_patch


class TestMultiIndex:
    """A stacked dimension has no patch coordinate, so it is refused."""

    def test_stacked_dimension_refused(self, random_patch):
        """Converting a stacked array says how to make it convertible."""
        pytest.importorskip("xarray")
        stacked = patch_to_xarray(random_patch).stack(z=("distance", "time"))
        with pytest.raises(dc.exceptions.PatchConversionError, match=r"unstack\('z'\)"):
            xarray_to_patch(stacked)

    def test_reset_index_converts(self, random_patch):
        """With the index reset, the levels convert as coordinates along z."""
        pytest.importorskip("xarray")
        stacked = patch_to_xarray(random_patch).stack(z=("distance", "time"))
        patch = xarray_to_patch(stacked.reset_index("z"))
        assert patch.dims == ("z",)
        assert patch.coords.dim_map["distance"] == ("z",)
