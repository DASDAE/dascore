"""Tests for converting patches to and from xarray objects."""

from __future__ import annotations

import re

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


class TestPublishedPaths:
    """The conversions stay importable from where the docs published them."""

    def test_utils_io_reexports(self):
        """dascore.utils.io keeps the names it published before the move."""
        from dascore.utils import io as utils_io  # noqa: PLC0415
        from dascore.xarray import patch_to_xarray, xarray_to_patch  # noqa: PLC0415

        assert utils_io.patch_to_xarray is patch_to_xarray
        assert utils_io.xarray_to_patch is xarray_to_patch


class TestCFUnits:
    """Units follow the CF ``units`` attribute both ways."""

    @pytest.fixture
    def xr(self):
        """The xarray module, or skip."""
        return pytest.importorskip("xarray")

    @pytest.fixture
    def units_patch(self, random_patch):
        """A patch with data and distance units."""
        return random_patch.set_units("m/s", distance="m")

    def test_data_units_written_as_units(self, units_patch):
        """Data units are written as strings, also under ``units``."""
        array = patch_to_xarray(units_patch)
        assert array.attrs["units"] == array.attrs["data_units"] == "m / s"

    def test_units_read_as_data_units(self, xr, random_patch):
        """A data variable's ``units`` become the patch's data units."""
        array = patch_to_xarray(random_patch).assign_attrs(units="1/s")
        patch = xarray_to_patch(array)
        assert patch.attrs.data_units == dc.get_quantity("1/s")
        assert "units" not in patch.attrs.model_dump()

    def test_round_trip_no_stray_units(self, units_patch):
        """A round trip gives the same patch, with no ``units`` attr."""
        out = xarray_to_patch(patch_to_xarray(units_patch))
        assert out == units_patch
        assert "units" not in out.attrs.model_dump()

    def test_coord_units_are_unit_strings(self, units_patch):
        """A coordinate's units are written without a magnitude."""
        array = patch_to_xarray(units_patch)
        assert array.coords["distance"].attrs["units"] == "m"

    @pytest.mark.parametrize("units", ["degrees_north", "m s-1", "("])
    def test_unknown_coord_units_dropped(self, xr, random_patch, units):
        """A coordinate unit string that cannot be parsed warns and is dropped."""
        array = patch_to_xarray(random_patch)
        array.coords["distance"].attrs["units"] = units
        with pytest.warns(UserWarning, match=r"distance.*" + re.escape(units)):
            patch = xarray_to_patch(array)
        assert patch.get_coord("distance").units is None

    def test_unknown_data_units_dropped(self, xr, random_patch):
        """Data units that cannot be parsed warn and are dropped."""
        array = patch_to_xarray(random_patch).assign_attrs(units="degrees_Celsius")
        with pytest.warns(UserWarning, match="degrees_Celsius"):
            patch = xarray_to_patch(array)
        assert patch.attrs.data_units is None
        assert "units" not in patch.attrs.model_dump()

    @pytest.mark.parametrize("writer", ["to_netcdf", "to_zarr"])
    def test_xarray_can_write(self, units_patch, tmp_path, writer):
        """Xarray's own writers accept the converted array."""
        pytest.importorskip("h5netcdf" if writer == "to_netcdf" else "zarr")
        array = patch_to_xarray(units_patch).rename("data")
        path = tmp_path / ("out.nc" if writer == "to_netcdf" else "out.zarr")
        kwargs = {"engine": "h5netcdf"} if writer == "to_netcdf" else {}
        getattr(array.to_dataset(), writer)(path, **kwargs)
