"""Tests for converting patches to and from xarray objects."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

import dascore as dc
from dascore.units import get_quantity_str
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

    @pytest.fixture(autouse=True)
    def xr(self):
        """The xarray module, or skip."""
        return pytest.importorskip("xarray")

    @pytest.fixture
    def units_patch(self, random_patch):
        """A patch with data and distance units."""
        return random_patch.set_units("m/s", distance="m")

    @pytest.mark.parametrize(
        ("units", "cf"),
        [
            ("m/s**2", "m s-2"),
            ("m/s", "m s-1"),
            ("rad", "rad"),
            ("strain", "1"),
            ("strain/s", "s-1"),
            ("nanostrain", "1e-09"),
            ("dimensionless", "1"),
        ],
    )
    def test_data_units_written_as_cf(self, random_patch, units, cf):
        """``units`` is the CF string; ``data_units`` reads back exactly."""
        patch = random_patch.set_units(units)
        array = patch_to_xarray(patch)
        assert array.attrs["units"] == cf
        assert array.attrs["data_units"] == get_quantity_str(patch.attrs.data_units)
        out = xarray_to_patch(array)
        assert out == patch
        assert "units" not in out.attrs.model_dump()

    @pytest.mark.parametrize(("units", "cf"), [("m", "m"), ("dimensionless", "1")])
    def test_coord_units_written_as_cf(self, random_patch, units, cf):
        """A coordinate's ``units`` is its CF string."""
        array = patch_to_xarray(random_patch.set_units(distance=units))
        assert array.coords["distance"].attrs["units"] == cf

    @pytest.mark.parametrize(
        ("units", "expected"),
        [
            ("m s-1", "m/s"),
            ("kg m-2 s-1", "kg/m**2/s"),
            ("m2", "m**2"),
            ("s-1", "1/s"),
            ("1e-09", "1e-09"),
            ("degrees_north", "degree"),
            ("degrees_east", "degree"),
            ("degree_N", "degree"),
            ("degreesE", "degree"),
            ("(m/s)2", "m**2/s**2"),
        ],
    )
    def test_cf_units_read(self, xr, random_patch, units, expected):
        """CF exponent notation and degrees_north/east parse, data and coords."""
        array = patch_to_xarray(random_patch).assign_attrs(units=units)
        array.coords["distance"].attrs["units"] = units
        patch = xarray_to_patch(array)
        assert patch.attrs.data_units == dc.get_quantity(expected)
        assert patch.get_coord("distance").units == dc.get_quantity(expected)

    @pytest.mark.parametrize("units", ["(", "()", "1/0", "m/0", "bogus"])
    def test_unknown_coord_units_dropped(self, xr, random_patch, units):
        """A coordinate unit string that cannot be parsed warns and is dropped."""
        array = patch_to_xarray(random_patch)
        array.coords["distance"].attrs["units"] = units
        with pytest.warns(UserWarning, match=r"distance.*" + re.escape(units)):
            patch = xarray_to_patch(array)
        assert patch.get_coord("distance").units is None

    @pytest.mark.parametrize("units", ["degrees_Celsius", "()", "1/0", "m/0"])
    def test_unknown_data_units_kept(self, xr, random_patch, units):
        """Data units that cannot be parsed warn and stay a plain attr."""
        array = patch_to_xarray(random_patch).assign_attrs(units=units)
        with pytest.warns(UserWarning, match=re.escape(units)) as record:
            patch = xarray_to_patch(array)
        assert patch.attrs.data_units is None
        assert patch.attrs.model_dump()["units"] == units
        assert record[0].filename == __file__

    def test_edited_units_win(self, xr, units_patch):
        """An edited ``units`` overrides the ``data_units`` it disagrees with."""
        array = patch_to_xarray(units_patch).assign_attrs(units="s-1")
        assert xarray_to_patch(array).attrs.data_units == dc.get_quantity("1/s")

    def test_unreadable_edit_clears_data_units(self, xr, units_patch):
        """An edited ``units`` which cannot be read leaves no stale data units."""
        array = patch_to_xarray(units_patch).assign_attrs(units="bogus")
        with pytest.warns(UserWarning, match="bogus"):
            patch = xarray_to_patch(array)
        assert patch.attrs.data_units is None
        assert patch.attrs.model_dump()["units"] == "bogus"

    def test_empty_data_units(self, xr, random_patch):
        """An empty ``data_units`` lets ``units`` state the data units."""
        array = patch_to_xarray(random_patch).assign_attrs(units="m", data_units="")
        assert xarray_to_patch(array).attrs.data_units == dc.get_quantity("m")

    def test_temporal_units_ignored(self, xr, random_patch):
        """A time coordinate's units attribute is not parsed as units."""
        array = patch_to_xarray(random_patch)
        array.coords["time"].attrs["units"] = "seconds since 2000-01-01"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            patch = xarray_to_patch(array)
        assert patch.get_coord("time") == random_patch.get_coord("time")

    @pytest.mark.parametrize("units", [None, "bogus"])
    def test_lazy_coord_units_cleared(self, xr, units_patch, units):
        """A lazy coordinate whose units are absent or dropped has none."""
        array = patch_to_xarray(units_patch, lazy_coords=True)
        array.coords["distance"].attrs.pop("units")
        if units is not None:
            array.coords["distance"].attrs["units"] = units
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            patch = xarray_to_patch(array)
        assert patch.get_coord("distance").units is None

    @pytest.mark.parametrize("writer", ["to_netcdf", "to_zarr"])
    def test_xarray_can_write(self, units_patch, tmp_path, writer):
        """Xarray's own writers accept the converted array."""
        pytest.importorskip("h5netcdf" if writer == "to_netcdf" else "zarr")
        array = patch_to_xarray(units_patch).rename("data")
        path = tmp_path / ("out.nc" if writer == "to_netcdf" else "out.zarr")
        kwargs = {"engine": "h5netcdf"} if writer == "to_netcdf" else {}
        getattr(array.to_dataset(), writer)(path, **kwargs)
