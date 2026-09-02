"""Conversions between a Patch and an xarray DataArray."""

from __future__ import annotations

import dascore as dc
from dascore.constants import PatchType
from dascore.utils.misc import optional_import


def patch_to_xarray(patch: PatchType):
    """Return a data array with patch contents."""
    xr = optional_import("xarray")
    # Omit None-valued attrs because xarray backends may reject them during
    # NetCDF serialization, while a missing attr round-trips cleanly.
    attrs = {
        key: value for key, value in dict(patch.attrs).items() if value is not None
    }
    patch_dims = patch.dims
    coords = {}
    for name, coord in patch.coords.coord_map.items():
        if coord._partial:
            continue
        dims = patch.coords.dim_map[name]
        coords[name] = (dims, coord.values)
    # Need to exclude non-coords
    return xr.DataArray(patch.data, attrs=attrs, dims=patch_dims, coords=coords)


def xarray_to_patch(data_array) -> dc.Patch:
    """Convert an xarray dataarray to a patch."""
    # this cant work if xarray isn't installed. This ensures it is.
    _ = optional_import("xarray")

    return dc.Patch(
        coords={i: (x.dims, x.values) for i, x in data_array.coords.items()},
        attrs=dict(data_array.attrs.items()),
        dims=data_array.dims,
        data=data_array.data,
    )
