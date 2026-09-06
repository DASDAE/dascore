"""Conversions between a Patch and an xarray DataArray."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import PatchType
from dascore.core.coords import get_coord
from dascore.utils.misc import optional_import


def _register_accessor() -> None:
    """Put the `.dc` accessor on what DASCore hands back.

    Registering it means importing xarray, which DASCore will not do on
    its own; here xarray is already imported, so anything converted
    carries the accessor without a user asking for it.
    """
    from dascore.xarray import accessor  # noqa: F401, PLC0415


def patch_to_xarray(patch: PatchType):
    """Return a data array with patch contents."""
    xr = optional_import("xarray")
    _register_accessor()
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
        coords={i: _coord_from(data_array, i, x) for i, x in data_array.coords.items()},
        attrs=dict(data_array.attrs.items()),
        dims=data_array.dims,
        data=data_array.data,
    )


def _coord_from(data_array, name, coord):
    """
    The dims and values a patch coordinate is built from.

    A lazily indexed coordinate states its samples rather than storing
    them, and reading its values would spell out every one -- which for
    a tree spanning a long acquisition is what the lazy index exists to
    avoid. The range it describes rebuilds the same coordinate from
    three numbers.
    """
    index = data_array.xindexes.get(name)
    transform = getattr(index, "transform", None)
    if transform is not None and hasattr(transform, "start_ns"):
        start = np.asarray(transform.start_ns).astype(transform.dtype)
        step = np.asarray(transform.step_ns).astype("timedelta64[ns]")
        size = transform.dim_size[name]
        return coord.dims, get_coord(start=start, step=step, stop=start + step * size)
    return coord.dims, coord.values
