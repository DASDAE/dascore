"""Conversions between a Patch and an xarray DataArray."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

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


def _lazy_temporal_index(name, coord):
    """
    Return a lazy xarray index for an evenly sampled temporal coordinate.

    None when the coordinate cannot be served lazily — an irregular
    (segmented or array) coordinate, a descending one, or a numeric one,
    whose materialized values are short in practice.
    """
    from dascore.core.coords import CoordRange  # noqa: PLC0415

    step = getattr(coord, "step", None)
    if not isinstance(coord, CoordRange) or step is None or pd.isnull(step):
        return None
    if not (
        np.issubdtype(coord.dtype, np.datetime64)
        or np.issubdtype(coord.dtype, np.timedelta64)
    ):
        return None
    if np.asarray(step).astype("int64") <= 0:
        return None
    # The index holds one whole-tick step; a fractional grid would be
    # relabelled by it, so its values travel instead.
    if getattr(coord, "step_denominator", None) not in (None, 1):
        return None
    # function-level: xarray is an optional dependency
    from dascore.xarray.index import TemporalRangeIndex  # noqa: PLC0415

    return TemporalRangeIndex.from_coord(name, coord)


def patch_to_xarray(patch: PatchType, lazy_coords: Collection[str] = ()):
    """
    Return a data array with patch contents.

    Parameters
    ----------
    patch
        The patch to convert.
    lazy_coords
        Names of coordinates to serve by the range which states them
        rather than by an array of every label. Such a coordinate then
        costs three numbers however long the acquisition, and its labels
        are computed on demand. A name which does not belong to an
        evenly sampled temporal dimension coordinate of this patch is
        ignored; anything not named spells its labels out, which is what
        xarray aligns arithmetic on. See
        `dascore.xarray.index.TemporalRangeIndex` for what a lazily
        served coordinate does not yet support.

    Notes
    -----
    A DataArray states a coordinate by its labels, so a coordinate of a
    single sample cannot say how far apart its samples would be: such a
    step is lost unless the coordinate is served lazily, which states it.
    """
    xr = optional_import("xarray")
    _register_accessor()
    # Omit None-valued attrs because xarray backends may reject them during
    # NetCDF serialization, while a missing attr round-trips cleanly.
    attrs = {
        key: value for key, value in dict(patch.attrs).items() if value is not None
    }
    patch_dims = patch.dims
    coords, units, lazy = {}, {}, []
    for name, coord in patch.coords.coord_map.items():
        if coord._partial:
            continue
        dims = patch.coords.dim_map[name]
        # An index labels a dimension, so only a coordinate which defines
        # one can be served by it; a coordinate merely riding a dimension
        # states its values as any other does.
        if name in lazy_coords and dims == (name,):
            if (index := _lazy_temporal_index(name, coord)) is not None:
                lazy.append(index)
                continue
        if coord.units is not None and not _is_temporal(coord.dtype):
            # a coordinate's units are its own; xarray states them the
            # way the CF conventions do, as an attribute beside it.
            # A datetime says its units in its dtype, and xarray spends
            # that same attribute on saying how to serialize it.
            units[name] = str(coord.units)
        coords[name] = (dims, coord.values)
    # Need to exclude non-coords
    out = xr.DataArray(patch.data, attrs=attrs, dims=patch_dims, coords=coords)
    for index in lazy:
        out = out.assign_coords(xr.Coordinates.from_xindex(index))
    for name, value in units.items():
        out.coords[name].attrs["units"] = value
    return out


def _is_temporal(dtype) -> bool:
    """Whether a dtype states its own units, as a time or a duration does."""
    return np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)


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
    units = coord.attrs.get("units")
    size = None if transform is None else transform.dim_size.get(name)
    # A range needs somewhere to go: a selection which kept no samples
    # has no start, step and stop to be rebuilt from, and its values are
    # the empty array they say they are.
    if transform is not None and hasattr(transform, "start_ns") and size:
        # scalars, not zero-dimensional arrays: a coordinate's step is
        # divided into and compared against, which an array shape breaks
        start = np.asarray(transform.start_ns).astype(transform.dtype)[()]
        step = np.timedelta64(int(transform.step_ns), "ns")
        values = get_coord(
            start=start, step=step, stop=start + step * size, units=units
        )
        return coord.dims, values
    values = coord.values
    if units is not None and not _is_temporal(values.dtype):
        return coord.dims, get_coord(values=values, units=units)
    return coord.dims, values
