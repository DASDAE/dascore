"""Conversions between a Patch and an xarray DataArray."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np

import dascore as dc
from dascore.constants import PatchType
from dascore.core.coords import BaseCoord, get_coord
from dascore.utils.misc import optional_import


def _register_accessor() -> None:
    """Put the `.dc` accessor on what DASCore hands back.

    Registering it means importing xarray, which DASCore will not do on
    its own; here xarray is already imported, so anything converted
    carries the accessor without a user asking for it.
    """
    from dascore.xarray import accessor  # noqa: F401, PLC0415


def _lazy_index(name, coord, held: bool = False):
    """
    A lazy xarray index for a range or segmented coordinate, else None.

    With ``held``, any coordinate is served, holding its labels, so an
    index built over labels by hand comes back as the index it was.
    """
    # function-level: xarray is an optional dependency
    from dascore.xarray.index import (  # noqa: PLC0415
        CoordIndex,
        CoordTransform,
        is_servable,
    )

    if held or is_servable(coord):
        return CoordIndex(CoordTransform(name, coord))
    return None


def patch_to_xarray(patch: PatchType, lazy_coords: bool | Collection[str] = False):
    """
    Convert a patch to an xarray DataArray.

    Parameters
    ----------
    patch
        The patch to convert.
    lazy_coords
        Which dimension coordinates to serve through
        `dascore.xarray.index.CoordIndex`, which computes labels on demand from
        the coordinate instead of storing them: True for every evenly sampled or
        segmented one, False (the default) for none, or their names, served
        whatever their kind (one holding its labels keeps them in the index).
        Other coordinates are materialized.

    Notes
    -----
    A lazy coordinate is the patch's own coordinate, so it converts back exactly,
    exact sampling grid included; a one-sample materialized coordinate cannot
    retain its step. xarray aligns a lazy index only with lazy indexes: combining
    a lazy result with an array whose index is materialized, or reindexing it to
    new labels, raises an AlignmentError. Materialized labels, the default, cost
    8 bytes a sample along each dimension, which beside a patch's data is little.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.xarray import patch_to_xarray
    >>> patch = dc.get_example_patch()
    >>> # Materialized labels, as a plain xarray index holds them.
    >>> array = patch_to_xarray(patch)
    >>> # Lazy labels; the patch's own coordinates come back exactly.
    >>> lazy = patch_to_xarray(patch, lazy_coords=True)
    >>> assert lazy.xindexes["time"].coordinate == patch.get_coord("time")
    """
    xr = optional_import("xarray")
    _register_accessor()
    # Omit None-valued attrs because xarray backends may reject them during
    # NetCDF serialization, while a missing attr round-trips cleanly.
    attrs = {
        key: value for key, value in dict(patch.attrs).items() if value is not None
    }
    patch_dims = patch.dims
    named = not isinstance(lazy_coords, bool)
    if not named:
        lazy_coords = patch_dims if lazy_coords else ()
    coords, units, lazy = {}, {}, []
    for name, coord in patch.coords.coord_map.items():
        if coord._partial:
            continue
        dims = patch.coords.dim_map[name]
        if coord.units is not None and not _is_temporal(coord.dtype):
            # a coordinate's units are its own; xarray states them the
            # way the CF conventions do, as an attribute beside it.
            # A datetime says its units in its dtype, and xarray spends
            # that same attribute on saying how to serialize it.
            units[name] = str(coord.units)
        # An index labels a dimension, so only a coordinate which defines
        # one can be served by it; a coordinate merely riding a dimension
        # states its values as any other does.
        if name in lazy_coords and dims == (name,):
            if (index := _lazy_index(name, coord, held=named)) is not None:
                lazy.append(index)
                continue
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

    A lazily indexed coordinate is served by the DASCore coordinate it
    was built from, which comes back as it is: reading its values would
    spell out every sample, which is what the lazy index exists to avoid.
    """
    index = data_array.xindexes.get(name)
    units = coord.attrs.get("units")
    if isinstance(served := getattr(index, "coordinate", None), BaseCoord):
        if units is not None and not _is_temporal(served.dtype):
            served = served.set_units(units)
        return coord.dims, served
    values = coord.values
    if units is not None and not _is_temporal(values.dtype):
        return coord.dims, get_coord(values=values, units=units)
    return coord.dims, values
