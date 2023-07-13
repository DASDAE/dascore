"""
Module for performing integration on patches.
"""
from __future__ import annotations

from operator import mul
from typing import Sequence

import numpy as np

from dascore.constants import PatchType
from dascore.utils.misc import broadcast_for_index, iterate
from dascore.utils.patch import (
    _get_data_units_from_dims,
    _get_dx_or_spacing_and_axes,
    patch_function,
)


def _quasi_mean(array):
    """Get a quasi mean value from an array. Works with datetimes."""
    dtype = array.dtype
    if np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64):
        out = array.view("i8").mean().astype(array.dtype)
    else:
        out = np.mean(array)
    return np.array([out], dtype=array.dtype)


def _get_new_coords_and_array(patch, array, dims, axes, keep_dims):
    """Get new coordinates with smashed (or not) coordinates."""

    if keep_dims:
        new_coords = {x: _quasi_mean(patch.get_coord(x).data) for x in dims}
        # also add related coords indicating start/stop
        for name in dims:
            coord = patch.get_coord(name).data
            new_coords[f"{name}_min"] = (name, np.array([coord.min()]))
            new_coords[f"{name}_max"] = (name, np.array([coord.max()]))
    else:
        # Need to collapse empty dimensions
        ndim = len(patch.dims)
        indexer = broadcast_for_index(ndim, axes, 0)
        array = array[indexer]
        new_coords = {x: None for x in dims}
    cm = patch.coords.update_coords(**new_coords)
    return array, cm


@patch_function()
def integrate(
    patch: PatchType, dim: Sequence[str] | str | None, keep_dims=True
) -> PatchType:
    """
    Integrate along a specified dimension using composite trapezoidal rule.

    Parameters
    ----------
    patch
        Patch object for integration.
    dim
        The dimension(s) along which to integrate.
    keep_dims
        If False collapse (remove) the integration dimension(s).

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # integrate along time axis, preserve time coordinate
    >>> time_integrated = patch.tran.integrate(dim="time", keep_dims=True)
    >>> assert "time" in time_integrated.dims
    >>> # integrate along distance axis, drop distance coordinate
    >>> dist_integrated = patch.tran.integrate(dim="distance", keep_dims=False)
    >>> assert "distance" not in dist_integrated.dims
    """
    dims = iterate(dim if dim is not None else patch.dims)
    dxs_or_vals, axes = _get_dx_or_spacing_and_axes(patch, dims)
    ndims = len(patch.dims)
    indexers = [broadcast_for_index(ndims, x, None) for x in axes]
    array = patch.data
    for dxs_or_val, ax, inds in zip(dxs_or_vals, axes, indexers):
        if isinstance(dxs_or_val, np.ndarray):
            array = np.trapz(array, x=dxs_or_val, axis=ax)[inds]
        else:
            array = np.trapz(array, dx=dxs_or_val, axis=ax)[inds]
    array, coords = _get_new_coords_and_array(patch, array, dims, axes, keep_dims)
    new_units = _get_data_units_from_dims(patch, dims, mul)
    attrs = patch.attrs.update(data_units=new_units)
    return patch.new(data=array, attrs=attrs, coords=coords)
