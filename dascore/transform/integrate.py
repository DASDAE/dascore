"""Module for performing integration on patches."""

from __future__ import annotations

from collections.abc import Sequence
from operator import mul

import numpy as np

from dascore.compat import is_array
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
    return np.asarray([out], dtype=array.dtype)


def _get_definite_integral(patch, dxs_or_vals, dims, axes):
    """Get a definite integral along axes."""

    def _get_new_coords_and_array(patch, array, dims):
        """Get new coordinates with smashed (or not) coordinates."""
        new_coords = {x: _quasi_mean(patch.get_coord(x).data) for x in dims}
        # also add related coords indicating start/stop
        for name in dims:
            coord = patch.get_coord(name).data
            new_coords[f"pre_integrate_{name}_min"] = (name, np.asarray([coord.min()]))
            new_coords[f"pre_integrate_{name}_max"] = (name, np.asarray([coord.max()]))
        cm = patch.coords.update(**new_coords)
        return array, cm

    array = patch.data
    ndims = len(patch.shape)
    for dxs_or_val, ax in zip(dxs_or_vals, axes):
        # Numpy 2/3 compat code
        trap = getattr(np, "trapezoid", getattr(np, "trapz"))
        indexer = broadcast_for_index(ndims, ax, None, fill=slice(None))
        if is_array(dxs_or_val):
            array = trap(array, x=dxs_or_val, axis=ax)[indexer]
        else:
            array = trap(array, dx=dxs_or_val, axis=ax)[indexer]
    array, coords = _get_new_coords_and_array(patch, array, dims)
    return array, coords


def _get_indefinite_integral(patch, dxs_or_vals, axes):
    """
    Get indefinite integral along dimensions.

    We need to calculate the distance weighted average for the areas between
    samples for the integration dimension.
    """
    out = np.zeros_like(patch.data)
    array = patch.data
    for dx_or_val, ax in zip(dxs_or_vals, axes):
        # if coordinate values are provided need to get diffs.
        if is_array(dx_or_val):
            dx_or_val = dx_or_val[1:] - dx_or_val[:-1]
        ndim = len(out.shape)
        # get diffs along dimension
        stop_indexer = broadcast_for_index(ndim, ax, slice(1, None), fill=slice(None))
        start_indexer = broadcast_for_index(ndim, ax, slice(None, -1), fill=slice(None))
        # first_indexer = broadcast_for_index(ndim, ax, slice(0, 1), fill=slice(None))
        # get average values of trapezoid between points
        avs = (array[stop_indexer] + array[start_indexer]) * (dx_or_val / 2)
        out[stop_indexer] = np.cumsum(avs, axis=ax)
    return out, patch.coords  # coords shouldn't change


@patch_function()
def integrate(
    patch: PatchType,
    dim: Sequence[str] | str | None,
    definite: bool = False,
) -> PatchType:
    """
    Integrate along a specified dimension using composite trapezoidal rule.

    Parameters
    ----------
    patch
        Patch object for integration.
    dim
        The dimension(s) along which to integrate. If None, integrate along
        all dimensions.
    definite
        If True, consider the integration to be defined from the minimum to
        the maximum value along specified dimension(s). In essence, this
        collapses the integrated dimensions to a length of 1.
        If define is False, the shape of the patch is preserved and a
        "cumulative" type integration in performed.

    Notes
    -----
    The number of dimensions will always remain the same regardless of `definite`
    value. To remove dimensions with length 1, use
    [`Patch.squeeze`](`dascore.Patch.squeeze`).

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # integrate along time axis, preserve patch shape with indefinite integral
    >>> time_integrated = patch.integrate(dim="time", definite=False)
    >>> # integrate along distance axis, collapse distance coordinate
    >>> dist_integrated = patch.integrate(dim="distance", definite=True)
    >>> # integrate along all dimensions.
    >>> all_integrated = patch.integrate(dim=None, definite=False)
    """
    dims = iterate(dim if dim is not None else patch.dims)
    dxs_or_vals, axes = _get_dx_or_spacing_and_axes(patch, dims)
    if definite:
        array, coords = _get_definite_integral(patch, dxs_or_vals, dims, axes)
    else:
        array, coords = _get_indefinite_integral(patch, dxs_or_vals, axes)
    new_units = _get_data_units_from_dims(patch, dims, mul)
    # we need to reset coords so coords has priority in update.
    attrs = patch.attrs.update(data_units=new_units, coords={})
    return patch.new(data=array, attrs=attrs, coords=coords)
