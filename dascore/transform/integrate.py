"""Module for performing integration on patches."""

from __future__ import annotations

from operator import mul
from typing import Any

import numpy as np

from dascore.compat import is_array
from dascore.core.coords import get_coord
from dascore.core.processor import PatchProcessor
from dascore.utils.misc import broadcast_for_index, iterate
from dascore.utils.patch import (
    _get_data_type_from_dims,
    _get_data_units_from_dims,
    _get_dx_or_spacing_and_axes,
    _refuse_segments,
)

_TRAP_FUNC = getattr(np, "trapezoid" if hasattr(np, "trapezoid") else "trapz")


def _quasi_mean(array):
    """Get a quasi mean value from an array. Works with datetimes."""
    dtype = array.dtype
    if np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64):
        out = array.view("i8").mean().astype(array.dtype)
    else:
        out = np.mean(array)
    return np.asarray([out], dtype=array.dtype)


def _get_definite_coords(meta, dims):
    """Return coords with each dim collapsed to one sample, keeping its min and max."""
    new_coords, coords = {}, meta.coords
    for name in dims:
        coord = meta.get_coord(name)
        if len(coord) > 1:  # collapsed, so no warning
            coords = coords.drop_associated(name, None)
        # the label and the bounds keep the dimension's units
        new_coords[name] = coord.update(values=_quasi_mean(coord.values))
        for end in ("min", "max"):
            value = np.asarray([getattr(coord.values, end)()])
            bound = get_coord(data=value, units=coord.units)
            new_coords[f"pre_integrate_{name}_{end}"] = (name, bound)
    return coords.update(**new_coords)


def _get_trapezoids(array, dx_or_val, ax):
    """Return the area between each pair of adjacent samples along an axis."""
    if is_array(dx_or_val):
        intervals = dx_or_val[1:] - dx_or_val[:-1]
        indexer = broadcast_for_index(array.ndim, ax, slice(None), fill=None)
        dx_or_val = intervals[indexer]
    ndim = array.ndim
    stop_indexer = broadcast_for_index(ndim, ax, slice(1, None), fill=slice(None))
    start_indexer = broadcast_for_index(ndim, ax, slice(None, -1), fill=slice(None))
    return (array[stop_indexer] + array[start_indexer]) * (dx_or_val / 2)


def _get_definite_integral(array, dxs_or_vals, axes, skip_nan=False):
    """Get a definite integral along axes; with skip_nan, NaN intervals add zero."""
    ndims = len(array.shape)
    for dxs_or_val, ax in zip(dxs_or_vals, axes):
        if skip_nan:
            areas = _get_trapezoids(array, dxs_or_val, ax)
            array = np.nansum(areas, axis=ax, keepdims=True)
            continue
        # Numpy 2/3 compat code
        indexer = broadcast_for_index(ndims, ax, None, fill=slice(None))
        key = "x" if is_array(dxs_or_val) else "dx"
        array = _TRAP_FUNC(array, **{key: dxs_or_val}, axis=ax)[indexer]
    return array


def _get_indefinite_integral(array, dxs_or_vals, axes, skip_nan=False):
    """
    Get indefinite integral along dimensions.

    We need to calculate the distance weighted average for the areas between
    samples for the integration dimension. With skip_nan, intervals touching
    a NaN add zero and NaN samples stay NaN.
    """
    for dx_or_val, ax in zip(dxs_or_vals, axes):
        out = np.zeros_like(array)
        avs = _get_trapezoids(array, dx_or_val, ax)
        if skip_nan:
            avs[np.isnan(avs)] = 0
        stop_indexer = broadcast_for_index(
            out.ndim, ax, slice(1, None), fill=slice(None)
        )
        out[stop_indexer] = np.cumsum(avs, axis=ax)
        if skip_nan:
            out[np.isnan(array)] = np.nan
        array = out
    return array


class Integrate(PatchProcessor):
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

    NaN samples are treated as missing: an interval with a NaN end adds
    nothing, a cumulative integral carries on past a gap, and NaN samples
    stay NaN.

    Integer and boolean data are converted to float64 before integration.
    Floating-point and complex data are used without an explicit dtype cast.

    The output `data_type` is mapped through the pairs an integral is
    known to relate, which are those of
    [differentiate](`dascore.Patch.differentiate`) read backwards: along
    `time` strain_rate becomes strain and acceleration becomes velocity;
    along `distance` strain becomes displacement. An integral the pairs
    cannot name all the way through clears `data_type` rather than
    leaving a stale one on it.

    A dimension of more than one segment (from `concatenate`, or samples
    removed from its middle) raises; use [fill_gaps](`dascore.Patch.fill_gaps`)
    or [split_gaps](`dascore.Patch.split_gaps`) first. One list of labels,
    with or without a step, is an irregular axis, integrated by its label spacing.

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

    __version__ = "1.3"
    dim: Any
    definite: Any = False

    def get_metadata(self, meta):
        """Return the integral's coords, units and data_type, and the spacing."""
        dims = iterate(self.dim if self.dim is not None else meta.dims)
        _refuse_segments(meta, dims, "integrate")
        dxs_or_vals, axes = _get_dx_or_spacing_and_axes(meta, dims)
        coords = _get_definite_coords(meta, dims) if self.definite else meta.coords
        new_units = _get_data_units_from_dims(meta, dims, mul)
        data_type = _get_data_type_from_dims(meta, dims, differentiate=False)
        attrs = meta.attrs.update(data_units=new_units, data_type=data_type)
        out = meta.new(attrs=attrs, coords=coords)
        return out, {"axes": axes, "spacing": dxs_or_vals}

    def numpy_kernel(self, data, *, axes, spacing):
        """Return the integral of the data along the axes."""
        if axes and data.dtype.kind in "biu":
            data = data.astype(np.float64)
        func = _get_definite_integral if self.definite else _get_indefinite_integral
        out = func(data, spacing, axes)
        # A NaN sample reaches the last sample of each integrated axis longer
        # than one, so only that slice is checked before scanning the input.
        last = broadcast_for_index(out.ndim, axes, -1, fill=slice(None))
        single = any(data.shape[ax] == 1 for ax in axes)
        maybe_nan = out.size and (single or np.isnan(out[last]).any())
        if maybe_nan and np.isnan(data).any():
            out = func(data, spacing, axes, skip_nan=True)
        return out
