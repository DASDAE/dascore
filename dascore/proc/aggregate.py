"""Module for applying aggregations (reductions) along a specified axis."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from dascore.constants import _AGG_FUNCS, DIM_REDUCE_DOCS, PatchType
from dascore.core.coords import CoordPartial
from dascore.exceptions import ParameterError
from dascore.utils.array import _apply_aggregator, is_numpy
from dascore.utils.array_api import (
    asarray_like,
    backend_name,
    to_numpy,
    warn_numpy_fallback,
)
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import _get_nullish
from dascore.utils.patch import patch_function
from dascore.utils.time import dtype_time_like

AGG_DOC_STR = f"""
patch
    The input Patch.
dim
    The dimension along which aggregations are to be performed.
    If None, apply aggregation to all dimensions sequentially.
    If a sequence, apply sequentially in order provided.
{DIM_REDUCE_DOCS}
"""

AGG_NOTES = """
Notes
-----
See [`Patch.aggregate`](`dascore.Patch.aggregate`) for examples
and more details.
"""


@patch_function()
@compose_docstring(params=AGG_DOC_STR, options=sorted(_AGG_FUNCS))
def aggregate(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    method: str | Callable = "mean",
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Aggregate values along a specified dimension.

    Notes
    -----
    Whether an aggregation can be applied by the patch's own array backend
    depends on the method, so this function makes no promise about the
    backend of its output. Most shortcuts, such as
    [`Patch.mean`](`dascore.proc.aggregate.mean`), do keep the data on their
    backend; use those where one fits.

    Parameters
    ----------
    {params}
    method
        The aggregation to apply along dimension. Options are:
            {options}

    See Also
    --------
    - See also the aggregation shortcut methods in the
      [aggregate module](`dascore.proc.aggregate`).

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc

    >>> patch = dc.get_example_patch()
    >>>
    >>> # Calculate mean along time axis
    >>> patch_time = patch.aggregate("time", method=np.nanmean)
    >>>
    >>> # Calculate median distance along distance dimension
    >>> patch_dist = patch.aggregate("distance", method=np.nanmedian)
    >>>
    >>> # Calculate the mean, and remove the associated dimension
    >>> patch_mean_no_dim = patch.aggregate(
    ...     "time", method="mean", dim_reduce="squeeze"
    ... )
    >>>
    >>> # Aggregate by the min value and keep the mean of the dimension
    >>> patch_mean_min = patch.aggregate(
    ...     "distance", method="min", dim_reduce="mean",
    ... )
    """
    func = _AGG_FUNCS.get(method, method)
    return _apply_aggregator(patch, dim, func, dim_reduce)


@patch_function()
@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
def min(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Calculate the minimum along one or more dimensions.

    Parameters
    ----------
    {params}

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Get minimum along time dimension
    >>> min_patch = patch.min(dim='time')
    >>> assert min_patch.size < patch.size

    {notes}
    """
    return aggregate.func(patch, dim=dim, method=np.nanmin, dim_reduce=dim_reduce)


@patch_function()
@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
def max(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Calculate the maximum along one or more dimensions.

    Parameters
    ----------
    {params}

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Get maximum along distance dimension
    >>> max_patch = patch.max(dim='distance')
    >>> assert max_patch.size < patch.size

    {notes}
    """
    return aggregate.func(patch, dim=dim, method=np.nanmax, dim_reduce=dim_reduce)


@patch_function()
@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
def mean(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Calculate the mean along one or more dimensions.

    Parameters
    ----------
    {params}

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Get mean along time dimension
    >>> time_mean = patch.mean(dim='time')
    >>> assert time_mean.size < patch.size

    {notes}
    """
    return aggregate.func(patch, dim=dim, method=np.nanmean, dim_reduce=dim_reduce)


@patch_function()
@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
def median(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Calculate the median along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """
    return aggregate.func(patch, dim=dim, method=np.nanmedian, dim_reduce=dim_reduce)


@patch_function()
@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
def std(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Calculate the standard deviation along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """
    return aggregate.func(patch, dim=dim, method=np.nanstd, dim_reduce=dim_reduce)


@patch_function()
@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
def first(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Get the first value along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """
    func = _AGG_FUNCS["first"]
    return aggregate.func(patch, dim=dim, method=func, dim_reduce=dim_reduce)


@patch_function()
@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
def last(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Get the last value along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """
    func = _AGG_FUNCS["last"]
    return aggregate.func(patch, dim=dim, method=func, dim_reduce=dim_reduce)


@patch_function()
@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
def sum(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Sum the values along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """
    return aggregate.func(patch, dim=dim, method=np.nansum, dim_reduce=dim_reduce)


@patch_function(data_type="")
@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
def any(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Perform boolean any operation along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """
    return aggregate.func(patch, dim=dim, method=np.any, dim_reduce=dim_reduce)


@patch_function(data_type="")
@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
def all(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Perform boolean all operation along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """
    return aggregate.func(patch, dim=dim, method=np.all, dim_reduce=dim_reduce)


IDX_DOC_STR = f"""
patch
    The input Patch.
dim
    The name of the single dimension to reduce. None or a sequence,
    which the other aggregations accept, raises here.
{DIM_REDUCE_DOCS}
"""

IDX_NOTES = """
Notes
-----
- The data become coordinate values rather than the values found there,
  so they take the coordinate's dtype. Use
  [`Patch.max`](`dascore.proc.aggregate.max`) or
  [`Patch.min`](`dascore.proc.aggregate.min`) for the values themselves.

- NaN and NaT samples are skipped. A slice with none left has no
  coordinate to point at, so it yields a null; an integer coordinate is
  widened to float64 to hold one, which loses exactness above 2**53, and
  a coordinate which can hold no null, such as a string one, raises.

- Ties go to the first occurrence, as in NumPy.
"""


def _extreme_index(data, axis, want_max):
    """
    Return the index of the extreme along axis, and the all-missing slices.

    The index of an all-missing slice is arbitrary; the mask says which
    those are so the caller can null them out.
    """
    if dtype_time_like(data.dtype):
        # NaT does not compare, but its int64 view is the smallest int64,
        # so the view orders correctly once the missing are filled away.
        info = np.iinfo(np.int64)
        missing, values = np.isnat(data), data.view(np.int64)
        fill = info.min if want_max else info.max
    else:
        values, fill = data, (-np.inf if want_max else np.inf)
        # Integers and booleans have no value which means "missing".
        inexact = np.issubdtype(data.dtype, np.inexact)
        missing = np.isnan(data) if inexact else None
    if missing is None or not missing.any():
        return (np.argmax if want_max else np.argmin)(values, axis=axis), None
    filled = np.where(missing, fill, values)
    extreme = filled.max(axis=axis) if want_max else filled.min(axis=axis)
    # Match the extreme against the unfilled data rather than taking the
    # arg of the filled data: a real -inf equals the fill for a max, and a
    # first-wins tie would then answer with the missing sample. Nothing
    # missing can match here, since NaN equals nothing and NaT is only the
    # extreme when the whole slice is missing.
    hit = values == np.expand_dims(extreme, axis)
    return hit.argmax(axis=axis), missing.all(axis=axis)


def _fill_empty(values, empty):
    """Put a null where a slice had nothing to point at."""
    dtype = values.dtype
    if not (dtype_time_like(dtype) or np.issubdtype(dtype, np.number)):
        msg = (
            f"A slice with no valid sample has no {dtype} coordinate to "
            "point at. Drop or fill the empty slices first."
        )
        raise ParameterError(msg)
    # An integer coordinate cannot hold NaN, so where widens it to float.
    return np.where(empty, _get_nullish(dtype), values)


def _idx_aggregate(patch, dim, want_max, dim_reduce):
    """Shared implementation of idxmax and idxmin."""
    name = "idxmax" if want_max else "idxmin"
    if not isinstance(dim, str):
        msg = f"{name} reduces a single dimension; dim must be its name."
        raise ParameterError(msg)
    coord = patch.get_coord(dim)
    if isinstance(coord, CoordPartial):
        # The coord the default dim_reduce leaves behind holds no values,
        # so indexing it would quietly null the whole result.
        msg = (
            f"The '{dim}' coordinate holds no values for {name} to return; "
            "the dimension has already been reduced."
        )
        raise ParameterError(msg)
    coord_values = coord.values

    def _func(data, axis):
        original = data
        if not is_numpy(data):
            # The indexing below is numpy only, so say so rather than
            # quietly pulling a lazy or device array into memory.
            warn_numpy_fallback(name, backend_name(data))
            data = to_numpy(data)
        index, empty = _extreme_index(data, axis, want_max)
        out = coord_values[index]
        out = out if empty is None else _fill_empty(out, empty)
        return out if data is original else asarray_like(out, original)

    out = _apply_aggregator(patch, dim, _func, dim_reduce)
    # A time coord's units describe its step, not its magnitude, so
    # labelling nanoseconds "s" would scale any unit maths by a billion.
    units = None if dtype_time_like(coord.dtype) else coord.units
    return out.update_attrs(data_units=units)


@patch_function(data_type="")
@compose_docstring(params=IDX_DOC_STR, notes=IDX_NOTES)
def idxmax(
    patch: PatchType,
    dim: str,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Return the coordinate value where the data are largest along a dimension.

    Parameters
    ----------
    {params}

    {notes}

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> patch = dc.get_example_patch()
    >>>
    >>> # The time of each channel's largest sample.
    >>> peak_time = patch.idxmax("time")
    >>>
    >>> # Drop the reduced dimension, as xarray's idxmax does.
    >>> squeezed = patch.idxmax("time", dim_reduce="squeeze")

    See Also
    --------
    - [`Patch.idxmin`](`dascore.proc.aggregate.idxmin`)
    - [`Patch.max`](`dascore.proc.aggregate.max`)
    """
    return _idx_aggregate(patch, dim, True, dim_reduce)


@patch_function(data_type="")
@compose_docstring(params=IDX_DOC_STR, notes=IDX_NOTES)
def idxmin(
    patch: PatchType,
    dim: str,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Return the coordinate value where the data are smallest along a dimension.

    Parameters
    ----------
    {params}

    {notes}

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> trough_time = dc.get_example_patch().idxmin("time")

    See Also
    --------
    - [`Patch.idxmax`](`dascore.proc.aggregate.idxmax`)
    - [`Patch.min`](`dascore.proc.aggregate.min`)
    """
    return _idx_aggregate(patch, dim, False, dim_reduce)
