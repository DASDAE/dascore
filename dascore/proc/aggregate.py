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
    The name of the dimension along which to find the extreme value.
    Unlike the other aggregations this takes exactly one dimension;
    None or a sequence raises a ParameterError.
{DIM_REDUCE_DOCS}
"""

IDX_NOTES = """
Notes
-----
- The returned data are coordinate values, not the data values found
  there, so they take the coordinate's dtype and units. Use
  [`Patch.max`](`dascore.proc.aggregate.max`) or
  [`Patch.min`](`dascore.proc.aggregate.min`) for the values themselves.

- Missing samples (NaN in float data, NaT in time-like data) are skipped.
  A slice which is missing everywhere has no coordinate to point at, so
  it yields NaT for a time-like coordinate and NaN otherwise. An integer
  coordinate is widened to float64 to hold that NaN, which loses
  exactness above 2**53; a coordinate which can hold neither, such as a
  string one, raises instead.

- Ties go to the first occurrence along the dimension, as in NumPy.
"""


def _missing_mask(data):
    """Return a mask of missing samples, or None if the dtype has none."""
    if dtype_time_like(data.dtype):
        return np.isnat(data)
    if np.issubdtype(data.dtype, np.inexact):
        return np.isnan(data)
    # Integers and booleans have no value which means "missing".
    return None


def _comparable(data):
    """Return data in a form NumPy can order, and its extreme fills."""
    if dtype_time_like(data.dtype):
        # NaT does not compare, but its int64 view is simply the smallest
        # int64, so the view orders correctly once NaT is filled.
        info = np.iinfo(np.int64)
        return data.view(np.int64), info.min, info.max
    return data, -np.inf, np.inf


def _extreme_index(data, axis, want_max):
    """
    Return the index of the extreme along axis, and the all-missing slices.

    The index for an all-missing slice is arbitrary; the mask says which
    ones those are so the caller can null them out.
    """
    missing = _missing_mask(data)
    if missing is None or not missing.any():
        arg = np.argmax if want_max else np.argmin
        return arg(_comparable(data)[0], axis=axis), None
    values, least, most = _comparable(data)
    filled = np.where(missing, least if want_max else most, values)
    extreme = filled.max(axis=axis) if want_max else filled.min(axis=axis)
    # Match the extreme in the unfilled data rather than taking the arg of
    # the filled data: a genuine -inf equals the fill for a max, and an arg
    # reduction breaking that tie first-wins would point at the missing
    # sample instead of the real one. A missing sample cannot match here,
    # since NaN equals nothing and NaT views as the smallest int64, which
    # is only ever the extreme when the whole slice is missing.
    hit = values == np.expand_dims(extreme, axis)
    return hit.argmax(axis=axis), missing.all(axis=axis)


def _fill_empty(values, empty):
    """Put a null where a slice had nothing to point at."""
    dtype = values.dtype
    if dtype_time_like(dtype) or np.issubdtype(dtype, np.inexact):
        return np.where(empty, _get_nullish(dtype), values)
    if np.issubdtype(dtype, np.integer) and not np.issubdtype(dtype, np.bool_):
        # Integers cannot hold a null, so widen as Patch.pad does.
        return np.where(empty, np.nan, values.astype(np.float64))
    msg = (
        f"A slice with no valid sample has no {dtype} coordinate to point "
        "at. Drop or fill the empty slices before calling idxmax/idxmin."
    )
    raise ParameterError(msg)


def _index_to_coord(coord, want_max, name):
    """Build a reduction mapping data to the coord value at its extreme."""
    coord_values = coord.values

    def _func(data, axis):
        original = data
        if not is_numpy(data):
            # The index gymnastics below are numpy only, so say so rather
            # than quietly pulling a lazy or device array into memory.
            warn_numpy_fallback(name, backend_name(data))
            data = to_numpy(data)
        index, empty = _extreme_index(data, axis, want_max)
        out = coord_values[index]
        if empty is not None:
            out = _fill_empty(out, empty)
        return out if data is original else asarray_like(out, original)

    return _func


def _idx_aggregate(patch, dim, want_max, name, dim_reduce):
    """Shared implementation of idxmax and idxmin."""
    if not isinstance(dim, str):
        msg = f"{name} reduces a single dimension; dim must be its name."
        raise ParameterError(msg)
    coord = patch.get_coord(dim)
    if isinstance(coord, CoordPartial):
        # A partial coord, such as the one the default dim_reduce leaves
        # behind, would index as NaN and quietly null the whole result.
        msg = (
            f"The '{dim}' coordinate holds no values for {name} to return. "
            "This happens when the dimension has already been reduced."
        )
        raise ParameterError(msg)
    func = _index_to_coord(coord, want_max, name)
    out = _apply_aggregator(patch, dim, func, dim_reduce)
    # datetime64 and timedelta64 carry their unit in the dtype, and the
    # coord's unit describes the step rather than the magnitude, so
    # labelling nanoseconds "s" would silently scale any unit maths.
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
    return _idx_aggregate(patch, dim, True, "idxmax", dim_reduce)


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
    >>> patch = dc.get_example_patch()
    >>>
    >>> # The time of each channel's smallest sample.
    >>> trough_time = patch.idxmin("time")

    See Also
    --------
    - [`Patch.idxmax`](`dascore.proc.aggregate.idxmax`)
    - [`Patch.min`](`dascore.proc.aggregate.min`)
    """
    return _idx_aggregate(patch, dim, False, "idxmin", dim_reduce)
