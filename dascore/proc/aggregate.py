"""Module for applying aggregations (reductions) along a specified axis."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from dascore.constants import _AGG_FUNCS, DIM_REDUCE_DOCS, PatchType
from dascore.exceptions import ParameterError
from dascore.utils.array import _apply_aggregator
from dascore.utils.docs import compose_docstring
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
    The dimension along which to find the extreme value.
{DIM_REDUCE_DOCS}
"""

IDX_NOTES = """
Notes
-----
- NaN values are ignored. A slice which is entirely NaN has no extreme
  value to point at, so it yields NaT for a time-like coordinate and NaN
  otherwise; an integer coordinate is upcast to float so it can hold the
  NaN.

- The returned data are coordinate values, not the data values found
  there, so the patch's data units become the coordinate's units. Use
  [`Patch.max`](`dascore.proc.aggregate.max`) or
  [`Patch.min`](`dascore.proc.aggregate.min`) for the values themselves.

- Ties go to the first occurrence along the dimension, as in NumPy.
"""


def _nan_fill(dtype):
    """Return the null value and output dtype used for all-NaN slices."""
    if dtype_time_like(dtype):
        return np.array("NaT", dtype=dtype), dtype
    if np.issubdtype(dtype, np.floating):
        return np.array(np.nan, dtype=dtype), dtype
    # Integer and everything else cannot hold NaN, so widen to float.
    return np.float64(np.nan), np.dtype(np.float64)


def _index_to_coord(coord_values, arg_func, extreme_func):
    """
    Build a reduction mapping data to the coord value at its extreme.

    `arg_func` is the nan-aware argmin/argmax and `extreme_func` its
    nan-aware counterpart, used only to locate all-NaN slices.
    """

    def _func(data, axis):
        data = np.asarray(data)
        # nanargmin/nanargmax raise on an all-NaN slice, so find those first
        # and hand the reduction a slice it can answer.
        if np.issubdtype(data.dtype, np.inexact):
            empty = np.all(np.isnan(data), axis=axis)
            safe = np.where(np.isnan(data), extreme_func, data)
        else:
            empty, safe = None, data
        out = np.asarray(coord_values)[arg_func(safe, axis=axis)]
        if empty is not None and empty.any():
            fill, dtype = _nan_fill(out.dtype)
            out = np.where(empty, fill, out.astype(dtype))
        return out

    return _func


def _idx_aggregate(patch, dim, arg_func, extreme_func, dim_reduce):
    """Shared implementation of idxmax and idxmin."""
    if not isinstance(dim, str):
        msg = "idxmax/idxmin reduce a single dimension; dim must be its name."
        raise ParameterError(msg)
    coord = patch.get_coord(dim)
    func = _index_to_coord(coord.values, arg_func, extreme_func)
    out = _apply_aggregator(patch, dim, func, dim_reduce)
    # The data are now coordinate values, so they carry the coord's units
    # and none of the original data's meaning.
    return out.update_attrs(data_units=coord.units, data_type="")


@patch_function()
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
    >>> # Drop the reduced dimension entirely, as xarray's idxmax does.
    >>> peak_time = patch.idxmax("time", dim_reduce="squeeze")

    See Also
    --------
    - [`Patch.idxmin`](`dascore.proc.aggregate.idxmin`)
    - [`Patch.max`](`dascore.proc.aggregate.max`)
    """
    return _idx_aggregate(patch, dim, np.argmax, -np.inf, dim_reduce)


@patch_function()
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
    return _idx_aggregate(patch, dim, np.argmin, np.inf, dim_reduce)
