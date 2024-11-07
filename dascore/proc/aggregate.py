"""Module for applying aggregations (reductions) along a specified axis."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np

import dascore as dc
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import iterate
from dascore.utils.patch import get_dim_axis_value, patch_function
from dascore.utils.time import (
    dtype_time_like,
    is_datetime64,
    to_datetime64,
    to_timedelta64,
)

_AGG_FUNCS = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "min": np.nanmin,
    "max": np.nanmax,
    "sum": np.nansum,
    "std": np.nanstd,
    "first": partial(np.take, indices=0),
    "last": partial(np.take, indices=-1),
}


AGG_DOC_STR = """
patch
    The input Patch.
dim
    The dimension along which aggregations are to be performed.
    If None, apply aggregation to all dimensions sequentially.
    If a sequence, apply sequentially in order provided.
dim_reduce
    How to reduce the dimensional coordinate associated with the 
    aggregated axis. Can be the name of any valid aggregator, a callable,
    "empty" (the default) - which returns and empty coord, or "squeeze" 
    which drops the coordinate. For dimensions with datetime or timedelta 
    datatypes, if the operation fails it will automatically be applied 
    to the coordinates converted to floats then the output converted back 
    to the appropriate time type. 
"""

AGG_NOTES = """
Notes
-----
See [`Patch.aggregate`](`dascore.Patch.aggregate`) for examples
and more details.
"""


def _get_new_coord(coord, dim_reduce):
    """Get the new coordinate."""

    def _maybe_handle_datatypes(func, data):
        """Maybe handle the complexity of date times here."""
        try:  # First try function directly
            out = func(data)
        except Exception:  # Fall back to floats and re-packing.
            float_data = dc.to_float(data)
            dfunc = to_datetime64 if is_datetime64(data) else to_timedelta64
            out = dfunc(func(float_data))
        return np.atleast_1d(out)

    if dim_reduce == "empty":
        new_coord = coord.update(shape=(1,), start=None, stop=None, data=None)
    elif dim_reduce == "squeeze":
        return None
    elif (func := _AGG_FUNCS.get(dim_reduce)) or callable(dim_reduce):
        func = dim_reduce if callable(dim_reduce) else func
        coord_data = coord.data
        if dtype_time_like(coord_data):
            result = _maybe_handle_datatypes(func, coord_data)
        else:
            result = func(coord.data)
        new_coord = coord.update(data=result)
    else:
        msg = "dim_reduce must be 'empty', 'squeeze' or valid aggregator."
        raise ParameterError(msg)
    return new_coord


@patch_function()
@compose_docstring(params=AGG_DOC_STR, options=list(_AGG_FUNCS))
def aggregate(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    method: str | Callable = "mean",
    dim_reduce: str | Callable = "empty",
) -> PatchType:
    """
    Aggregate values along a specified dimension.

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
    data = patch.data
    dims = tuple(iterate(patch.dims if dim is None else dim))
    dfo = get_dim_axis_value(patch, args=dims, allow_multiple=True)
    # Iter all specified dimensions.
    for dim, axis, value in dfo:
        new_coord = _get_new_coord(patch.get_coord(dim), dim_reduce=dim_reduce)
        if new_coord is None:
            coords = patch.coords.drop_coords(dim)[0]
            data = func(data, axis=axis)
        else:
            coords = patch.coords.update(**{dim: new_coord})
            data = np.expand_dims(func(data, axis=axis), axis)
        patch = patch.new(data=data, coords=coords)
    return patch


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
