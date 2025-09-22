"""Module for applying aggregations (reductions) along a specified axis."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from dascore.constants import _AGG_FUNCS, DIM_REDUCE_DOCS, PatchType
from dascore.utils.array import _apply_aggregator
from dascore.utils.docs import compose_docstring
from dascore.utils.patch import patch_function

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


@patch_function()
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


@patch_function()
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
