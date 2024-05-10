"""Module for applying aggregations along a specified axis."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np

from dascore.compat import is_array
from dascore.constants import PatchType
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import iterate
from dascore.utils.patch import patch_function

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
"""


@patch_function()
@compose_docstring(params=AGG_DOC_STR, options=list(_AGG_FUNCS))
def aggregate(
    patch: PatchType,
    dim: str | Sequence[str] | None = None,
    method: str | Callable = "mean",
) -> PatchType:
    """
    Aggregate values along a specified dimension.

    Parameters
    ----------
    {params}
    method
        The aggregation to apply along dimension. Options are:
            {options}

    Notes
    -----
    - The old dimension is kept but its coordiante values are removed.
      use [`Patch.squeeze`](`dascore.Patch.squeeze`) to remove them.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc

    >>> patch = dc.get_example_patch()
    >>> # Calculate mean along time axis
    >>> patch_time = patch.aggregate("time", method=np.nanmean)
    >>> # Calculate median distance along distance dimension
    >>> patch_dist = patch.median("distance", method=np.nanmedian)
    """
    func = _AGG_FUNCS.get(method, method)
    data = patch.data
    # iterate all
    for current_dim in iterate(patch.dims if dim is None else dim):
        axis = patch.dims.index(current_dim)
        # Use expand dims
        data = np.expand_dims(func(data, axis=axis), axis)
        coords = patch.coords.update(**{current_dim: 1})
        patch = patch.new(data=data, coords=coords)
    return patch


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def min(
    patch: PatchType,
    dim: str | None = None,
) -> PatchType:
    """
    Calculate the minimum along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    return aggregate.func(patch, dim=dim, method=np.nanmin)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def max(
    patch: PatchType,
    dim: str | None = None,
) -> PatchType:
    """
    Calculate the maximum along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    return aggregate.func(patch, dim=dim, method=np.nanmax)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def mean(
    patch: PatchType,
    dim: str | None = None,
) -> PatchType:
    """
    Calculate the mean along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    return aggregate.func(patch, dim=dim, method=np.nanmean)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def median(
    patch: PatchType,
    dim: str | None = None,
) -> PatchType:
    """
    Calculate the median along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    return aggregate.func(patch, dim=dim, method=np.nanmedian)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def std(
    patch: PatchType,
    dim: str | None = None,
) -> PatchType:
    """
    Calculate the standard deviation along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    return aggregate.func(patch, dim=dim, method=np.nanstd)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def first(
    patch: PatchType,
    dim: str | None = None,
) -> PatchType:
    """
    Get the first value along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    func = _AGG_FUNCS["first"]
    return aggregate.func(patch, dim=dim, method=func)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def last(
    patch: PatchType,
    dim: str | None = None,
) -> PatchType:
    """
    Get the last value along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    func = _AGG_FUNCS["last"]
    return aggregate.func(patch, dim=dim, method=func)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def sum(
    patch: PatchType,
    dim: str | None = None,
) -> PatchType:
    """
    Sum the values along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    return aggregate.func(patch, dim=dim, method=np.nansum)
