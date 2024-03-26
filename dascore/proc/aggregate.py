"""Module for applying aggregations along a specified axis."""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np

import dascore.core
from dascore.constants import PatchType
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import iterate
from dascore.utils.patch import patch_function
from dascore.utils.time import is_datetime64, to_datetime64, to_float


def _take_first(data, axis):
    """Just take the first values of data along an axis."""
    return np.take(data, 0, axis=axis)


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
keep_dims
    If True, keep the dimension(s) specified by dims argument. Otherwise, 
    the aggregated dimension(s) will be removed.  
"""


@patch_function()
@compose_docstring(params=AGG_DOC_STR, options=list(_AGG_FUNCS))
def aggregate(
    patch: PatchType,
    dim: str | None = None,
    keep_dims: bool = False,
    method: Literal["mean", "median", "min", "max", "first", "last"]
    | Callable = "mean",
) -> PatchType:
    """
    Aggregate values along a specified dimension.

    Parameters
    ----------
    {params}
    method
        The aggregation to apply along dimension. Options are:
            {options}

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # Calculate mean along time axis, keep same shape of patch.
    >>> patch_time = patch.mean("time", keep_dims=True)
    >>> # Calculate median distance, discard aggregated dimensions.
    >>> patch_dist = patch.median("distance", keep_dims=False)
    """
    func = _AGG_FUNCS.get(method, method)
    if keep_dims:
        out = _aggregate_keep_dims(patch, dim, func)
    else:
        out = _aggregate_reduce_dims(patch, dim, func)
    return out


def _aggregate_keep_dims(patch, dims, func):
    """Aggregate while keeping dimensions."""
    data, coords = patch.data, patch.coords
    for dim in iterate(patch.dims if dims is None else dims):
        axis = patch.dims.index(dim)
        data = np.expand_dims(func(data, axis=axis), axis)
        coord_array = coords.get_array(dim)
        # Need to account for taking mean of datetime arrays.
        if is_datetime64(coord_array):
            ns = to_float(patch.coords.get_array(dim))
            coord = dascore.core.get_coord(values=[to_datetime64(np.mean(ns))])
        else:
            coord = dascore.core.get_coord(values=[np.mean(coord_array)])
        coords = coords.update(**{dim: coord})
    return patch.new(data=data, coords=coords)


def _aggregate_reduce_dims(patch, dims, func):
    """Remove dimensions from the patch while aggregating."""
    for dim in iterate(patch.dims if dims is None else dims):
        axis = patch.dims.index(dim)
        data = func(patch.data, axis=axis)
        # In this case we have reduced all the dimensions. Just return scalar.
        if not isinstance(data, np.ndarray):
            return data
        coords = patch.coords.update(**{dim: None})
        patch = patch.new(data=data, coords=coords)
    return patch


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def min(
    patch: PatchType,
    dim: str | None = None,
    keep_dims: bool = False,
) -> PatchType:
    """
    Calculate the minimum along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    return aggregate.func(patch, dim=dim, keep_dims=keep_dims, method=np.nanmean)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def max(
    patch: PatchType,
    dim: str | None = None,
    keep_dims: bool = False,
) -> PatchType:
    """
    Calculate the maximum along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    return aggregate.func(patch, dim=dim, keep_dims=keep_dims, method=np.nanmax)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def mean(
    patch: PatchType,
    dim: str | None = None,
    keep_dims: bool = False,
) -> PatchType:
    """
    Calculate the mean along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    return aggregate.func(patch, dim=dim, keep_dims=keep_dims, method=np.nanmean)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def median(
    patch: PatchType,
    dim: str | None = None,
    keep_dims: bool = False,
) -> PatchType:
    """
    Calculate the median along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    return aggregate.func(patch, dim=dim, keep_dims=keep_dims, method=np.nanmedian)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def std(
    patch: PatchType,
    dim: str | None = None,
    keep_dims: bool = False,
) -> PatchType:
    """
    Calculate the standard deviation along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    return aggregate.func(patch, dim=dim, keep_dims=keep_dims, method=np.nanstd)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def first(
    patch: PatchType,
    dim: str | None = None,
    keep_dims: bool = False,
) -> PatchType:
    """
    Get the first value along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    func = _AGG_FUNCS["first"]
    return aggregate.func(patch, dim=dim, keep_dims=keep_dims, method=func)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def last(
    patch: PatchType,
    dim: str | None = None,
    keep_dims: bool = False,
) -> PatchType:
    """
    Get the last value along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    func = _AGG_FUNCS["last"]
    return aggregate.func(patch, dim=dim, keep_dims=keep_dims, method=func)


@patch_function()
@compose_docstring(params=AGG_DOC_STR)
def sum(
    patch: PatchType,
    dim: str | None = None,
    keep_dims: bool = False,
) -> PatchType:
    """
    Sum the values along one or more dimensions.

    Parameters
    ----------
    {params}
    """
    func = _AGG_FUNCS["sum"]
    return aggregate.func(patch, dim=dim, keep_dims=keep_dims, method=func)
