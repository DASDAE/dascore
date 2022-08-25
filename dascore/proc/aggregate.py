"""
Module for applying aggregations along a specified axis.
"""
from functools import partial
from typing import Literal

import numpy as np

from dascore.constants import PatchType
from dascore.utils.patch import copy_attrs, patch_function
from dascore.utils.time import to_datetime64


def _take_first(data, axis):
    """just take the first values of data along an axis."""
    return np.take(data, 0, axis=axis)


_AGG_FUNCS = {
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "first": partial(np.take, indices=0),
    "last": partial(np.take, indices=-1),
}


@patch_function()
def aggregate(
    patch: PatchType,
    dim: str,
    method: Literal["mean", "median", "min", "max", "first", "last"] = "mean",
) -> PatchType:
    """
    Aggregate values along a specified dimension.

    Parameters
    ----------
    dim
        The dimension along which aggregations are to be performed.
    method
        The aggregation to apply along dimension. Options are:
            mean, min, max, median, first, last
    """
    axis = patch.dims.index(dim)
    func = _AGG_FUNCS[method]
    new_data = np.expand_dims(func(patch.data, axis=axis), axis)
    # update coords with new dimension (use mean)
    coords = patch.coords
    if dim == "time":  # need to convert time to ints
        ns = patch.coords[dim].astype(int) / 1_000_000_000
        new_coord_val = to_datetime64(np.mean(ns))
    else:
        new_coord_val = np.mean(patch.coords[dim])
    new_coords = {
        name: coords[name] if name != dim else np.array([new_coord_val])
        for name in patch.dims
    }
    # set
    new_attrs = copy_attrs(patch.attrs)
    new_attrs[f"{dim}_min"] = new_coord_val
    new_attrs[f"{dim}_max"] = new_coord_val
    new_attrs[f"d_{dim}"] = np.NaN
    return patch.__class__(new_data, attrs=new_attrs, coords=new_coords)
