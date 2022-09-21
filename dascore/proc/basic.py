"""
Basic operations for patches.
"""
from typing import Literal

import numpy as np

from dascore.constants import PatchType
from dascore.utils.patch import patch_function


@patch_function()
def abs(patch: PatchType) -> PatchType:
    """
    Take the absolute value of the patch data.
    """
    new_data = np.abs(patch.data)
    return patch.new(data=new_data)


@patch_function()
def transpose(self: PatchType, *dims: str) -> PatchType:
    """
    Transpose the data array to any dimension order desired.

    Parameters
    ----------
    *dims
        Dimension names which define the new data axis order.
    """
    return self.__class__(self._data_array.transpose(*dims))


@patch_function()
def squeeze(self: PatchType, dim=None, drop=False):
    """
    Return a new object with len one dimensions flattened.

    Parameters
    ----------
    dim
        Selects a subset of the length one dimensions. If a dimension
        is selected with length greater than one, an error is raised.
        If None, all length one dimensions are squeezed.
    drop
        If True, drop squeezed coordinates instead of making them scalar.


    Notes
    -----
    Simply calls `xr.squeeze`.
    """
    dar = self._data_array
    out = dar.squeeze(dim=dim, drop=drop)
    return self.__class__(out)


@patch_function()
def rename(self: PatchType, **names) -> PatchType:
    """
    Rename coordinate or dimensions of Patch.

    Parameters
    ----------
    **names
        The mapping from old names to new names

    Examples
    --------
    >>> from dascore.examples import get_example_patch
    >>> pa = get_example_patch()
    >>> # rename dim "distance" to "fragrance"
    >>> pa2 = pa.rename(distance='fragrance')
    >>> assert 'fragrance' in pa2.dims
    """
    new_data_array = self._data_array.rename(**names)
    return self.__class__(new_data_array)


@patch_function()
def normalize(
    self: PatchType,
    dim: str,
    norm: Literal["l1", "l2", "max"] = "l2",
) -> PatchType:
    """
    Normalize a patch along a specified dimension.

    Parameters
    ----------
    dim
        The dimension along which the normalization takes place.
    norm
        Determines the value to divide each sample by along a given axis.
        Options are:
            l1 - divide each sample by the l1 of the axis.
            l2 - divide each sample by the l2 of the axis.
            max - divide each sample by the maximum of the absolute value of the axis.
    """
    axis = self.dims.index(dim)
    data = self.data
    if norm in {"l1", "l2"}:
        order = int(norm[-1])
        norm_values = np.linalg.norm(self.data, axis=axis, ord=order)
    elif norm == "max":
        norm_values = np.max(data, axis=axis)
    else:
        msg = (
            f"Norm value of {norm} is not supported. "
            f"Supported values are {('l1', 'l2', 'max')}"
        )
        raise ValueError(msg)
    new_data = data / np.expand_dims(norm_values, axis=axis)
    return self.new(data=new_data)
