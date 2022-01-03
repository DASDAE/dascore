"""
Basic operations for patches.
"""
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
