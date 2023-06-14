"""
Function for querying Patchs
"""
from dascore.constants import PatchType
from dascore.utils.patch import patch_function


@patch_function(history=None)
def select(patch: PatchType, *, copy=False, **kwargs) -> PatchType:
    """
    Return a subset of the trace based on query parameters.

    Any dimension of the data can be passed as key, and the values
    should either be a Slice or a tuple of (min, max) for that
    dimension.

    The time dimension is handled specially in that either floats,
    datetime64 or datetime objects can be used to specify relative
    or absolute times, respectively.

    Parameters
    ----------
    copy
        If True, copy the resulting data. This is needed so the old
        array can get gc'ed and memory freed.
    **kwargs
        Used to specify the dimension and slices to select on.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.examples import get_example_patch
    >>> tr = get_example_patch()
    >>> # select meters 50 to 300
    >>> new = tr.select(distance=(50, 300))
    """
    new_coords, inds = patch.coords.select(**kwargs)
    data = patch.data[inds]
    # no slicing was performed, just return original.
    if data.shape == patch.data.shape:
        return patch
    if copy:
        data = data.copy()
    attrs, dims = patch.attrs, patch.dims
    return patch.__class__(data, attrs=attrs, coords=new_coords, dims=dims)
