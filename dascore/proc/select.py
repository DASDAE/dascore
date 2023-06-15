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
    >>> import dascore as dc
    >>> from dascore.examples import get_example_patch
    >>> patch = get_example_patch()
    >>> # select meters 50 to 300
    >>> new_distance = patch.select(distance=(50, 300))
    >>> # select time (1 second from start to -1 second from end)
    >>> t1 = patch.attrs.time_min + dc.to_timedelta64(1)
    >>> t2 = patch.attrs.time_max - dc.to_timedelta64(1)
    >>> new_time = patch.select(time=(t1, t2))
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
