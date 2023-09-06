"""Function for querying Patchs."""
from __future__ import annotations

from dascore.constants import PatchType
from dascore.utils.patch import patch_function


@patch_function(history=None)
def select(
    patch: PatchType, *, copy=False, relative=False, samples=False, **kwargs
) -> PatchType:
    """
    Return a subset of the patch.

    Any dimension of the data can be passed as key, and the values
    should either be a Slice or a tuple of (min, max) for that
    dimension. None and ... both indicate open intervals.

    Parameters
    ----------
    patch
        The patch object.
    copy
        If True, copy the resulting data. This is needed so the old
        array can get gc'ed and memory freed.
    relative
        If True, select ranges are relative to the start of coordinate, if
        possitive, or the end of the coordinate, if negative.
    samples
        If True, the query meaning is in samples.
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
    >>> # select channels less than 300
    >>> lt_dist = patch.select(distance=(..., 300))
    >>> # select time (1 second from start to -1 second from end)
    >>> t1 = patch.attrs.time_min + dc.to_timedelta64(1)
    >>> t2 = patch.attrs.time_max - dc.to_timedelta64(1)
    >>> new_time1 = patch.select(time=(t1, t2))
    >>> # this can be accomplished more simply using the relative keyword
    >>> new_time2 = patch.select(time=(1, -1), relative=True)
    >>> # filter 1 second from start time to 3 seconds from start time
    >>> new_time3 = patch.select(time=(1, 3), relative=True)
    >>> # filter 6 second from end time to 1 second from end time
    >>> new_time4 = patch.select(time=(-6, -1), relative=True)
    >>> # Select first 10 distance indices
    >>> new_distance1 = patch.select(distance=(..., 10), samples=True)
    >>> # Select last time row/column
    >>> new_distance2 = patch.select(time=-1, samples=True)
    """
    new_coords, data = patch.coords.select(
        **kwargs,
        array=patch.data,
        relative=relative,
        samples=samples,
    )
    # no slicing was performed, just return original.
    if data.shape == patch.data.shape:
        return patch
    if copy:
        data = data.copy()
    return patch.new(data=data, coords=new_coords)
