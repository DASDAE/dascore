"""
Processing operations that have much to do with coordinates.
"""
from dascore.constants import PatchType
from dascore.utils.patch import patch_function


@patch_function()
def snap_coords(patch: PatchType, *coords, reverse=False):
    """
    Snap coordinates to evenly spaced samples.

    This ensures all of the specified coordinates are evenly spaced and
    monotonic. First, the patch is sorted along specified coordinates,
    then coordinates are assumed evenly-sampled from their min to max value.
    Doing this can introduce some error since the coordinate labels are moved
    and the data (apart from the sorting) are left unchanged.
    Consider using [interpolate](`dascore.Patch.interpolate`) for a more expensive
    but more accurate linear interpolation.

    Parameters
    ----------
    *coords
        Used to specify the dimension names to convert to CoordRanges. If not
        specified convert all dimensional coordinates.
    reverse
        If True, reverse the sorting of the coordinates.

    Examples
    --------
    >>> import dascore as dc
    >>> # get an example patch which has unevenly sampled coords time, distance
    >>> patch = dc.get_example_patch("wacky_dim_coords_patch")
    >>> # snap time dimension
    >>> time_snap = patch.snap_coords("time")
    >>> # snap the distance dimension
    >>> dist_snap = patch.snap_coords("distance")
    """
    cman, data = patch.coords.snap(*coords, array=patch.data, reverse=reverse)
    return patch.new(data=data, coords=cman)


@patch_function()
def sort_coords(patch: PatchType, *coords, reverse=False):
    """
    Sort one or more coordinates.

    Sorts the specified coordinates in the patch. An error will be raised
    if the coordinates have overlapping dimensions since it may not be
    possible to sort each. An error is also raised in any of the coordinates
    are multidimensional.

    Parameters
    ----------
    *coords
        Used to specify the coordinates to sort.
    reverse
        If True, sort in descending order, else ascending.

    Examples
    --------
    >>> import dascore as dc
    >>> # get an example patch which has unevenly sampled coords time, distance
    >>> patch = dc.get_example_patch("wacky_dim_coords_patch")
    >>> # sort time coordinate (dimension) in ascending order
    >>> time_snap = patch.sort_coords("time")
    >>> assert time_snap.coords.coord_map['time'].sorted
    >>> # sort distance coordinate (dimension) in descending order
    >>> dist_snap = patch.sort_coords("distance", reverse=True)
    >>> assert dist_snap.coords.coord_map['distance'].reverse_sorted
    """
    cman, data = patch.coords.sort(*coords, array=patch.data, reverse=reverse)
    return patch.new(data=data, coords=cman)
