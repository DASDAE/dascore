"""Processing operations that have much to do with coordinates."""
from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import Self

from dascore.constants import PatchType
from dascore.core.coords import BaseCoord
from dascore.exceptions import CoordError
from dascore.utils.misc import get_parent_code_name, iterate
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


def get_coord(
    self: PatchType,
    name: str,
    require_sorted: bool = False,
    require_evenly_sampled: bool = False,
) -> BaseCoord:
    """
    Get a managed coordinate, raising if it doesn't meet requirements.

    Parameters
    ----------
    name
        Name of the coordinate to fetch.
    require_sorted
        If True, require the coordinate to be sorted or raise Error.
    require_evenly_sampled
        If True, require the coordinate to be evenly sampled or raise Error.
    """
    coord = self.coords.coord_map[name]
    if require_evenly_sampled and coord.step is None:
        extra = f"as required by {get_parent_code_name()}"  # adds caller name
        msg = f"Coordinate {name} is not evenly sampled {extra}"
        raise CoordError(msg)
    if require_sorted and not coord.sorted or coord.reverse_sorted:
        extra = f"as required by {get_parent_code_name()}"  # adds caller name
        msg = f"Coordinate {name} is not sorted {extra}"
        raise CoordError(msg)
    return coord


def assert_has_coords(self: PatchType, coord_names: Sequence[str] | str) -> Self:
    """Raise an error if patch doesn't have required coordinates."""
    required_coords = set(iterate(coord_names))
    current_coords = set(self.coords.coord_map)
    if missing := required_coords - current_coords:
        msg = f"Patch does not have required coordinate(s): {missing}"
        raise CoordError(msg)
    return self


@patch_function()
def rename_coords(self: PatchType, **kwargs) -> PatchType:
    """
    Rename coordinate of Patch.

    Parameters
    ----------
    **kwargs
        The mapping from old names to new names

    Examples
    --------
    >>> import dascore as dc
    >>> pa = dc.get_example_patch()
    >>> # rename dim "distance" to "fragrance"
    >>> pa2 = pa.rename_coords(distance='fragrance')
    >>> assert 'fragrance' in pa2.dims
    """
    new_coord = self.coords.rename_coord(**kwargs)
    attrs = self.attrs.rename_dimension(**kwargs)
    return self.new(coords=new_coord, dims=new_coord.dims, attrs=attrs)


@patch_function()
def update_coords(self: PatchType, **kwargs) -> PatchType:
    """
    Update the coordiantes of a patch.

    Will either add new coordinates, or update existing ones.

    Parameters
    ----------
    **kwargs
        The mapping from old names to new names

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> pa = dc.get_example_patch()
    >>> # Add 1 to all distance coords
    >>> new_dist = pa.coords['distance'] + 1
    >>> pa2 = pa.update_coords(distance=new_dist)
    >>> assert np.all(pa2.coords['distance'] == (pa.coords['distance'] + 1))
    """
    new_coord = self.coords.update_coords(**kwargs)
    return self.new(coords=new_coord, dims=new_coord.dims)
