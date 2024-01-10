"""Processing operations that have much to do with coordinates."""
from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing_extensions import Self

from dascore.constants import PatchType
from dascore.core.coords import BaseCoord
from dascore.exceptions import CoordError, ParameterError
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
    >>> new_dist = pa.coords.get_array('distance') + 1
    >>> pa2 = pa.update_coords(distance=new_dist)
    >>> assert np.allclose(pa2.coords.get_array('distance'), new_dist)
    """
    new_coord = self.coords.update(**kwargs)
    return self.new(coords=new_coord, dims=new_coord.dims)


@patch_function()
def drop_coords(self: PatchType, *coords: str | Collection[str]) -> PatchType:
    """
    Update the coordiantes of a patch.

    Will either add new coordinates, or update existing ones.

    Parameters
    ----------
    *coords
        One or more coordinates to drop.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> pa = dc.get_example_patch("random_patch_with_lat_lon")
    >>> # Drop non-dimensional coordinate latitude
    >>> pa_no_lat = pa.drop_coords("latitude")
    """
    if dim_coords := set(coords) & set(self.dims):
        msg = f"Cannot drop dimensional coordinates: {dim_coords}"
        raise ParameterError(msg)
    new_coord, data = self.coords.drop_coords(*coords, array=self.data)
    return self.new(coords=new_coord, dims=new_coord.dims, data=data)


@patch_function(history="method_name")
def coords_from_df(
    self: PatchType,
    dataframe: pd.DataFrame,
    units: dict[str, str] | None = None,
    extrapolate: bool = False,
) -> PatchType:
    """
    Update non-dimensional coordinate of a patch using a dataframe.

    Parameters
    ----------
    dataframe
        Table with a column matching in title to one of patch.dims along with other
        coordinates to associate with dimension. Example one column matching distance
        axis and then latitude and longitude attached to the distances.
    units
        Dictionary mapping column name in dataframe to its units.
    extrapolate
        If True, extrapolate outside provided range in dataframe.

    Examples
    --------
    >>> import dascore as dc
    >>> import pandas as pd
    >>> # get example patch and create example dataframe
    >>> pa = dc.get_example_patch()
    >>> distance = pa.coords.get_array("distance")[::10]
    >>> df = pd.DataFrame(distance, columns=['distance'])
    >>> df['x'] = df['distance'] * 3 + 10
    >>> df['y'] = df['distance'] * 2.5 - 10
    >>> # attach dataframe to patch, interpolating when needed. This
    >>> # adds coordinates x and y which are associated with dimension distance.
    >>> patch_with_coords = pa.coords_from_df(df)

    Notes
    -----
    * Exactly one of the column names in the dataframe must map to one of
      the patch.dims. This will either add new coordinates, or update existing
      ones if they already exist.

    * This function uess linear extrapolation between the nearest two points
      to get values in patch coords that aren't in the dataframe.

    """
    # match dataframe headings to dims
    anchor_dim = set(self.dims) & set(dataframe.columns)
    if len(anchor_dim) != 1:
        msg = "Exactly one column has to match with an existing dimension"
        raise ParameterError(msg)

    # Get coordinates of axis being updated
    anchor_dim = next(iter(anchor_dim))
    coords = self.coords
    axis_coords = coords.get_array(anchor_dim)

    # make a dictionary from coordinates("(axis, coordinate array)") as input to
    # update_coords
    # coordinate array is an interpolation to match existing coords being updated
    new_coords = {}

    for coord in set(dataframe.columns) - {anchor_dim}:
        if extrapolate:
            f = interp1d(
                pd.to_numeric(dataframe[anchor_dim]),
                pd.to_numeric(dataframe[coord]),
                fill_value="extrapolate",
            )
            new_coords[coord] = (anchor_dim, f(axis_coords))
        else:
            new_coords[coord] = (
                anchor_dim,
                np.interp(
                    axis_coords,
                    pd.to_numeric(dataframe[anchor_dim]),
                    pd.to_numeric(dataframe[coord]),
                    left=float("nan"),
                    right=float("nan"),
                ),
            )

    out = self.update_coords.func(self, **new_coords)

    if units is not None:
        out = out.convert_units.func(out, **units)

    return out
