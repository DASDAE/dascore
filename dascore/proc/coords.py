"""Processing operations that have much to do with coordinates."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from dascore.constants import PatchType, select_values_description
from dascore.core.coords import BaseCoord
from dascore.exceptions import (
    CoordError,
    ParameterError,
    PatchCoordinateError,
    PatchError,
)
from dascore.utils.docs import compose_docstring
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
    >>>
    >>> # snap time dimension
    >>> time_snap = patch.snap_coords("time")
    >>>
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
    >>>
    >>> # sort time coordinate (dimension) in ascending order
    >>> time_snap = patch.sort_coords("time")
    >>> assert time_snap.coords.coord_map['time'].sorted
    >>>
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
    Get a managed coordinate from the patch.

    Parameters
    ----------
    name
        The name of the coordinate to fetch from the patch.
    require_sorted
        If True, require the coordinate to be sorted or raise Error.
    require_evenly_sampled
        If True, require the coordinate to be evenly sampled or raise Error.

    Raises
    ------
    [`CoordError`](`dascore.exceptions.CoordError`) if the coordinate does
    not exist or does not meet the imposed requirements.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Get the the distance coordinate from the patch.
    >>> distance = patch.get_coord("distance")
    >>>
    >>> # Get the time coordinate from the patch, raise CoordError if it
    >>> # is not evenly sampled.
    >>> time = patch.get_coord("time", require_evenly_sampled=True)

    See Also
    --------
    [get_array](`dascore.Patch.get_array`).

    """
    if (coord := self.coords.coord_map.get(name)) is None:
        coords = sorted(self.coords.coord_map)
        msg = f"Coordinate '{name}' not found in Patch coordinates: {coords}"
        raise CoordError(msg)
    if require_evenly_sampled and coord.step is None:
        extra = f"as required by {get_parent_code_name()}"  # adds caller name
        msg = f"Coordinate {name} is not evenly sampled {extra}"
        raise CoordError(msg)
    if require_sorted and not (coord.sorted or coord.reverse_sorted):
        extra = f"as required by {get_parent_code_name()}"  # adds caller name
        msg = f"Coordinate {name} is not sorted {extra}"
        raise CoordError(msg)
    return coord


def get_array(
    self: PatchType,
    name: str | None = None,
    require_sorted: bool = False,
    require_evenly_sampled: bool = False,
) -> BaseCoord:
    """
    Get an array associated with patch data or a coordinate.

    Parameters
    ----------
    name
        The name of the coordinate to fetch. If None return patch data.
    require_sorted
        If True, require the coordinate to be sorted or raise Error.
    require_evenly_sampled
        If True, require the coordinate to be evenly sampled or raise Error.

    Raises
    ------
    [`CoordError`](`dascore.exceptions.CoordError`) if the coordinate does
    not exist or does not meet the imposed requirements.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Get the patch data array.
    >>> data = patch.get_array()  # same as patch.data
    >>>
    >>> # Get an array of distance values
    >>> distance_array = patch.get_array("distance")
    >>>
    >>> # Get an array of time values. Raise an error if they arent sorted.
    >>> time_array = patch.get_array("time", require_sorted=True)

    See Also
    --------
    [Patch.get_coord](`dascore.Patch.get_coord`)
    """
    if name is None:
        return self.data
    coord = get_coord(
        self,
        name,
        require_sorted=require_sorted,
        require_evenly_sampled=require_evenly_sampled,
    )
    return coord.data


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
    >>>
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
    Update the coordinates of a patch.

    Will either add new coordinates, or update existing ones.

    Parameters
    ----------
    kwargs
        The name of the coordinate (key) and coordinate values. Values
        can either be a sequence (eg array) or a single int. If an int
        is used it will create a non-coord.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> pa = dc.get_example_patch()
    >>>
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
    Update the coordinates of a patch.

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


@patch_function()
def drop_private_coords(self: PatchType) -> PatchType:
    """
    Drop all private coords in the patch.

    Parameters
    ----------
    self
        Patch

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> pa = (
    ...     dc.get_example_patch("random_das")
    ...     .update_coords(_private=(None, np.array([1,2,3])))
    ... )
    >>> pa_no_private = pa.drop_private_coords()
    >>> assert "_private" not in pa_no_private.coords.coord_map
    """
    new_coord, data = self.coords.drop_private_coords(array=self.data)
    return self.new(coords=new_coord, dims=new_coord.dims, data=data)


@patch_function()
def make_broadcastable_to(
    self: PatchType,
    shape: tuple[int, ...],
    drop_coords=False,
) -> PatchType:
    """
    Update the coordinates of a patch.

    Will either add new coordinates, or update existing ones.

    Parameters
    ----------
    shape
        The new shape the patch should be able to broadcast with.
    drop_coords
        If True, drop coords that need to be broadcasted up, otherwise
        only NonCoordinate dimensions can change shape.

    Examples
    --------
    >>> import dascore as dc
    >>> pa = dc.get_example_patch("random_das")
    >>> # Get a patch with non-coordinate dimensions
    >>> patch = pa.mean()
    >>> out = patch.make_broadcastable_to(shape=(2, 3))
    >>> assert out.shape == (2, 3)
    """
    coords, data = self.coords.make_broadcastable_to(
        shape, array=self.data, drop_coords=drop_coords
    )
    return self.new(coords=coords, data=data)


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


@patch_function(history=None)
@compose_docstring(select_params=select_values_description)
def select(
    patch: PatchType, *, copy=False, relative=False, samples=False, **kwargs
) -> PatchType:
    """
    Return a subset of the patch.

    {select_params}

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
        Used to specify the coordinate on which data are selected.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> from dascore.examples import get_example_patch
    >>> patch = get_example_patch()
    >>>
    >>> # select meters 50 to 300
    >>> new_distance = patch.select(distance=(50, 300))
    >>>
    >>> # select channels less than 300
    >>> lt_dist = patch.select(distance=(..., 300))
    >>>
    >>> # select time (1 second from start to -1 second from end)
    >>> t1 = patch.attrs.time_min + dc.to_timedelta64(1)
    >>> t2 = patch.attrs.time_max - dc.to_timedelta64(1)
    >>> new_time1 = patch.select(time=(t1, t2))
    >>>
    >>> # this can be accomplished more simply using the relative keyword
    >>> new_time2 = patch.select(time=(1, -1), relative=True)
    >>>
    >>> # filter 1 second from start time to 3 seconds from start time
    >>> new_time3 = patch.select(time=(1, 3), relative=True)
    >>>
    >>> # filter 6 second from end time to 1 second from end time
    >>> new_time4 = patch.select(time=(-6, -1), relative=True)
    >>>
    >>> # Select first 10 distance indices
    >>> new_distance1 = patch.select(distance=(..., 10), samples=True)
    >>>
    >>> # Select last time row/column
    >>> new_distance2 = patch.select(time=-1, samples=True)
    >>>
    >>> # only include certain rows/columns based on a boolean array.
    >>> time = patch.get_array("time")
    >>> new_time_5 = patch.select(time=time>time[2])
    >>>
    >>> # Select only specific values along a dimension
    >>> distance = patch.get_array("distance")
    >>> new_distance_3 = patch.select(distance=distance[1::2])

    Notes
    -----
    - It is important to remember select will not change the order of the
      patch, only fiter values. If the order of the patch should change, or
      multiple rows/columns need to be repeated,
      See [`Patch.order`](`dascore.Patch.order`).

    """
    # Check for and raise on invalid kwargs.
    if invalid_coords := set(kwargs) - set(patch.coords.coord_map):
        invalid_list = sorted(invalid_coords)
        valid_list = sorted(patch.coords.coord_map)
        msg = (
            f"Coordinate(s) {invalid_list} not found in patch coordinates: {valid_list}"
        )
        raise PatchCoordinateError(msg)

    new_coords, data = patch.coords.select(
        **kwargs,
        array=patch.data,
        relative=relative,
        samples=samples,
    )
    # no slicing was performed, just return original if coordinates also unchanged.
    if data.shape == patch.data.shape and new_coords == patch.coords:
        return patch
    if copy:
        data = data.copy()
    return patch.new(data=data, coords=new_coords)


@patch_function(history=None)
@compose_docstring(select_params=select_values_description)
def order(
    patch: PatchType, *, copy=False, relative=False, samples=False, **kwargs
) -> PatchType:
    """
    Re-order the patch contents based on coordinate values or indices.

    Parameters
    ----------
    patch
        The patch object.
    copy
        If True, copy the resulting data. This is needed so the old
        array can get gc'ed and memory freed.
    relative
        If True, order values are relative to the start/end of the coordinates.
    samples
        If True, the
    **kwargs
        Used to specify the coordinate and values on which the coordinates
        are ordered.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.examples import get_example_patch
    >>> patch = get_example_patch()
    >>>
    >>> # Sub-select only a section of the distance and ensure order.
    >>> dist = patch.get_array("distance")
    >>> new_dist = dist[1:5][::-1]
    >>> patch_1 = patch.order(distance=new_dist)
    >>>
    >>> # Get duplicate the first time row or column
    >>> patch_2 = patch.order(time=[0, 0, 0], samples=True)

    Notes
    -----
    - This function is similar to [`Patch.select`](`dascore.Patch.select`)
      but it will also change the patch order to match the inputs exactly.
      If there are repeated values in the requsted values or in the patch
      coordinate arrays, the data will end up being repeated as well.
    """
    new_coords, data = patch.coords.order(
        **kwargs,
        array=patch.data,
        relative=relative,
        samples=samples,
    )
    if copy:
        data = data.copy()
    return patch.new(data=data, coords=new_coords)


@patch_function(history=None)
def transpose(self: PatchType, *dims: str) -> PatchType:
    """
    Transpose the data array to any dimension order desired.

    Parameters
    ----------
    *dims
        Dimension names which define the new data axis order.
        Can also include ... to indicate diemsnions that should be left
        alone.

    Examples
    --------
    >>> import dascore # import dascore library
    >>> pa = dascore.get_example_patch() # generate example patch
    >>>
    >>> # Transpose the time and data array dimensions in the example patch
    >>> out = pa.transpose("time", "distance")
    >>>
    >>> # Set "distance" as the last dimension
    >>> out = pa.transpose(..., "distance")
    >>>
    >>> # Set distance as the first dimension.
    >>> out = pa.transpose("distance", ...)
    """
    dims = tuple(dims)
    old_dims = self.coords.dims
    new_coord = self.coords.transpose(*dims)
    new_dims = new_coord.dims
    axes = tuple(old_dims.index(x) for x in new_dims)
    new_data = np.transpose(self.data, axes)
    return self.new(data=new_data, coords=new_coord)


@patch_function(history=None)
def append_dims(patch: PatchType, *empty_dims, **dim_kwargs) -> PatchType:
    """
    Insert dimensions at the end of the patch.

    Parameters
    ----------
    empty_dims
        Used to pass the name of empty dimensions.
    dim_kwargs
        Used to pass keys (new dim names) and values. Values can either be
        an int specifying the length of the new dimension or a sequence
        specifying the coordinate values. If an int is used, the new dimension
        will be a non-coordinate dimension.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Add two dummy dimensions to patch named "end" and "stop"
    >>> new = patch.append_dims("end", "stop")
    >>>
    >>> # Add a dummy dimension called "face" to end of patch
    >>> # which has a coordinate value of [1].
    >>> new = patch.append_dims(face=[1])
    >>>
    >>> # Same thing as above, but with a larger coords which broadcasts
    >>> # the data to shape appropriate to mach coordinates.
    >>> new = patch.append_dims(face=[1, 2])
    >>>
    >>> # Add a dummy dimension of length 3 to end of patch.
    >>> # the data to shape appropriate to mach coordinates.
    >>> new = patch.append_dims(face=3)

    Notes
    -----
    - This tries to be more simple than numpy and xarray's expand_dims.
    - Use [`Patch.transpose`](`dascore.Patch.transpose`) to re-arrange dimensions.
    - If dimension with the same name already exists nothing will happen.
    """
    dim_dict = {x: 1 for x in empty_dims}
    dim_dict.update(dim_kwargs)
    # Remove duplicate dims and convert non ints to arrays.
    kwargs = {
        i: (i, np.atleast_1d(v) if not isinstance(v, int) else v)
        for i, v in dim_dict.items()
        if i not in patch.dims
    }
    # Nothing to do.
    if not kwargs:
        return patch
    ndim = patch.ndim
    # First get data with empty dimensions
    insert_inds = [x + ndim for x in range(len(kwargs))]
    data = np.expand_dims(patch.data, insert_inds)
    shapes = list(data.shape)
    for ind, (_, cdata) in zip(insert_inds, kwargs.values()):
        shapes[ind] = cdata if isinstance(cdata, int) else len(cdata)
    data = np.broadcast_to(data, shapes)
    coords = patch.coords.update(**kwargs)
    return patch.update(data=data, coords=coords)


@patch_function()
def squeeze(self: PatchType, dim=None) -> PatchType:
    """
    Return a new object with len one dimensions flattened.

    Parameters
    ----------
    dim
        Selects a subset of the length one dimensions. If a dimension
        is selected with length greater than one, an error is raised.
        If None, all length one dimensions are squeezed.

    Examples
    --------
    >>> import dascore as dc
    >>> import numpy as np
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Create a patch with a length-1 dimension by selecting time slice
    >>> time_array = patch.coords.get_array("time")
    >>> single_time = patch.select(time=(time_array[0], time_array[0]))
    >>>
    >>> # Squeeze the length-1 time dimension
    >>> squeezed = single_time.squeeze(dim="time")
    """
    coords = self.coords.squeeze(dim)
    if dim is None:
        axes = None
    else:
        axes = tuple(self.get_axis(x) for x in iterate(dim))
    data = np.squeeze(self.data, axis=axes)
    return self.new(data=data, coords=coords)


@patch_function()
def add_distance_to(
    patch: PatchType, origin: pd.Series, ord=None, prefix: str = "origin"
) -> PatchType:
    """
    Calculate the distance to "origin" and create new coordinate.

    A new coordinate called `origin_distance` (or another name controlled
    by the pre-fix argument) is added to the output patch to specify the
    exact distance. Coordinates representing the origin location
    (eg origin_x, origin_y, origin_z) are also added as non-associated
    coordinates.

    Parameters
    ----------
    patch
        The patch object which contains some overlap in coordinates as
        index names in origin.
    origin
        A series which contains index names that overlap with patch coordinates.
        All the referenced coordinates must be associated with the same
        dimension.
    ord
        Controls the norm type. Default is Frobenius norm, see the norm
        function of numpy.linalg for supported options.
    prefix
        The prefix name for the added coordinates and attributes.

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> import dascore as dc
    >>>
    >>> # Add a coordinate specifying the distance to a theoretical shot.
    >>> shot = pd.Series({"x": 10, "y": 10, "z": 0})
    >>> patch = dc.get_example_patch("random_patch_with_xyz")
    >>> patch_with_origin_dist = patch.add_distance_to(shot)
    >>> # Now the new coordinates of distance and shot origin exist.
    >>> dist = patch_with_origin_dist.get_array("origin_distance")
    >>> origin_x = patch_with_origin_dist.get_array("origin_x")
    >>>
    >>> # Of course, the new coordinate can be used for sorting.
    >>> sorted_patch = patch_with_origin_dist.sort_coords("origin_distance")
    """
    # Ensure all index values are represented in coord map.
    if missing_coords := (set(origin.index) - set(patch.coords.coord_map)):
        msg = f"Indices {missing_coords} are not patch coordinates."
        raise PatchError(msg)
    # Ensure all coordinates have the same associated dimension.
    associated_dims = {patch.coords.dim_map[x] for x in origin.index}
    if len(associated_dims) > 1:
        dims = {i: v for i, v in patch.coords.dim_map.items() if i in origin.index}
        msg = (
            "All coordinate must be associated with the same dimension to "
            f"calculate distance. Relevant dimension mappings are {dims}"
        )
        raise PatchError(msg)
    # Create 2d arrays from coords and origin.
    coord_array = np.stack([patch.get_array(x) for x in origin.index], axis=1)
    origin_array = np.atleast_2d(origin.values)
    # Translate coords to origin and take norm.
    distance = np.linalg.norm(origin_array - coord_array, axis=1, ord=ord)
    # Add attrs and coords to new patch
    dims = next(iter(associated_dims))
    new_coords = {f"{prefix}_{i}": (None, np.atleast_1d(v)) for i, v in origin.items()}
    new_coords[f"{prefix}_distance"] = (dims, distance)
    out = patch.update_coords.func(patch, **new_coords)
    return out


def get_axis(self: PatchType, dim: str) -> int:
    """
    Get the axis corresponding to a Patch dimension. Raise error if not found.

    Parameters
    ----------
    self
        The Patch object.
    dim
        The dimension name.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> axis = patch.get_axis("time")
    >>> assert axis == patch.get_axis("time")
    """
    return self.coords.get_axis(dim)
