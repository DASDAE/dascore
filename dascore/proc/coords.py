"""Processing operations that have much to do with coordinates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ConfigDict
from scipy.interpolate import interp1d

import dascore as dc
from dascore.constants import PatchType, select_values_description
from dascore.core.coords import BaseCoord, CoordSegmented
from dascore.exceptions import (
    CoordError,
    ParameterError,
    PatchCoordinateError,
    PatchError,
)
from dascore.utils.array_api import array_namespace
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import get_parent_code_name, iterate
from dascore.utils.patch import patch_function
from dascore.utils.time import dtype_time_like
from dascore.utils.transformatter import FourierTransformatter
from dascore.workflow.processor import (
    PatchProcessor,
    register_implementation,
)


@patch_function()
def snap_coords(patch: PatchType, *coords, reverse: bool = False) -> PatchType:
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
    patch
        The patch whose coordinates should be snapped.
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
    # Nothing changed; return the original patch to avoid a rebuild.
    if cman is patch.coords and data is patch.data:
        return patch
    return patch.new(data=data, coords=cman)


@patch_function()
def sort_coords(patch: PatchType, *coords, reverse: bool = False) -> PatchType:
    """
    Sort one or more coordinates.

    Sorts the specified coordinates in the patch. An error will be raised
    if the coordinates have overlapping dimensions since it may not be
    possible to sort each. An error is also raised in any of the coordinates
    are multidimensional.

    Parameters
    ----------
    patch
        The patch whose coordinates should be sorted.
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
    # Nothing changed; return the original patch to avoid a rebuild.
    if cman is patch.coords and data is patch.data:
        return patch
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
) -> np.ndarray:
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
    >>> # Get an array of time values. Raise an error if they aren't sorted.
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
    return RenameCoords(**kwargs)._apply(self)


class RenameCoords(PatchProcessor):
    """
    Give coordinates other names.

    No kernel: the data are what they were, and only what they are
    called changes. A processor which defines no kernel says exactly
    that, and `_apply` hands the array through untouched.
    """

    # The renames arrive as whatever the caller named them, so the fields
    # cannot be known in advance; `model_values` folds the extras back
    # into the parameters, so the fingerprint and the document still hold
    # every one of them.
    model_config = ConfigDict(extra="allow", frozen=True)

    def derive_meta(self, meta):
        """Return the coordinates under their new names."""
        coords = meta.coords.rename_coord(**self._params())
        return meta.update(coords=coords)


register_implementation("rename_coords", RenameCoords)


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
def drop_coords(self: PatchType, *coords: str | Iterable[str]) -> PatchType:
    """
    Update the coordinates of a patch.

    Will either add new coordinates, or update existing ones.

    Parameters
    ----------
    *coords
        One or more coordinates to drop. Each can be a coordinate name or
        a sequence of them.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> pa = dc.get_example_patch("random_patch_with_lat_lon")
    >>> # Drop non-dimensional coordinate latitude
    >>> pa_no_lat = pa.drop_coords("latitude")
    >>> # A sequence of names works as well.
    >>> pa_no_lat = pa.drop_coords(["latitude"])
    """
    names = {x for coord in coords for x in iterate(coord)}
    if dim_coords := names & set(self.dims):
        msg = f"Cannot drop dimensional coordinates: {dim_coords}"
        raise ParameterError(msg)
    new_coord, data = self.coords.drop_coords(*names, array=self.data)
    if new_coord is self.coords:  # none of the named coords were here
        return self
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
    if new_coord is self.coords:  # there were no private coords
        return self
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

    * This function uses linear extrapolation between the nearest two points
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


def _check_coord_names(patch: PatchType, kwargs) -> None:
    """Refuse a name the patch has no coordinate for, naming what it has."""
    if not (invalid := set(kwargs) - set(patch.coords.coord_map)):
        return
    valid_list = sorted(patch.coords.coord_map)
    msg = (
        f"Coordinate(s) {sorted(invalid)} not found in patch coordinates: {valid_list}"
    )
    raise PatchCoordinateError(msg)


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
        positive, or the end of the coordinate, if negative.
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
    >>> t1 = patch.get_coord("time").min() + dc.to_timedelta64(1)
    >>> t2 = patch.get_coord("time").max() - dc.to_timedelta64(1)
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
      patch, only filter values. If the order of the patch should change, or
      multiple rows/columns need to be repeated,
      See [`Patch.order`](`dascore.Patch.order`).

    - A range of values includes both of its endpoints, but a range of
      samples excludes its upper bound, like python's slicing. This means
      -1 at the end of a sample range excludes the last sample, even
      though -1 on its own selects it. Using the example patch, which has
      300 distance channels and 2000 time samples:

      >>> import dascore as dc
      >>> patch = dc.get_example_patch()
      >>> # Both endpoints included; 11 channels.
      >>> len(patch.select(distance=(0, 10)).get_array("distance"))
      11
      >>> # Upper bound excluded; 10 samples.
      >>> len(patch.select(time=(0, 10), samples=True).get_array("time"))
      10
      >>> # -1 as a range end drops the last sample.
      >>> len(patch.select(time=(0, -1), samples=True).get_array("time"))
      1999
      >>> # But -1 on its own selects it.
      >>> len(patch.select(time=-1, samples=True).get_array("time"))
      1

      A slice can be used in place of a tuple, and makes the half-open
      behavior of sample ranges more obvious:

      >>> len(patch.select(time=slice(0, -1), samples=True).get_array("time"))
      1999

    """
    _check_coord_names(patch, kwargs)
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
def unselect(
    patch: PatchType, *, copy=False, relative=False, samples=False, **kwargs
) -> PatchType:
    """
    Return the patch outside a selection.

    The complement of [`Patch.select`](`dascore.Patch.select`): it takes
    the same selectors and removes the samples that selection would have
    kept. With one coordinate named that is exactly the complement; with
    several, each is complemented on its own — see the note below.

    Parameters
    ----------
    patch
        The patch object.
    copy
        If True, copy the resulting data. This is needed so the old
        array can get gc'ed and memory freed.
    relative
        If True, unselect ranges are relative to the start of coordinate, if
        positive, or the end of the coordinate, if negative.
    samples
        If True, the query meaning is in samples.
    **kwargs
        Used to specify the coordinate on which data are unselected.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.examples import get_example_patch
    >>> patch = get_example_patch()
    >>>
    >>> # Drop meters 50 to 300, keeping what lies outside them.
    >>> outside = patch.unselect(distance=(50, 300))
    >>>
    >>> # Drop the first ten distance samples.
    >>> trimmed = patch.unselect(distance=(..., 10), samples=True)

    Notes
    -----
    - Removing a range from the middle of a coordinate leaves a hole in
      it, so the result is no longer evenly sampled and the coordinate
      becomes a monotonic array. That is exactly what
      [`Spool.unselect`](`dascore.core.spool.Spool.unselect`) refuses the
      patches' *own* coordinates for: at spool level the complement of a
      range is a hole in every patch rather than a choice between
      patches. The coordinates an attached inventory defines along the
      fiber it does accept, since removing one of those chooses which
      channels a patch holds.

    - Each named coordinate is complemented on its own. Selecting on two
      coordinates keeps the samples in both ranges, and everything
      outside that is a frame around them rather than a block, which no
      array can hold — so `unselect` removes each named range instead,
      which is the part of the complement that is expressible. Two
      coordinates along one dimension therefore both take their range
      out of it, leaving what neither removed.
    """
    _check_coord_names(patch, kwargs)
    keep: dict[str, np.ndarray] = {}
    for name, value in kwargs.items():
        coord = patch.coords.coord_map[name]
        dims = patch.coords.dim_map[name]
        if len(dims) != 1:
            msg = (
                f"Coordinate {name!r} spans {list(dims)}, so removing a range "
                "of it does not name samples of one dimension to drop."
            )
            raise PatchCoordinateError(msg)
        # Asking select itself which samples it would keep is what stops
        # the two from drifting: one selector cannot come to mean
        # different things in select and its complement.
        _, indexer = coord.select(value, relative=relative, samples=samples)
        selected = np.zeros(len(coord), dtype=bool)
        selected[indexer] = True
        keep[dims[0]] = keep.get(dims[0], True) & ~selected
    # Kept as sample numbers along each dimension rather than as a mask
    # per coordinate: coordinates sharing a dimension are applied in
    # separate passes, so the second mask would meet an already trimmed
    # axis, and a dimension carrying no values of its own takes samples
    # where it would refuse an array.
    trims = {dim: np.flatnonzero(mask) for dim, mask in keep.items()}
    return patch.select(**trims, samples=True, copy=copy)


@patch_function(history=None)
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
        If True, the values are indices along the coordinate rather than
        values in it.
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
      If there are repeated values in the requested values or in the patch
      coordinate arrays, the data will end up being repeated as well.
    """
    new_coords, data = patch.coords.order(
        **kwargs,
        array=patch.data,
        relative=relative,
        samples=samples,
    )
    if copy:
        assert data is not None  # order returns an array when given one
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
        Can also include ... to indicate dimensions that should be left
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
    return Transpose(dims=tuple(dims))._apply(self)


class Transpose(PatchProcessor):
    """Put the dimensions of a patch into another order."""

    # Typed loosely because `...` is a legal element: `transpose(...,
    # "distance")` means "distance last, the rest as they were".
    dims: tuple[Any, ...] = ()

    def derive_meta(self, meta):
        """
        Return the coordinates in their new order.

        The coord manager hands back the very object it was given when
        the order asked for is the order already held, and that is what
        tells `_apply` the operation did nothing.
        """
        old_dims = meta.coords.dims
        named = [x for x in self.dims if x is not ...]
        if invalid := set(named) - set(old_dims):
            msg = (
                f"Dimension(s) {sorted(invalid)} not found in Patch "
                f"dimensions: {sorted(old_dims)}"
            )
            raise ParameterError(msg)
        coords = meta.coords.transpose(*self.dims)
        return meta if coords is meta.coords else meta.update(coords=coords)

    def kernel(self, data, meta, out_meta):
        """Return the data with its axes permuted to the new order."""
        if out_meta is meta:
            return data
        axes = tuple(meta.dims.index(x) for x in out_meta.dims)
        return array_namespace(data).permute_dims(data, axes)


register_implementation("transpose", Transpose)


@patch_function(history=None)
def append_dims(patch: PatchType, *empty_dims, **dim_kwargs) -> PatchType:
    """
    Insert dimensions at the end of the patch.

    Parameters
    ----------
    patch
        The patch to add dimensions to.
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
    data = np.expand_dims(patch.data, tuple(insert_inds))
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

    Notes
    -----
    [`Patch.squeeze_coords`](`dascore.Patch.squeeze_coords`) does the
    analogous thing for non-dimensional coordinates whose values are all
    the same: it drops them and stores the value in the attrs.
    """
    coords = self.coords.squeeze(dim)
    # Nothing to squeeze; the coord manager returned self, so reuse this patch.
    if coords is self.coords:
        return self
    if dim is None:
        axes = tuple(i for i, x in enumerate(self.shape) if x == 1)
    else:
        axes = tuple(self.get_axis(x) for x in iterate(dim))
    xp = array_namespace(self.data)
    data = xp.squeeze(self.data, axis=axes)
    return self.new(data=data, coords=coords)


# Sentinel for a coord with no lone value, since a lone value can itself
# be falsey.
_NO_VALUE = object()


# Names the attrs already use, so a coord value cannot be stored under
# them. Asked of the class rather than listed: a declared field validates
# or coerces whatever it is given (`data_type=3.0` becomes `"3.0"`), a
# method shadows the value on the way back out, `dims` is dropped on the
# way in and `coords` refused there, and the decorator stamps `history`
# and both ids over once this returns.
def _reserved_attr(name: str) -> bool:
    """Whether the patch attrs already use a name for something else."""
    attrs = dc.PatchAttrs
    if name in {"coords", "dims"} or name in attrs.model_fields:
        return True
    return hasattr(attrs, name)


# What an attr can hold and the rest of DASCore can write down: numbers,
# booleans, text, times. An object coord holds python objects, which need
# not even compare elementwise (an entry which is itself a sequence
# broadcasts instead), and a complex value is read back as a time by the
# spool index.
_STATEABLE_KINDS = frozenset("biufSUMm")


def _get_lone_value(coord: BaseCoord):
    """Return the single distinct value a coord holds, or _NO_VALUE if it has none."""
    if not coord.size:
        return _NO_VALUE
    values = np.asarray(coord.values).reshape(-1)
    if values.dtype.kind not in _STATEABLE_KINDS:
        return _NO_VALUE
    value = values[0]
    # A nullish value is what a coord says when it doesn't know, which a
    # partial coord says for every sample. Not something to state as an
    # attr, so such a coord keeps its shape and stays a coord. Asked
    # first: a partial coord's values are a broadcast view, and the
    # comparison below would make a real array the size of the coord.
    if pd.isnull(value):
        return _NO_VALUE
    if values.size > 1 and not bool(np.all(values == value)):
        return _NO_VALUE
    # Times stay numpy scalars, matching how the other attrs store them;
    # everything else stores better as a python scalar.
    if not dtype_time_like(values.dtype) and hasattr(value, "item"):
        value = value.item()
    return value


@patch_function()
def squeeze_coords(self: PatchType, *coords: str | Iterable[str]) -> PatchType:
    """
    Convert coordinates which hold a single value to attributes.

    A coordinate whose values are all the same says one thing about the
    patch rather than one thing about each sample. Each such coordinate is
    dropped and its value stored in the patch attrs under the coordinate's
    name.

    Parameters
    ----------
    *coords
        The names of the coordinates to squeeze, or sequences of them. If
        none are given, every non-dimensional coordinate which holds a
        single value is squeezed.

    Raises
    ------
    CoordError
        If a named coordinate is not in the patch, is a dimension, is
        private, is named for something the patch attrs already use, is
        needed to invert a transform, or does not hold the same non-null
        value the attrs can state for every sample.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>>
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Add a coordinate which has the same value for each distance.
    >>> quality = np.ones(patch.coord_shapes["distance"])
    >>> patch = patch.update_coords(quality=("distance", quality))
    >>>
    >>> # The coordinate becomes an attribute.
    >>> out = patch.squeeze_coords()
    >>> assert "quality" not in out.coords.coord_map
    >>> assert out.attrs.get("quality") == 1

    Notes
    -----
    - Dimensions are never squeezed, even when they have length one; use
      [`Patch.squeeze`](`dascore.Patch.squeeze`) for those.
    - Coordinate units are not kept, only the value.
    - A coordinate whose only value is null (NaN or NaT), or whose values
      are of a type the attrs cannot state (an object or a complex
      number), is left alone.
    - So are the coordinates which are not the patch's to state: a
      private one (whose name begins with an underscore), one needed to
      invert a transform ([`Patch.idft`](`dascore.Patch.idft`) reads
      `time` off a patch whose dimension is `ft_time`), and one named
      for something the attrs already use, such as `tag` or `history`.
      All of these are passed over by the sweep, and raise when named.
    - An attribute of the same name is overwritten.
    - Which coordinates the sweep takes depends on the values a patch
      holds, so patches in a spool can come out with different
      coordinates. Name the coordinates to squeeze the same ones
      everywhere.
    """
    cm = self.coords
    # True when the caller named coords at all: an explicitly empty
    # sequence squeezes nothing rather than sweeping every coord.
    given = tuple(x for x in coords if x is not None)
    named = bool(given)
    names = (
        tuple(x for coord in given for x in iterate(coord))
        if named
        else tuple(cm.coord_map)
    )
    values = {}
    for name in names:
        value, refusal = _get_squeezed_value(cm, name)
        # A named coord must qualify, so one which doesn't raises; the
        # sweep skips it silently.
        if refusal:
            if named:
                raise CoordError(refusal)
            continue
        values[name] = value
    # Nothing qualified, so nothing happened.
    if not values:
        return self
    new_coords, _ = cm.drop_coords(*values)
    return self.new(coords=new_coords, attrs=self.attrs.update(**values))


def _get_squeezed_value(cm, name: str) -> tuple[Any, str]:
    """Return the value a coord becomes, and why it cannot become one."""
    if name not in cm.coord_map:
        # Said as `Patch.get_coord` says it, listing what is there.
        coords = sorted(cm.coord_map)
        msg = f"Coordinate '{name}' not found in Patch coordinates: {coords}"
        return _NO_VALUE, msg
    if name in cm.dims:
        msg = (
            f"Cannot squeeze coordinate {name} because it is a dimension. "
            f"Use Patch.squeeze to remove length one dimensions."
        )
        return _NO_VALUE, msg
    if _reserved_attr(name):
        msg = (
            f"Cannot squeeze coordinate {name} because the patch attrs "
            f"already use that name. Rename the coordinate first."
        )
        return _NO_VALUE, msg
    # A private coord belongs to whichever operation put it there --
    # `stft` reads its window back in `istft` -- so it is not the
    # patch's to state as an attr.
    if name.startswith("_"):
        return _NO_VALUE, f"Cannot squeeze coordinate {name} because it is private."
    # A transform leaves the coordinate it consumed behind under its own
    # name so the transform can be undone: `idft` reads `time` off a
    # patch whose dimension is `ft_time`.
    if name in FourierTransformatter().rename_dims(cm.dims, forward=False):
        msg = (
            f"Cannot squeeze coordinate {name} because the transform "
            f"which made this patch needs it to be inverted."
        )
        return _NO_VALUE, msg
    value = _get_lone_value(cm.coord_map[name])
    if value is _NO_VALUE:
        msg = (
            f"Cannot squeeze coordinate {name} because its values are "
            f"not all the same non-null value."
        )
        return _NO_VALUE, msg
    return value, ""


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


def split_gaps(self: PatchType, dim: str | None = None) -> dc.Spool:
    """
    Split the patch into contiguous patches at coordinate gaps.

    Dimensional coordinates that are segmented
    ([`CoordSegmented`](`dascore.core.coords.CoordSegmented`), e.g. produced
    by concatenating nearly-contiguous data) mark where the patch is not
    contiguous. This splits the patch at every segment boundary so each
    output patch has a plain, contiguous coordinate.

    Parameters
    ----------
    self
        The Patch object.
    dim
        The dimension to split along. If None (default), split along every
        dimension with a segmented coordinate. Patches without segmented
        coordinates come back unchanged (as a length 1 spool).

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> from dascore.core.coords import concat_coords, get_coord
    >>>
    >>> # A patch whose distance coordinate has a gap.
    >>> dist = concat_coords(
    ...     get_coord(start=0.0, stop=10.0, step=1.0),
    ...     get_coord(start=15.0, stop=25.0, step=1.0),
    ... )
    >>> patch = dc.Patch(
    ...     data=np.zeros((len(dist), 5)),
    ...     coords={"distance": dist, "time": dc.to_datetime64(np.arange(5))},
    ...     dims=("distance", "time"),
    ... )
    >>> spool = patch.split_gaps()
    >>> assert len(spool) == 2
    """
    if dim is not None and dim not in self.dims:
        msg = f"split_gaps dim must be one of {self.dims}, got {dim!r}."
        raise ParameterError(msg)
    dims = (dim,) if dim is not None else self.dims
    patches = [self]
    for dname in dims:
        out = []
        for patch in patches:
            coord = patch.get_coord(dname)
            if not isinstance(coord, CoordSegmented):
                out.append(patch)
                continue
            offset = 0
            for seg in coord.segments:
                stop = offset + len(seg)
                out.append(patch.select(**{dname: (offset, stop)}, samples=True))
                offset = stop
        patches = out
    return dc.spool(patches)
