"""Processing operations that have much to do with coordinates."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import ConfigDict
from scipy.interpolate import interp1d

import dascore as dc
from dascore.constants import PatchType, select_values_description
from dascore.core.coords import BaseCoord, CoordSegmented, _fill_layout
from dascore.core.processor import PatchProcessor
from dascore.exceptions import (
    CoordError,
    ParameterError,
    PatchCoordinateError,
    PatchError,
)
from dascore.utils.array_api import array_namespace, to_numpy
from dascore.utils.docs import compose_docstring
from dascore.utils.indexing import get_indexers, label_indexer
from dascore.utils.misc import (
    _apply_union_indexers,
    broadcast_for_index,
    get_parent_code_name,
    iterate,
)
from dascore.utils.patch import (
    drop_associated_coords,
    get_dim_axis_value,
    patch_function,
)


class _IndexRecorder:
    """
    An array's stand-in, which records the indexing applied to it.

    [`CoordManager.sort`](`dascore.core.CoordManager.sort`) and `snap`
    reorder an array they are handed but never say how, and the metadata
    half of an operation runs before the data are touched. Standing in for
    the array collects exactly the indexing the kernel has to replay,
    rather than a second copy of the manager's sorting logic.
    """

    def __init__(self):
        self.keys: list[tuple] = []

    def __getitem__(self, key):
        """Record an indexing operation and stand in for its result."""
        self.keys.append(key)
        return self


class _Reorder(PatchProcessor):
    """The half snapping and sorting share: putting coordinates in order."""

    coords: tuple[str, ...] = ()
    reverse: bool = False

    name = None
    _var_positional = "coords"
    # Whether the coordinates are snapped to even samples once sorted.
    _snap: ClassVar[bool] = False

    def get_metadata(self, meta):
        """Return the sorted coordinates and the indexing they imply."""
        recorder = _IndexRecorder()
        method = meta.coords.snap if self._snap else meta.coords.sort
        coords, _ = method(*self.coords, array=recorder, reverse=self.reverse)
        # The manager hands back itself when there was nothing to sort.
        if coords is meta.coords:
            return meta, {}
        return meta.new(coords=coords), {"keys": tuple(recorder.keys)}

    def kernel(self, data, *, keys=()):
        """Return the data reordered the way the coordinates were."""
        for key in keys:
            data = data[key]
        return data


class SnapCoords(_Reorder):
    """
    Snap coordinates to evenly spaced samples.

    Sorts each specified coordinate, then replaces its labels with evenly spaced
    values between its endpoints in the selected sort direction. Data remain
    unchanged after sorting, so snapping can shift labels. Use
    [interpolate](`dascore.Patch.interpolate`) when linear interpolation is preferable.

    Parameters
    ----------
    *coords
        Dimensions to snap. By default, snap every dimensional coordinate.
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

    _snap = True


snap_coords = SnapCoords.patch_function


class SortCoords(_Reorder):
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


sort_coords = SortCoords.patch_function


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
    if require_evenly_sampled and not coord.evenly_sampled:
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


class RenameCoords(PatchProcessor):
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

    # The renames arrive under whatever names the caller used.
    model_config = ConfigDict(extra="allow", frozen=True)

    def get_metadata(self, meta):
        """Return the coordinates under their new names; there is no kernel."""
        return meta.new(coords=meta.coords.rename_coord(**self.kwargs)), {}


rename_coords = RenameCoords.patch_function


class UpdateCoords(PatchProcessor):
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

    # The coordinates arrive under whatever names the caller used.
    model_config = ConfigDict(extra="allow", frozen=True)

    def get_metadata(self, meta):
        """Return the new coordinates; the data they label do not move."""
        return meta.new(coords=meta.coords.update(**self.kwargs)), {}


update_coords = UpdateCoords.patch_function


class DropCoords(PatchProcessor):
    """
    Drop one or more non-dimensional coordinates.

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

    coords: tuple[Any, ...] = ()

    _var_positional = "coords"

    def get_metadata(self, meta):
        """Return the coordinates which survive; a dimension cannot go."""
        names = {x for coord in self.coords for x in iterate(coord)}
        if dim_coords := names & set(meta.dims):
            msg = f"Cannot drop dimensional coordinates: {dim_coords}"
            raise ParameterError(msg)
        # Only non-dimensional coordinates get here, so no axis is emptied
        # and the data are untouched: there is nothing for a kernel to do.
        coords, _ = meta.coords.drop_coords(*names)
        if coords is meta.coords:  # none of the named coords were here
            return meta, {}
        return meta.new(coords=coords), {}


drop_coords = DropCoords.patch_function


class DropPrivateCoords(PatchProcessor):
    """
    Drop all private coords in the patch.

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

    def get_metadata(self, meta):
        """Return the coordinates whose names are public, and what that costs."""
        coords, _ = meta.coords.drop_private_coords()
        if coords is meta.coords:  # there were no private coords
            return meta, {}
        # Unlike drop_coords, nothing here refuses a private *dimension*, and
        # dropping one takes its axis with it: the manager empties that axis
        # rather than removing it, so the data are emptied to match. Private
        # coordinates which are not dimensions leave every axis alone.
        private = {x for x in meta.dims if x.startswith("_")}
        if not private:
            return meta.new(coords=coords), {}
        indexer = tuple(
            slice(0, 0) if dim in private else slice(None) for dim in meta.dims
        )
        return meta.new(coords=coords), {"indexer": indexer}

    def kernel(self, data, *, indexer=None):
        """Return the data, emptied along any dimension which was private."""
        return _selected(data, indexer)


drop_private_coords = DropPrivateCoords.patch_function


class MakeBroadcastableTo(PatchProcessor):
    """
    Stretch a patch until it broadcasts against a shape.

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

    shape: tuple[int, ...]
    drop_coords: bool = False

    def get_metadata(self, meta):
        """Return the stretched coordinates, and the shape to stretch to."""
        coords, _ = meta.coords.make_broadcastable_to(
            self.shape, None, drop_coords=self.drop_coords
        )
        target = np.broadcast_shapes(meta.coords.shape, self.shape)
        return meta.new(coords=coords), {"shape": tuple(target)}

    def kernel(self, data, *, shape):
        """Return the data broadcast to the target shape."""
        return array_namespace(data).broadcast_to(data, shape)


make_broadcastable_to = MakeBroadcastableTo.patch_function


class CoordsFromDf(PatchProcessor):
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

    dataframe: pd.DataFrame
    # Typed loosely because a unit is written as a string or as a unit object.
    units: dict[str, Any] | None = None
    extrapolate: bool = False

    history = "method_name"

    def get_metadata(self, meta):
        """Return the coordinates the table interpolates onto a dimension."""
        dataframe = self.dataframe
        # match dataframe headings to dims
        anchor_dim = set(meta.dims) & set(dataframe.columns)
        if len(anchor_dim) != 1:
            msg = "Exactly one column has to match with an existing dimension"
            raise ParameterError(msg)

        # Get coordinates of axis being updated
        anchor_dim = next(iter(anchor_dim))
        axis_coords = meta.coords.get_array(anchor_dim)

        # make a dictionary from coordinates("(axis, coordinate array)") as input to
        # update_coords
        # coordinate array is an interpolation to match existing coords being updated
        new_coords = {}

        for coord in set(dataframe.columns) - {anchor_dim}:
            if self.extrapolate:
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

        coords = meta.coords.update(**new_coords)
        # Only coordinates are named, so the conversion never scales data.
        if self.units is not None:
            coords = coords.convert_units(**self.units)
        return meta.new(coords=coords), {}


coords_from_df = CoordsFromDf.patch_function


def _check_coord_names(patch: PatchType, kwargs) -> None:
    """Refuse a name the patch has no coordinate for, naming what it has."""
    if not (invalid := set(kwargs) - set(patch.coords.coord_map)):
        return
    valid_list = sorted(patch.coords.coord_map)
    msg = (
        f"Coordinate(s) {sorted(invalid)} not found in patch coordinates: {valid_list}"
    )
    raise PatchCoordinateError(msg)


def _dimension_indexers(coords, queries, **kwargs):
    """
    Return what a selection leaves, and the indexer along each dimension.

    [`select_indexers`](`dascore.core.CoordManager.select_indexers`) is the
    published form of this for plain selection; ordering and positional
    indexing need the same seam, which the manager only reaches through
    `_select`.
    """
    found: dict[str, Any] = {}
    out, _ = coords._select(queries, _indexers=found, **kwargs)
    return out, tuple(found.get(dim, slice(None)) for dim in coords.dims)


def _selected(data, indexer, copy=False):
    """Return the samples an indexer picks out of the data."""
    if indexer is None:  # nothing was filtered out
        return data
    out = _apply_union_indexers(indexer, data)
    return out.copy() if copy else out


with warnings.catch_warnings():
    # Pydantic warns that a field named `copy` shadows `BaseModel.copy`, the
    # deprecated v1 method. The name is part of these operations' public
    # signatures and nothing here calls that method. Declared on the shared
    # base so the warning is dealt with once rather than at each operation.
    warnings.filterwarnings("ignore", 'Field name "copy"', UserWarning)

    class _Query(PatchProcessor):
        """What select, unselect and order share: a query per coordinate."""

        # The queries arrive under whatever coordinates the caller named.
        model_config = ConfigDict(extra="allow", frozen=True)

        copy: bool = False
        relative: bool = False
        samples: bool = False

        name = None
        history = None
        _positional_fields = ()

        def kernel(self, data, *, indexer=None):
            """Return the samples the query picks out."""
            return _selected(data, indexer, self.copy)


@compose_docstring(select_params=select_values_description)
class Select(_Query):
    """
    Return a subset of the patch.

    {select_params}

    For xarray-compatible indexing, use [`Patch.sel`](`dascore.Patch.sel`) for labels or
    [`Patch.isel`](`dascore.Patch.isel`) for sample positions. These methods preserve
    indexer order and repetitions and remove dimensions selected with scalar indexers.
    `select` preserves dimensions and filters values in source order.

    Parameters
    ----------
    copy
        Copy the result so it does not retain the original data array.
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
    >>> from dascore.examples import get_example_patch
    >>> patch = get_example_patch()
    >>>
    >>> # Coordinate values and open bounds.
    >>> new_distance = patch.select(distance=(50, 300))
    >>> lt_dist = patch.select(distance=(..., 300))
    >>>
    >>> # One second from the start through one second before the end.
    >>> new_time = patch.select(time=(1, -1), relative=True)
    >>>
    >>> # Sample ranges and scalar sample indices.
    >>> new_distance1 = patch.select(distance=(..., 10), samples=True)
    >>> new_distance2 = patch.select(time=-1, samples=True)
    >>>
    >>> # Boolean masks and explicit coordinate values.
    >>> time = patch.get_array("time")
    >>> new_time_5 = patch.select(time=time > time[2])
    >>> distance = patch.get_array("distance")
    >>> new_distance_3 = patch.select(distance=distance[1::2])

    Notes
    -----
    Selection filters values without reordering or repeating them; use
    [`Patch.order`](`dascore.Patch.order`) for those operations.

    Value ranges include both endpoints. Sample ranges are half-open like
    Python slices, so ``-1`` as a range end excludes the final sample while
    the scalar ``-1`` selects it:

      >>> import dascore as dc
      >>> patch = dc.get_example_patch()
      >>> len(patch.select(distance=(0, 10)).get_array("distance"))
      11
      >>> len(patch.select(time=(0, 10), samples=True).get_array("time"))
      10
      >>> len(patch.select(time=(0, -1), samples=True).get_array("time"))
      1999
      >>> len(patch.select(time=-1, samples=True).get_array("time"))
      1

    See Also
    --------
    [Patch.sel](`dascore.Patch.sel`) : Xarray-compatible label indexing.
    [Patch.isel](`dascore.Patch.isel`) : Xarray-compatible positional indexing.
    """

    def get_metadata(self, meta):
        """Return the selected coordinates and the indexers behind them."""
        queries = self.model_extra or {}
        _check_coord_names(meta, queries)
        coords, indexer = _dimension_indexers(
            meta.coords, queries, relative=self.relative, samples=self.samples
        )
        # No slicing was performed, so the patch is its own answer.
        if coords == meta.coords:
            return meta, {}
        return meta.new(coords=coords), {"indexer": indexer}


select = Select.patch_function


class Isel(PatchProcessor):
    """
    Select sample positions with xarray-compatible dimension indexing.

    Supports the `DataArray.isel` operations described below. Use
    [`Patch.select`](`dascore.Patch.select`) for DASCore's tuple range notation,
    relative selections, and filtering that preserves dimensions and source order.

    Parameters
    ----------
    indexers
        Mapping of dimension names to integer positions, slices, or 1D integer
        arrays or boolean masks. Supply this or keyword indexers.
    drop
        Drop coordinates made scalar by indexing. By default they are retained
        as scalar coordinates. Scalar indexers remove their dimension either way;
        use a one-element list to retain a length-one dimension.
    missing_dims
        How to handle absent dimensions: ``"raise"``, ``"warn"``, or ``"ignore"``.
    **kwargs
        Dimension indexers supplied as keywords.

    Notes
    -----
    Slices use Python's exclusive stop and support strides and negative indices.
    Evenly sampled slice results stay compact; floating coordinate values can
    differ from xarray by rounding relative to the original range, as with `select`.
    Arrays preserve order and repetitions; arrays on multiple dimensions select
    every combination of positions. Out-of-bounds scalar and array indices raise.
    Labelled xarray indexers and multidimensional indexer arrays are not supported.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> window = patch.isel(time=slice(0, 100, 2), distance=[3, 1, 3])
    >>> assert window.shape == (3, 50)
    >>> channel = patch.isel(distance=3)
    >>> assert channel.dims == ("time",)
    >>> assert channel.get_array("distance").shape == ()

    See Also
    --------
    [Patch.sel](`dascore.Patch.sel`) : Xarray-compatible label indexing.
    [Patch.select](`dascore.Patch.select`) : Range filtering and relative selections.
    """

    # The indexers may also arrive as keywords, one per dimension.
    model_config = ConfigDict(extra="allow", frozen=True)

    # Typed loosely so `get_indexers` refuses anything else in its own words,
    # rather than a field validator refusing it in pydantic's.
    indexers: Any = None
    drop: bool = False
    missing_dims: str = "raise"

    history = None

    def get_metadata(self, meta):
        """Return the indexed coordinates and the indexers behind them."""
        requested = get_indexers(
            self.indexers, self.model_extra or {}, meta.dims, self.missing_dims
        )
        coords, indexer = _dimension_indexers(
            meta.coords, requested, operation="isel", drop=self.drop
        )
        # Nothing was named, so nothing moves.
        if coords is meta.coords:
            return meta, {}
        return meta.new(coords=coords), {"indexer": indexer}

    def kernel(self, data, *, indexer=None):
        """Return the samples at the requested positions."""
        return _selected(data, indexer)


isel = Isel.patch_function


class Sel(PatchProcessor):
    """
    Select coordinate labels with xarray-compatible dimension indexing.

    Supports the `DataArray.sel` operations described below. Use
    [`Patch.select`](`dascore.Patch.select`) for DASCore's tuple range notation,
    relative selections, and filtering that preserves dimensions and source order.

    Parameters
    ----------
    indexers
        Mapping of dimension names to scalar labels, slices, or 1D label arrays.
        Supply this or keyword indexers. Quantities convert to coordinate units.
    method
        ``None`` requires exact matches; ``"nearest"`` selects the nearest label.
    tolerance
        Maximum distance allowed for nearest matches, in coordinate units or as
        a quantity. Datetime tolerances are durations. Not supported with slices.
    drop
        Drop coordinates made scalar by indexing instead of retaining them.
        Scalar indexers remove their dimension regardless of this option.
    **kwargs
        Dimension indexers supplied as keywords.

    Notes
    -----
    Label slices include both endpoints. Arrays preserve order and repetitions,
    selecting every combination when several dimensions have array indexers.
    Missing scalar or array labels raise KeyError. Slices follow coordinate
    order, including descending coordinates, and require sliceable labels.
    Datetime strings follow pandas' partial-date selection rules.
    Evenly sampled coordinates stay compact during lookup and slicing.
    Floating slice labels can differ from xarray by rounding relative to the
    original range.
    Array/nearest lookup on long grids below floating-point resolution raises
    instead of expanding the grid to check whether every label is unique.

    This supports dimension coordinates, not arbitrary auxiliary coordinates or
    MultiIndexes. Labelled xarray indexers and multidimensional indexer arrays
    are not supported. Dimensions without labels use positional indexing.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> window = patch.sel(distance=slice(10, 20))
    >>> assert window.shape[0] == 11
    >>> channel = patch.sel(distance=10.2, method="nearest", tolerance=0.5)
    >>> assert channel.get_array("distance") == 10

    See Also
    --------
    [Patch.isel](`dascore.Patch.isel`) : Xarray-compatible positional indexing.
    [Patch.select](`dascore.Patch.select`) : Range filtering and relative selections.
    """

    # The indexers may also arrive as keywords, one per dimension.
    model_config = ConfigDict(extra="allow", frozen=True)

    # Typed loosely so each is refused where it is used, in that code's own
    # words: `get_indexers` for the indexers, `get_metadata` for the method.
    indexers: Any = None
    method: Any = None
    tolerance: Any = None
    drop: bool = False

    history = None

    def get_metadata(self, meta):
        """Resolve each label to a position, then index by position."""
        if self.method not in (None, "nearest"):
            raise ValueError("method must be None or 'nearest'.")
        requested = get_indexers(self.indexers, self.model_extra or {}, meta.dims)
        positions = {
            dim: label_indexer(
                meta.coords.get_coord(dim), value, self.method, self.tolerance
            )
            for dim, value in requested.items()
        }
        return Isel(indexers=positions, drop=self.drop).get_metadata(meta)

    def kernel(self, data, *, indexer=None):
        """Return the samples the labels resolved to."""
        return _selected(data, indexer)


sel = Sel.patch_function


class Unselect(_Query):
    """
    Return the patch outside a selection.

    The complement of [`Patch.select`](`dascore.Patch.select`): it takes
    the same selectors and removes the samples that selection would have
    kept. With one coordinate named that is exactly the complement; with
    several, each is complemented on its own — see the note below.

    Parameters
    ----------
    copy
        Copy the result so it does not retain the original data array.
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

    def get_metadata(self, meta):
        """Return what selecting the complement of each range leaves."""
        queries = self.model_extra or {}
        _check_coord_names(meta, queries)
        keep: dict[str, np.ndarray] = {}
        for name, value in queries.items():
            coord = meta.coords.coord_map[name]
            dims = meta.coords.dim_map[name]
            if len(dims) != 1:
                msg = (
                    f"Coordinate {name!r} spans {list(dims)}, so removing a range "
                    "of it does not name samples of one dimension to drop."
                )
                raise PatchCoordinateError(msg)
            # Asking select itself which samples it would keep is what stops
            # the two from drifting: one selector cannot come to mean
            # different things in select and its complement.
            _, indexer = coord.select(
                value, relative=self.relative, samples=self.samples
            )
            selected = np.zeros(len(coord), dtype=bool)
            selected[indexer] = True
            keep[dims[0]] = keep.get(dims[0], True) & ~selected
        # Kept as sample numbers along each dimension rather than as a mask
        # per coordinate: coordinates sharing a dimension are applied in
        # separate passes, so the second mask would meet an already trimmed
        # axis, and a dimension carrying no values of its own takes samples
        # where it would refuse an array.
        # Typed as the selectors they are: each key is a coordinate name, so
        # the values never land on select's own bool fields.
        trims: dict[str, Any] = {
            dim: np.flatnonzero(mask) for dim, mask in keep.items()
        }
        return Select(**trims, samples=True).get_metadata(meta)


unselect = Unselect.patch_function


class Order(_Query):
    """
    Re-order the patch contents based on coordinate values or indices.

    Parameters
    ----------
    copy
        Copy the result so it does not retain the original data array.
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

    def get_metadata(self, meta):
        """Return the re-ordered coordinates and the indexers behind them."""
        coords, indexer = _dimension_indexers(
            meta.coords,
            self.model_extra or {},
            operation="order",
            relative=self.relative,
            samples=self.samples,
        )
        return meta.new(coords=coords), {"indexer": indexer}


order = Order.patch_function


class Transpose(PatchProcessor):
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

    # Typed loosely because `...` is a legal element: `transpose(...,
    # "distance")` means "distance last, the rest as they were".
    dims: tuple[Any, ...] = ()

    history = None
    _var_positional = "dims"

    def get_metadata(self, meta):
        """Return the coordinates in their new order, and the permutation."""
        old_dims = meta.dims
        named = [x for x in self.dims if x is not ...]
        if invalid := set(named) - set(old_dims):
            msg = (
                f"Dimension(s) {sorted(invalid)} not found in Patch "
                f"dimensions: {sorted(old_dims)}"
            )
            raise ParameterError(msg)
        coords = meta.coords.transpose(*self.dims)
        if coords is meta.coords:
            return meta, {}
        axes = tuple(old_dims.index(x) for x in coords.dims)
        return meta.new(coords=coords), {"axes": axes}

    def kernel(self, data, *, axes=()):
        """Return the data with its axes permuted, or as it was."""
        if axes == tuple(range(len(axes))):
            return data
        return array_namespace(data).permute_dims(data, axes)


transpose = Transpose.patch_function


class AppendDims(PatchProcessor):
    """
    Insert dimensions at the end of the patch.

    Parameters
    ----------
    *empty_dims
        Used to pass the name of empty dimensions.
    **kwargs
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

    # The named dimensions arrive under their own names.
    model_config = ConfigDict(extra="allow", frozen=True)

    empty_dims: tuple[Any, ...] = ()

    history = None
    _var_positional = "empty_dims"

    def get_metadata(self, meta):
        """Return the longer coordinates, and the shape they describe."""
        if bad := [x for x in self.empty_dims if not isinstance(x, str)]:
            # Reached by `append_dims(empty_dims=[1, 2])`, which named a
            # dimension `empty_dims` before this was a processor: a keyword
            # never fills a `*args` parameter, but the generated function
            # hands the extras back to this class as plain keywords, where
            # the field of that name takes them. Said plainly here rather
            # than left to `TypeError: keywords must be strings`.
            msg = f"append_dims takes dimension names; got {bad}."
            raise ParameterError(msg)
        dim_dict = {x: 1 for x in self.empty_dims}
        dim_dict.update(self.model_extra or {})
        # Remove duplicate dims and convert non ints to arrays.
        kwargs = {
            i: (i, np.atleast_1d(v) if not isinstance(v, int) else v)
            for i, v in dim_dict.items()
            if i not in meta.dims
        }
        # Nothing to do.
        if not kwargs:
            return meta, {}
        ndim = len(meta.dims)
        axes = tuple(range(ndim, ndim + len(kwargs)))
        lengths = tuple(
            cdata if isinstance(cdata, int) else len(cdata)
            for _, cdata in kwargs.values()
        )
        coords = meta.coords.update(**kwargs)
        shape = tuple(meta.coords.shape) + lengths
        return meta.new(coords=coords), {"axes": axes, "shape": shape}

    def kernel(self, data, *, axes=(), shape=()):
        """Return the data stretched over the dimensions which were added."""
        if not axes:
            return data
        return np.broadcast_to(np.expand_dims(data, axes), shape)


append_dims = AppendDims.patch_function


class Squeeze(PatchProcessor):
    """
    Return a new object with len one dimensions flattened.

    Parameters
    ----------
    dim
        Selects a subset of the length one dimensions. If a dimension
        is selected with length greater than one, an error is raised.
        If None, all length one dimensions are squeezed.

    Raises
    ------
    CoordError
        If a selected dimension does not exist or has more than one sample.
    ParameterError
        If squeezing would remove every dimension from the patch.

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

    dim: Any = None

    def get_metadata(self, meta):
        """Return the shorter coordinates, and the axes which go with them."""
        coords = meta.coords.squeeze(self.dim)
        # Nothing to squeeze; the coord manager returned itself.
        if coords is meta.coords:
            return meta, {}
        if not coords.dims:
            msg = "Cannot squeeze all dimensions; at least one dimension must remain."
            raise ParameterError(msg)
        if self.dim is None:
            axes = tuple(i for i, x in enumerate(meta.coords.shape) if x == 1)
        else:
            axes = tuple(meta.coords.get_axis(x) for x in iterate(self.dim))
        return meta.new(coords=coords), {"axes": axes}

    def kernel(self, data, *, axes=()):
        """Return the data without the axes which were squeezed out."""
        if not axes:
            return data
        return array_namespace(data).squeeze(data, axis=axes)


squeeze = Squeeze.patch_function


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


def _fill_scalar(value, dtype) -> np.ndarray:
    """The fill value as the data's dtype, raising if the cast changes it."""
    ref = np.asarray(value)
    try:
        with np.errstate(invalid="ignore", over="ignore"):
            cast = ref.astype(dtype)
        same = bool(pd.isnull(ref) and pd.isnull(cast)) or bool(cast == ref)
        if not same and cast.dtype.kind in "fc":
            # a float may round the value, but never overflow it to inf
            tol = np.finfo(cast.dtype).resolution
            same = bool(np.isfinite(cast) and np.isclose(cast, ref, rtol=tol, atol=0))
    except (TypeError, ValueError):
        same = False
    if not same:
        msg = (
            f"Cannot fill data of dtype {np.dtype(dtype)} with {value!r}. Pass "
            "a value of that dtype, or cast the data first (eg to float for NaN)."
        )
        raise ParameterError(msg)
    return cast


def _place_blocks(data, axis: int, length: int, blocks, fill) -> np.ndarray:
    """Copy each ``(source start, source stop, target start)`` block; fill the rest."""
    shape = list(data.shape)
    shape[axis] = length
    out = np.empty(shape, dtype=data.dtype)
    index = partial(broadcast_for_index, len(shape), axis)
    end = 0
    for start, stop, target in blocks:
        out[index(slice(end, target))] = fill
        end = target + stop - start
        out[index(slice(target, end))] = data[index(slice(start, stop))]
    out[index(slice(end, None))] = fill
    return out


@patch_function()
def fill_gaps(
    patch: PatchType,
    *args,
    value: Any = np.nan,
    samples: bool = False,
    **kwargs,
) -> PatchType:
    """
    Fill the holes along a dimension with a constant value.

    Places runs of samples on one evenly sampled grid and writes `value`
    where no sample sits, so a segmented coordinate (for example from
    [`Spool.chunk`](`dascore.Spool.chunk`) with `snap_coords=False` across a
    gap) becomes a plain range, unless a limit leaves wider holes as seams.

    Parameters
    ----------
    patch
        The patch to fill.
    *args
        The dimension to fill, eg `patch.fill_gaps("time")`.
    value
        The value written at filled positions. It must fit the data's
        dtype: NaN cannot fill integer data, so pass an integer or cast
        the data to float first.
    samples
        If True, the limit given with the dimension counts missing samples.
    **kwargs
        The dimension and the widest hole to fill, eg `time=10` fills holes
        missing up to ten seconds of samples and leaves wider ones as seams.
        A hole's width is its missing samples times the step, one step less
        than the jump between the labels either side. Give the limit in the
        coordinate's units (seconds for time), or as a quantity or
        timedelta; None fills every hole.

    Notes
    -----
    The coordinate needs a declared step: a segmented coordinate whose
    runs share one step (different steps raise; resample first), or an
    array declared with a step. A sample or run off the grid moves to the
    nearest position, by at most half a step.

    Non-dimensional coordinates along the dimension are dropped with a
    warning, since their values at the filled positions are unknown.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> from dascore.core.coords import concat_coords, get_coord
    >>>
    >>> # A patch whose distance coordinate misses three samples.
    >>> dist = concat_coords(
    ...     get_coord(start=0.0, stop=5.0, step=1.0),
    ...     get_coord(start=8.0, stop=10.0, step=1.0),
    ... )
    >>> patch = dc.Patch(
    ...     data=np.ones((len(dist), 3)),
    ...     coords={"distance": dist, "time": dc.to_datetime64(np.arange(3))},
    ...     dims=("distance", "time"),
    ... )
    >>> filled = patch.fill_gaps("distance")
    >>> assert filled.shape == (10, 3)
    >>> assert np.isnan(filled.data[5:8]).all()
    >>>
    >>> # Fill only holes of at most two missing samples: this one stays.
    >>> assert patch.fill_gaps(distance=2, samples=True).shape == patch.shape
    >>>
    >>> # Fill with zeros instead of NaN.
    >>> assert (patch.fill_gaps("distance", value=0).data[5:8] == 0).all()
    """
    dim, axis, limit = get_dim_axis_value(patch, args=args, kwargs=kwargs)[0]
    layout = _fill_layout(patch.get_coord(dim), limit, samples=samples)
    if layout is None:
        return patch
    coord, blocks = layout
    data = to_numpy(patch.data)
    fill = _fill_scalar(value, data.dtype)
    data = _place_blocks(data, axis, len(coord), blocks, fill)
    coords = drop_associated_coords(patch.coords, dim, "Filling gaps along")
    return patch.new(data=data, coords=coords._update_grid(dim, **{dim: coord}))
