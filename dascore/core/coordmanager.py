"""
Module for managing coordinates.

The coordmanager is a simple class for tracking multidimensional coordinates
and labels of ndarrays. Various methods exist for trimming, sorting, and
filtering the managed arrays based on coordinates.

# Initializing CoordinateManagers

Coordinate managers are initialized using the
[get_coords](`dascore.core.coordmanager.get_coord_manager`) function. They take
a combination of coordinate dictionaries, dimensions, and attributes. They can
also be extracted from example patches.

```{python}
import dascore as dc
from rich import print

patch = dc.get_example_patch()
data = patch.data
cm = patch.coords
print(cm)
```

```{python}
import numpy as np
# Get array of coordinate values
time = cm['time']

# Filter data array
# this gets a new coord manager and a view of the trimmed data.
_sorted_time = np.sort(time)
t1 = _sorted_time[5]
t2 = _sorted_time[-5]
new_cm, new_data = cm.select(time=(t1, t2), array=data)
print(f"Old data shape: {data.shape}")
print(f"New data shape: {new_data.shape}")
print(new_cm)
```
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence, Sized
from functools import reduce
from operator import and_, or_
from typing import Annotated, Any, TypeVar

import numpy as np
from pydantic import field_validator, model_validator
from rich.text import Text
from typing_extensions import Self

import dascore as dc
from dascore.constants import dascore_styles
from dascore.core.coords import BaseCoord, CoordSummary, get_coord
from dascore.exceptions import (
    CoordDataError,
    CoordError,
    CoordMergeError,
    CoordSortError,
    ParameterError,
)
from dascore.utils.display import get_nice_text
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    _matches_prefix_suffix,
    all_close,
    cached_method,
    iterate,
    separate_coord_info,
)
from dascore.utils.models import (
    ArrayLike,
    DascoreBaseModel,
    frozen_dict_serializer,
    frozen_dict_validator,
)

MaybeArray = TypeVar("MaybeArray", ArrayLike, np.ndarray, None)


def _validate_select_coords(coord, coord_name):
    """Ensure multi-dims are not used."""
    if not len(coord.shape) == 1:
        msg = (
            "Only 1 dimensional coordinates can be used for selection "
            f"{coord_name} has {len(coord.shape)} dimensions."
        )
        raise CoordError(msg)


def _indirect_coord_updates(cm, dim_name, coord_name, reduction, new_coords):
    """
    Applies trim to coordinates.

    Assumes other associated coordinates are trimmed.
    """
    other_coords = set(cm.dim_to_coord_map[dim_name]) - {coord_name}
    # perform indirect updates.
    for icoord in other_coords:
        dims, coord = new_coords[icoord]
        axis = cm.dim_map[icoord].index(dim_name)
        new = coord.index(reduction, axis=axis)
        new_coords[icoord] = (dims, new)


def _to_slice(limits):
    """Convert slice or two len tuple to slice."""
    if isinstance(limits, slice):
        return limits
    # ints should be interpreted as Slice(int, int+1) to not collapse dim.
    if isinstance(limits, int):
        if limits == -1:  # -1 case needs open interval to work
            return slice(-1, None)
        return slice(limits, limits + 1)
    if limits is ... or limits is None:
        return slice(None, None)
    assert isinstance(limits, Sized) and len(limits) == 2
    val1, val2 = limits
    start = None if val1 is ... or val1 is None else val1
    stop = None if val2 is ... or val2 is None else val2
    return slice(start, stop)


def _get_indexers_and_new_coords_dict(cm, kwargs, samples=False, relative=False):
    """Function to get reductions for each dimension."""
    dim_reductions = {x: slice(None, None) for x in cm.dims}
    dimap = cm.dim_map
    new_coords = dict(cm._get_dim_array_dict(keep_coord=True))
    for coord_name, limits in kwargs.items():
        # this is not a selectable coord, just skip.
        if coord_name not in cm.coord_map or not len(cm.dim_map[coord_name]):
            continue
        coord = cm.coord_map[coord_name]
        _validate_select_coords(coord, coord_name)
        dim_name = dimap[coord_name][0]
        # different logic if we are using indices or values
        if not samples:
            new_coord, reductions = coord.select(limits, relative=relative)
        else:
            reductions = _to_slice(limits)
            new_coord = coord[reductions]
        # this handles the case of out-of-bound selections.
        # These should be converted to degenerate coords.
        dim_reductions[dim_name] = reductions
        new_coords[coord_name] = (dimap[coord_name], new_coord)
        # update other coords affected by change.
        _indirect_coord_updates(cm, dim_name, coord_name, reductions, new_coords)
    indexers = tuple(dim_reductions[x] for x in cm.dims)
    return new_coords, indexers


class CoordManager(DascoreBaseModel):
    """
    Class for managing coordinates.

    Attributes
    ----------
    dims
        A tuple of dimension names.
    coord_map
        A mapping of {coord_name: Coord}
    dim_map
        A mapping of {coord_name: (dimensions, ...)}
    """

    dims: tuple[str, ...]
    coord_map: Annotated[
        FrozenDict[str, BaseCoord],
        frozen_dict_validator,
        frozen_dict_serializer,
    ]
    dim_map: Annotated[
        FrozenDict[str, tuple[str, ...]],
        frozen_dict_validator,
        frozen_dict_serializer,
    ]

    @model_validator(mode="before")
    @classmethod
    def _validate_coords(cls, values):
        """Validate the coordinates and dimensions."""
        coord_map, dim_map = values["coord_map"], values["dim_map"]
        dims = values["dims"]
        try:
            dim_shapes = {dim: coord_map[dim].shape for dim in dims}
        except KeyError:
            missing = set(dims) - set(coord_map)
            msg = f"All dimensions must have coordinates, {missing} are missing."
            raise CoordError(msg)
        # ensure non-dimensional coordinates have the same length as
        # corresponding coordinates.
        for name, coord_dims in dim_map.items():
            expected_shape = tuple(dim_shapes[x][0] for x in coord_dims)
            shape = coord_map[name].shape
            if tuple(expected_shape) == shape or not expected_shape:
                continue
            msg = (
                f"coordinate: {name} has a shape of {shape} which does not "
                f"match the dimension(s) of {coord_dims} which have a shape "
                f"of {expected_shape}"
            )
            raise CoordError(msg)
        # Ensure dimensional coordinates are associated as such
        for dim in dims:
            assert dim_map[dim] == (dim,), f"Incorrect dim association on {dim}"

        return values

    @field_validator("coord_map", "dim_map")
    @classmethod
    def _convert_to_frozen_dicts(cls, v):
        """Ensure mapping fields are immutable."""
        return FrozenDict(v)

    def __getitem__(self, item) -> np.ndarray:
        # in order to not break backward compatibility, we need to return
        # the array. Start the deprecation cycle to be consistent.
        msg = (
            "Currently coords[coord_name] returns a numpy array, but in a "
            "future dascore version it will return a BaseCoord instance. "
            "Use patch.coords.get_array(coord_name) instead."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        return self._get_coord(item).values

    def __getattr__(self, item) -> BaseCoord:
        try:
            return super().__getattr__(item)
        except AttributeError:
            # unlike get item, get attr returns the base coordinate.
            return self._get_coord(item)

    def __iter__(self):
        for key in self.coord_map:
            yield (key, self.get_array(key))

    def __contains__(self, key):
        return key in self.coord_map

    def update_coords(self, **kwargs) -> Self:
        """
        Update the coordinates, return a new Coordinate Manager.

        Input values can be of the same form as initialization.
        To drop coordinates, simply pass {coord_name: None}
        """

        def _get_dim_change_drop(coord_map, dim_map):
            """
            Determine which coords must be dropped because their corresponding
            dimension size changed.
            """
            out = []
            for i, _v in coord_map.items():
                if i not in self.dims:
                    continue
                assert len(dim_map[i]) == 1, "only dealing with 1D dimensions"
                if len(coord_map[i]) == len(self.coord_map[i]):
                    continue
                for coord_name, dims in self.dim_map.items():
                    if i in dims:
                        out.append(coord_name)
            return out

        def _divide_kwargs(kwargs):
            """Divide kwargs into coords to update, drop, add, etc."""
            prefix = self.dims
            suffix = ("step", "min", "max")
            coord_updates = {
                x: kwargs.pop(x)
                for x in set(kwargs)
                if _matches_prefix_suffix(x, prefixes=prefix, suffixes=suffix)
            }
            coords_to_drop = [i for i, v in kwargs.items() if v is None]
            # convert input to coord_map/dim_map
            coords_to_add = {i: v for i, v in kwargs.items() if v is not None}
            return coord_updates, coords_to_drop, coords_to_add

        coord_updates, coord_to_drop, coord_to_add = _divide_kwargs(kwargs)
        # get coords to drop from selecting None
        coord_map, dim_map = _get_coord_dim_map(coord_to_add, self.dims)
        # find coords to drop because their dimension changed.
        indirect_coord_drops = _get_dim_change_drop(coord_map, dim_map)
        # drop coords then call get_coords to handle adding new ones.
        coords, _ = self.drop_coords(*(coord_to_drop + indirect_coord_drops))
        out = coords._get_dim_array_dict()
        out.update({i: v for i, v in kwargs.items() if i not in coord_to_drop})
        # update based on keywords
        for item, value in coord_updates.items():
            coord_name, attr = item.split("_")
            new = list(out[coord_name])
            coord = get_coord(values=new[1])
            new[1] = coord.update(**{attr: value})
            out[coord_name] = tuple(new)
        dims = tuple(x for x in self.dims if x not in coord_to_drop)
        return get_coord_manager(out, dims=dims)

    def update_from_attrs(
        self, attrs: Mapping | dc.PatchAttrs
    ) -> tuple[Self, dc.PatchAttrs]:
        """
        Update coordinates from attrs.

        This will also return a PatchAttrs which conforms to coords.

        Parameters
        ----------
        attrs
            The attribute source, either PatchAttrs instance or mapping.
        """
        coord_info, attr_info = separate_coord_info(attrs, dims=self.dims)
        out = dict(self.coord_map)
        for name in set(coord_info) & set(out):
            maybe_updates = coord_info[name]
            coord = self.coord_map[name]
            # convert values to dict to determine which should be updated.
            model_contents = coord.to_summary().model_dump(exclude_defaults=True)
            # see what has changed
            diff = {
                i: v for i, v in maybe_updates.items() if v != model_contents.get(i)
            }
            out[name] = coord.update(**diff)
        coords = self.new(coord_map=out)
        # anything not used in coord_info should be put back.
        # for example, data_units might get put in its own coord called data.
        for unused_dim in set(coord_info) - set(coords.dims):
            for key, val in coord_info[unused_dim].items():
                attr_info[f"{unused_dim}_{key}"] = val
        attr_info["coords"] = coords.to_summary_dict()
        attr_info["dims"] = coords.dims
        attrs = dc.PatchAttrs.from_dict(attr_info)
        return coords, attrs

    def sort(
        self, *coords, array: MaybeArray = None, reverse: bool = False
    ) -> tuple[Self, MaybeArray]:
        """
        Sort coordinates.

        Parameters
        ----------
        *coords
            Specify which dimensions to sort. If None, sort along dimensional
            coordinates.
        array
            The array to sort.
        reverse
            If true, sort in descending order, else ascending.

        Raises
        ------
        CoordSortError if sorting is not possible.
        """

        def _validate_coords(coords):
            """Coordinates should be one dimensional."""
            dims = [self.dim_map[x] for x in coords]
            if any(len(x) > 1 for x in dims):
                bad_coord = [y for x, y in zip(dims, coords, strict=True) if len(y) > 1]
                msg = (
                    f"cannot simultaneously sort coordinate(s) {bad_coord}"
                    f" because they are associated with more than one dimension."
                )
                raise CoordSortError(msg)
            if len(coords) == 1:  # no need to check conflict if one dim
                return
            flat_dims = [x for y in dims for x in y]
            if len(flat_dims) != len(set(flat_dims)):
                msg = f"cannot merge {coords} because they share a dimension"
                raise CoordSortError(msg)

        def _get_dimensional_sorts(coords2sort):
            """Create the indexer for each coord."""
            new_coords = dict(self.coord_map)
            indexes = []
            for name, coord in coords2sort.items():
                dim = self.dim_map[name][0]
                new_coord, indexer = coord.sort(reverse=reverse)
                new_coords[name] = new_coord
                indexes.append(self._get_indexer(self.dims.index(dim), indexer))
                # also sort related coords.
                _sort_related(name, dim, indexer, new_coords)
            return new_coords, tuple(indexes)

        def _sort_related(name, dim, indexer, new_coords):
            """Find and apply sorting to related coords."""
            related_coords = set(self.dim_to_coord_map[dim]) - {name}
            for r_coord_name in related_coords:
                coord = new_coords[r_coord_name]
                ind = self.dim_map[r_coord_name].index(dim)
                slicer = [slice(None, None) for _ in coord.shape]
                slicer[ind] = indexer
                new_coords[r_coord_name] = coord[tuple(slicer)]

        # get coords that need sorting
        attr = "reverse_sorted" if reverse else "sorted"
        coords = coords if coords else self.dims
        _validate_coords(coords)
        cmap = self.coord_map
        assert set(coords).issubset(set(cmap))
        coords2sort = {x: cmap[x] for x in coords if not getattr(cmap[x], attr)}
        new_coords, indexers = _get_dimensional_sorts(coords2sort)
        if array is not None:
            for index in indexers:
                array = array[index]
        out = self.new(coord_map=new_coords)
        assert out.shape == self.shape
        return out, array

    def snap(
        self, *coords, array: MaybeArray = None, reverse: bool = False
    ) -> tuple[Self, MaybeArray]:
        """
        Force the specified coordinates to be monotonic and evenly sampled.

        Coordinates are first sorted, then snapped.

        Parameters
        ----------
        *coords
            Specify which dimensions to sort. If None, sort along dimensional
            coordinates.
        array
            The array to sort/snap.
        reverse
            If true, sort in descending order, else ascending.
        """
        coords = self.dims if len(coords) == 0 else coords
        cm, array = self.sort(*coords, array=array, reverse=reverse)
        # now the arrays are sorted it should be correct to snap dimensions.
        cmap = dict(cm.coord_map)
        for coord_name in coords:
            cmap[coord_name] = cmap[coord_name].snap()
        out = cm.new(coord_map=cmap)
        assert out.shape == self.shape
        return out, array

    def new(self, dims=None, coord_map=None, dim_map=None) -> Self:
        """
        Return a new coordmanager with specified attributes replaced.

        Parameters
        ----------
        dims
            A tuple of dimension strings.
        coord_map
            A mapping of {name: coord}
        dim_map
            A mapping of {coord_name:
        """
        out = self.__class__(
            dims=dims if dims is not None else self.dims,
            coord_map=coord_map if coord_map is not None else self.coord_map,
            dim_map=dim_map if dim_map is not None else self.dim_map,
        )
        return out

    def drop_coords(
        self,
        *coords: str | Sequence[str],
        array: MaybeArray = None,
    ) -> tuple[Self, MaybeArray]:
        """
        Drop one or more coordinates.

        If the coordinate is a dimension, also drop other coords that depend
        on that dimension.

        Parameters
        ----------
        *coords
            The name of the coordinate or dimension.
        """
        dim_drop_list = []
        coords_to_drop = {x for x in iterate(coords)}
        # If there are either no coords to drop or this cm doesn't have them.
        if not coords_to_drop or not (set(self.coord_map) & coords_to_drop):
            return self, array
        # figure out which coords to drop are dimensional coordinates.
        for name, dims in self.dim_map.items():
            if coords_to_drop & set(dims):
                dim_drop_list.append(name)
        dims_to_drop = set(dim_drop_list)
        # combine dims and plain coordinates that should be dropped.
        total_drops = dims_to_drop | coords_to_drop
        coord_map = {i: v for i, v in self.coord_map.items() if i not in total_drops}
        dim_map = {i: v for i, v in self.dim_map.items() if i not in total_drops}
        dims = tuple(x for x in self.dims if x not in dims_to_drop)
        index = tuple(
            slice(None, None) if x not in coords_to_drop else slice(0, 0)
            for x in self.dims
        )
        new = self.__class__(coord_map=coord_map, dim_map=dim_map, dims=dims)
        return new, self._get_new_data(index, array)

    def disassociate_coord(self, *coord: str) -> Self:
        """
        Disassociate some coordinates from dimensions.

        These coordinates will no longer be associated with a dimension in
        the coord manager but can still be retrieved.

        Parameters
        ----------
        coord
            The coordinate name(s) to disassociated from their dimensions.
        """
        new = {x: (None, self.coord_map[x]) for x in coord}
        return self.drop_coords(*coord)[0].update_coords(**new)

    def drop_disassociated_coords(self) -> Self:
        """Drop all coordinates not associated with a dimension."""
        cmap = self.coord_map
        dim_map = self.dim_map
        no_dim_coords = [x for x in cmap if dim_map[x] == ()]
        return self.drop_coords(*no_dim_coords)

    def set_dims(self, **kwargs: str) -> Self:
        """
        Set dimension to non-dimensional coordinate.

        Parameters
        ----------
        kwargs
            A mapping indicating old_dim: new_dim where new_dim refers to
            the name of a non-dimensional coordinate.
        """
        dims = list(self.dims)
        coord_map = dict(self.coord_map)
        dim_map = dict(self.dim_map)
        for old_dim, new_dim in kwargs.items():
            if new_dim not in coord_map or old_dim not in dims:
                msg = (
                    f"{old_dim} is not a dimension or {new_dim} is not a "
                    f"coordinate."
                )
                raise CoordError(msg)
            # ensure coords have the same shape
            old_coord, new_coord = coord_map[old_dim], coord_map[new_dim]
            if not old_coord.shape == new_coord.shape or len(new_coord.shape) != 1:
                msg = (
                    f"New coordinate {new_coord} with a shape of {new_coord.shape} "
                    f"does not match the shape of {old_coord} ({old_coord.shape})"
                )
                raise CoordError(msg)

            # swap out old dims
            dims[dims.index(old_dim)] = new_dim
        # now loop over dim_maps and swap out old dims with new
        old_to_new = {i: v for i, v in zip(self.dims, dims, strict=True)}
        for coord_name, coord_dims in dim_map.items():
            dim_map[coord_name] = tuple(old_to_new[x] for x in coord_dims)
        return self.__class__(dims=dims, coord_map=coord_map, dim_map=dim_map)

    def select(
        self, array: MaybeArray = None, relative=False, samples=False, **kwargs
    ) -> tuple[Self, MaybeArray]:
        """
        Perform value-based selection on coordinates.

        Parameters
        ----------
        array
            An array to which the selection will be applied.
        relative
            If True, coordinate updates are relative.
        samples
            If True, the query meaning is in samples.
        **kwargs
            Used to specify select arguments. Can be of the form
            {coord_name: (lower_limit, upper_limit)}.
        """
        new_coords, indexers = _get_indexers_and_new_coords_dict(
            self, kwargs, samples=samples, relative=relative
        )
        new_cm = self.update_coords(**new_coords)
        return new_cm, self._get_new_data(indexers, array)

    def _get_new_data(self, indexer, array: MaybeArray) -> MaybeArray:
        """Get new data array after applying some trimming."""
        if array is None:  # no array passed, just return.
            return array
        return array[indexer]

    def __rich__(self) -> str:
        """Rich formatting for the coordinate manager."""
        dc_blue = dascore_styles["dc_blue"]
        header_text = Text("➤ ") + Text("Coordinates", style=dc_blue) + Text(" (")
        lens = {x: self.coord_map[x].shape[0] for x in self.dims}
        dim_texts = Text(", ").join(
            [
                Text(x, style="bold")
                + Text(": ")
                + Text(f"{lens[x]}", style="underline")
                for x in self.dims
            ]
        )
        out = [header_text, dim_texts, ")"]
        # sort coords by dims, coords
        non_dim_coords = sorted(set(self.coord_map) - set(self.dims))
        names = list(self.dims) + non_dim_coords
        for name in names:
            coord = self.coord_map[name]
            coord_dims = self.dim_map[name]
            if name in self.dims:
                base = Text.assemble("\n    * ", Text(name, style="bold"), ": ")
            else:
                base = Text(f"\n    {name} {coord_dims}: ")
            text = Text.assemble(base, coord.__rich__())
            out.append(text)
        return Text.assemble(*out)

    def __str__(self):
        return str(self.__rich__())

    def equals(self, other) -> bool:
        """Return True if other coordinates are approx equal."""
        if not isinstance(other, self.__class__):
            return False
        if not set(self.coord_map) == set(other.coord_map):
            return False
        coord_1, coord_2 = self.coord_map, other.coord_map
        for name, _coord in self.coord_map.items():
            if not all_close(coord_1[name].values, coord_2[name].values):
                return False
        return True

    __eq__ = equals

    @property
    def shape(self):
        """Return the shape of the dimensions."""
        out = tuple(len(self.coord_map[x]) for x in self.dims)
        # empty arrays return (0,) as their shape, so we must do the same.
        if not out:
            return (0,)
        return out

    @property
    def size(self):
        """Return the shape of the dimensions."""
        return np.prod(self.shape)

    def validate_data(self, data):
        """Ensure data conforms to coordinates."""
        data = np.array([]) if data is None else data
        if self.shape != data.shape:
            msg = (
                f"Data array has a shape of {data.shape} which doesnt match "
                f"the coordinate manager shape of {self.shape}."
            )
            raise CoordDataError(msg)
        return data

    def _get_dim_array_dict(self, keep_coord=False):
        """
        Get the coord map in the form:
        {coord_name = ((dims,), array)}.

        if keep_coord, just keep the coordinate as second arg.
        """
        out = {}
        for name, coord in self.coord_map.items():
            dims = self.dim_map[name]
            out[name] = (dims, coord if keep_coord else coord.data)
        return out

    def set_units(self, **kwargs):
        """Set the units of the coordinate manager."""
        new_coords = dict(self.coord_map)
        for name, units in kwargs.items():
            new_coords[name] = new_coords[name].set_units(units)
        return self.new(coord_map=new_coords)

    def convert_units(self, **kwargs):
        """
        Convert units in coords according to kwargs. Will raise if incompatible
        Coordinates are specified.
        """
        new_coords = dict(self.coord_map)
        for name, units in kwargs.items():
            new_coords[name] = new_coords[name].convert_units(units)
        return self.new(coord_map=new_coords)

    def simplify_units(self):
        """Simplify all units in the coordinates."""
        new_coords = dict(self.coord_map)
        for name, coord in new_coords.items():
            new_coords[name] = coord.simplify_units()
        return self.new(coord_map=new_coords)

    def transpose(self, *dims: str | type(Ellipsis)) -> Self:
        """Transpose the coordinates."""

        def _get_transpose_dims(new, old):
            """Get a full tuple of dimensions for transpose, allowing for ..."""
            new_set = set(new)
            new_list, old_list = list(new), list(old)
            if len(new_set) != len(new):
                msg = "Transpose cannot use duplicate dimensions or ... more than once."
                raise ParameterError(msg)
            # iterate new list, pop out values from old list.
            for value in new_set - {...}:
                old_list.remove(value)
            # sub in remaining values for ...
            if ... in new_set:
                ind = new_list.index(...)
                new_list = new_list[:ind] + old_list + new_list[ind + 1 :]
            # ensure all values are accounted for
            if set(new_list) != set(old):
                msg = (
                    "You must specify all dimensions in a transpose operation "
                    f"or use ... You passed {new} but dimensions are {old}"
                )
                raise ParameterError(msg)
            return tuple(new_list)

        dims = _get_transpose_dims(new=dims or self.dims[::-1], old=self.dims)
        return self.new(dims=dims)

    def rename_coord(self, **kwargs) -> Self:
        """
        Rename the coordinates or dimensions.

        Parameters
        ----------
        **kwargs
            Used to rename coord {old_coord_name: new_coord_name}
        """

        def _adjust_dim_map_to_new_dim(dim_map, old_name, new_name):
            """Get the updated dim map."""
            out = dim_map
            # replace all mentions of
            for key, value in out.items():
                if old_name not in value:
                    continue
                new_value = tuple(x if x != old_name else new_name for x in value)
                out[key] = new_value

            return out

        dims, coord_map = list(self.dims), dict(self.coord_map)
        dim_map = dict(self.dim_map)
        for old_name, new_name in kwargs.items():
            if old_name == new_name:
                continue
            if old_name in dims:  # only
                dims[dims.index(old_name)] = new_name
                dim_map = _adjust_dim_map_to_new_dim(dim_map, old_name, new_name)
            coord_map[new_name] = coord_map.pop(old_name)
            dim_map[new_name] = dim_map.pop(old_name)

        out = dict(dims=dims, coord_map=coord_map, dim_map=dim_map)
        return self.__class__(**out)

    def squeeze(self, dim: Sequence[str] | None = None) -> Self:
        """
        Squeeze length one dimensions.

        Parameters
        ----------
        dim
            The dimension name, or sequence of dimension names, to drop.
            If None, drop all length 1 dims

        Raises
        ------
        CoordError if the selected dimension has a length gt 1.
        """
        to_drop = []
        if dim is None:
            dim = [x for x in self.dims if len(self.coord_map[x]) <= 1]
        bad_dim = set(iterate(dim)) - set(self.dims)
        if bad_dim:
            msg = (
                f"The following dims cannot be dropped because they don't "
                f"exist in the coordinate manager {bad_dim}"
            )
            raise CoordError(msg)
        for name in iterate(dim):
            coord = self.coord_map[name]
            if len(coord) > 1:
                msg = f"cant squeeze dim {name} because it has non-zero length"
                raise CoordError(msg)
            to_drop.append(name)
        return self.drop_coords(*to_drop)[0]

    def decimate(self, **kwargs) -> tuple[Self, tuple[slice, ...]]:
        """
        Evenly subsample along some dimension.

        Parameters
        ----------
        **kwargs
            Used to specify the dimension to decimate.

        Notes
        -----
        Removes any coordinate which depended on the decimated dimension.
        """
        assert len(kwargs) == 1
        (dim, value) = next(iter(kwargs.items()))
        assert dim in self.dims
        dim_slice = slice(None, None, int(value))
        new_array = self.coord_map[dim][dim_slice]
        new = self.update_coords(**{dim: new_array})
        slices = tuple(
            slice(None, None) if d != dim else slice(None, None, value)
            for d in new.dims
        )
        return new, slices

    @property
    @cached_method
    def coord_shapes(self) -> dict[str, tuple[int, ...]]:
        """Get a dict of {coord_name: shape}."""
        return {i: v.shape for i, v in self.coord_map.items()}

    @property
    @cached_method
    def dim_to_coord_map(self) -> FrozenDict[str, tuple[str, ...]]:
        """Get a dimension to coordinate map."""
        out = defaultdict(list)
        for coord, dims in self.dim_map.items():
            for dim in dims:
                out[dim].append(coord)
        return FrozenDict({i: tuple(v) for i, v in out.items()})

    def get_coord_tuple_map(self) -> dict[str, tuple[tuple[str, ...], BaseCoord]]:
        """
        Return a mapping of {coord_name: (dims, coord)}.

        This is helpful because it can be passed directly to
        get_coordinate_manager.
        """
        dims = self.dim_map
        coords = self.coord_map
        return {x: (dims[x], c) for x, c in coords.items()}

    def _get_coord_dims_tuple(self):
        """Return a tuple of ((coord, dims...,), ...)."""
        dim_map = self.dim_map
        return tuple((name, *dim_map[name]) for name in self.coord_map)

    def _get_indexer(self, ind: int | None = None, value=None):
        """
        Get an indexer for the appropriate data shape.

        This is useful for generating a tuple that can be used

        ind is a list of indices to substitute in values.
        """
        out = [slice(None, None) for _ in self.shape]
        out[ind] = value
        return tuple(out)

    def to_summary_dict(self) -> dict[str, CoordSummary]:
        """Convert the contents of the coordinate manager to a summary dict."""
        dim_map = self.dim_map
        out = {}
        for name, coord in self.coord_map.items():
            out[name] = coord.to_summary(dims=dim_map[name])
        return out

    def _get_coord(self, coord_name):
        """Get a coord. Raise an error if the coordinate is not in the coordmanager."""
        if coord_name not in self.coord_map:
            msg = (
                f"No coordinate named {coord_name} in coord manager. "
                f"Valid coordinates are {list(self.coord_map)}."
            )
            raise AttributeError(msg)
        return self.coord_map[coord_name]

    def min(self, coord_name):
        """Return the minimum value of a coordinate."""
        return self._get_coord(coord_name).min()

    def max(self, coord_name):
        """Return the maximum value of a coordinate."""
        return self._get_coord(coord_name).max()

    def step(self, coord_name):
        """Return the coordinate step."""
        return self._get_coord(coord_name).step

    def get_array(self, coord_name) -> np.ndarray:
        """Return the coordinate values as a numpy array."""
        return np.array(self._get_coord(coord_name))


def get_coord_manager(
    coords: Mapping[str, BaseCoord | np.ndarray] | CoordManager | None = None,
    dims: tuple[str, ...] | None = None,
    attrs: dc.PatchAttrs | dict[str, Any] | None = None,
) -> CoordManager:
    """
    Create a coordinate manager.

    Parameters
    ----------
    coords
        Information about coordinates. These can be a mapping of the
        form: {name, array}, {name: (dim_name, array)}, or
        {name: ((dim_names,) array). Can also be a
        [`CoordManager`](`dascore.core.CoordManager`).
    dims
        Tuple specify dimension names
    attrs
        Attributes which can be used to create coordinates.
        Cannot be used with coords argument.
        If you want to update [`CoordManager`](`dascore.core.CoordManager`)
        use [`update_from_attrs`](`dascore.core.CoordManager.update_from_attrs`).

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> from dascore.core.coordmanager import get_coord_manager
    >>> # initialize coordinates from dict of arrays
    >>> distance = np.arange(0, 100)
    >>> time = dc.to_datetime64(np.arange(0, 100_000, 1_000))
    >>> coords = {"distance": distance, "time": time}
    >>> dims = ("time", "distance")
    >>> cm = get_coord_manager(coords=coords, dims=dims)
    >>> # add non-dimension 1D coordinate. Note the coord dict must be a tuple
    >>> # with the first element specifying the dimensional coord the
    >>> # non-dimensional coord is attached to.
    >>> latitude = np.random.random(distance.shape)
    >>> coords['latitude'] = ('distance', latitude)
    >>> cm = get_coord_manager(coords=coords, dims=dims)
    >>> # Add two-D non-dimensional coord.
    >>> quality = np.random.random((len(distance), len(time)))
    >>> coords['quality'] = (("distance", "time"), quality)
    >>> cm = get_coord_manager(coords=coords, dims=dims)
    >>> # Get coordinate manager from typical patch attribute dict
    >>> attrs = dc.get_example_patch().attrs
    >>> cm = get_coord_manager(attrs=attrs)
    """
    if coords is not None and attrs is not None:
        msg = (
            "Cannot use both attrs and coords in get_coord_manager. "
            "Perhaps you want CoordManager.update_from_attrs?"
        )
        raise ParameterError(msg)
    # return coords if we already have a coord manager.
    if isinstance(coords, CoordManager):
        # maybe try to rename dims.
        if dims is not None and dims != coords.dims:
            kwargs = {i: v for i, v in zip(coords.dims, dims, strict=True)}
            coords = coords.rename_coord(**kwargs)
        return coords
    # this allows a simple dict without dims to be passed and dims pulled
    # from dict keys.
    if dims is None:
        if isinstance(coords, Mapping) and all(isinstance(x, str) for x in coords):
            dims = tuple(coords.keys())
        elif attrs is not None and "dims" in attrs or hasattr(attrs, "dims"):
            dims = tuple(attrs["dims"].split(","))
        else:
            dims = ()
    coords = {} if coords is None else coords
    coord_map, dim_map = _get_coord_dim_map(coords, dims)
    if attrs:
        coord_updates, _ = separate_coord_info(attrs, dims)
        updateable_coords = set(coord_updates) - set(coord_map)
        for name in updateable_coords:
            coord_map[name] = get_coord(**coord_updates[name])
            dim_map[name] = (name,)
    out = CoordManager(coord_map=coord_map, dim_map=dim_map, dims=dims)
    return out


def _get_coord_dim_map(coords, dims):
    """Get coord_map and dim_map from coord input."""

    def _get_coord(coord):
        """Get a coordinate from various inputs."""
        if hasattr(coord, "model_dump"):
            coord = coord.model_dump()
        if isinstance(coord, Mapping):  # input is a dict
            out = get_coord(**coord)
        else:
            out = get_coord(values=coord)
        return out

    def _coord_from_simple(name, coord):
        """Get coordinates from {coord_name: coord} where coord_name is dim name."""
        if name not in dims:
            msg = (
                "Coordinates that are not named the same as dimensions "
                "must be passed as a tuple of the form: "
                "(dimension, coord) "
            )
            raise CoordError(msg)
        out = _get_coord(coord)
        return out, (name,)

    def _maybe_coord_from_nested(coord):
        """
        Get coordinates from {coord_name: (dim_name, coord)} or
        {coord_name: ((dim_names...,), coord)}.
        """
        if not len(coord) == 2:
            msg = (
                "Second input for coords must be length two of the form:"
                " (dimension, coord) or ((dimensions,...), coord)"
            )
            raise CoordError(msg)
        dim_names = iterate(coord[0])
        # all dims must be in the input dims.
        if not (d1 := set(dim_names)).issubset(d2 := set(dims)):
            bad_dims = d2 - d1
            msg = (
                f"Coordinate specified invalid dimension(s) {bad_dims}."
                f" Valid dimensions are {dims}"
            )
            raise CoordError(msg)
        # pull out any relevant info from attrs.
        coord_out = _get_coord(coord[1])
        assert coord_out.shape == np.shape(coord[1])
        return coord_out, dim_names

    assert not isinstance(coords, CoordManager)
    # this is a dead code path, but may be needed later, leaving for now.
    # if isinstance(coords, CoordManager):
    #     coords_dump = coords.model_dump()
    #     return dict(coords_dump["coord_map"]), dict(coords_dump["dim_map"])

    c_map, d_map = {}, {}
    # iterate coords, get coordinate output.
    for name, coord in coords.items():
        if not isinstance(coord, tuple):
            c_map[name], d_map[name] = _coord_from_simple(name, coord)
        else:
            c_map[name], d_map[name] = _maybe_coord_from_nested(coord)
    return c_map, d_map


def merge_coord_managers(
    coord_managers: Sequence[CoordManager],
    dim: str,
    snap_tolerance: float | None = None,
) -> CoordManager:
    """
    Merger coordinate managers along a specified dimension.

    Parameters
    ----------
    coord_managers
        A sequence of coord_managers to merge.
    dim
        The dimension along which to merge.
    snap_tolerance
        The tolerance for snapping CoordRanges together. E.G, allows
        coord ranges that have snap_tolerances differences from their
        start/end to be joined together. If they don't meet this requirement
        an [CoordMergeError](`dascore.exceptions.CoordMergeError`) is raised.
        If None, no checks are performed.
    """

    def _get_dims(managers):
        """Ensure all managers have same dimensions."""
        dims = {x.dims for x in managers}
        if len(dims) != 1:
            msg = (
                "Can't merge coord managers, they don't all have the "
                "same dimensions!"
            )
            raise CoordMergeError(msg)
        return managers[0].dims

    def _drop_unshared_coordinates(managers):
        """Any coordinates not shared between managers should be dropped."""
        # gets [{(coord, dims, ...), (coord, dims, ...)}, ...] to ensure
        # both the coords name and their dimensions are common between managers
        coord_sets = [set(x._get_coord_dims_tuple()) for x in managers]
        common_coords = reduce(and_, coord_sets)
        all_coords = reduce(or_, coord_sets)
        if not (drop_coords := all_coords - common_coords):
            return managers
        coords_to_drop = [x[0] for x in drop_coords]
        return [x.drop_coords(*coords_to_drop)[0] for x in managers]

    def _get_non_merge_coords(managers, non_merger_names):
        """Ensure all non-merge coords are equal."""
        out = {}
        for coord_name in non_merger_names:
            first = managers[0].coord_map[coord_name]
            if all([first == x.coord_map[coord_name] for x in managers]):
                dims = managers[0].dim_map[coord_name]
                out[coord_name] = (dims, first)
                continue
            msg = (
                f"Non merging coordinates {coord_name} are not equal. "
                "Coordinate managers cannot be merged."
            )
            raise CoordMergeError(msg)
        return out

    def _snap_coords(coord_list):
        """Snap coordinates together."""
        if snap_tolerance is None:
            return coord_list  # skip snapping if no snap tolerance.
        for ind in range(1, len(coord_list)):
            c_coord = coord_list[ind - 1]
            n_coord = coord_list[ind]
            tolerance = snap_tolerance * c_coord.step
            assumed_start = c_coord.max() + c_coord.step
            diff = np.abs(assumed_start - n_coord.min())
            # snap is close enough, update coord.
            if diff > 0 and diff <= tolerance:
                coord_list[ind] = n_coord.update_limits(min=assumed_start)
            # snap is too far off, bail out.
            elif diff > tolerance:
                msg = (
                    f"Cannot merge. Snap tolerance: {get_nice_text(tolerance)}"
                    f" not met"
                )
                raise CoordMergeError(msg)
        return coord_list

    def _get_merged_coords(managers, coords_to_merge):
        """Get the merged coordinates."""
        out = {}
        for coord_name in coords_to_merge:
            merge_coords = [x.coord_map[dim] for x in managers]
            axis = managers[0].dim_map[coord_name].index(dim)
            if len(units := {x.units for x in merge_coords}) != 1:
                # TODO: we might try to convert all the units to a common
                # unit in the future.
                msg = (
                    f"Cannot merge coordinates {coord_name}, they dont all "
                    f"share the same units. Units found are: {set(units)}"
                )
                raise CoordMergeError(msg)
            snap_coords = _snap_coords(merge_coords)
            datas = [x.data for x in snap_coords]
            dims = managers[0].dim_map[dim]
            new_data = np.concatenate(datas, axis=axis)
            out[coord_name] = (dims, new_data)
        return out

    def _get_new_coords(managers) -> dict[str, tuple[tuple[str, ...], ArrayLike]]:
        """Merge relevant coordinates together."""
        # build up merged coords.
        coords_to_merge = managers[0].dim_to_coord_map[dim]
        coords_not_to_merge = set(managers[0].coord_map) - set(coords_to_merge)
        # non-merging coordinates should be identical.
        coords_dict = _get_non_merge_coords(managers, coords_not_to_merge)
        # merge coordinates
        coords_dict.update(_get_merged_coords(managers, coords_to_merge))
        return coords_dict

    dims = _get_dims(coord_managers)
    coord_managers = _drop_unshared_coordinates(coord_managers)
    sort_managers = sorted(coord_managers, key=lambda x: x.coord_map[dim].min())
    merged_coords = _get_new_coords(sort_managers)
    return get_coord_manager(merged_coords, dims=dims)
