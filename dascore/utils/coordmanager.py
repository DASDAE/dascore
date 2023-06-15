"""
Module for managing coordinates.
"""
from contextlib import suppress
from typing import Mapping, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from pydantic import root_validator, validator
from rich.text import Text
from typing_extensions import Self

from dascore.constants import DC_BLUE
from dascore.core.schema import PatchAttrs
from dascore.exceptions import CoordError
from dascore.utils.coords import BaseCoord, get_coord, get_coord_from_attrs
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import iterate
from dascore.utils.models import ArrayLike, DascoreBaseModel

MaybeArray = TypeVar("MaybeArray", ArrayLike, np.ndarray, None)


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
        A mapping of {coord_name: (dimensions, )
    """

    dims: Tuple[str, ...]
    coord_map: FrozenDict[str, BaseCoord]
    dim_map: FrozenDict[str, Tuple[str, ...]]

    def __getitem__(self, item) -> np.ndarray:
        # in order to not break backward compatibility, we need to return
        # the array. However, using coords.coord_map[item] gives us the
        # coordinate.
        return self.coord_map[item].values

    def __iter__(self):
        for key in self.coord_map:
            yield (key, self[key])

    def __contains__(self, key):
        return key in self.coord_map

    @validator("coord_map", "dim_map")
    def _convert_to_frozen_dicts(cls, v):
        """Ensure mapping fields are immutable."""
        return FrozenDict(v)

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
            for i, v in coord_map.items():
                # we treat non-dimensional coords elsewhere.
                if i not in self.dims:
                    continue
                assert len(dim_map[i]) == 1, "only dealing with 1D dimensions"
                if len(coord_map[i]) == len(self.coord_map[i]):
                    continue
                for coord_name, dims in self.dim_map.items():
                    if i in dims:
                        out.append(coord_name)
            return out

        # get coords to drop from selecting None
        coords_to_drop = [i for i, v in kwargs.items() if i is None]
        # convert input to coord_map/dim_map
        _coords_to_add = {i: v for i, v in kwargs.items() if i is not None}
        coord_map, dim_map = _get_coord_dim_map(_coords_to_add, self.dims)
        # find coords to drop because their dimension changed.
        dim_change_drops = _get_dim_change_drop(coord_map, dim_map)
        # drop coords then call get_coords to handle adding new ones.
        coords, _ = self.drop_coord(coords_to_drop + dim_change_drops)
        out = coords._get_dim_array_dict()
        out.update(kwargs)
        return get_coord_manager(out, dims=self.dims)

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

    def drop_coord(self, coord: Union[str, Sequence[str]]) -> Tuple[Self, Tuple]:
        """
        Drop one or more coordinates.

        If the coordinate is a dimension, also drop other coords that depend
        on that dimension.

        Parameters
        ----------
        coord
            The name of the coordinate or dimension.
        """
        coord_name_to_kill = []
        coords_to_kill = set(iterate(coord))
        if not coords_to_kill:
            return self, tuple(slice(None, None) for _ in self.dims)
        for name, dims in self.dim_map.items():
            if coords_to_kill & set(dims):
                coord_name_to_kill.append(name)
        coord_map = {
            i: v for i, v in self.coord_map.items() if i not in coord_name_to_kill
        }
        dim_map = {i: v for i, v in self.dim_map.items() if i not in coord_name_to_kill}
        dims = tuple(x for x in self.dims if x not in coords_to_kill)
        index = tuple(
            slice(None, None) if x not in coords_to_kill else 0 for x in self.dims
        )
        new = self.__class__(coord_map=coord_map, dim_map=dim_map, dims=dims)
        return new, index

    def select(self, array: MaybeArray = None, **kwargs) -> Tuple[Self, MaybeArray]:
        """
        Perform selection on coordinates.

        Parameters
        ----------
        array
            An array to which the selection will be applied.
        **kwargs
            Used to specify select arguments. Can be of the form
            {coord_name: (lower_limit, upper_limit)}.
        """

        def _validate_coords(coord, coord_name):
            """Ensure multi-dims are not used."""
            if not len(coord.shape) == 1:
                msg = (
                    "Only 1 dimensional coordinates can be used for selection "
                    f"{coord_name} has {len(coord.shape)} dimensions."
                )
                raise CoordError(msg)

        def _get_dim_reductions():
            """Function to get reductions for each dimension."""
            dim_reductions = {}
            for coord_name, limits in kwargs.items():
                coord = self.coord_map[coord_name]
                _validate_coords(coord, coord_name)
                dim_name = self.dim_map[coord_name][0]
                _, reductions = coord.filter(limits)
                dim_reductions[dim_name] = reductions
            return dim_reductions

        def _get_new_coords(dim_reductions):
            """Function to create dict of trimmed coordinates."""
            new_coords = dict(self.coord_map)
            for coord_name, dims in self.dim_map.items():
                if not set(dims) & set(kwargs):
                    continue  # no overlap, dont redo coord.
                inds = tuple(
                    slice(None, None) if x not in dim_reductions else dim_reductions[x]
                    for x in dims
                )
                coord = self.coord_map[coord_name]
                new_coords[coord_name] = coord[inds]
            return new_coords

        dim_reductions = _get_dim_reductions()
        new_coords = _get_new_coords(dim_reductions)
        # iterate each input and apply reductions.
        # now iterate each dimension and update.
        new = self.__class__(dims=self.dims, dim_map=self.dim_map, coord_map=new_coords)
        if array is not None:
            inds = tuple(dim_reductions.get(x, slice(None, None)) for x in self.dims)
            array = array[inds]
        return new, array

    def __rich__(self) -> str:
        header_text = Text("➤ ") + Text("Coordinates", style=DC_BLUE) + Text(" (")
        lens = {x: self.coord_map[x].shape[0] for x in self.dims}
        dim_texts = Text(", ").join(
            [
                Text(x, style="bold")
                + Text(": ")
                + Text(f"{lens[x]}", style="underline")
                for x in self.dims
            ]
        )

        out = [header_text] + [dim_texts] + [")"]
        for name, coord in self.coord_map.items():
            coord_dims = self.dim_map[name]
            if name in self.dims:
                base = Text.assemble("\n    ", Text(name, style="bold"), ": ")
            else:
                base = Text(f"\n    {name} {coord_dims}: ")
            text = Text.assemble(base, coord.__rich__())
            out.append(text)
        return Text.assemble(*out)

    def __str__(self):
        return str(self.__rich__())

    @root_validator(pre=True)
    def _validate_coords(cls, values):
        """Validate the coordinates and dimensions."""
        coord_map, dim_map = values["coord_map"], values["dim_map"]
        dims = values["dims"]
        dim_shapes = {dim: coord_map[dim].shape for dim in dims}
        for name, coord_dims in dim_map.items():
            expected_shape = tuple(dim_shapes[x][0] for x in coord_dims)
            shape = coord_map[name].shape
            if tuple(expected_shape) == shape:
                continue
            msg = (
                f"coordinate: {name} has a shape of {shape} which does not "
                f"match the dimension(s) of {coord_dims} which have a shape "
                f"of {expected_shape}"
            )
            raise CoordError(msg)
        return values

    def equals(self, other) -> bool:
        """Return True if other coordinates are approx equal."""
        if not isinstance(other, self.__class__):
            return False
        if not set(self.coord_map) == set(other.coord_map):
            return False
        coord_1, coord_2 = self.coord_map, other.coord_map
        for name, coord in self.coord_map.items():
            if not np.all(coord_1[name] == coord_2[name]):
                if not np.allclose(coord_1[name], coord_2[name]):
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

    def validate_data(self, data):
        """Ensure data conforms to coordinates."""
        data = np.array([]) if data is None else data
        assert self.shape == data.shape
        return data

    def _get_dim_array_dict(self):
        """
        Get the coord map in the form:
        {coord_name = ((dims,), array)}
        """
        out = {}
        for name, coord in self.coord_map.items():
            dims = self.dim_map[name]
            out[name] = (dims, coord.data)
        return out

    def set_units(self, **kwargs):
        """Set the units of the coordinate manager."""
        new_coords = dict(self.coord_map)
        for name, units in kwargs.items():
            new_coords[name] = new_coords[name].set_units(units)
        out = dict(dims=self.dims, coord_map=new_coords, dim_map=self.dim_map)
        return self.__class__(**out)

    def update_from_attrs(self, attrs: Mapping) -> Self:
        """Update coordinates based on new attributes."""
        # a bit wasteful, but we need the coercion from initing a PatchAttrs.
        # this enables, for example, conversion of specified fields to datetime
        validated_attrs = PatchAttrs(**dict(attrs))
        attrs = {i: v for i, v in validated_attrs.items() if i in dict(attrs)}
        out = dict(self.coord_map)
        for name, coord in self.coord_map.items():
            start, stop = attrs.get(f"{name}_min"), attrs.get(f"{name}_max")
            step = attrs.get(f"d_{name}")
            # quick path for not updating.
            all_none = all([x is None for x in (start, stop, step)])
            limits_equal = coord.max == stop and coord.min == start
            if all_none or limits_equal:
                continue
            out[name] = coord.update_limits(start, stop, step)
        return self.new(coord_map=out)

    def update_to_attrs(self, attrs: PatchAttrs = None) -> PatchAttrs:
        """Update attrs from information in coordinates."""
        attr_dict = {} if attrs is None else dict(attrs)
        attr_dict["dims"] = self.dims
        for dim in self.dims:
            coord = self.coord_map[dim]
            attr_dict[f"{dim}_min"] = coord.min
            attr_dict[f"{dim}_max"] = coord.max
            # passing None messes up the validation for d_{dim}
            if coord.step is not None:
                attr_dict[f"d_{dim}"] = coord.step
        return PatchAttrs(**attr_dict)

    def transpose(self, dims: Tuple[str, ...]) -> Self:
        """Transpose the coordinates."""
        if set(dims) != set(self.dims):
            msg = (
                "You must specify all dimensions in a transpose operation. "
                f"You passed {dims} but dimensions are {self.dims}"
            )
            raise CoordError(msg)
        assert set(dims) == set(self.dims), "You must pass all dimensions."
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
            """Get the updated dim map"""
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

    def squeeze(self, dim: Optional[Sequence[str]] = None) -> Self:
        """
        Squeeze length one dimensions.

        Parameters
        ----------
        dim
            The dimension name, or sequence of dimension names, to drop.

        Raises
        ------
        CoordError if the selected dimension has a length gt 1.
        """
        to_drop = []
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
        return self.drop_coord(to_drop)[0]

    def decimate(self, **kwargs) -> Tuple[Self, Tuple[slice, ...]]:
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
        (dim, value) = tuple(kwargs.items())[0]
        assert dim in self.dims
        dim_slice = slice(None, None, int(value))
        new_array = self.coord_map[dim][dim_slice]
        new = self.update_coords(**{dim: new_array})
        slices = tuple(
            slice(None, None) if d != dim else slice(None, None, value)
            for d in new.dims
        )
        return new, slices


def get_coord_manager(
    coords: Optional[Mapping[str, Union[BaseCoord, np.ndarray]]] = None,
    dims: Optional[Tuple[str, ...]] = None,
    attrs: Optional[PatchAttrs] = None,
) -> CoordManager:
    """
    Create a coordinate manager.

    Parameters
    ----------
    coords
        A mapping with coordinates. These can be of the form: {name, array},
        {name: (dim_name, array)}, or {name: ((dim_names,) array.
    dims
        Tuple specify dimension names
    attrs
        Attributes which can be used to augment/create coordinates.
    """

    def _coord_from_attrs(coord_map, dim_map, attrs, names):
        """Try to get coordinates from attributes."""
        for name in names:
            with suppress(CoordError):
                coord = get_coord_from_attrs(attrs, name)
                coord_map[name] = coord
                dim_map[name] = (name,)

    def _update_units(coord_manager, attrs):
        """Update coordinates to include units from attrs."""
        if attrs is None:
            return coord_manager
        kwargs = {}
        for name, coord in coord_manager.coord_map.items():
            attrs_units = attrs.get(f"{name}_units")
            if coord.units is None and attrs_units is not None:
                kwargs[name] = attrs_units
        return coord_manager.set_units(**kwargs)

    def _check_and_fill_coords(coord_map, dim_map, dims, attrs):
        """Ensure dims are here and fill missing values from attrs if needed."""
        coord_set = set(coord_map)
        dim_set = set(dims)
        missing = dim_set - coord_set
        if not missing:
            return
        # we have missing dimensional coordinates.
        if attrs is None:
            msg = (
                f"All dimensions must have coordinates. The following "
                f"are missing {missing}"
            )
            raise CoordError(msg)
        # Try to fill in data from attributes, recurse to raise error if any missed.
        _coord_from_attrs(coord_map, dim_map, attrs, missing)
        return _check_and_fill_coords(coord_map, dim_map, dims, attrs=None)

    # nothing to do but return coords if not other info provided.
    if isinstance(coords, CoordManager) and dims is None and attrs is None:
        return coords
    coords = {} if coords is None else coords
    dims = () if dims is None else dims
    coord_map, dim_map = _get_coord_dim_map(coords, dims)
    _check_and_fill_coords(coord_map, dim_map, dims, attrs)
    out = CoordManager(coord_map=coord_map, dim_map=dim_map, dims=dims)
    out = _update_units(out, attrs)
    return out


def _get_coord_dim_map(coords, dims):
    """
    Get coord_map and dim_map from coord input.
    """

    def _coord_from_simple(name, coord):
        """
        Get coordinates from {coord_name: coord} where coord_name is dim name.
        """
        if name not in dims:
            msg = (
                "Coordinates that are not named the same as dimensions "
                "must be passed as a tuple of the form: "
                "(dimension, coord) "
            )
            raise CoordError(msg)
        assert name in dims
        return get_coord(values=coord), (name,)

    def _maybe_coord_from_nested(coord):
        """
        Get coordinates from {coord_name: (dim_name, coord)} or
        {coord_name: ((dim_names...,), coord)}
        """
        if not len(coord) == 2:
            msg = (
                "Second input for coords must be length two of the form:"
                " (dimension, coord) or ((dimensions,...), coord)"
            )
            raise CoordError(msg)
        dim_names = iterate(coord[0])
        # all dims must be in the input dims.
        if not (d1 := set(dim_names)).issubset((d2 := set(dims))):
            bad_dims = d2 - d1
            msg = (
                f"Coordinate specified invalid dimension(s) {bad_dims}."
                f" Valid dimensions are {dims}"
            )
            raise CoordError(msg)
        coord_out = get_coord(values=coord[1])
        return coord_out, dim_names

    # no need to do anything if we already have coord manager.
    if isinstance(coords, CoordManager):
        return dict(coords.coord_map), dict(coords.dim_map)
    # otherwise create coord and dim maps.
    coord_map, dim_map = {}, {}
    for name, coord in coords.items():
        if not isinstance(coord, tuple):
            coord_map[name], dim_map[name] = _coord_from_simple(name, coord)
        else:
            coord_map[name], dim_map[name] = _maybe_coord_from_nested(coord)
    return coord_map, dim_map
