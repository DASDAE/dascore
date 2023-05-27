"""
Module for managing coordinates.
"""
from typing import Dict, Sequence, Tuple, Union

import numpy as np
from contextlib import suppress
from typing_extensions import Self
from pydantic import root_validator, ValidationError

from dascore.exceptions import CoordError
from dascore.utils.coords import BaseCoord, get_coord, get_coord_from_attrs
from dascore.utils.misc import iterate
from dascore.utils.models import ArrayLike, DascoreBaseModel


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
    coord_map: Dict[str, BaseCoord]
    dim_map: Dict[str, Tuple[str, ...]]

    def __getitem__(self, item):
        return self.coord_map[item]

    def update(self, **kwargs) -> Self:
        """
        Update the coordinates, return a new Coordinate Manager.

        Input values can be of the same form as initialization.
        To drop coordinates, simply pass {coord_name: None}
        """
        coords = dict(self.coord_map)
        for item, value in kwargs.items():
            if value is None:
                coords.pop(item, None)
            else:
                coords[item] = value
        out = dict(coord_map=coords, dim_map=self.dim_map, dims=self.dims)
        return self.__class__(**out)

    def drop_dim(self, dim: Union[str, Sequence[str]]) -> Tuple[Self, Tuple]:
        """Drop one or more dimension."""
        coord_name_to_kill = []
        dims_to_kill = set(iterate(dim))
        for name, dims in self.dim_map.items():
            if dims_to_kill & set(dims):
                coord_name_to_kill.append(name)
        coord_map = {
            i: v for i, v in self.coord_map.items() if i not in coord_name_to_kill
        }
        dim_map = {i: v for i, v in self.dim_map.items() if i not in coord_name_to_kill}
        dims = tuple(x for x in self.dims if x not in dims_to_kill)
        index = tuple(
            slice(None, None) if x not in dims_to_kill else 0 for x in self.dims
        )
        new = self.__class__(coord_map=coord_map, dim_map=dim_map, dims=dims)
        return new, index

    def select(self, **kwargs) -> Tuple[Self, Tuple]:
        """
        Perform selection on coordinates.

        Parameters
        ----------
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
            """Function to get reductions for each dimension. """
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
        inds = tuple(dim_reductions.get(x, slice(None, None)) for x in self.dims)
        new = self.__class__(dims=self.dims, dim_map=self.dim_map, coord_map=new_coords)
        return new, inds

    def __rich__(self) -> str:
        out = ["[bold] Coordinates [/bold]"]
        for name, coord in self.coord_map.items():
            dims = self.dim_map[name]
            new = f"    {name}: {dims}: {coord}"
            out.append(new)
        return "\n".join(out)

    @root_validator(pre=True)
    def _validate_coords(cls, values):
        """Validate the coordinates and dimensions."""
        coord_map, dim_map = values['coord_map'], values['dim_map']
        dims = values['dims']
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
            # TODO should we raise a pydantic.ValidatorError directly?
            raise CoordError(msg)
        return values

    @property
    def shape(self):
        """Return the shape of the dimensions."""
        return tuple(len(self.coord_map[x]) for x in self.dims)


def get_coord_manager(coord_dict, dims, attrs=None) -> CoordManager:
    """
    Create a coordinate manager.
    """

    def _coord_from_attrs(attrs, names):
        """Try to get coordinates from attributes."""
        out = {}
        for name in names:
            with suppress(CoordError):
                out[name] = get_coord_from_attrs(attrs, name)
        return out

    def _check_n_fill_coords(coord_dict, dims, attrs):
        coord_set = set(coord_dict)
        dim_set = set(dims)
        missing = dim_set - coord_set
        if not missing:
            return coord_dict
        # we have missing dimensional coordinates.
        if attrs is None:
            msg = (
                f"All dimensions must have coordinates. The following "
                f"are missing {missing}"
            )
            raise CoordError(msg)
        # Try to fill in data from attributes, recurse to raise error if any missed.
        out = dict(coord_dict)
        out.update(_coord_from_attrs(attrs, missing))
        return _check_n_fill_coords(out, dims, attrs=None)

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

    def _coord_from_nested(coord):
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

    coord_dict = _check_n_fill_coords(coord_dict, dims, attrs)
    coord_map, dim_map = {}, {}
    for name, coord in coord_dict.items():
        if isinstance(coord, (BaseCoord, ArrayLike, np.ndarray)):
            coord_map[name], dim_map[name] = _coord_from_simple(name, coord)
        else:
            coord_map[name], dim_map[name] = _coord_from_nested(coord)
    return CoordManager(coord_map=coord_map, dim_map=dim_map, dims=dims)
