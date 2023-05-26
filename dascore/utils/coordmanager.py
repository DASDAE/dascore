from typing import Dict, Sequence, Tuple, Union

import numpy as np
from typing_extensions import Self

from dascore.exceptions import CoordError
from dascore.utils.coords import BaseCoord, get_coord
from dascore.utils.misc import iterate
from dascore.utils.models import ArrayLike, DascoreBaseModel


class CoordManager(DascoreBaseModel):
    """
    Class for managing coordinates.

    Attributes
    ----------
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

    def drop_dim(self, dim: Union[str, Sequence[str]]) -> Self:
        """Drop one or more dimension."""
        coord_name_to_kill = []
        dims_to_kill = set(iterate(dim))
        for name, dims in self.dims.items():
            if dims_to_kill & set(dims):
                coord_name_to_kill.append(name)
        coord_map = {
            i: v for i, v in self.coord_map.items() if i not in coord_name_to_kill
        }
        dim_map = {i: v for i, v in self.dim_map.items() if i not in coord_name_to_kill}
        dims = tuple(x for x in self.dims if x not in dims_to_kill)
        return self.__class__(coord_map=coord_map, dim_map=dim_map, dims=dims)

    def __rich__(self) -> str:
        out = ["[bold] Coordinates [/bold]"]
        for name, coord in self.coord_map.items():
            dims = self.dim_map[name]
            new = f"    {name}: {dims}: {coord}"
            out.append(new)
        return "\n".join(out)


def get_coord_manager(coord_dict, dims) -> CoordManager:
    """
    Create a coordinate manager.
    """

    def _coord_from_simple(name, coord):
        """
        Get coordinates from {coord_name: coord} where coord_name is dim name.
        """
        if name not in dims:
            msg = (
                "Coordinates that are not named the same as dimensions"
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

    def _validate_coords(coord_map, dim_map, dim):
        """Validate the coordinates and dimensions."""

    # each dimension must have a coordinate
    if not (cset := set(coord_dict)).issuperset((dset := set(dims))):
        missing = cset - dset
        msg = (
            f"All dimensions must have coordinates. The following "
            f"are missing {missing}"
        )
        raise CoordError(msg)

    coord_map, dim_map = {}, {}
    for name, coord in coord_dict.items():
        if isinstance(coord, (BaseCoord, ArrayLike, np.ndarray)):
            coord_map[name], dim_map[name] = _coord_from_simple(name, coord)
        else:
            coord_map[name], dim_map[name] = _coord_from_nested(coord)
    _validate_coords(coord_map, dim_map, dims)
    return CoordManager(coord_map=coord_map, dim_map=dim_map, dims=dims)
