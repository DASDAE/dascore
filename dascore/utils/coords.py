"""
Utilities for working with coordinates on Patches.
"""
from __future__ import annotations

from typing_extensions import Self

from dascore.constants import PatchType
from dascore.utils.mapping import FrozenDict


def assign_coords(patch: PatchType, **kwargs) -> PatchType:
    """
    Add non-dimensional coordinates to a patch.

    Parameters
    ----------
    patch
        The patch to which coordinates will be added.
    **kwargs
        Used to specify the name, dimension, and values of the new
        coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> patch_1 = dc.get_example_patch()
    >>> coords = patch_1.coords
    >>> dist = coords['distance']
    >>> time = coords['time']
    >>> # Add a single coordinate associated with distance dimension
    >>> lat = np.arange(0, len(dist)) * .001 -109.857952
    >>> out_1 = patch_1.assign_coords(latitude=('distance', lat))
    >>> # Add multiple coordinates associated with distance dimension
    >>> lon = np.arange(0, len(dist)) *.001 + 41.544654
    >>> out_2 = patch_1.assign_coords(
    ...     latitude=('distance', lat),
    ...     longitude=('distance', lon),
    ... )
    >>> # Add multi-dimensional coordinates
    >>> quality = np.ones_like(patch_1.data)
    >>> out_3 = patch_1.assign_coords(
    ...     quality=(patch_1.dims, quality)
    ... )
    """
    coords = {x: patch.coords[x] for x in patch.coords}
    for coord_key, (dimension, value) in kwargs.items():
        coords[coord_key] = (dimension, value)
    return patch.new(coords=coords, dims=patch.dims)


class Coords:
    """
    A class to simplify the handling of coordinates.

    Also helps in supporting non-dimensional coordinates and inferring
    dimensions.

    Attributes
    ----------
    array_dict
        A dict of {coord_name: array}.
    dims_dict
        A dict of {coord_name: tuple[dim_1, dim_2...]}.
    dims
        A tuple of the dimensions for the coordinates.
    """

    # --- Init stuff
    dims = ()
    array_dict = FrozenDict()
    dims_dict = FrozenDict()

    def __init__(self, coords, dims=None):
        if coords is None:
            return
        # Another coord as input
        if isinstance(coords, Coords):
            self.__dict__.update(coords.__dict__)
            return
        # hande a dict being passed
        if isinstance(coords, (dict, FrozenDict)):
            array_dict, dim_dict = self._coord_dict_from_dict(coords)
            self.dims = dims
        # handle xarray coordinate
        else:
            array_dict, dim_dict = self._coord_dict_from_xr(coords)
            self.dims = coords.dims
        self.array_dict = FrozenDict(array_dict)
        self.dims_dict = FrozenDict(dim_dict)

    def _coord_dict_from_xr(self, coord):
        """Get the coordinates from an xarray coordinate object."""
        array_dict = {}
        dim_dict = {}
        for i, v in coord.items():
            array_dict[i] = v.data
            dims = v.dims
            dim_dict[i] = (dims,) if isinstance(dims, str) else tuple(dims)
        return array_dict, dim_dict

    def _coord_dict_from_dict(self, coord_dict):
        """Get the coordinate dict from a dictionary"""
        arrays = {}
        dimensions = {}
        for i, v in coord_dict.items():
            # determine if of the form (dim_name(s), value) or just value
            if len(v) and isinstance(v[0], (str, tuple, list)):
                assert len(v) == 2  # should be a sequence of dims, array
                arrays[i] = v[1]
                dims = v[0]
                dimensions[i] = (dims,) if isinstance(dims, str) else tuple(dims)
            else:
                arrays[i] = v
                dimensions[i] = (i,)
        return arrays, dimensions

    def __str__(self):
        # TODO: Once I am convinced Coords will stay we need a better str
        msg = f"Coords managing {list(self.dims_dict)}"
        return msg

    __repr__ = __str__

    def get(self, item, default=None):
        """Return item or None if not in coord. Same as dict.get."""
        return self.array_dict.get(item, default=default)

    def __iter__(self):
        return self.array_dict.__iter__()

    def __getitem__(self, item):
        return self.array_dict[item]

    def to_nested_dict(self):
        """
        Return a dict of {coord_name: ((*dims), data)}.

        This is useful for input into xarray.
        """
        out = {i: (self.dims_dict[i], self.array_dict[i]) for i in self.dims_dict}
        return out

    def update(self, **kwargs) -> Self:
        """Set the data of an existing coordinate."""
        assert set(kwargs).issubset(set(self.array_dict))
        coords = self.to_nested_dict()
        for item, val in kwargs.items():
            data_list = list(coords[item])
            data_list[1] = val
            coords[item] = data_list
        out = self.__class__(coords)
        return out
