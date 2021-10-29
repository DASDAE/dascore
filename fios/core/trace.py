"""
A 2D trace object.
"""
from typing import Optional, Mapping

import numpy as np
from numpy.typing import ArrayLike

from fios.constants import DEFAULT_ATTRS
from fios.exceptions import IncompatibleCoords, MissingDimensions
from fios.utils.mapping import FrozenDict


def _get_attrs(attr=None):
    """Get the attribute dict, add required keys if not yet defined."""
    out = {} if attr is None else dict(attr)
    # add default values
    for missing in set(DEFAULT_ATTRS) - set(attr):
        out[missing] = DEFAULT_ATTRS[missing]
    return FrozenDict(out)


def _get_coords(data, coords):
    """Get and validate coordinates from data."""
    out = {}
    for expected_len, (name, array) in zip(data.shape, coords.items()):
        if not expected_len == len(array):
            coord_shape = {i: len(v) for i, v in coords.items()}
            msg = (
                f"array of shape {array.shape} is not compatible with "
                f"coordinate shapes: {coord_shape}"
            )
            raise IncompatibleCoords(msg)
        out[name] = array
    return out


class Trace2D:
    """
    A Class for storing and accessing 2D fiber data.
    """

    def __init__(
        self,
        data: ArrayLike,
        coords: Mapping[str, ArrayLike],
        attrs: Optional[Mapping] = None,
    ):
        self._data = data
        self.coords = _get_coords(data, coords)
        self.dims = list(self.coords)
        self._attrs = _get_attrs(attrs)

    def __eq__(self, other):
        """
        Compare one Trace2D to another.

        Parameters
        ----------
        other

        Returns
        -------

        """
        return self.equals(other)

    def equals(self, other: "Trace2D", only_required_attrs=True) -> bool:
        """
        Determine if the current trace equals the other trace.

        Parameters
        ----------
        other
            A Trace2D object
        only_required_attrs
            If True, only compare required attributes.
        """

        if only_required_attrs:
            attrs1 = {k: v for k, v in self.attrs.items() if k in DEFAULT_ATTRS}
            attrs2 = {k: v for k, v in other.attrs.items() if k in DEFAULT_ATTRS}
        else:
            attrs1, attrs2 = dict(self.attrs), dict(other.attrs)
        if attrs1 != attrs2:
            return False
        return np.equal(self.data, other.data).all()

    def new(self, data=None, coords=None, attrs=None):
        """
        Return a copy of the trace with data, coords, or attrs updated.
        """
        data = data if data is not None else self.data
        coords = coords if coords is not None else self.coords
        attrs = attrs if attrs is not None else self.attrs
        return self.__class__(data=data, coords=coords, attrs=attrs)

    def update_attrs(self, **kwargs) -> "Trace2D":
        """
        Update attrs and return a new trace2D.
        """
        attrs = dict(self._attrs)
        attrs.update(**kwargs)
        return self.__class__(self.data, coords=self.coords, attrs=attrs)

    def select(self, **kwargs):
        """
        Return a subset of the trace based on query parameters.

        Any dimension of the data can be passed as key, and the values
        should either be a Slice or a tuple of (min, max) for that
        dimension.

        The time dimension is handled specially in that either floats,
        datetime64 or datetime objects can be used to specify relative
        or absolute times, respectively.

        Examples
        --------
        > from fios.examples import get_example_trace
        > tr = get_example_trace()
        """
        if not len(kwargs):
            return self
        assert len(kwargs) <= 1, "only one dim supported for now"
        dim = list(kwargs)[0]
        vals = kwargs[dim]
        coord = self.coords[dim]
        start = vals[0] if vals[0] is not None else coord.min()
        stop = vals[1] if vals[1] is not None else coord.max()
        missing_dimension = set(kwargs) - set(self.coords)
        if missing_dimension:
            msg = f"Trace does to have dimension(s): {missing_dimension}"
            raise MissingDimensions(msg)
        index1 = np.searchsorted(coord, start, side="left")
        index2 = np.searchsorted(coord, stop, side="right")
        # create new coords and slice arrays
        coords = dict(self.coords)
        coords[dim] = coords[dim][slice(index1, index2)]
        # slice np array
        slices = [slice(None)] * len(self.data.shape)
        dim_ind = self.dims.index(dim)
        slices[dim_ind] = slice(index1, index2)
        data = self.data[tuple(slices)]

        return self.new(data=data, coords=coords)

    @property
    def iloc(self):
        """Return an Ilocator for selecting based on index, not values."""

    @property
    def data(self):
        """Return the data array."""
        return self._data

    @property
    def attrs(self):
        """Return the attributes of the trace."""
        return self._attrs
