"""
A 2D trace object.
"""
from typing import Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from xarray import DataArray

from fios.constants import DEFAULT_PATCH_ATTRS, PatchType, COMPARE_ATTRS
from fios.proc.filter import pass_filter, stop_filter
from fios.proc.resample import decimate, detrend
from fios.utils.mapping import FrozenDict
from fios.utils.time import to_datetime64, to_timedelta64
from fios.utils.patch import get_relative_deltas
from fios.viz.core import TraceViz

from .coords import Coords


def _get_attrs(attr=None, coords=None, kwargs=None):
    """Get the attribute dict, add required keys if not yet defined."""
    out = {} if attr is None else dict(attr)
    # add default values if they are not in out or attrs yet
    for missing in (set(DEFAULT_PATCH_ATTRS) - set(attr)) - set(out):
        value = DEFAULT_PATCH_ATTRS[missing]
        out[missing] = value if not callable(value) else value()
    # Update time and distance range if they are in coords
    if coords is not None and "time" in coords:
        time = getattr(coords["time"], "values", coords["time"])
        # set absolute values from time array
        if np.issubdtype(time.dtype, np.datetime64) and len(time) > 0:
            out["time_min"] = time.min()
            out["time_max"] = time.max()
        elif len(time) > 0:  # update coords to be in timedelta64 and update endtime
            time = to_timedelta64(time)
            if not pd.isnull(out["time_min"]):
                out["time_max"] = time.max() + out["time_min"]
            coords["time"] = to_timedelta64(time)

    if coords is not None and "distance" in coords:
        dist = getattr(coords["distance"], "values", coords["distance"])
        out["distance_min"] = dist.min()
        out["distance_max"] = dist.max()

    return FrozenDict(out)


def _condition_coords(coords):
    """
    Condition the coordinates before using them.

    This is mainly to enforce common time conventions
    """

    if coords is None or (time := coords.get("time")) is None:
        return coords
    # Convert datetime arrays into time deltas assuming first element is start
    if np.issubdtype(time.dtype, np.datetime64):
        time = coords["time"] - coords["time"][0]
    if not np.issubdtype(time.dtype, np.timedelta64):
        time = pd.to_timedelta(to_timedelta64(time))
    coords["time"] = time
    return coords


class Patch:
    """
    A Class for storing and accessing 2D fiber data.
    """

    def __init__(
        self,
        data: Union[ArrayLike, DataArray] = None,
        coords: Mapping[str, ArrayLike] = None,
        dims: Tuple[str] = None,
        attrs: Optional[Mapping] = None,
    ):
        if isinstance(data, DataArray):
            self._data_array = data
            return
        attrs = _get_attrs(attrs, coords)
        coords = _condition_coords(coords)
        dims = dims if dims is not None else list(coords)
        # get xarray coords from custom coords object
        if isinstance(coords, Coords):
            coords = coords._coords
        self._data_array = DataArray(data=data, dims=dims, coords=coords, attrs=attrs)

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

    def equals(self, other: PatchType, only_required_attrs=True) -> bool:
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
            attrs1 = {k: v for k, v in self.attrs.items() if k in COMPARE_ATTRS}
            attrs2 = {k: v for k, v in other.attrs.items() if k in COMPARE_ATTRS}
        else:
            attrs1, attrs2 = dict(self.attrs), dict(other.attrs)
        if attrs1 != attrs2:
            return False
        return np.equal(self.data, other.data).all()

    def new(self: PatchType, data=None, coords=None, attrs=None) -> PatchType:
        """
        Return a copy of the trace with data, coords, or attrs updated.
        """
        data = data if data is not None else self.data
        attrs = attrs if attrs is not None else self.attrs
        if coords is None:
            coords = getattr(self.coords, "_coords", self.coords)
        return self.__class__(data=data, coords=coords, attrs=attrs)

    def update_attrs(self: PatchType, **attrs) -> PatchType:
        """
        Update attrs and return a new Patch.

        Parameters
        ----------
        **attrs
            attrs to add/update.
        """
        new_attrs = dict(self.attrs)
        new_attrs.update(**attrs)
        return self.__class__(self.data, coords=self.coords, attrs=new_attrs)

    def select(self: PatchType, **kwargs) -> PatchType:
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
        >>> # select meters 50 to 300
        >>> import numpy as np
        >>> from fios.examples import get_example_patch
        >>> tr = get_example_patch()
        >>> new = tr.select(distance=(50,300))
        """
        new_attrs = dict(self.attrs)
        # do special thing for time, else just use DataArray select
        if "time" in kwargs:
            tmin = self._data_array.attrs["time_min"]
            tmax = self._data_array.attrs["time_max"]
            time = get_relative_deltas(kwargs["time"], tmin, tmax)
            if time.start is not None:
                new_attrs['time_min'] = tmin + time.start
            if time.stop is not None:
                new_attrs['time_max'] = tmin + time.stop
            kwargs['time'] = time
        # convert tuples into slices
        kwargs = {
            i: slice(v[0], v[1]) if isinstance(v, tuple) else v
            for i, v in kwargs.items()
        }
        new = self._data_array.sel(**kwargs)
        new.attrs = _get_attrs(new_attrs, new.coords, kwargs)
        return self.__class__(new)

    def transpose(self: PatchType, *dims: str) -> PatchType:
        """
        Transpose the data array to any dimension order desired.

        Parameters
        ----------
        *dims
            Dimension names which define the new data axis order.
        """
        return self.__class__(self._data_array.transpose(*dims))

    def rename(self: PatchType, **names) -> PatchType:
        """
        Rename coordinate or dimensions of Patch.

        Parameters
        ----------
        **names
            The mapping from old names to new names

        Examples
        --------
        >>> from fios.examples import get_example_patch
        >>> pa = get_example_patch()
        >>> # rename dim "distance" to "fragrance"
        >>> pa2 = pa.rename(distance='fragrance')
        >>> assert 'fragrance' in pa2.dims
        """
        new_data_array = self._data_array.rename(**names)
        return self.__class__(new_data_array)

    @property
    def iloc(self):
        """Return an index locator for selecting based on index, not values."""

    @property
    def data(self):
        """Return the data array."""
        return self._data_array.data

    @property
    def coords(self):
        """Return the data array."""
        return Coords(self._data_array.coords)

    @property
    def dims(self) -> Tuple[str, ...]:
        """Return the data array."""
        return self._data_array.dims

    @property
    def attrs(self) -> FrozenDict:
        """Return the attributes of the trace."""
        return FrozenDict(self._data_array.attrs)

    @property
    def viz(self) -> TraceViz:
        """Return a TraceViz instance bound to this Trace"""
        return TraceViz(self)

    def __str__(self):
        xarray_str = str(self._data_array)
        class_name = self.__class__.__name__
        return xarray_str.replace("xarray.DataArray", f"fios.{class_name}")

    # add patch processing methods.

    decimate = decimate
    pass_filter = pass_filter
    stop_filter = stop_filter
    detrend = detrend

    __repr__ = __str__
