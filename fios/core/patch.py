"""
A 2D trace object.
"""
from typing import Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from xarray import DataArray

from fios.constants import DEFAULT_ATTRS
from fios.proc.filter import pass_filter, stop_filter
from fios.proc.resample import decimate, detrend
from fios.utils.mapping import FrozenDict
from fios.utils.time import to_datetime64, to_timedelta64
from fios.viz.core import TraceViz

from .coords import Coords


def _get_attrs(attr=None, coords=None):
    """Get the attribute dict, add required keys if not yet defined."""
    out = {} if attr is None else dict(attr)
    # Update time and distance range if they are in coords
    # this needs to change if we use timedelta64 here
    if coords is not None and "time" in coords:
        time = getattr(coords["time"], "values", coords["time"])
        out["time_min"] = time.min()
        out["time_max"] = time.max()
    if coords is not None and "distance" in coords:
        dist = getattr(coords["distance"], "values", coords["distance"])
        out["distance_min"] = dist.min()
        out["distance_max"] = dist.max()
    # add default values if they are not in out or attrs yet
    for missing in (set(DEFAULT_ATTRS) - set(attr)) - set(out):
        out[missing] = DEFAULT_ATTRS[missing]
    return FrozenDict(out)


def _condition_coords(coords):
    """
    Condition the coordinates before using them.

    This is mainly to enforce common time conventions
    """

    if coords is None or (time := coords.get("time")) is None:
        return coords
    if not np.issubdtype(time.dtype, np.datetime64):
        coords["time"] = pd.to_datetime(to_datetime64(time))
    return coords


def _time_to_absolute(time, attrs):
    """Convert time to absolute time for slicing."""
    if isinstance(time, slice):
        raise NotImplementedError("Time dimension doesnt yet support slicing.")
    starttime = attrs["time_min"]  # get reference time for trace
    start = time[0]
    if isinstance(start, (float, np.timedelta64, int)):
        start = starttime + to_timedelta64(start)
    stop = time[1]
    if isinstance(stop, (float, np.timedelta64, int)):
        stop = starttime + to_timedelta64(stop)
    if start is not None and stop is not None:
        assert start <= stop, "start time must be less than stop time"
    return slice(start, stop)


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
        coords = _condition_coords(coords)
        dims = dims if dims is not None else list(coords)
        attrs = _get_attrs(attrs, coords)
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

    def equals(self, other: "Patch", only_required_attrs=True) -> bool:
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
        attrs = attrs if attrs is not None else self.attrs
        if coords is None:
            coords = getattr(self.coords, "_coords", self.coords)
        return self.__class__(data=data, coords=coords, attrs=attrs)

    def update_attrs(self, **kwargs) -> "Patch":
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
        >>> # select meters 50 to 300
        >>> import numpy as np
        >>> from fios.examples import get_example_trace
        >>> tr = get_example_trace()
        >>> new = tr.select(distance=(50,300))
        """
        # do special thing for time, else just use DataArray select
        if "time" in kwargs:
            kwargs["time"] = _time_to_absolute(kwargs["time"], self.attrs)
            pass
        # convert tuples into slices
        kwargs = {
            i: slice(v[0], v[1]) if isinstance(v, tuple) else v
            for i, v in kwargs.items()
        }
        new = self._data_array.sel(**kwargs)
        new.attrs = _get_attrs(new.attrs, new.coords)
        return self.__class__(new)

    def transpose(self, *dims: str):
        """
        Transpose the data array to any dimension order desired.

        Parameters
        ----------
        *dims
            Dimension names which define the new data axis order.
        """
        return self.__class__(self._data_array.transpose(*dims))

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
    def attrs(self):
        """Return the attributes of the trace."""
        return self._data_array.attrs

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
