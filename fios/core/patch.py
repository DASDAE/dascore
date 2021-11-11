"""
A 2D trace object.
"""
from functools import cached_property
from typing import Mapping, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from xarray import DataArray

from fios.constants import COMPARE_ATTRS, PatchType
from fios.proc import ProcessingPatchNamespace
from fios.transform import TransformPatchNameSpace
from fios.utils.mapping import FrozenDict
from fios.utils.patch import Coords, _AttrsCoordsMixer
from fios.utils.time import get_select_time
from fios.viz import VizPatchNameSpace


class Patch:
    """
    A Class for storing and accessing 2D fiber data.
    """

    # --- Dunder methods

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
        dims = dims if dims is not None else list(coords)
        mixer = _AttrsCoordsMixer(attrs, coords, dims)
        attrs, coords = mixer()
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

    def __str__(self):
        xarray_str = str(self._data_array)
        class_name = self.__class__.__name__
        return xarray_str.replace("xarray.DataArray", f"fios.{class_name}")

    __repr__ = __str__

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
        dar = self._data_array
        mixer = _AttrsCoordsMixer(dar.attrs, dar.coords, dar.dims)
        mixer.update_attrs(**attrs)
        attrs, coords = mixer()
        return self.__class__(self.data, coords=coords, attrs=attrs)

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
            times = tuple(
                get_select_time(x, tmin, tmax) if x is not None else x
                for x in kwargs["time"]
            )
            kwargs["time"] = times
        # convert tuples into slices
        kwargs = {
            i: slice(v[0], v[1]) if isinstance(v, tuple) else v
            for i, v in kwargs.items()
        }
        new = self._data_array.sel(**kwargs)
        attrs, coords = _AttrsCoordsMixer(new.attrs, new.coords, new.dims)()
        return self.__class__(new.data, attrs=attrs, coords=coords, dims=self.dims)

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

    # --- Method Namespaces

    @cached_property
    def viz(self) -> VizPatchNameSpace:
        """The visualization namespace."""
        return VizPatchNameSpace(self)

    @cached_property
    def tran(self) -> TransformPatchNameSpace:
        """The transformation namespace."""
        return TransformPatchNameSpace(self)

    @cached_property
    def proc(self) -> ProcessingPatchNamespace:
        """The transformation namespace."""
        return ProcessingPatchNamespace(self)
