"""
A 2D trace object.
"""
from typing import Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from xarray import DataArray

import dascore.proc
from dascore.constants import DEFAULT_PATCH_ATTRS, PatchType
from dascore.io import PatchIO
from dascore.transform import TransformPatchNameSpace
from dascore.utils.mapping import FrozenDict
from dascore.utils.patch import Coords, _AttrsCoordsMixer
from dascore.viz import VizPatchNameSpace


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
        coords = {} if coords is None else coords
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
        return xarray_str.replace("xarray.DataArray", f"dascore.{class_name}")

    __repr__ = __str__

    def equals(self, other: PatchType, only_required_attrs=True) -> bool:
        """
        Determine if the current trace equals the other trace.

        Parameters
        ----------
        other
            A Trace2D object
        only_required_attrs
            If True, only compare required attributes. This helps avoid issues
            with comparing histories or custom attrs of patches, for example.
        """

        if only_required_attrs:
            attrs_to_compare = set(DEFAULT_PATCH_ATTRS) - {"history"}
            attrs1 = {x: self.attrs.get(x, None) for x in attrs_to_compare}
            attrs2 = {x: other.attrs.get(x, None) for x in attrs_to_compare}
        else:
            attrs1, attrs2 = dict(self.attrs), dict(other.attrs)
        if set(attrs1) != set(attrs2):  # attrs don't have same keys; not equal
            return False
        if attrs1 != attrs2:
            # see if some values are NaNs, these should be counted equal
            not_equal = {
                x
                for x in attrs1
                if attrs1[x] != attrs2[x]
                and not (pd.isnull(attrs1[x]) and pd.isnull(attrs2[x]))
            }
            if not_equal:
                return False
        # check coords, names and values
        coord1 = {x: self.coords[x] for x in self.coords}
        coord2 = {x: other.coords[x] for x in other.coords}
        if not set(coord2) == set(coord1):
            return False
        for name in coord1:
            if not np.all(coord1[name] == coord2[name]):
                return False
        # handle transposed case; patches that are identical but transposed
        # should still be equal.
        if self.dims != other.dims and set(self.dims) == set(other.dims):
            other = other.transpose(*self.dims)
        return np.equal(self.data, other.data).all()

    def new(self: PatchType, data=None, coords=None, attrs=None) -> PatchType:
        """
        Return a copy of the trace with data, coords, or attrs updated.
        """
        data = data if data is not None else self.data
        attrs = attrs if attrs is not None else self.attrs
        if coords is None:
            coords = getattr(self.coords, "_coords", self.coords)
            dims = self.dims
        else:
            dims = list(coords)
        return self.__class__(data=data, coords=coords, attrs=attrs, dims=dims)

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
        return self.__class__(self.data, coords=coords, attrs=attrs, dims=self.dims)

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

    def to_xarray(self):
        """
        Return a data array with patch contents.
        """
        # Note this is here in case we decide to remove xarray there will
        # still be a way to get a DataArray object with an optional import
        return self._data_array

    squeeze = dascore.proc.squeeze
    rename = dascore.proc.rename
    transpose = dascore.proc.transpose

    # --- processing funcs

    select = dascore.proc.select
    decimate = dascore.proc.decimate
    detrend = dascore.proc.detrend
    pass_filter = dascore.proc.pass_filter
    aggregate = dascore.proc.aggregate
    abs = dascore.proc.abs
    resample = dascore.proc.resample
    iresample = dascore.proc.iresample
    interpolate = dascore.proc.interpolate

    # --- Method Namespaces
    # Note: these can't be cached_property (from functools) or references
    # to self stick around and keep large arrays in memory.

    @property
    def viz(self) -> VizPatchNameSpace:
        """The visualization namespace."""
        return VizPatchNameSpace(self)

    @property
    def tran(self) -> TransformPatchNameSpace:
        """The transformation namespace."""
        return TransformPatchNameSpace(self)

    @property
    def io(self) -> PatchIO:
        """Return a patch IO object for saving patches to various formats."""
        return PatchIO(self)
