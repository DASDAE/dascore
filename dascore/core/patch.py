"""
A 2D trace object.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from xarray import DataArray

import dascore.proc
from dascore.constants import PatchType
from dascore.core.schema import PatchAttrs
from dascore.io import PatchIO
from dascore.transform import TransformPatchNameSpace
from dascore.utils.coords import Coords, assign_coords

# from dascore.utils.mapping import FrozenDict
from dascore.utils.patch import _AttrsCoordsMixer
from dascore.viz import VizPatchNameSpace


class Patch:
    """
    A Class for managing data and metadata.

    Parameters
    ----------
    data
        An array-like containing data, an xarray DataArray object, or a Patch.
    coords
        The coordinates, or dimensional labels for the data. These can be
        passed in three forms:
        {coord_name: data}
        {coord_name: ((dimensions,), data)}
        {coord_name: (dimensions, data)}
    dims
        A sequence of dimension strings. The first entry cooresponds to the
        first axis of data, the second to the second dimension, and so on.
    attrs
        Optional attributes (non-coordinate metadata) passed as a dict.

    Notes
    -----
    Unless data is a DataArray or Patch, data, coords, and dims are required.
    """

    data: ArrayLike
    coords: Mapping[str, ArrayLike]
    dims: tuple[str, ...]
    attrs: PatchAttrs

    def __init__(
        self,
        data: ArrayLike | DataArray | None = None,
        coords: Mapping[str, ArrayLike] | None = None,
        dims: Sequence[str] | None = None,
        attrs: Optional[Mapping] = None,
    ):
        non_attrs = [x is None for x in [data, coords, dims]]
        if isinstance(data, (DataArray, self.__class__)):
            dar = data if isinstance(data, DataArray) else data._data_array
            self._data_array = dar
            return
        elif any(non_attrs) and not all(non_attrs):
            msg = "data, coords, and dims must be defined to init Patch."
            raise ValueError(msg)
        mixer = _AttrsCoordsMixer(attrs, coords, dims)
        attrs, coords = mixer()
        # get xarray coords from custom coords object
        xr_coords = coords.to_nested_dict()
        self._data_array = DataArray(
            data=data, dims=dims, coords=xr_coords, attrs=attrs
        )

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
            attrs_to_compare = set(PatchAttrs.get_defaults()) - {"history"}
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

    def new(
        self: PatchType,
        data: None | ArrayLike = None,
        coords: None | dict[str | Sequence[str], ArrayLike] = None,
        dims: None | Sequence[str] = None,
        attrs: None | Mapping = None,
    ) -> PatchType:
        """
        Return a copy of the Patch with updated data, coords, dims, or attrs.

        Parameters
        ----------
        data
            An array-like containing data, an xarray DataArray object, or a Patch.
        coords
            The coordinates, or dimensional labels for the data. These can be
            passed in three forms:
            {coord_name: data}
            {coord_name: ((dimensions,), data)}
            {coord_name: (dimensions, data)}
        dims
            A sequence of dimension strings. The first entry cooresponds to the
            first axis of data, the second to the second dimension, and so on.
        attrs
            Optional attributes (non-coordinate metadata) passed as a dict.
        """
        data = data if data is not None else self.data
        attrs = attrs if attrs is not None else self.attrs
        if coords is None:
            coords = getattr(self.coords, "_coords", self.coords)
            dims = self.dims
        else:
            dims = dims or list(coords)
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
        """Return a dict of coordinate data {coord_name: data}"""
        return Coords(self._data_array.coords, dims=self.dims).array_dict

    @property
    def coord_dims(self):
        """Return a dict of coordinate dimensions {coord_name: (**dims)}"""
        return Coords(self._data_array.coords, dims=self.dims).dims_dict

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the dimensions contained in patch."""
        return self._data_array.dims

    @property
    def attrs(self) -> PatchAttrs:
        """Return the attributes of the trace."""
        return PatchAttrs(**self._data_array.attrs)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the data array."""
        return self._data_array.shape

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
    sobel_filter = dascore.proc.sobel_filter
    median_filter = dascore.proc.median_filter
    aggregate = dascore.proc.aggregate
    abs = dascore.proc.abs
    resample = dascore.proc.resample
    iresample = dascore.proc.iresample
    interpolate = dascore.proc.interpolate
    normalize = dascore.proc.normalize
    standardize = dascore.proc.standardize

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

    def pipe(self, func: Callable[["Patch", ...], "Patch"], *args, **kwargs) -> "Patch":
        """
        Pipe the patch to a function.

        This is primarily useful for maintaining a chain of patch calls for
        a function.

        Parameters
        ----------
        func
            The function to pipe the patch. It must take a patch instance as
            the first argument followed by any number of positional or keyword
            arguments, then return a patch.
        *args
            Positional arguments that get passed to func.
        **kwargs
            Keyword arguments passed to func.
        """
        return func(self, *args, **kwargs)

    # Bind assign_coords as method
    assign_coords = assign_coords
