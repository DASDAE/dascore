"""
A 2D trace object.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from rich.text import Text

import dascore.proc
from dascore.compat import DataArray, array
from dascore.constants import PatchType
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.schema import PatchAttrs
from dascore.io import PatchIO
from dascore.transform import TransformPatchNameSpace
from dascore.utils.display import array_to_text, attrs_to_text, get_dascore_text
from dascore.utils.misc import optional_import
from dascore.utils.models import ArrayLike
from dascore.viz import VizPatchNameSpace


class Patch:
    """
    A Class for managing data and metadata.

    Parameters
    ----------
    data
        The data representing fiber optic measurements.
    coords
        The coordinates, or dimensional labels for the data. These can be
        passed in three forms:
        {coord_name: coord}
        {coord_name: ((dimensions,), coord)}
        {coord_name: (dimensions, coord)}
    dims
        A sequence of dimension strings. The first entry corresponds to the
        first axis of data, the second to the second dimension, and so on.
    attrs
        Optional attributes (non-coordinate metadata) passed as a dict or
        [PatchAttrs](`dascore.core.schema.PatchAttrs')

    Notes
    -----
    - If coordinates and dims are not provided, they will be extracted from
    attrs, if possible.

    - If coords and attrs are provided, attrs will have priority. This means
    if there is a conflict between information contained in both, the coords
    will be recalculated. However, any missing data in attrs will be filled in
    if available in coords.
    """

    data: ArrayLike
    coords: CoordManager
    dims: tuple[str, ...]
    attrs: Union[PatchAttrs, Mapping]

    def __init__(
        self,
        data: ArrayLike | DataArray | None = None,
        coords: Mapping[str, ArrayLike] | None | CoordManager = None,
        dims: Sequence[str] | None = None,
        attrs: Optional[Union[Mapping, PatchAttrs]] = None,
    ):
        if isinstance(data, (DataArray, self.__class__)):
            data, attrs, coords = data.data, data.attrs, data.coords
        if dims is None and isinstance(coords, CoordManager):
            dims = coords.dims
        # Try to generate coords from ranges in attrs
        if coords is None and attrs is not None:
            coords = PatchAttrs.coords_from_dims(attrs)
            dims = dims if dims is not None else attrs.dim_tuple
        # Ensure required info is here
        non_attrs = [x is None for x in [data, coords, dims]]
        if any(non_attrs) and not all(non_attrs):
            msg = "data, coords, and dims must be defined to init Patch."
            raise ValueError(msg)
        self._coords = get_coord_manager(coords, dims, attrs)
        self._attrs = PatchAttrs.from_dict(attrs, self.coords)
        self._data = array(self.coords.validate_data(data))

    def __eq__(self, other):
        """
        Compare one Patch.
        """
        return self.equals(other)

    def __add__(self, other):
        return dascore.proc.apply_operator(self, other, np.add)

    def __sub__(self, other):
        return dascore.proc.apply_operator(self, other, np.subtract)

    def __floordiv__(self, other):
        return dascore.proc.apply_operator(self, other, np.floor_divide)

    def __truediv__(self, other):
        return dascore.proc.apply_operator(self, other, np.divide)

    def __mul__(self, other):
        return dascore.proc.apply_operator(self, other, np.multiply)

    def __pow__(self, other):
        return dascore.proc.apply_operator(self, other, np.power)

    def __rich__(self):
        dascore_text = get_dascore_text()
        patch_text = Text("Patch ⚡", style="bold")
        header = Text.assemble(dascore_text, " ", patch_text)
        line = Text("-" * len(header))
        coords = self.coords.__rich__()
        data = array_to_text(self.data)
        attrs = attrs_to_text(self.attrs)
        out = Text("\n").join([header, line, coords, data, attrs])
        return out

        pass

    def __str__(self):
        out = self.__rich__()
        return str(out)

    __repr__ = __str__

    def equals(self, other: PatchType, only_required_attrs=True) -> bool:
        """
        Determine if the current patch equals the other patch.

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
        if not self.coords == other.coords:
            return False
        return np.equal(self.data, other.data).all()

    def new(
        self: PatchType,
        data: None | ArrayLike = None,
        coords: None | dict[str | Sequence[str], ArrayLike] | CoordManager = None,
        dims: None | Sequence[str] = None,
        attrs: None | Mapping | PatchAttrs = None,
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
            A sequence of dimension strings. The first entry corresponds to the
            first axis of data, the second to the second dimension, and so on.
        attrs
            Optional attributes (non-coordinate metadata) passed as a dict.
        """
        data = data if data is not None else self.data
        coords = coords if coords is not None else self.coords
        if dims is None:
            dims = coords.dims if isinstance(coords, CoordManager) else self.dims
        coords = get_coord_manager(coords, dims)
        if attrs:
            # need to figure out what changed and just pass that to update coords
            new, old = dict(attrs), dict(self.attrs)
            diffs = {i: v for i, v in new.items() if new[i] != old.get(i, type)}
            coords = coords.update_from_attrs(diffs)
        attrs = attrs or self.attrs
        return self.__class__(data=data, coords=coords, attrs=attrs, dims=coords.dims)

    def _fast_attr_update(self, attrs):
        """A fast method for just squashing the attrs and returning new patch."""
        new = self.__new__(self.__class__)
        new._data = self.data
        new._attrs = attrs
        new._coords = self.coords
        return new

    def update_attrs(self: PatchType, **attrs) -> PatchType:
        """
        Update attrs and return a new Patch.

        Parameters
        ----------
        **attrs
            attrs to add/update.
        """
        # since we update history so often, we make a fast track for it.
        new_attrs = dict(self.attrs)
        new_attrs.update(attrs)
        if len(attrs) == 1 and "history" in attrs:
            return self._fast_attr_update(PatchAttrs(**new_attrs))
        new_coords = self.coords.update_from_attrs(attrs)
        out = dict(coords=new_coords, attrs=new_attrs, dims=self.dims)
        return self.__class__(self.data, **out)

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the dimensions contained in patch."""
        return self.coords.dims

    @property
    def coord_shapes(self) -> Dict[str, tuple[int, ...]]:
        """Return a dict of coordinate: (shape, ...)"""
        return self.coords.coord_shapes

    @property
    def attrs(self) -> PatchAttrs:
        """Return the dimensions contained in patch."""
        return self._attrs

    @property
    def coords(self) -> CoordManager:
        """Return the dimensions contained in patch."""
        return self._coords

    @property
    def data(self) -> ArrayLike:
        """Return the dimensions contained in patch."""
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the data array."""
        return self.coords.shape

    @property
    def size(self) -> tuple[int, ...]:
        """Return the shape of the data array."""
        return self.coords.size

    def to_xarray(self):
        """
        Return a data array with patch contents.
        """
        xr = optional_import("xarray")
        attrs = dict(self.attrs)
        dims = self.dims
        coords = self.coords._get_dim_array_dict()
        return xr.DataArray(self.data, attrs=attrs, dims=dims, coords=coords)

    squeeze = dascore.proc.squeeze
    transpose = dascore.proc.transpose
    snap_coords = dascore.proc.snap_coords
    sort_coords = dascore.proc.sort_cords
    rename_coords = dascore.proc.rename_coords
    update_coords = dascore.proc.update_coords
    assign_coords = dascore.proc.update_coords

    set_units = dascore.proc.set_units
    convert_units = dascore.proc.convert_units
    simplify_units = dascore.proc.simplify_units

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
    taper = dascore.proc.taper

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
