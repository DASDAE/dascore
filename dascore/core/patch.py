"""
A 2D trace object.
"""
from __future__ import annotations

import warnings
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
from rich.text import Text

import dascore.proc
from dascore.compat import DataArray, array
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.schema import PatchAttrs
from dascore.io import PatchIO
from dascore.transform import TransformPatchNameSpace
from dascore.utils.display import array_to_text, attrs_to_text, get_dascore_text
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
        return dascore.proc.equals(self, other)

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
        attrs = self.attrs
        data = array_to_text(self.data, units=attrs.get("data_units"))
        attrs = attrs_to_text(self.attrs)
        out = Text("\n").join([header, line, coords, data, attrs])
        return out

        pass

    def __str__(self):
        out = self.__rich__()
        return str(out)

    __repr__ = __str__

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

    # --- basic patch functionality.

    new = dascore.proc.new
    equals = dascore.proc.equals
    update_attrs = dascore.proc.update_attrs
    assert_has_coords = dascore.proc.assert_has_coords
    get_coord = dascore.proc.get_coord
    to_xarray = dascore.proc.to_xarray
    pipe = dascore.proc.pipe
    set_dims = dascore.proc.set_dims
    squeeze = dascore.proc.squeeze
    transpose = dascore.proc.transpose
    snap_coords = dascore.proc.snap_coords
    sort_coords = dascore.proc.sort_coords
    rename_coords = dascore.proc.rename_coords
    update_coords = dascore.proc.update_coords

    def assign_coords(self, *args, **kwargs):
        """Deprecated method for update_coords"""
        msg = "assign_coords is deprecated, use update_coords instead."
        warnings.warn(msg, DeprecationWarning)
        return self.update_coords(*args, **kwargs)

    set_units = dascore.proc.set_units
    convert_units = dascore.proc.convert_units
    simplify_units = dascore.proc.simplify_units

    # --- processing funcs

    select = dascore.proc.select
    iselect = dascore.proc.iselect
    decimate = dascore.proc.decimate
    detrend = dascore.proc.detrend
    pass_filter = dascore.proc.pass_filter
    sobel_filter = dascore.proc.sobel_filter
    median_filter = dascore.proc.median_filter
    aggregate = dascore.proc.aggregate
    abs = dascore.proc.abs
    real = dascore.proc.real
    imag = dascore.proc.imag
    angle = dascore.proc.angle
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
