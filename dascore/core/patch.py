"""A 2D trace object."""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from rich.text import Text
from typing_extensions import Self

import dascore as dc
import dascore.proc.coords
import dascore.utils.io
from dascore import transform
from dascore.compat import DataArray, array
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.coords import BaseCoord
from dascore.utils.display import array_to_text, attrs_to_text, get_dascore_text
from dascore.utils.models import ArrayLike
from dascore.viz import VizPatchNameSpace


class Patch:
    """
    A Class for managing data and metadata.

    See the [patch tutorial](/tutorial/patch.qmd) for examples.

    Parameters
    ----------
    data
        The array data representing fiber optic measurements.
    coords
        The coordinates, or dimensional labels for the data.
        A few types of input are permitted. If a mapping (eg dict) the value
        should conform to one of the following three forms:
        {coord_name: coord}
        {coord_name: ((dimensions,), coord)}
        {coord_name: (dimensions, coord)}
        Where coord can be a numpy array or a
        [`BaseCoord`](`dascore.core.coords.BaseCoord`) object.
        A [`CoordManager`](`dascore.core.coordmanager.CoordManager`) is also acceptable.
    dims
        A sequence of dimension strings. The first entry corresponds to the
        first axis of data, the second to the second dimension, and so on.
    attrs
        Optional attributes (non-coordinate metadata) passed as a dict or
        [PatchAttrs](`dascore.core.attrs.PatchAttrs')

    Notes
    -----
    - If coordinates and dims are not provided, they will be extracted from
    attrs, if possible.

    - If coords and attrs are provided, attrs will have priority. This means
    if there is a conflict between information contained in both, the coords
    will be recalculated.
    """

    data: ArrayLike
    coords: CoordManager
    dims: tuple[str, ...]
    attrs: PatchAttrs | Mapping

    def __init__(
        self,
        data: ArrayLike | DataArray | None = None,
        coords: Mapping[str, ArrayLike | BaseCoord] | CoordManager | None = None,
        dims: Sequence[str] | None = None,
        attrs: Mapping | PatchAttrs | None = None,
    ):
        if isinstance(data, DataArray | self.__class__):
            data, attrs, coords = data.data, data.attrs, data.coords
        if dims is None and isinstance(coords, CoordManager):
            dims = coords.dims
        # Try to generate coords from ranges in attrs
        if coords is None and attrs is not None:
            attrs = dc.PatchAttrs.from_dict(attrs)
            coords = attrs.coords_from_dims()
            dims = dims if dims is not None else attrs.dim_tuple
        # Ensure required info is here
        non_attrs = [x is None for x in [data, coords, dims]]
        if any(non_attrs) and not all(non_attrs):
            msg = "data, coords, and dims must be defined to init Patch."
            raise ValueError(msg)
        coords = get_coord_manager(coords, dims=dims)
        # the only case we allow attrs to include coords is if they are both
        # dicts, in which case attrs might have unit info for coords.
        if isinstance(attrs, Mapping) and attrs:
            coords, attrs = coords.update_from_attrs(attrs)
        else:
            # ensure attrs conforms to coords
            attrs = dc.PatchAttrs.from_dict(attrs).update(coords=coords)
        self._coords = coords
        self._attrs = attrs
        self._data = array(self.coords.validate_data(data))

    def __eq__(self, other):
        """Compare one Patch."""
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
    def coord_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return a dict of coordinate: (shape, ...)."""
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

    update = dascore.proc.update
    # Before 0.1.0 update was called new, this is for backwards compatibility.
    new = dascore.proc.update
    equals = dascore.proc.equals
    update_attrs = dascore.proc.update_attrs
    assert_has_coords = dascore.proc.assert_has_coords
    get_coord = dascore.proc.get_coord
    pipe = dascore.proc.pipe
    set_dims = dascore.proc.set_dims
    squeeze = dascore.proc.squeeze
    transpose = dascore.proc.transpose
    snap_coords = dascore.proc.snap_coords
    sort_coords = dascore.proc.sort_coords
    rename_coords = dascore.proc.rename_coords
    update_coords = dascore.proc.update_coords
    drop_coords = dascore.proc.drop_coords
    coords_from_df = dascore.proc.coords_from_df

    def assign_coords(self, *args, **kwargs):
        """Deprecated method for update_coords."""
        msg = "assign_coords is deprecated, use update_coords instead."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self.update_coords(*args, **kwargs)

    set_units = dascore.proc.set_units
    convert_units = dascore.proc.convert_units
    simplify_units = dascore.proc.simplify_units

    # --- processing funcs

    select = dascore.proc.select

    def iselect(self, *args, **kwargs):
        """Deprecated  form of select."""
        msg = "patch.iselect is deprecated. Use patch.select with samples=True"
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self.select(*args, samples=True, **kwargs)

    correlate = dascore.proc.correlate
    decimate = dascore.proc.decimate
    detrend = dascore.proc.detrend
    dropna = dascore.proc.dropna
    pass_filter = dascore.proc.pass_filter
    sobel_filter = dascore.proc.sobel_filter
    median_filter = dascore.proc.median_filter
    savgol_filter = dascore.proc.savgol_filter
    gaussian_filter = dascore.proc.gaussian_filter
    aggregate = dascore.proc.aggregate
    abs = dascore.proc.abs
    real = dascore.proc.real
    imag = dascore.proc.imag
    angle = dascore.proc.angle
    resample = dascore.proc.resample

    def iresample(self, *args, **kwargs):
        """Deprecated method."""
        msg = (
            "patch.iresample is deprecated. Please use patch.resample "
            "with samples=True"
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self.resample(*args, samples=True, **kwargs)

    interpolate = dascore.proc.interpolate
    normalize = dascore.proc.normalize
    standardize = dascore.proc.standardize
    taper = dascore.proc.taper
    rolling = dascore.proc.rolling
    whiten = dascore.proc.whiten

    # --- transformation functions
    differentiate = transform.differentiate
    rfft = transform.rfft
    dft = transform.dft
    idft = transform.idft
    integrate = transform.integrate
    spectrogram = transform.spectrogram
    velocity_to_strain_rate = transform.velocity_to_strain_rate
    dispersion_phase_shift = transform.dispersion_phase_shift

    # --- Method Namespaces
    # Note: these can't be cached_property (from functools) or references
    # to self stick around and keep large arrays in memory.

    @property
    def viz(self) -> VizPatchNameSpace:
        """The visualization namespace."""
        return VizPatchNameSpace(self)

    @property
    def tran(self) -> Self:
        """The transformation namespace."""
        msg = (
            "The tran namespace is deprecated. Its methods can now be "
            "accessed as normal path methods (eg patch.dft)"
        )
        warnings.warn(msg, DeprecationWarning)
        return self

    @property
    def io(self) -> dc.io.PatchIO:
        """Return a patch IO object for saving patches to various formats."""
        return dc.io.PatchIO(self)
