"""A 2D trace object."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal, Self

import numpy as np

import dascore as dc
import dascore.proc.basic
import dascore.proc.coords
import dascore.utils.io
from dascore import transform
from dascore.compat import DataArray, array
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.patch_meta import PatchMeta, _as_dtype
from dascore.core.processor import check_patch_listings
from dascore.core.source import PatchSource
from dascore.models import ArrayLike
from dascore.proc.adaptive_spectral_filter import AdaptiveSpectralFilter
from dascore.proc.tile_apply import TileApply
from dascore.utils.array import (
    PatchUFunc,
    apply_ufunc,
    patch_array_function,
    patch_array_ufunc,
)
from dascore.utils.array_api import backend_name, to_numpy
from dascore.utils.display import (
    Repr,
    array_to_text,
    attrs_to_text,
    get_header_text,
    split_block,
)
from dascore.utils.namespace import NamespaceOwner


class Patch(NamespaceOwner, PatchMeta):
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
        [PatchAttrs](`dascore.core.attrs.PatchAttrs`)
    dtype
        Optional. When given, the data are refused unless that is what
        they hold; a patch's dtype is always its data's.

    source
        Internal source metadata supplied by the I/O framework.

    Notes
    -----
    Coordinates are owned by the patch/coord manager, not by attrs.
    Use `Patch.summary` when you need a combined view of attrs plus
    coordinate summary metadata.

    Coords and a dtype describe data without holding any, which is a
    [`PatchMeta`](`dascore.PatchMeta`); see
    [`drop_data`](`dascore.Patch.drop_data`).
    """

    data: ArrayLike
    _data: ArrayLike

    _namespace_entry_point_group: Final[str] = "dascore.patch_namespace"

    def __init__(
        self,
        data: ArrayLike | DataArray | None = None,
        coords: Mapping[str, Any] | CoordManager | None = None,
        dims: Sequence[str] | None = None,
        attrs: Mapping | PatchAttrs | None = None,
        dtype: Any = None,
        source: PatchSource | None = None,
    ):
        # Init empty patch
        if all(x is None for x in (data, coords, dims, attrs)):
            data = np.asarray([], dtype=dtype)
            coords = {}
            dims = ()
            attrs = dc.PatchAttrs()
        # Init Patch from Patch-like
        if isinstance(data, Patch):
            data, attrs, coords = data._data, data.attrs, data.coords
        elif isinstance(data, DataArray):
            data, attrs, coords = data.data, data.attrs, data.coords
        elif isinstance(data, PatchMeta):
            # Metadata describes data; it is not data, and no patch can be
            # made of it without an array to go with it.
            msg = (
                "A PatchMeta holds no data, so a Patch cannot be built from "
                "one. Use meta.to_patch(data) for the patch it describes."
            )
            raise ValueError(msg)
        if dims is None and isinstance(coords, CoordManager):
            dims = coords.dims
        # By this point, everything should be defined.
        if data is None or coords is None or dims is None:
            msg = (
                "data, coords, and dims must be defined to init Patch; "
                "coords and a dtype alone describe data, which is a PatchMeta."
            )
            raise ValueError(msg)
        data = array(data)
        coords = get_coord_manager(coords, dims=dims, shape=data.shape)
        data = array(coords.validate_data(data))
        if dtype is not None and _as_dtype(dtype) != _as_dtype(data.dtype):
            msg = f"The data are {data.dtype}, not the dtype given: {dtype}."
            raise ValueError(msg)
        self._data = data
        self._set_state(coords, attrs, source)

    def __add__(self, other):
        return apply_ufunc(np.add, self, other)

    def __sub__(self, other):
        return apply_ufunc(np.subtract, self, other)

    def __floordiv__(self, other):
        return apply_ufunc(np.floor_divide, self, other)

    def __truediv__(self, other):
        return apply_ufunc(np.divide, self, other)

    def __mul__(self, other):
        return apply_ufunc(np.multiply, self, other)

    def __pow__(self, other):
        return apply_ufunc(np.power, self, other)

    def __mod__(self, other):
        return apply_ufunc(np.mod, self, other)

    def __gt__(self, other):
        return apply_ufunc(np.greater, self, other)

    def __ge__(self, other):
        return apply_ufunc(np.greater_equal, self, other)

    def __lt__(self, other):
        return apply_ufunc(np.less, self, other)

    def __le__(self, other):
        return apply_ufunc(np.less_equal, self, other)

    def __bool__(self):
        return dascore.proc.basic.bool_patch(self)

    # Also add reverse operators

    __radd__ = __add__

    def __rsub__(self, other):
        """Reverse subtraction: other - self."""
        return apply_ufunc(np.subtract, other, self)

    __rmul__ = __mul__

    def __rpow__(self, other):
        """Reverse power: other ** self."""
        return apply_ufunc(np.power, other, self)

    def __rtruediv__(self, other):
        """Reverse true division: other / self."""
        return apply_ufunc(np.divide, other, self)

    def __rfloordiv__(self, other):
        """Reverse floor division: other // self."""
        return apply_ufunc(np.floor_divide, other, self)

    def __rmod__(self, other):
        """Reverse modulo: other % self."""
        return apply_ufunc(np.mod, other, self)

    def __neg__(self):
        # Through the ufunc, not `update`: `-patch` and `np.negative(patch)`
        # are one operation, and `update` records nothing.
        return apply_ufunc(np.negative, self)

    # Numpy Compatibility things
    __array_ufunc__ = patch_array_ufunc
    __array_function__ = patch_array_function
    __array_priority__ = 1000.0  # Prefer Patch in mixed ops.

    def __array__(self, dtype=None, copy=None):
        """Used to convert Patches to arrays."""
        # dascore.utils.misc.to_object_array stores patch references in numpy
        # object arrays. When that function is called it tries to make a copy
        # of the array data with dtype == object, which takes a TON of memory.
        # For now, just don't let this method convert to object dtype arrays.
        if dtype is not None and np.issubdtype(dtype, np.dtype(object)):
            out = np.empty((), dtype=object)
            out[()] = self
            return out
        data = to_numpy(self.data)
        out = data.astype(dtype) if dtype is not None else data
        out = out if not copy else np.copy(out)
        return out

    def _repr_node(self) -> Repr:
        """The banner, the coordinates, the data and the attributes."""
        attrs = self.attrs
        return Repr(
            header=get_header_text("Patch ⚡"),
            body=(
                self.coords._repr_section(),
                split_block(array_to_text(self._data, units=attrs.get("data_units"))),
                split_block(attrs_to_text(attrs)),
            ),
        )

    @property
    def data(self) -> ArrayLike:
        """
        Return the data contained in patch.

        Examples
        --------
        >>> import dascore as dc
        >>> patch = dc.get_example_patch()
        >>> data = patch.data
        >>> assert data.shape == patch.shape
        """
        return self._data

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the data array."""
        return self._data.dtype

    @property
    def backend(self) -> str:
        """Return the array backend the data are in."""
        return backend_name(self._data)

    def drop_data(self) -> PatchMeta:
        """
        Return this patch's metadata, without its data.

        The result keeps the coords, attrs and dtype, so it still describes
        the data. The inverse of
        [`to_patch`](`dascore.PatchMeta.to_patch`), which gives a
        description its data back.

        Examples
        --------
        >>> import dascore as dc
        >>> patch = dc.get_example_patch()
        >>> described = patch.drop_data()
        >>> assert described.shape == patch.shape
        >>> assert described.dtype == patch.dtype
        >>> assert described.to_patch(patch.data).equals(patch)
        """
        out = PatchMeta(
            coords=self.coords,
            attrs=self.attrs,
            dtype=self.dtype,
            backend=self.backend,
            source=self._source,
        )
        # What `to_patch` builds: this patch's class, so an operation which
        # drops the data and fills them again keeps a subclass.
        out._patch_type = type(self)
        return out

    def _new_like(self, data, coords: CoordManager, attrs: PatchAttrs, dtype=None):
        """Return a patch of this one's class; without new data it keeps its own."""
        # A dtype is handed on only when one was asked for: a subclass's
        # `__init__` need not take one, and the data already state theirs.
        extra = {} if dtype is None else {"dtype": dtype}
        data = self._data if data is None else data
        return type(self)(data=data, coords=coords, attrs=attrs, **extra)

    def _reattach(self, out: PatchMeta, attrs: PatchAttrs) -> Patch:
        """Return `out` under `attrs`, carrying this patch's unchanged data."""
        return out.update(attrs=attrs).to_patch(self._data)

    def to_patch(self, data) -> Patch:
        """
        Return this patch holding `data` instead of its own.

        What [`PatchMeta.to_patch`](`dascore.PatchMeta.to_patch`) means for
        something which already has data, and the same as `new(data=...)`.

        Examples
        --------
        >>> import dascore as dc
        >>> patch = dc.get_example_patch()
        >>> assert patch.to_patch(patch.data * 2).shape == patch.shape
        """
        # Not inherited: the base reads `_patch_type`, which describes the
        # patch a `PatchMeta` came from and which a patch never sets, so a
        # subclass would come back a plain `Patch`.
        return self.new(data=data)

    @property
    def T(self):  # noqa: N802
        """Transpose the Patch."""
        # This isnt a great name but keeps the numpy tradition.
        return self.transpose()

    # --- basic patch functionality.

    equals = dascore.proc.equals
    get_array = dascore.proc.get_array
    split_gaps = dascore.proc.coords.split_gaps
    fill_gaps = dascore.proc.coords.fill_gaps
    add_distance_to = dascore.proc.coords.add_distance_to
    enrich = dascore.proc.enrich
    radians_to_strain = dascore.transform.radians_to_strain
    full = dascore.proc.full

    # The operations written as `PatchProcessor` subclasses which compute
    # data. Written here rather than attached at import so that a reader, an
    # IDE and a type checker all see the patch's surface; each body builds
    # its processor and runs it, and the framework refuses a method which is
    # missing, on the wrong class, or whose parameters have drifted from the
    # processor's fields. The operation is documented once, with its class,
    # and that docstring replaces the summary line below at import.

    def squeeze(self, dim=None) -> Self:
        """Return a patch with length-one dimensions removed."""
        return dascore.proc.coords.Squeeze(dim=dim).run(self)

    def append_dims(self, /, *empty_dims, **dim_kwargs) -> Self:
        """Insert dimensions at the end of the patch."""
        return dascore.proc.coords.AppendDims.from_names(empty_dims, dim_kwargs).run(
            self
        )

    def transpose(self, *dims) -> Self:
        """Transpose the data array to any dimension order."""
        return dascore.proc.coords.Transpose(dims=dims).run(self)

    def snap_coords(self, *coords, reverse: bool = False) -> Self:
        """Snap coordinates to evenly sampled versions of themselves."""
        return dascore.proc.coords.SnapCoords(coords=coords, reverse=reverse).run(self)

    def sort_coords(self, *coords, reverse: bool = False) -> Self:
        """Sort the patch along one or more coordinates."""
        return dascore.proc.coords.SortCoords(coords=coords, reverse=reverse).run(self)

    def drop_private_coords(self) -> Self:
        """Drop coordinates whose names start with an underscore."""
        return dascore.proc.coords.DropPrivateCoords().run(self)

    def make_broadcastable_to(self, shape: tuple[int, ...], drop_coords=False) -> Self:
        """Make the patch broadcastable to a given shape."""
        return dascore.proc.coords.MakeBroadcastableTo(
            shape=shape, drop_coords=drop_coords
        ).run(self)

    def apply_ufunc(self, ufunc, *args, **kwargs) -> Patch:
        """
        Apply a ufunc with the patch as its first operand.

        Parameters
        ----------
        ufunc
            The ufunc to apply.
        *args
            The remaining operands, which can contain patches.
        **kwargs
            Keyword arguments which configure the operation, such as `dim`
            for a reduction or accumulation.

        Examples
        --------
        >>> import numpy as np
        >>> import dascore as dc
        >>> patch = dc.get_example_patch()
        >>>
        >>> # Take the absolute value of the patch.
        >>> abs_patch = patch.apply_ufunc(np.abs)
        >>>
        >>> # Multiply the patch by 10.
        >>> scaled_patch = patch.apply_ufunc(np.multiply, 10)

        See Also
        --------
        [`apply_ufunc`](`dascore.utils.array.apply_ufunc`)
        """
        # The module-level function, not this method.
        return apply_ufunc(ufunc, self, *args, **kwargs)

    set_units = dascore.proc.set_units
    convert_units = dascore.proc.convert_units
    simplify_units = dascore.proc.simplify_units

    # --- processing funcs

    def sel(
        self,
        /,
        indexers: Mapping[str, Any] | None = None,
        method: Literal["nearest"] | None = None,
        tolerance: Any = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> Self:
        """Select values by coordinate label, as xarray does."""
        return dascore.proc.coords.Sel(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kwargs,
        ).run(self)

    def isel(
        self,
        /,
        indexers: Mapping[str, Any] | None = None,
        drop: bool = False,
        missing_dims: str = "raise",
        **indexers_kwargs: Any,
    ) -> Self:
        """Select values by integer index, as xarray does."""
        return dascore.proc.coords.Isel(
            indexers=indexers,
            drop=drop,
            missing_dims=missing_dims,
            **indexers_kwargs,
        ).run(self)

    def select(self, /, *, copy=False, relative=False, samples=False, **kwargs) -> Self:
        """Return a subset of the patch."""
        return dascore.proc.coords.Select(
            copy=copy, relative=relative, samples=samples, **kwargs
        ).run(self)

    def unselect(
        self, /, *, copy=False, relative=False, samples=False, **kwargs
    ) -> Self:
        """Return the patch with the selected range removed."""
        return dascore.proc.coords.Unselect(
            copy=copy, relative=relative, samples=samples, **kwargs
        ).run(self)

    def order(self, /, *, copy=False, relative=False, samples=False, **kwargs) -> Self:
        """Order the patch along coordinates by a set of values."""
        return dascore.proc.coords.Order(
            copy=copy, relative=relative, samples=samples, **kwargs
        ).run(self)

    correlate = dascore.proc.correlate
    correlate_shift = dascore.proc.correlate_shift
    decimate = dascore.proc.decimate
    demedian = dascore.proc.demedian
    detrend = dascore.proc.detrend
    dropna = dascore.proc.dropna
    fillna = dascore.proc.fillna
    pass_filter = dascore.proc.pass_filter
    hampel_filter = dascore.proc.hampel_filter
    sobel_filter = dascore.proc.sobel_filter
    median_filter = dascore.proc.median_filter
    notch_filter = dascore.proc.notch_filter
    savgol_filter = dascore.proc.savgol_filter
    gaussian_filter = dascore.proc.gaussian_filter
    slope_filter = dascore.proc.slope_filter
    wiener_filter = dascore.proc.wiener_filter
    reassemble = dascore.proc.reassemble
    angle = dascore.proc.angle
    resample = dascore.proc.resample
    pad = dascore.proc.pad
    roll = dascore.proc.roll
    where = dascore.proc.where
    flip = dascore.proc.flip
    align_to_coord = dascore.proc.align_to_coord

    interpolate = dascore.proc.interpolate

    def abs(self) -> Self:
        """Return a patch with the absolute value of its data."""
        return dascore.proc.basic.Abs().run(self)

    def conj(self) -> Self:
        """Return a patch with the complex conjugate of its data."""
        return dascore.proc.basic.Conj().run(self)

    def real(self) -> Self:
        """Return a patch with the real part of its data."""
        return dascore.proc.basic.Real().run(self)

    def imag(self) -> Self:
        """Return a patch with the imaginary part of its data."""
        return dascore.proc.basic.Imag().run(self)

    def demean(self, dim: str = "time") -> Self:
        """Remove the mean along a dimension."""
        return dascore.proc.basic.Demean(dim=dim).run(self)

    def normalize(
        self,
        dim: str,
        norm: str = "l2",
        window: Any | None = None,
        samples: bool = False,
    ) -> Self:
        """Normalize a patch along a specified dimension."""
        return dascore.proc.basic.Normalize(
            dim=dim, norm=norm, window=window, samples=samples
        ).run(self)

    def standardize(self, dim: str) -> Self:
        """Standardize a patch along a dimension."""
        return dascore.proc.basic.Standardize(dim=dim).run(self)

    def adaptive_spectral_filter(
        self,
        /,
        *,
        overlap: Any = None,
        exponent: float = 0.8,
        normalize_power: bool = False,
        samples: bool = False,
        engine: str = "auto",
        **kwargs,
    ) -> Self:
        """Apply an adaptive spectral filter to the patch."""
        return AdaptiveSpectralFilter(
            overlap=overlap,
            exponent=exponent,
            normalize_power=normalize_power,
            samples=samples,
            engine=engine,
            **kwargs,
        ).run(self)

    def tile_apply(
        self,
        /,
        function: Callable,
        *,
        mode: str = "overlap_add",
        overlap: Any = None,
        taper: Any = None,
        analysis: Any = None,
        samples: bool = False,
        engine: str = "auto",
        **kwargs,
    ) -> Self:
        """Apply a function to overlapping tiles of the patch."""
        return TileApply(
            function=function,
            mode=mode,
            overlap=overlap,
            taper=taper,
            analysis=analysis,
            samples=samples,
            engine=engine,
            **kwargs,
        ).run(self)

    pow_coord = dascore.proc.pow_coord
    taper = dascore.proc.taper
    taper_range = dascore.proc.taper_range
    line_mute = dascore.proc.line_mute
    slope_mute = dascore.proc.slope_mute
    rolling = dascore.proc.rolling
    whiten = dascore.proc.whiten

    # --- Patch aggregations shortcuts.
    aggregate = dascore.proc.agg.aggregate
    min = dascore.proc.agg.min
    max = dascore.proc.agg.max
    mean = dascore.proc.agg.mean
    median = dascore.proc.agg.median
    std = dascore.proc.agg.std
    sum = dascore.proc.agg.sum
    any = dascore.proc.agg.any
    all = dascore.proc.agg.all
    first = dascore.proc.agg.first
    last = dascore.proc.agg.last
    idxmax = dascore.proc.agg.idxmax
    idxmin = dascore.proc.agg.idxmin

    # --- Universal functions
    add = PatchUFunc(np.add)
    subtract = PatchUFunc(np.subtract)
    multiply = PatchUFunc(np.multiply)
    divide = PatchUFunc(np.divide)
    exp = PatchUFunc(np.exp)
    log = PatchUFunc(np.log)
    log10 = PatchUFunc(np.log10)
    log2 = PatchUFunc(np.log2)
    is_finite = PatchUFunc(np.isfinite)
    isnan = PatchUFunc(np.isnan)
    isinf = PatchUFunc(np.isinf)
    maximum = PatchUFunc(np.maximum)
    minimum = PatchUFunc(np.minimum)

    # --- transformation functions
    differentiate = transform.differentiate
    dft = transform.dft
    fbe = transform.fbe
    idft = transform.idft
    stft = transform.stft
    istft = transform.istft
    integrate = transform.integrate
    stalta = transform.stalta
    kurtosis = transform.kurtosis
    velocity_to_strain_rate = transform.velocity_to_strain_rate
    velocity_to_strain_rate_edgeless = transform.velocity_to_strain_rate_edgeless
    dispersion_phase_shift = transform.dispersion_phase_shift
    tau_p = transform.tau_p
    hilbert = transform.hilbert
    envelope = transform.envelope
    phase_weighted_stack = transform.phase_weighted_stack
    median_frequency = transform.median_frequency
    spectral_centroid = transform.spectral_centroid
    spectral_peak_frequency = transform.spectral_peak_frequency
    spectral_peak_amplitude = transform.spectral_peak_amplitude
    spectral_entropy = transform.spectral_entropy
    spectral_kurtosis = transform.spectral_kurtosis
    spectral_flatness = transform.spectral_flatness


# Both classes list their operations by hand, and this is what refuses a
# listing which is missing or on the wrong class. Deferred to here because
# the classes are created while this module is still importing.
check_patch_listings(Patch, PatchMeta)
