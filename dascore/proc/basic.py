"""Basic operations for patches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.fft import next_fast_len
from scipy.ndimage import correlate1d

from dascore.compat import array
from dascore.constants import PatchType, samples_arg_description
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import (
    CoordManager,
    CoordManagerInput,
    get_coord_manager,
)
from dascore.core.coords import get_coord
from dascore.exceptions import ParameterError
from dascore.models import ArrayLike
from dascore.units import Quantity, get_quantity
from dascore.utils.array import _apply_binary_ufunc
from dascore.utils.array_api import (
    _real_dtype,
    array_namespace,
    asarray_like,
    backend_name,
    is_numpy,
    nan_reduce,
    to_numpy,
    warn_numpy_fallback,
)
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import _get_nullish
from dascore.utils.moving import move_max
from dascore.utils.patch import (
    align_patch_coords,
    get_dim_axis_value,
    patch_function,
)
from dascore.utils.time import dtype_time_like
from dascore.utils.window import resolve_window
from dascore.workflow.processor import (
    PatchProcessor,
    register_implementation,
)

# The dtypes which promise, without the values being looked at, that there
# is no imaginary part: bool, signed and unsigned integers, and floats.
_REAL_KINDS = ("b", "i", "u", "f")


def _known_real(data) -> bool:
    """
    Whether the dtype alone says the data has no imaginary part.

    Object arrays are not among them: their dtype says nothing about the
    elements, and `np.conj` really does conjugate a complex object held in
    one. Anything whose dtype cannot be read is treated the same way --
    not known to be real, so the operation runs.
    """
    dtype = getattr(data, "dtype", None)
    if (kind := getattr(dtype, "kind", None)) is not None:
        return kind in _REAL_KINDS
    if dtype is None:
        return False
    # A backend whose dtypes carry no `kind` -- the standard does not ask
    # for one -- is asked through its own namespace instead. Without this
    # every such array reads as "not known to be real", and the operation
    # runs `real`/`conj` on data the standard forbids them for.
    with suppress(Exception):
        return not array_namespace(data).isdtype(dtype, "complex floating")
    return False


def _as_float(data):
    """
    Promote data which cannot hold a fraction to floats.

    Numpy promotes when dividing or subtracting a float; the array API
    standard has no mixed kind promotion, so it has to be explicit.
    """
    xp = array_namespace(data)
    if xp.isdtype(data.dtype, ("real floating", "complex floating")):
        return data
    return xp.astype(data, xp.float64)


def set_dims(self: PatchType, **kwargs: str) -> PatchType:
    """
    Set dimension to non-dimensional coordinate.

    Parameters
    ----------
    **kwargs
        A mapping indicating old_dim: new_dim where new_dim refers to
        the name of a non-dimensional coordinate which will become a
        dimensional coordinate. The old dimensional coordinate will
        become a non-dimensional coordinate.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # add new coordinate, random numbers length of time dim
    >>> my_coord = np.random.random(patch.coord_shapes["time"])
    >>> out = (
    ...    patch.update_coords(my_coord=("time", my_coord))  # add my_coord
    ...    .set_dims(time="my_coord") # set mycoord as dim (rather than time)
    ... )
    >>> assert "my_coord" in out.dims
    """
    cm = self.coords.set_dims(**kwargs)
    return self.new(coords=cm)


def pipe(self: PatchType, func: Callable[..., PatchType], *args, **kwargs) -> PatchType:
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

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Define a custom function that squares the data
    >>> def square_data(patch):
    ...     return patch.new(data=patch.data ** 2)
    >>>
    >>> # Use pipe to apply the function
    >>> squared = patch.pipe(square_data)
    >>>
    >>> # Can also chain with other methods
    >>> result = patch.pipe(square_data).mean(dim="time")
    """
    return func(self, *args, **kwargs)


def update_attrs(self: PatchType, **attrs) -> PatchType:
    """
    Update patch attrs and return a new Patch.

    Parameters
    ----------
    **attrs
        Attrs to add/update. Nested `coords` payloads are not accepted here;
        use `patch.update_coords(...)` for coordinate changes.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Update existing attributes
    >>> updated = patch.update_attrs(instrument_type="DAS")
    >>>
    >>> # Add new custom attributes
    >>> with_custom = patch.update_attrs(processing_date="2024-01-01")
    """
    new_attrs = self.attrs.model_dump(exclude_unset=True)
    new_attrs.update(attrs)
    validated = PatchAttrs.from_dict(new_attrs)
    return self.__class__(
        self._data, coords=self.coords, attrs=validated, dims=self.dims
    )


# Which data a patch is and what was done to it are not part of what it
# *is*: two patches holding the same data are equal however they were made.
_LINEAGE = {"patch_id", "processing_id"}


def equals(self: PatchType, other: Any, only_required_attrs=True, close=False) -> bool:
    """
    Determine if the current patch equals another.

    Parameters
    ----------
    other
        A Patch (could be equal) or some other type (not equal)
    only_required_attrs
        If True, only compare required attributes. This helps avoid issues
        with comparing histories or custom attrs of patches, for example.
    close
        If True, the data can be "close" using np.allclose, otherwise
        all data must be equal.

    Examples
    --------
    >>> import dascore as dc
    >>> patch1 = dc.get_example_patch()
    >>> patch2 = dc.get_example_patch()
    >>>
    >>> # Test equality
    >>> are_equal = patch1.equals(patch2)
    >>>
    >>> # Test with modified patch
    >>> modified = patch1.update_attrs(custom_attr="test")
    >>> equal_ignoring_custom = patch1.equals(modified, only_required_attrs=True)
    >>>
    >>> # Test with close comparison for numerical data
    >>> noisy = patch1.new(data=patch1.data + 1e-10)
    >>> close_equal = patch1.equals(noisy, close=True)
    """
    # different types are not equal
    if not isinstance(other, type(self)):
        return False
    # Different coords are not equal; can pop out coords from attrs
    if not self.coords == other.coords:
        return False
    if only_required_attrs:  # only include default fields
        # The ids are not part of what a patch *is*: two patches with the
        # same data, coords and attrs are equal however they were made.
        attrs_to_compare = set(PatchAttrs.model_fields) - {"history"} - _LINEAGE
        attrs1 = self.attrs.model_dump(include=attrs_to_compare)
        attrs2 = other.attrs.model_dump(include=attrs_to_compare)
    else:
        # The ids are excluded here too: comparing every attr is about
        # the user's attrs, not about where the data came from.
        attrs1 = self.attrs.model_dump(exclude=_LINEAGE)
        attrs2 = other.attrs.model_dump(exclude=_LINEAGE)
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
    # Test data equality or proximity.
    if self.data.shape != other.data.shape:
        return False
    if close and not np.allclose(self.data, other.data):
        return False
    if not close and not np.equal(self.data, other.data).all():
        return False
    return True


def bool_patch(self: PatchType):
    """
    Get the boolean value of a patch.

    This follows the NumPy convention of raising a ValueError if the patch
    has more than one element. Otherwise, it returns the truthiness of the
    one element.
    """
    return bool(self.data)


def update(
    self: PatchType,
    data: ArrayLike | np.ndarray | None = None,
    coords: CoordManagerInput | CoordManager | None = None,
    dims: Sequence[str] | None = None,
    attrs: Mapping | PatchAttrs | None = None,
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
    data = data if data is not None else self._data
    coords = coords if coords is not None else self.coords
    if dims is None:
        dims = coords.dims if isinstance(coords, CoordManager) else self.dims
    coords = get_coord_manager(coords, dims)
    if attrs is not None:
        attrs = PatchAttrs.from_dict(attrs)
    else:
        attrs = self.attrs
    return self.__class__(data=data, coords=coords, attrs=attrs)


@patch_function()
def abs(patch: PatchType) -> PatchType:
    """
    Take the absolute value of the patch data.

    Examples
    --------
    >>> import dascore # import dascore library
    >>> pa = dascore.get_example_patch() # generate example patch
    >>> out = pa.abs() # take absolute value of generated example patch data
    """
    return Abs()._apply(patch)


class Abs(PatchProcessor):
    """Take the absolute value of the data."""

    def kernel(self, data, meta, out_meta):
        """Return the magnitude of every sample."""
        return array_namespace(data).abs(data)


register_implementation("abs", Abs)


@patch_function()
def conj(patch: PatchType) -> PatchType:
    """
    Apply the complex conjugate of the patch data.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()
    >>>
    >>> # Example 1
    >>> dft = pa.dft(None)  # multi-dim dft
    >>> conj = dft.conj()
    """
    return Conj()._apply(patch)


class Conj(PatchProcessor):
    """Flip the sign of the imaginary part."""

    def kernel(self, data, meta, out_meta):
        """
        Return the conjugate, or the data unchanged.

        Real data is its own conjugate, and handing back the very array
        which came in is what tells the caller nothing happened.
        """
        if _known_real(data):
            return data
        return array_namespace(data).conj(data)


register_implementation("conj", Conj)


@patch_function()
def real(patch: PatchType) -> PatchType:
    """
    Return a new patch with the real part of the data array.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()
    >>> out = pa.real()
    """
    return Real()._apply(patch)


class Real(PatchProcessor):
    """Keep only the real part of the data."""

    def kernel(self, data, meta, out_meta):
        """Return the real part, or the data which is already only that."""
        if _known_real(data):
            return data
        return array_namespace(data).real(data)


register_implementation("real", Real)


@patch_function()
def imag(patch: PatchType) -> PatchType:
    """
    Return a new patch with the imaginary part of the data array.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()
    >>> out = pa.imag()
    """
    return Imag()._apply(patch)


class Imag(PatchProcessor):
    """Keep only the imaginary part of the data."""

    def kernel(self, data, meta, out_meta):
        """
        Return the imaginary part, which is zero for real data.

        Asked for explicitly rather than through `imag`: numpy answers
        zero for a real array, and the standard refuses the question, so
        the answer numpy gives has to be built here to mean the same
        thing on every backend.
        """
        xp = array_namespace(data)
        if _known_real(data):
            return xp.zeros_like(data)
        return xp.imag(data)


register_implementation("imag", Imag)


@patch_function(data_type="")
def angle(patch: PatchType) -> PatchType:
    """
    Return a new patch with the phase angles from the data array.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()
    >>> out = pa.angle()
    """
    return patch.new(data=np.angle(patch.data))


@patch_function(data_type="")
@compose_docstring(sample_explanation=samples_arg_description)
def normalize(
    self: PatchType,
    dim: str,
    norm: Literal["l1", "l2", "max", "bit"] = "l2",
    window: float | Quantity | None = None,
    samples: bool = False,
) -> PatchType:
    """
    Normalize a patch along a specified dimension.

    By default each slice along `dim` is divided by a single norm of that
    whole slice. Giving a `window` divides each sample by the norm of a
    window centered on it instead, which is automatic gain control: late,
    weak arrivals come up to the amplitude of early, strong ones.

    NaN values are ignored when computing the norm. They remain NaN in the
    output but do not affect any other sample. Slices, or windows, with a
    norm of zero -- meaning they contain nothing but zeros and NaN -- are
    returned unscaled.

    Parameters
    ----------
    dim
        The dimension along which the normalization takes place.
    norm
        Determines the value to divide each sample by along a given axis.
        Options are:
            l1 - divide each sample by the l1 of the axis.
            l2 - divide each sample by the l2 of the axis.
            max - divide each sample by the maximum of the absolute value of the axis.
            bit - sample-by-sample normalization (-1/+1)
    window
        The length of the moving window, in units of `dim` unless `samples`
        is True. If None, the whole slice is one window. Not supported for
        `norm="bit"`, which is already a sample-by-sample operation.
    samples
        {sample_explanation}

    Notes
    -----
    - A window is centered on the sample it scales, so an even window length
      is raised to the next odd one.

    - The windowed norms are means rather than sums: `l2` divides by the
      window's RMS and `l1` by its mean absolute value. Were they sums, the
      output would scale with the window length, and the reflected windows
      at the edges of the dimension would not line up with the interior.
      Over a whole slice the two differ only by a constant, so the
      unwindowed norms are left as the sums they have always been.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # L2 normalization along time dimension
    >>> l2_norm = patch.normalize(dim="time", norm="l2")
    >>>
    >>> # Max normalization along distance dimension
    >>> max_norm = patch.normalize(dim="distance", norm="max")
    >>>
    >>> # Bit normalization (sign only)
    >>> bit_norm = patch.normalize(dim="time", norm="bit")
    >>>
    >>> # Automatic gain control: divide by the RMS of a 1 second window.
    >>> agc = patch.normalize(dim="time", norm="l2", window=1)
    >>>
    >>> # The same, with the window given in samples.
    >>> agc = patch.normalize(dim="time", norm="l2", window=251, samples=True)
    """
    return Normalize(dim=dim, norm=norm, window=window, samples=samples)._apply(self)


class Normalize(PatchProcessor):
    """Scale each slice along a dimension by a norm of that slice."""

    dim: str
    norm: str = "l2"
    window: Any | None = None
    samples: bool = False

    def kernel(self, data, meta, out_meta):
        """Return the data with each slice divided by its norm."""
        axis = meta.get_axis(self.dim)
        if self.window is None:
            return _normalize_kernel(data, axis, self.norm)
        if self.norm == "bit":
            msg = (
                "normalize(norm='bit') scales each sample by its own magnitude, "
                "so a window means nothing. Drop the window, or pick another norm."
            )
            raise ParameterError(msg)
        # A window has to be centered on the sample it scales, so it must
        # hold an odd number of them.
        window = resolve_window(
            meta,
            {self.dim: self.window},
            samples=self.samples,
            allow_multiple=False,
            require_odd=True,
            require_evenly_sampled=False,
            enforce_lt_coord=True,
        )
        return _windowed_normalize_kernel(data, axis, self.norm, window.size[0])


register_implementation("normalize", Normalize)


def _window_mean(data, window: int, axis: int):
    """
    Return the mean of a centered window, summed a window at a time.

    Not `dascore.utils.moving.move_mean`, and not for want of trying: both
    engines behind it run one accumulator along the axis, adding the sample
    which enters a window and subtracting the one which leaves. Squaring
    first, as `l2` does, squares the data's dynamic range, and what the
    accumulator then loses to cancellation is unbounded next to a window
    which should have come out near zero -- a mute, a dead channel, the
    quiet before an arrival. It reads as a small negative number, whose
    square root is null, so a single loud sample can blank the rest of its
    trace. Correlating against an explicit kernel sums each window on its
    own and cannot drift, at the cost of reading the window rather than
    stepping it.
    """
    weights = np.full(window, 1.0 / window, dtype=data.dtype)
    return correlate1d(data, weights, axis=axis, mode="reflect")


def _windowed_normalize_kernel(data, axis: int, norm: str, window: int):
    """Divide each sample by a norm of the window centered on it."""
    if norm not in {"l1", "l2", "max"}:
        msg = (
            f"Norm value of {norm} is not supported. "
            f"Supported values are {('l1', 'l2', 'max')}"
        )
        raise ValueError(msg)
    original = data
    numpy_input = is_numpy(data)
    if not numpy_input:
        # The moving windows come from scipy, which numpy alone can feed.
        warn_numpy_fallback("normalize", backend_name(data))
        data = to_numpy(data)
    data = _as_float(data)
    if data.dtype == np.float16:
        # The moving windows are scipy filters, which have no float16.
        data = data.astype(np.float32)
    # Anything not finite is read as zero so the window sums skip it, and
    # counted separately so it does not drag the mean toward zero either.
    # Infinities are excluded along with the nulls because these windows
    # are computed by running arithmetic: one infinity inside a running
    # sum leaves NaN behind it long after the window has moved past.
    # The engine is pinned rather than chosen so that installing
    # bottleneck cannot change what a patch normalizes to at the edges.
    valid = np.isfinite(data)
    filled = np.where(valid, data, 0.0)
    if norm == "max":
        # An absolute value is never negative, so a null read as zero
        # cannot win a window which holds anything else.
        divisor = move_max(np.abs(filled), window, axis=axis, engine="scipy")
    else:
        order = int(norm[-1])
        powers = np.abs(filled) ** float(order)
        counted = _window_mean(valid.astype(powers.dtype), window, axis)
        summed = _window_mean(powers, window, axis)
        # A window of nothing but nulls sums to zero as well, so it falls
        # through to the zero divisor guard below rather than dividing here.
        safe_count = np.where(counted == 0, 1.0, counted)
        divisor = (summed / safe_count) ** (1.0 / order)
    # Not `== 0`: a window whose samples are all zero can leave a mean a
    # hair below it, which a fractional power would turn into a null.
    out = data / np.where(divisor <= 0, 1.0, divisor)
    return out if numpy_input else asarray_like(out, original)


@patch_function()
def pow_coord(patch: PatchType, relative: bool = True, **kwargs) -> PatchType:
    """
    Scale the data by coordinate values raised to a power.

    This is the deterministic counterpart of automatic gain control (see
    [`normalize`](`dascore.Patch.normalize`)): the gain depends only on where
    a sample sits along a coordinate, not on the amplitudes around it, so
    amplitudes stay comparable from one trace to the next. Raising time to a
    power of one or two is the usual correction for the geometric spreading
    and attenuation which make later arrivals weaker.

    Parameters
    ----------
    patch
        The patch to scale.
    relative
        If True, count the coordinate from its own start, so the gain curve
        begins at one and the first sample keeps its amplitude. If False,
        use the coordinate's absolute values, which raises the data units to
        match.
    **kwargs
        Dimension names and the power to raise each to, e.g. `time=2`.

    Notes
    -----
    - The relative curve is `((coord - coord[0]) / step + 1) ** power`, which
      is one, two, three ... raised to the power. Counting from one rather
      than zero is what keeps a power from zeroing the first sample.

    - An unevenly sampled coordinate has no one step, so its first is used:
      the curve is the offset from the start measured in first steps, plus
      one. Every sample still gets a distinct gain, but the spacing of the
      curve no longer follows the spacing of the coordinate.

    - That makes the curve a function of the sample, not of the physical
      span, so the same patch resampled gains differently: the sample one
      second in is the 250th at 250 Hz and the 125th at 125 Hz. Within one
      patch every trace is gained identically, which is what makes their
      amplitudes comparable; across patches of different sample rates they
      are not, so gain before resampling or not at all.

    - That curve is a ratio of coordinate values and so carries no units,
      which is why `relative=True` leaves the data units alone. An absolute
      curve does carry them, so `relative=False` multiplies the data units by
      the coordinate's, raised to the same power.

    - `relative=False` is refused for time coordinates. Their absolute values
      count from an epoch, and a power of the seconds since 1970 says nothing
      about the data.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Correct for spreading which grows as the square of traveltime.
    >>> gained = patch.pow_coord(time=2)
    >>>
    >>> # Along the fiber instead.
    >>> gained = patch.pow_coord(distance=1)
    >>>
    >>> # Both at once.
    >>> gained = patch.pow_coord(time=2, distance=1)
    """
    dim_axis_values = get_dim_axis_value(patch, kwargs=kwargs, allow_multiple=True)
    data = _as_float(patch.data)
    # The curve is built in the data's own precision, so a float32 patch is
    # not doubled in size by being gained.
    gain_dtype = np.float32 if _real_dtype(data) == np.float32 else np.float64
    data_units = get_quantity(patch.attrs.data_units)
    for dim, axis, power in dim_axis_values:
        # Sorted, because a gain curve says "further along the coordinate
        # means more gain", which an unsorted coordinate cannot mean.
        coord = patch.get_coord(dim, require_sorted=True)
        curve = _coord_gain_curve(coord, float(power), relative, dim, gain_dtype)
        shape = [1] * len(patch.dims)
        shape[axis] = curve.size
        data = data * asarray_like(curve.reshape(shape), data)
        coord_units = get_quantity(coord.units)
        if not relative and coord_units is not None:
            # Data carrying no units is dimensionless, as it is everywhere
            # else units are combined, rather than a reason to drop the
            # coordinate's.
            gain_units = coord_units ** float(power)
            data_units = gain_units if data_units is None else data_units * gain_units
    out = patch.new(data=data)
    if data_units != get_quantity(patch.attrs.data_units):
        # The units moved, so whatever the data was called -- velocity,
        # strain rate -- it is not that any more.
        out = out.update_attrs(data_units=data_units, data_type="")
    return out


def _coord_gain_curve(
    coord, power: float, relative: bool, dim: str, dtype
) -> np.ndarray:
    """Return the gain curve a coordinate raised to a power makes."""
    values = coord.values
    if not relative:
        if dtype_time_like(coord.dtype):
            msg = (
                f"pow_coord cannot raise the absolute values of '{dim}' to a "
                "power because they are times, counted from an epoch which "
                "has nothing to do with the data. Use relative=True."
            )
            raise ParameterError(msg)
        with np.errstate(all="ignore"):
            return _check_finite(np.asarray(values, dtype=dtype) ** power, dim, power)
    if values.size < 2:
        # A lone sample is the start of the coordinate, and its gain is one.
        return np.ones(values.size, dtype=dtype)
    step = coord.step if coord.evenly_sampled else values[1] - values[0]
    with np.errstate(all="ignore"):
        offsets = np.asarray((values - values[0]) / step, dtype=dtype)
        return _check_finite((offsets + 1.0) ** power, dim, power)


def _check_finite(curve: np.ndarray, dim: str, power: float) -> np.ndarray:
    """Return the gain curve, or say which coordinate could not make one."""
    if np.all(np.isfinite(curve)):
        return curve
    # One guard for every way the arithmetic can fail: a coordinate holding
    # something which is not finite, a negative power of a coordinate which
    # passes through zero, a fractional power of a negative one. Naming
    # them apart would not help the caller, who has one coordinate and one
    # power to look at either way.
    msg = (
        f"pow_coord cannot build a gain curve from '{dim}' raised to "
        f"{power}: the result is not finite everywhere. With relative=False "
        f"the coordinate's own values are raised to the power, so one which "
        f"crosses zero or runs negative has no curve for every power."
    )
    raise ParameterError(msg)


def _normalize_kernel(data, axis: int, norm: str):
    """Divide each slice along an axis by the norm named."""
    data = _as_float(data)
    xp = array_namespace(data)
    if norm in {"l1", "l2"}:
        order = int(norm[-1])
        # Equivalent to np.linalg.norm, but skips NaN rather than letting a
        # single null blank every sample sharing its slice. The float exponent
        # promotes ints so the powers cannot overflow a narrow dtype.
        powers = xp.abs(data) ** float(order)
        norm_values = nan_reduce("sum", powers, axis=axis) ** (1 / order)
        divisor = xp.expand_dims(norm_values, axis=axis)
    elif norm == "max":
        maxes = nan_reduce("max", xp.abs(data), axis=axis)
        divisor = xp.expand_dims(maxes, axis=axis)
    elif norm == "bit":
        divisor = xp.abs(data)
    else:
        msg = (
            f"Norm value of {norm} is not supported. "
            f"Supported values are {('l1', 'l2', 'max', 'bit')}"
        )
        raise ValueError(msg)
    # A zero divisor means there is nothing but zeros and nulls to scale, so
    # divide those by one; the zeros stay zero and the nulls stay null.
    one = xp.asarray(1, dtype=divisor.dtype)
    return data / xp.where(divisor == 0, one, divisor)


@patch_function(data_type="")
def standardize(
    self: PatchType,
    dim: str,
) -> PatchType:
    """
    Standardize data by removing the mean and scaling to unit variance.

    The standard score of a sample x is calculated as:

    z = (x - u) / s
    where u is the mean of the training samples or zero if with_mean=False,
    and s is the standard deviation of the training samples or one if with_std=False.

    NaN values are ignored when computing the mean and standard deviation. They
    remain NaN in the output but do not affect any other sample.

    Parameters
    ----------
    dim
        The dimension along which the normalization takes place.

    Examples
    --------
    ```{python}
    import dascore as dc

    patch = dc.get_example_patch()

    # standardize along the time axis
    standardized_time = patch.standardize('time')

    # standardize along the x axis
    standardized_distance = patch.standardize('distance')
    ```
    """
    return Standardize(dim=dim)._apply(self)


class Standardize(PatchProcessor):
    """Remove the mean and scale to unit variance along a dimension."""

    dim: str

    def kernel(self, data, meta, out_meta):
        """Return the data centred and scaled along its dimension."""
        axis = meta.get_axis(self.dim)
        data = _as_float(data)
        mean = nan_reduce("mean", data, axis=axis, keepdims=True)
        std = nan_reduce("std", data, axis=axis, keepdims=True)
        return (data - mean) / std


register_implementation("standardize", Standardize)


# This is left here to not break compatibility. It also forces `apply_ufunc`
# to be imported in to this module where it used to live. In the rare chance
# a user tries to access it directly from here it will still work.
apply_operator = _apply_binary_ufunc


@patch_function()
def dropna(
    patch: PatchType,
    dim,
    how: Literal["any", "all"] = "any",
    include_inf=True,
) -> PatchType:
    """
    Return a patch with nullish values dropped along dimension.

    Parameters
    ----------
    patch
        The patch which may contain nullish values.
    dim
        The dimension along which to drop nullish values.
    how
        "any" or "all". If "any" drop label if any null values.
        If "all" drop label if all values are nullish.
    include_inf
        If True, drop all non-finite values.

    Notes
    -----
    When include_inf is False, "nullish" is defined by `pandas.isnull`.
    When include_inf is True (default), "nullish" includes non-finite values
    (NaN, inf, -inf) as determined by `numpy.isfinite`

    Examples
    --------
    >>> import dascore as dc
    >>> # load an example patch which has some NaN values.
    >>> patch = dc.get_example_patch("patch_with_null")
    >>> # drop all time labels that have a single null value
    >>> out = patch.dropna("time", how="any")
    >>> # drop all distance labels that have all null values
    >>> out = patch.dropna("distance", how="all")
    """
    axis = patch.get_axis(dim)
    func = np.any if how == "any" else np.all
    if include_inf:
        to_drop = ~np.isfinite(patch.data)
    else:
        to_drop = pd.isnull(patch.data)
    # need to iterate each non-dim axis and collapse with func
    axes = set(range(len(patch.shape))) - {axis}
    to_drop = func(to_drop, axis=tuple(axes))
    if not np.any(to_drop):  # nothing nullish along this dimension
        return patch
    to_keep = ~to_drop
    assert len(to_keep.shape) == 1
    assert to_keep.shape[0] == patch.data.shape[axis]
    # get slices for trimming data.
    # Annotated because the entries are not all slices; ty reads the
    # list as list[slice] from its initializer otherwise.
    slices: list[Any] = [slice(None)] * len(patch.dims)
    slices[axis] = to_keep
    new_data = patch.data[tuple(slices)]
    coord = patch.get_coord(dim)
    cm = patch.coords.update(**{dim: coord[to_keep]})
    attrs = patch.attrs
    return patch.new(data=new_data, coords=cm, attrs=attrs)


@patch_function()
def fillna(patch: PatchType, value, include_inf=True) -> PatchType:
    """
    Return a patch with nullish values replaced by a value.

    Parameters
    ----------
    patch
        The patch which may contain nullish values.
    value
        The value to replace nullish values with.
    include_inf
        If True, also fill all non-finite values.

    Notes
    -----
    When include_inf is False, "nullish" is defined by `pandas.isnull`.
    When include_inf is True (default), "nullish" includes non-finite values
    (NaN, inf, -inf) as determined by `numpy.isfinite`

    Examples
    --------
    >>> import dascore as dc
    >>> # load an example patch which has some NaN values.
    >>> patch = dc.get_example_patch("patch_with_null")
    >>>
    >>> # Replace all occurrences of NaN with 0
    >>> out = patch.fillna(0)
    >>>
    >>> # Replace all occurrences of NaN with 5
    >>> out = patch.fillna(5)
    """
    if include_inf:
        to_replace = ~np.isfinite(patch.data)
    else:
        to_replace = pd.isnull(patch.data)
    if not np.any(to_replace):  # nothing nullish to fill
        return patch
    new_data = patch.data.copy()
    new_data[to_replace] = value

    return patch.new(data=new_data)


@patch_function()
def pad(
    patch: PatchType,
    mode: Literal["constant"] = "constant",
    constant_values: Any = 0,
    expand_coords=True,
    samples=False,
    **kwargs,
) -> PatchType:
    """
    Pad the patch data along specified dimensions.

    Parameters
    ----------
    patch
        The patch to pad.
    mode : str, optional
        The mode of padding, by default 'constant'.
    constant_values : scalar , optional
        A single scalar value used as the padding value across all dimensions.
        Defaults to 0.
    expand_coords : bool, optional
        Determines how coordinates are adjusted when padding is applied.
        If set to True, the coordinates will be expanded to maintain their
        order and even sampling (if evenly sampled), by extrapolating
        based on the coordinate's step size. If set to False, or coordinate
        is not evenly sampled, the new coordinates introduced by padding
        will be padded with NaN values.
    samples : bool, optional
        If True, the values in kwargs represent samples along a dimension
        and must be integers. Otherwise, they are assumed to have the same
        units as the specified dimension, or have units attached.
    **kwargs:
        Used to specify dimension and number of elements,
        either an integer or a tuple (before, after).
        In addition, the following strings are supported:

        "fft" - pad to the next fast fft length along the given dimension by
        adding values to the end of the axis.

        "correlate" - prepare the coordinate for correlation/convolution in
        the frequency domain by padding to the next fast fft length after
        2*n - 1 where n is the current dimension length by adding values
        to the end of the axis.

    Notes
    -----
    A coordinate measured on a padded dimension grows with it, saying
    nothing over the samples which were added: NaN or NaT for a number
    or a time, blank for text, and False for a membership flag.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Zero pad `time` dimension with 2 patch's time unit (e.g., sec)
    >>> # zeros before and 3 zeros after
    >>> padded_patch_1 = patch.pad(time=(2, 3))
    >>>
    >>> # Pad `distance` dimension with 1s 4 samples before and 4 after.
    >>> padded_patch_3 = patch.pad(distance=4, constant_values=1, samples=True)
    >>>
    >>> # Get patch ready for fast fft along time dimension.
    >>> padded_fft = patch.pad(time="fft")
    """

    def _get_pad_tuple(value, samples, coord):
        """
        Get a tuple, in samples, of (pad_to_start, pad_to_end).
        """
        if value in {"fft", "correlate"}:
            target_length = len(coord) if value == "fft" else 2 * len(coord) - 1
            # Determine value so that the output dim will be a fast length.
            value = (0, next_fast_len(target_length) - len(coord))
            samples = True  # ensure padding isn't interpreted as coord units.
        elif not isinstance(value, Sequence):
            value = (value, value)
        if not samples:  # Ensure values are in samples.
            value = tuple(coord.get_sample_count(x) for x in value)
        return value

    def _get_new_coord(coord, pad_tuple, expand_coords):
        """Get the new coordinate along the expanded axis."""
        # A pad of no samples leaves the coordinate exactly as it was,
        # rather than rebuilding it from its values -- which would widen
        # an integer coordinate to hold a NaN nothing is going to write.
        if not any(pad_tuple):
            return coord
        if expand_coords and coord.evenly_sampled:
            new_start = coord.min() - pad_tuple[0] * coord.step
            new_end = coord.max() + (pad_tuple[1] + 1) * coord.step
            assert coord.evenly_sampled, "expand_coords requires evenly sampled."
            new_coord = get_coord(
                start=new_start, stop=new_end, step=coord.step, units=coord.units
            )
        else:
            old_values = coord.values
            # Need to convert ints to float so NaN can be used.
            if np.issubdtype(old_values.dtype, np.integer):
                old_values = old_values.astype(np.float64)
            null_value = _get_nullish(old_values.dtype)
            added_nan_values = np.pad(
                old_values, pad_width=pad_tuple, constant_values=null_value
            )
            # Units passed rather than updated onto the coordinate: a
            # coordinate built from new data starts with none, so the
            # meters a distance was measured in would come off here.
            new_coord = get_coord(data=added_nan_values, units=coord.units)
        return new_coord

    if isinstance(constant_values, Sequence):
        raise ParameterError("constant_values must be a scalar, not a sequence.")

    def _pad_fill(dtype):
        """What a padded coordinate holds where nothing was measured."""
        # The spellings a projection onto uncovered channels already
        # uses: blank text, an unset number, and not a member.
        if dtype.kind in "US":
            return ""
        if dtype.kind == "b":
            return False
        return _get_nullish(dtype)

    def _get_associated_coords(pad_tuples):
        """Grow the coordinates measured on a padded dimension with it."""
        out = {}
        for name, coord_dims in patch.coords.dim_map.items():
            if name in pad_tuples or pad_tuples.keys().isdisjoint(coord_dims):
                continue
            widths = [pad_tuples.get(x, (0, 0)) for x in coord_dims]
            # A pad of no samples is not a pad: `pad(time="fft")` on a
            # patch already of a fast length asks for one, and widening
            # an integer coordinate there would change it for nothing.
            if not any(any(x) for x in widths):
                continue
            coord = patch.coords.coord_map[name]
            values = coord.values
            # An integer coordinate has to widen to hold the NaN which
            # says nothing is known there, as the padded dimension's own
            # does above.
            if np.issubdtype(values.dtype, np.integer):
                values = values.astype(np.float64)
            padded = np.pad(
                values, pad_width=widths, constant_values=_pad_fill(values.dtype)
            )
            grown = get_coord(data=padded, units=coord.units)
            out[name] = (coord_dims, grown)
        return out

    pad_width = [(0, 0)] * len(patch.shape)
    dimfo = get_dim_axis_value(patch, kwargs=kwargs, allow_multiple=True)
    new_coords = {}
    pad_tuples = {}

    for dim, axis, value in dimfo:
        coord = patch.get_coord(dim, require_evenly_sampled=not samples)
        pad_tuple = _get_pad_tuple(value, samples, coord)
        pad_width[axis] = pad_tuple
        pad_tuples[dim] = pad_tuple
        new_coords[dim] = _get_new_coord(coord, pad_tuple, expand_coords)
    new_coords |= _get_associated_coords(pad_tuples)

    # Pad data, update coord manager, and return.
    new_data = np.pad(patch.data, pad_width, mode=mode, constant_values=constant_values)
    new_coords = patch.coords.update(**new_coords)
    return patch.new(data=new_data, coords=new_coords)


@patch_function()
def roll(patch, samples=False, update_coord=False, **kwargs):
    """
    Roll patch array elements along a given dimension.

    Parameters
    ----------
    patch
        input patch
    samples
        if True, value indicates coordinate or value of dimension
    update_coord
        if True, updates coord based on rolled amount
    **kwargs
        specifies dimension and number of elements to roll

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # roll time dimension 5 elements
    >>> rolled_patch = patch.roll(time=5, samples=True)
    >>>
    >>> # roll distance dimension 30 meters(or units of distance in patch)
    >>> rolled_patch2 = patch.roll(distance=30, samples=False)
    >>>
    >>> # roll time dimension 5 elements and update coordinates
    >>> rolled_patch3 = patch.roll(time=5, samples=True, update_coord=True)
    """
    dim, axis, input_value = get_dim_axis_value(patch, kwargs=kwargs)[0]
    arr = patch.data
    coord = patch.get_coord(dim)
    value = coord.get_sample_count(input_value, samples=samples)

    roll_arr = np.roll(arr, value, axis=axis)

    # update coords if True
    if update_coord:
        roll_coord_arr = np.roll(coord.values, value)
        new_coord = coord.update(values=roll_coord_arr)
        patch = patch.update_coords(**{dim: new_coord})

    return patch.new(data=roll_arr)


@patch_function()
def where(
    patch: PatchType, cond: ArrayLike | PatchType, other: Any | PatchType = np.nan
) -> PatchType:
    """
    Return elements from patch where condition is True, else fill with other.

    Parameters
    ----------
    patch
        The input patch
    cond
        Condition array. Should be a boolean array with the same shape as patch data,
        or a patch with boolean data that is broadcastable to the patch's shape.
    other
        Value to use for locations where cond is False. Can be a scalar value,
        array, or patch that is broadcastable to the patch's shape. Default is NaN.

    Returns
    -------
    PatchType
        A new patch with values from patch where cond is True, and other elsewhere.

    Examples
    --------
    >>> import dascore as dc
    >>> import numpy as np
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Where data > 0 fill with original patch values else nan.
    >>> condition = patch.data > 0
    >>> out = patch.where(condition)
    >>>
    >>> # Use another patch as condition
    >>> threshold = patch.data.mean()
    >>> boolean_patch = patch.new(data=(patch.data > threshold))
    >>> out = patch.where(boolean_patch, other=0)
    >>>
    >>> # Replace values below threshold with 0
    >>> out = patch.where(patch.data > patch.data.mean(), other=0)
    """
    cls = patch.__class__  # Use this so it works with subclasses
    # Align patch and cond
    if isinstance(cond, cls):
        patch, cond = align_patch_coords(patch, cond)
    # Align patch and other, may need to re-align cond
    if isinstance(other, cls):
        patch, other = align_patch_coords(patch, other)
        if isinstance(cond, cls):
            patch, cond = align_patch_coords(patch, cond)

    cond = cond.data if isinstance(cond, cls) else cond
    other = other.data if isinstance(other, cls) else other
    cond_array, other_array = array(cond), array(other)

    # Ensure condition is boolean
    if not np.issubdtype(cond_array.dtype, np.bool_):
        msg = "Condition must be a boolean array or patch with boolean data"
        raise ValueError(msg)

    # Use numpy.where to apply condition
    new_data = np.where(cond_array, patch.data, other_array)
    return patch.new(data=new_data)


@patch_function()
def flip(patch, *dims, flip_coords=True):
    """
    Flip patch data and (optionally coords) along specified dimensions.

    Parameters
    ----------
    patch
        The patch to flip.
    *dims
        The dimensions over which to flip (mirror).
    flip_coords
        If True, also flip coords associated with dimensions, otherwise
        leave them unchanged.

    Examples
    --------
    >>> import dascore as dc
    >>> import numpy as np
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Flip patch over time axis
    >>> out = patch.flip("time")
    >>> assert np.all(patch.get_array("time") == out.get_array("time")[::-1])
    >>>
    >>> # Flip patch over all dimensions.
    >>> out = patch.flip(*patch.dims)
    """
    if not dims:
        return patch  # no-op
    axes = tuple(patch.get_axis(name) for name in dims)
    data = np.flip(patch.data, axis=axes) if dims else patch.data
    coords = patch.coords.flip(*dims) if flip_coords else patch.coords
    return patch.new(data=data, coords=coords)


@patch_function(data_type="")
def full(patch, fill_value):
    """
    Return an identical patch with the data replaced by fill_value.

    Parameters
    ----------
    patch
        The patch to fill.
    fill_value
        The value in the output patch.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Get a patch identical to original but with data array containing
    >>> # Only 1s.
    >>> one_patch = patch.full(1.0)
    >>>
    >>> # Same thing, except for 0s.
    >>> zero_patch = patch.full(0.0)
    """
    array = np.full(patch.data.shape, fill_value)
    return patch.update(data=array)


@patch_function()
def demedian(patch, dim: str = "time"):
    """
    Remove the median along a given dimension of a DASCore patch.

    NaN values are ignored when computing the median, consistent with
    [Patch.median](`dascore.proc.aggregate.median`). They remain NaN in the
    output but do not affect any other sample.

    Parameters
    ----------
    patch :
        The patch to remove the median from.
    dim : str
        Dimension name (e.g., "time", "distance").

    Example
    --------
    >>> import matplotlib.pyplot as plt
    >>> import dascore as dc
    >>> import numpy as np
    >>>
    >>> patch = dc.get_example_patch('example_event_2')
    >>> nx,nt = patch.data.shape
    >>>
    >>> # Add some periodic common-mode noise
    >>> x = np.linspace(0, 6 * np.pi, nt)
    >>> y = np.sin(x) * patch.data.max() / 30
    >>> Y = y[np.newaxis, :] * np.ones((nx,nt), dtype=float)
    >>> patch0 = patch + Y
    >>>
    >>>
    >>> # Prepare figure
    >>> fig, axs = plt.subplots(1, 3, figsize=(20,8), layout='constrained')
    >>>
    >>> # Show patch with common-mode noise
    >>> ax0 = patch0.viz.waterfall(ax = axs[0], show=False)
    >>> _ = ax0.set_title('Original with common-mode noise');
    >>>
    >>> # Show demedian applied patch
    >>> patch1 = patch0.demedian(dim='distance')
    >>> ax1 =  patch1.viz.waterfall(ax = axs[1], show=False)
    >>> _ = ax1.set_title('Removed common-mode noise');
    >>>
    >>> # Show difference
    >>> ax2 = (patch0-patch1).viz.waterfall(ax = axs[2], show=False)
    >>> _ = ax2.set_title('Difference');
    >>>
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close(fig)
    """
    axis = patch.get_axis(dim)
    data = patch.data

    # Compute median along axis, keep dims for broadcasting
    med = np.nanmedian(data, axis=axis, keepdims=True)

    new_data = data - med

    # Return a new patch with updated data
    return patch.new(data=new_data)


@patch_function()
def demean(patch, dim: str = "time"):
    """
    Remove the mean along a given dimension of a DASCore patch.

    NaN values are ignored when computing the mean, consistent with
    [Patch.mean](`dascore.proc.aggregate.mean`). They remain NaN in the output
    but do not affect any other sample.

    Parameters
    ----------
    patch :
        The patch to remove the mean from.
    dim : str
        Dimension name (e.g., "time", "distance").

    Example (note that the example patch is not ideal, but still shows improvements)
    --------
    >>> import matplotlib.pyplot as plt
    >>> import dascore as dc
    >>> import numpy as np
    >>>
    >>> patch = dc.get_example_patch('example_event_2')
    >>> nx,nt = patch.data.shape
    >>>
    >>> # Add some periodic common-mode noise
    >>> x = np.linspace(0, 6 * np.pi, nt)
    >>> y = np.sin(x) * patch.data.max() / 30
    >>> Y = y[np.newaxis, :] * np.ones((nx,nt), dtype=float)
    >>> patch0 = patch + Y
    >>>
    >>>
    >>> # Prepare figure
    >>> fig, axs = plt.subplots(1, 3, figsize=(20,8), layout='constrained')
    >>>
    >>> # Show patch with common-mode noise
    >>> ax0 = patch0.viz.waterfall(ax = axs[0], show=False)
    >>> _ = ax0.set_title('Original with common-mode noise');
    >>>
    >>> # Show demean applied patch
    >>> patch1 = patch0.demean(dim='distance')
    >>> ax1 =  patch1.viz.waterfall(ax = axs[1], show=False)
    >>> _ = ax1.set_title('Removed common-mode noise');
    >>>
    >>> # Show difference
    >>> ax2 = (patch0-patch1).viz.waterfall(ax = axs[2], show=False)
    >>> _ = ax2.set_title('Difference');
    >>>
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close(fig)
    """
    return Demean(dim=dim)._apply(patch)


class Demean(PatchProcessor):
    """Remove the mean along a dimension."""

    dim: str = "time"

    def kernel(self, data, meta, out_meta):
        """Return the data with the mean of each slice taken out."""
        data = _as_float(data)
        mean = nan_reduce("mean", data, axis=meta.get_axis(self.dim), keepdims=True)
        return data - mean


register_implementation("demean", Demean)
