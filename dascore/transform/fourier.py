"""
Module for Fourier transforms.

See the [FFT note](/notes/dft_notes.qmd) for discussion on the
implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from operator import mul, truediv
from typing import Any

import numpy as np
import numpy.fft as nft
import scipy.fft as sft
from pydantic import ConfigDict, Field

import dascore as dc
from dascore import units
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.core.processor import PatchProcessor
from dascore.exceptions import CoordError, ParameterError, PatchError
from dascore.proc.basic import Pad, _pad_array
from dascore.proc.tile_apply import Reassemble, TileApply
from dascore.proc.units import _replace_data_units
from dascore.units import Quantity, _quantities_equal, invert_quantity, percent
from dascore.utils.array_api import (
    _as_numpy_dtype,
    _result_dtype,
    array_namespace,
    asarray_like,
)
from dascore.utils.imports import lazy_import
from dascore.utils.misc import iterate
from dascore.utils.patch import (
    _get_data_units_from_dims,
    _get_dx_or_spacing_and_axes,
)
from dascore.utils.signal import get_window_nd
from dascore.utils.tiles import _taper_like
from dascore.utils.time import to_float
from dascore.utils.transformatter import FourierTransformatter
from dascore.utils.window import resolve_window

sp_detrend = lazy_import("scipy.signal", "detrend")

DFT_OUTPUT_DATA_TYPE_MAP = {
    "AS": "amplitude_spectrum",
    "PS": "power_spectrum",
    "PSD": "power_spectral_density",
}
DFT_OUTPUT_TYPES = ("FFT", *DFT_OUTPUT_DATA_TYPE_MAP)
# stft's default overlap, one object shared by the method and the class so
# that a call which leaves it unset has the id it had as a patch function.
_HALF_OVERLAP = 50 * percent


def _associated_prefix(dim: str) -> str:
    """Where dft parks the coordinates measured on a dimension it takes."""
    # Private, like the unpadded coordinate parked beside them, and named
    # after the dimension so idft knows what to put each one back on.
    return f"_{dim}_associated_"


def _get_dft_coord_units(units):
    """Get units for DFT coordinates."""
    new_units = invert_quantity(units)
    # This purposefully converts 1/s to Hz to be more conventional. See #693.
    if _quantities_equal(new_units, dc.get_quantity("1/s")):
        new_units = dc.get_quantity("Hz")
    return new_units


def _get_dft_coord_unit_product(patch, dims, transformed=False, coords=None):
    """Get the product of original or transformed coordinate units."""
    coord_source = patch if coords is None else coords
    names = [f"ft_{dim}" if transformed else dim for dim in iterate(dims)]
    return prod(
        unit
        for name in names
        if (unit := dc.get_quantity(coord_source.get_coord(name).units)) is not None
    )


def _get_dft_data_units(patch, dims, output="FFT", coords=None):
    """Get data units for DFT outputs."""
    data_units = dc.get_quantity(patch.attrs.data_units)
    if data_units is None:
        return None
    domain_units = _get_dft_coord_unit_product(patch, dims)
    if output == "FFT":
        return data_units * domain_units
    original_units = data_units / domain_units
    spectral_units = _get_dft_coord_unit_product(
        patch, dims, transformed=True, coords=coords
    )
    output_units = {
        "AS": original_units,
        "PS": original_units**2,
        "PSD": original_units**2 / spectral_units,
    }
    return output_units[output]


def _get_dft_new_coords(patch, dxs, dims, axes, real, original_cm=None):
    """
    Create coordinates based on dxs and patch shape.

    if original_cm is not none, it means the patch was padded.
    """
    # Note: We need original_cm and patch because patch may have undergone
    # padding.

    def _get_fft_coord(x_len, dx, units, is_real=False):
        """Get coord for fft frequency bins."""
        new_dx = 1.0 / (x_len * dx)
        start = 0 if is_real else -(x_len // 2) * new_dx
        stop = (x_len // 2 + 1) * new_dx if is_real else ((x_len - 1) // 2 + 1) * new_dx
        units = _get_dft_coord_units(units)
        return get_coord(start=start, stop=stop, step=new_dx, units=units)

    # first disassociate old coordinates. We do this rather than drop them
    # so the idft can find them and exactly restore old coords.
    # A coordinate on a transformed dimension goes the same way, under a
    # private name saying which dimension it came off, since it cannot
    # ride the frequency axis -- a real transform is not even the same
    # length. One spanning several dimensions has no such name, so it is
    # dropped by the disassociation as it always was.
    stashed = {
        name: cdims[0]
        for name, cdims in patch.coords.dim_map.items()
        if name not in dims and len(cdims) == 1 and cdims[0] in dims
    }
    old_cm = patch.coords.disassociate_coord(*dims, *stashed)
    new_coords = old_cm.get_coord_tuple_map()
    for name, dim in stashed.items():
        parked = f"{_associated_prefix(dim)}{name}"
        # A dimension and a coordinate name can be anything, so the two
        # of them joined is not one name only they could make: a
        # coordinate `b_associated_c` on dimension `a` parks where a
        # coordinate `c` on dimension `a_associated_b` would. Nobody
        # names a fiber axis that, but idft would read one of them back
        # as the other, so it is refused rather than resolved.
        if parked in new_coords:
            msg = (
                f"The coordinate {name!r} of dimension {dim!r} cannot be "
                f"kept for the inverse transform: {parked!r} is where it "
                "would go, and that is taken. Rename one of them."
            )
            raise PatchError(msg)
        new_coords[parked] = new_coords.pop(name)
    ft = FourierTransformatter()
    for i, dim in enumerate(dims):
        old_coord = patch.get_coord(dim)
        units = old_coord.units
        size = old_coord.shape[0]
        dx = dxs[i]
        new_name = ft.rename_dims(dim)[0]
        coord = _get_fft_coord(size, dx, units, is_real=dim == real)
        new_coords[new_name] = (new_name, coord)
        # Add padded coordinates
        if original_cm is not None:
            new_coords[f"_{dim}_unpadded"] = (None, original_cm.get_coord(dim))
    new_dims = ft.rename_dims(patch.dims, index=axes)
    cm = get_coord_manager(new_coords, dims=new_dims)
    return cm


def _get_dft_attrs(patch, dims, new_coords, pad=False, output="FFT"):
    """Get new attributes for transformed patch."""
    new = dict(patch.attrs)
    new["data_units"] = _get_dft_data_units(patch, dims)
    new["_pre_dft_data_type"] = new.get("data_type")
    new["data_type"] = "fourier_transform"
    new["_dft_output"] = output
    new["_dft_padded"] = pad
    return PatchAttrs(**new)


def _get_untransformed_dims(patch, dims):
    """Return dimensions which have not been transformed."""
    dim_set = set(patch.dims)
    out = []
    for dim in dims:
        # This dim has already been transformed.
        if (dim not in dim_set) and f"ft_{dim}" in dim_set:
            continue
        out.append(dim)
    return out


def _get_transformed_domain_extent(patch, dims):
    """Get the transformed-domain extent from the DFT bin spacing."""
    extent = 1
    preserve_units = patch.attrs.data_units is not None
    for dim in iterate(dims):
        ft_coord = patch.get_coord(f"ft_{dim}")
        # df = 1 / (n * dx), so 1 / df is the original-domain extent.
        # For multi-axis DFTs, the total extent is the product over axes.
        step = abs(ft_coord.step)
        if preserve_units and ft_coord.units is not None:
            step = step * ft_coord.units
        extent = extent / step
    return extent


def _spectral_amplitude_plan(meta, output, dims, db):
    """
    Return the metadata of a spectral output and how its kernel scales.

    A scale in the data units ("10 m") is folded into the data where
    quantity arithmetic on patches folded it: always for PS and PSD, which
    square the amplitude, but for AS only when the extent carries units,
    which needs units on both the data and the transformed coordinates.
    """
    extent = _get_transformed_domain_extent(meta, dims)
    data_units = _get_dft_data_units(meta, dims, output)
    fft_units = dc.get_quantity(meta.attrs.data_units)
    scale = None if fft_units is None else fft_units.magnitude
    has_quantity = isinstance(extent, Quantity)
    extent = extent.magnitude if has_quantity else extent
    if (output == "AS" and not has_quantity) or scale == 1:
        scale = None
    divisor = extent * extent if output == "PS" else extent
    if db:
        data_units = _replace_data_units(meta.attrs, units.dB).data_units
    attrs = meta.attrs.update(
        data_type=DFT_OUTPUT_DATA_TYPE_MAP[output], data_units=data_units
    )
    plan = {
        "scale": scale,
        "square": output != "AS",
        "divisor": divisor,
        "db": (20 if output == "AS" else 10) if db else None,
    }
    return meta.new(attrs=attrs), plan


def _spectral_amplitude(data, xp, *, scale, square, divisor, db):
    """Return Fourier coefficients as amplitudes, powers or densities."""
    amp = xp.abs(data)
    if scale is not None:
        amp = amp * _scalar(scale, amp)
    out = (amp * amp if square else amp) / _scalar(divisor, amp)
    if db is None:
        return out
    out = out + _scalar(xp.finfo(out.dtype).eps, out)
    return db * xp.log10(out)


def _fft_input(data, xp, complex_input: bool):
    """
    Return data in the floating dtype an FFT takes, cast as numpy casts it.

    NumPy converts integers, and real data for a complex transform, itself;
    the standard refuses them. Complex data for a real transform are left
    for the backend to refuse.
    """
    if xp.isdtype(data.dtype, "complex floating"):
        return data
    single = data.dtype == xp.float32
    if complex_input:
        return xp.astype(data, xp.complex64 if single else xp.complex128)
    if xp.isdtype(data.dtype, "real floating"):
        return data
    return xp.astype(data, xp.float64)


def _scalar(value, like):
    """Return a number as an operand `like` takes, with numpy's promotion."""
    # A numpy scalar sets the result's dtype, as a 0-d array does, where a
    # python number would not; some backends refuse numpy scalars outright.
    return asarray_like(value, like) if isinstance(value, np.generic) else value


class Dft(PatchProcessor):
    """
    Perform the discrete Fourier transform (dft) on specified dimension(s).

    Parameters
    ----------
    dim
        Dimension or dimensions to transform. None transforms all dimensions.
    real
        Dimension for a real FFT, True for the last requested dimension, or
        None for complex FFTs along every dimension.
    pad
        Pad each transformed dimension to its next fast FFT length.
    output
        Spectral representation for each frequency bin:
        - ``'FFT'``: Complex Fourier coefficients scaled by sample spacing.
        - ``'AS'``: Amplitude spectrum in the original data units.
        - ``'PS'``: Power spectrum whose bin sum gives mean square.
        - ``'PSD'``: Spectral density whose bin-width-weighted sum gives
          mean square.
    db
        Convert non-FFT output to decibels without a reference value: use
        ``20 * log10`` for AS and ``10 * log10`` for PS or PSD.

    Notes
    -----
    NumPy FFT output is scaled by each transformed dimension's sample spacing.
    Frequency coordinates remain ordered, use reciprocal units, and are named
    with an ``ft_`` prefix (for example, ``time`` becomes ``ft_time``).

    A non-dimensional coordinate measured on one transformed dimension is
    removed from the output coordinates but retained for
    [idft](`dascore.Patch.idft`) to restore. One spanning multiple
    dimensions is dropped.

    FFT data units combine the original data and transformed-dimension units;
    other outputs are normalized as described under ``output``.

    With ``real=True``, AS, PS, and PSD do not double non-DC or non-Nyquist
    bins for a one-sided spectrum; multiply the applicable bins when needed.

    If every requested dimension is already transformed, ``dft`` returns the
    input unchanged regardless of ``output``. See the
    [FFT notes](`dascore/docs/notes/dft_notes.qmd`) for details.

    See Also
    --------
    - [idft](`dascore.Patch.idft`)
    - [stft](`dascore.Patch.stft`)

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> dft_time = patch.dft(dim="time")
    >>> dft_time_real = patch.dft(dim="time", real=True)
    >>> dft_some_real = patch.dft(dim=("time", "distance"), real="time")
    >>> psd = patch.dft(dim="time", real=True, output="PSD")
    """

    dim: Any
    real: Any = None
    pad: Any = True
    output: Any = "FFT"
    db: Any = False

    _positional_fields = ("dim",)

    def get_metadata(self, meta):
        """Return the transformed metadata, and the padding, axes and scales."""
        output = self.output
        output_type = output.upper()
        if output_type not in DFT_OUTPUT_TYPES:
            msg = f"Unknown output={output!r}. Expected one of: {DFT_OUTPUT_TYPES}."
            raise ValueError(msg)
        if output_type == "FFT" and self.db:
            msg = "db=True is only supported for output='AS', 'PS', or 'PSD'."
            raise ParameterError(msg)
        dim, real = self.dim, self.real
        dims = list(iterate(dim if dim is not None else meta.dims))
        meta.check_coords(coords=dims)
        real = dims[-1] if real is True else real  # if true grab last dim
        dims = _get_untransformed_dims(meta, dims)
        real = real if real in dims else None  # may need to reset real
        plan: dict[str, Any] = dict(pad_width=None, axes=(), real=False, step=1.0)
        plan |= dict(scale=None, square=False, divisor=None, db=None)
        if not dims:  # no transformation needed.
            return meta, plan
        # Check before padding so errors name dft rather than pad.
        for name in dims:
            if not len(meta.get_coord(name)):
                msg = f"dft cannot transform {name}; the dimension is empty."
                raise CoordError(msg)
            meta.get_coord(name, require_evenly_sampled=True)
        # re-arrange list so real dim is last (if provided)
        if isinstance(real, str):
            assert real in dims, "real must be in provided dimensions."
            dims.append(dims.pop(dims.index(real)))
        padded = meta
        if self.pad:  # apply padding to avoid slow dft lengths.
            pad = Pad(**{x: "fft" for x in dims})
            padded, pad_plan = pad.get_metadata(meta)
            plan["pad_width"] = pad_plan["pad_width"]
        # get axes and spacing along desired dimensions.
        dxs, axes = _get_dx_or_spacing_and_axes(
            padded, dims, require_evenly_spaced=True
        )
        original_cm = meta.coords if self.pad else None
        new_coords = _get_dft_new_coords(
            padded, dxs, dims, axes, real, original_cm=original_cm
        )
        attrs = _get_dft_attrs(
            padded, dims, new_coords, pad=self.pad, output=output_type
        )
        out = padded.new(coords=new_coords, attrs=attrs)
        axes = tuple(int(x) for x in axes)
        step = np.prod(dxs)
        plan |= {"axes": axes, "real": real is not None, "step": step}
        # Complex, then promoted by the step it is scaled by.
        dtype = _result_dtype(meta.dtype, np.complex64, step)
        if output_type == "FFT":
            return out.new(dtype=dtype), plan
        out, spectral = _spectral_amplitude_plan(out, output_type, dims, self.db)
        return out.new(dtype=_result_dtype(dtype, real=True)), plan | spectral

    def kernel(self, data, **plan):
        """Return the transform on the data's backend, cast as numpy would cast it."""
        xp = array_namespace(data)
        return _dft_kernel(data, xp, xp.fft, cast=True, **plan)

    def numpy_kernel(self, data, **plan):
        """As `kernel`, but numpy casts the data, as it always has."""
        return _dft_kernel(data, array_namespace(data), nft, cast=False, **plan)


def _dft_kernel(
    data, xp, fft, *, cast, pad_width, axes, real, step, divisor, **scaling
):
    """Return the scaled, centred transform, or a spectrum made from it."""
    if not axes:
        return data
    if pad_width is not None:
        data = _pad_array(data, pad_width)
    func = fft.rfftn if real else fft.fftn
    data = _fft_input(data, xp, not real) if cast else data
    # Scaled by the sample spacing (see the dft note), then centred.
    data = func(data, axes=axes) * _scalar(step, data)
    # The one-sided axis of a real transform is not centred.
    if shifted := (axes[:-1] if real else axes):
        data = fft.fftshift(data, axes=shifted)
    # No divisor: output="FFT", the coefficients as they are.
    if divisor is None:
        return data
    return _spectral_amplitude(data, xp, divisor=divisor, **scaling)


def _get_idft_dims_steps_axis(patch, dim):
    """
    Get the dimensions, step sizes as a float, axis numbers and if an
    irft should be performed.
    """
    ft = FourierTransformatter()
    if dim is None:
        dim = [x for x in patch.dims if x.startswith("ft_")]
    # try to get pre-transformed names if used. EG "time" might refer to
    # ft_time for brevity.
    current_dims = set(patch.dims)
    dims = [x if x in current_dims else ft.rename_dims(x)[0] for x in iterate(dim)]
    patch.check_coords(dims=dims)
    coords = [patch.get_coord(x, require_evenly_sampled=True) for x in dims]
    is_real = [1 if to_float(x.min()) == 0 else 0 for x in coords]
    real_sum = sum(is_real)
    assert real_sum <= 1, "only one real axis allowed."
    has_real = bool(real_sum)
    # we need to move the real dim to the end of the list
    if has_real:
        real_ind = is_real.index(1)
        dims.append(dims.pop(real_ind))
    steps, axis = _get_dx_or_spacing_and_axes(patch, dims)
    return dims, steps, axis, has_real


def _get_idft_coords_and_sizes(patch, dims, new_dims, axes, real):
    """Get the new coords for the idft and expected sizes to pass to numpy."""
    shapes = patch.shape
    padded = patch.attrs.get("_dft_padded", False)
    coord_map = patch.coords.disassociate_coord(*dims).get_coord_tuple_map()
    sizes = []
    padding = {}
    for old_dim, new_dim, ax in zip(dims, new_dims, axes):
        # if old dim is stored
        ax_len = shapes[ax]
        potential_coord = coord_map.get(new_dim, (None, None))[1]
        if potential_coord is None:
            msg = (
                "Currently, IDFT can only be performed on patches which have"
                " been transformed to Fourier domain with dft method."
            )
            raise NotImplementedError(msg)
        if (len(potential_coord) == ax_len) or (real and old_dim == dims[-1]):
            sizes.append(len(potential_coord))
        coord_map[new_dim] = (new_dim, potential_coord)
        # Put back the coordinates dft parked when it took the dim away.
        prefix = _associated_prefix(new_dim)
        for name in [x for x in coord_map if x.startswith(prefix)]:
            coord_map[name[len(prefix) :]] = (new_dim, coord_map.pop(name)[1])
        if not padded:  # No padding, go to next dim.
            continue
        old_len = len(coord_map.pop(f"_{new_dim}_unpadded")[1])
        diff = old_len - len(coord_map[new_dim][1])
        if diff < 0:
            padding[new_dim] = (0, diff)
    ft = FourierTransformatter()
    new_dims = ft.rename_dims(patch.dims, index=axes, forward=False)
    cm = get_coord_manager(coord_map, dims=new_dims).drop_coords(*dims)[0]
    out_size = np.asarray(sizes) if len(sizes) else None
    return cm, out_size, padding


def _get_idft_attrs(patch, dims, new_coords):
    """Get new attributes for transformed patch."""
    # add all {dim}_min to new coords to ensure reverse ft can restore dims.
    new = dict(patch.attrs)
    new.pop("coords", None)
    new["data_units"] = _get_data_units_from_dims(patch, dims, mul)
    # Restore the pre-dft datatype.
    if "_pre_dft_data_type" in new:
        new["data_type"] = new.pop("_pre_dft_data_type", None)
    new.pop("_dft_output", None)
    new.pop("_dft_padded", None)
    return PatchAttrs(**new)


def _check_dft_output_invertible(patch):
    """Raise if patch DFT data are not invertible Fourier coefficients."""
    output = patch.attrs.get("_dft_output", "FFT")
    if output != "FFT":
        msg = f"Only dft(output='FFT') can be inverted with idft, not {output!r}."
        raise ValueError(msg)


class Idft(PatchProcessor):
    """
    Perform the inverse discrete Fourier transform (idft) on specified dimension(s).

    Currently, only patches transformed with [dft](`dascore.Patch.dft`)
    can be inverted. After dft, the transformed coordinates must not change
    (e.g., with [select](`dascore.Patch.select`)), otherwise idft won't work.

    Parameters
    ----------
    dim
        A single, or multiple dimensions over which to perform idft. If
        None, perform idft over all dimensions that have names starting
        with "ft_", which indicates they have already undergone a fourier
        transform.

    Notes
    -----
    - Real transforms are determined by transformed coordinates which have
      no negative values.

    - Non-dimensional coordinates measured on a transformed dimension are
      restored with it, provided the patch still carries what
      [dft](`dascore.Patch.dft`) parked for them. One
      spanning more than one dimension is not parked, so it does not come
      back.

    - See the [FFT note](dascore.org/notes/fft_notes.html) in Notes section
      of DASCore's documentation.

    See Also
    --------
    - [dft](`dascore.Patch.dft`)
    - [istft](`dascore.Patch.istft`)

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # perform dft (fft) on time axis
    >>> dft_time = patch.dft(dim="time")
    >>> # get inverse dft, transformed axis are ascertained automatically
    >>> idft = dft_time.idft()
    """

    dim: Any = None

    def get_metadata(self, meta):
        """Return the restored metadata, and the axes, sizes and trim."""
        _check_dft_output_invertible(meta)
        dims, _steps, axes, real = _get_idft_dims_steps_axis(meta, self.dim)
        new_dims = FourierTransformatter().rename_dims(dims, forward=False)
        # Get new coords, fft sizes, and padding to remove.
        coords, sizes, padding = _get_idft_coords_and_sizes(
            meta, dims, new_dims, axes, real
        )
        step = np.prod([to_float(coords.coord_map[x].step) for x in new_dims])
        out = meta.new(attrs=_get_idft_attrs(meta, dims, coords), coords=coords)
        indexer = None
        if padding:
            coords, found = out.coords.select_indexers(samples=True, **padding)
            indexer = tuple(found.get(x, slice(None)) for x in out.dims)
            out = out.new(coords=coords)
        sizes = None if sizes is None else tuple(int(x) for x in sizes)
        axes = tuple(int(x) for x in axes)
        plan = {"axes": axes, "real": real, "step": step, "sizes": sizes}
        # Divided by the step, then made complex unless there is nothing to invert.
        dtype = _result_dtype(meta.dtype, step, *([np.complex64] if axes else []))
        out = out.new(dtype=_result_dtype(dtype, real=real))
        return out, plan | {"indexer": indexer}

    def kernel(self, data, *, axes, real, step, sizes, indexer):
        """Return the inverse transform, trimmed of the padding dft added."""
        xp = array_namespace(data)
        # now unshift data and undo scaling
        data = data / _scalar(step, data)
        if shifted := (axes[:-1] if real else axes):
            data = xp.fft.ifftshift(data, axes=shifted)
        func = xp.fft.irfftn if real else xp.fft.ifftn
        # Along no axes numpy hands the data back as they are.
        data = func(_fft_input(data, xp, True) if axes else data, s=sizes, axes=axes)
        return data if indexer is None else data[indexer]


def _resolve_nfft(nfft, coord, window_samples: int) -> int:
    """
    Return the FFT length in samples: the window's, or a longer one to pad to.

    A bare number is a sample count whatever `samples` said; a quantity is
    read through the coordinate.
    """
    if nfft is None:
        return window_samples
    if isinstance(nfft, Quantity | np.timedelta64):
        count = coord.get_sample_count(nfft)
    elif isinstance(nfft, int | np.integer):
        count = int(nfft)
    else:
        msg = f"nfft must be a whole number of samples or a quantity; got {nfft!r}."
        raise ParameterError(msg)
    if count < window_samples:
        msg = (
            f"nfft must be at least the window length; a {count} point FFT of "
            f"a {window_samples} sample window would drop data."
        )
        raise ParameterError(msg)
    return count


def _centre_phase(cycles, sizes, steps, dtype) -> np.ndarray:
    """
    Return the factor which scales the spectra and refers each to its centre.

    An FFT refers phase to the window's first sample; the window's time
    coordinate is its centre, so the spectrum is rotated to say the phase
    there, as scipy's `ShortTimeFFT` does. `cycles` holds each windowed
    dimension's frequencies in cycles per sample; the product of `steps` is
    the scale, for compatibility with dft.
    """
    ndim = len(sizes)
    factor = np.prod(steps).astype(dtype)
    for axis, per_sample, size in zip(range(-ndim, 0), cycles, sizes):
        shape = [1] * ndim
        shape[axis] = -1
        phase = np.exp(2j * np.pi * per_sample * (size // 2)).astype(dtype)
        factor = factor * phase.reshape(shape)
    return factor


def _as_is(tiles: np.ndarray) -> np.ndarray:
    """The stack, untouched: `stft` transforms it once it has coordinates."""
    return tiles


def _swap_window_axes(data: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """
    Swap a stack's last axes, one per windowed dimension, with `axes`.

    A stack keeps the samples within a tile as its last axes; an stft keeps
    the frequencies where the transformed dimensions were and the window
    centres last. The swap is its own inverse.
    """
    tail = tuple(range(-len(axes), 0))
    return np.moveaxis(data, (*axes, *tail), (*tail, *axes))


class Stft(PatchProcessor):
    """
    Perform a short-time fourier transform.

    Parameters
    ----------
    taper_window
        The taper each window is multiplied by before its transform: a name,
        an array, or a ``(name, parameter)`` tuple `get_window` knows, for
        every windowed dimension, or a list with one per dimension.
    overlap
        The overlap between windows. Can be a number (assumed to be in units of
        the transformed dimension if `samples`==False), a percent, or None for
        0 overlap.
    samples
        If True, the window length (provided in kwargs) and overlap parameters
        are in samples (or explicit units).
    detrend
        If True, detrend each time window before performing fourier transform.
        This can lead to nicer looking spectrograms, but means the istft is
        no longer possible.
    nfft
        The length of the FFT taken of each window, in samples, or as a
        quantity or timedelta in the transformed dimension's units; a mapping
        gives each dimension its own. None, the default, is the window
        length. A longer FFT zero pads each window, which samples the same
        spectrum at more, closer frequencies; it adds no resolution, since
        the window holds no more data. Must be at least the window length.
    **kwargs
        The dimensions to window and the window along each, in coordinate
        units or, with `samples`, in samples. More than one dimension gives
        each window's spectrum over all of them: ``distance=32, time=64``
        is a local frequency-wavenumber spectrum.

    Examples
    --------
    >>> from scipy.signal import get_window
    >>> import dascore as dc
    >>> from dascore.units import second, percent
    >>> patch = dc.get_example_patch("chirp", channel_count=2)
    >>>
    >>> # Simple stft with 10 second window and 4 seconds overlap
    >>> pa1 = patch.stft(time=10*second, overlap=4*second)
    >>>
    >>> # Same as above, but using a boxcar window and 10% overlap.
    >>> pa2 = patch.stft(time=10*second, taper_window="boxcar", overlap=10*percent)
    >>>
    >>> # Using a custom window array and specifying window/overlap in samples.
    >>> window = get_window(("tukey", 0.1), 1000)
    >>> pa2 = patch.stft(time=1000, taper_window=window, overlap=100, samples=True)
    >>>
    >>> # Zero pad each 1000 sample window to a 4096 point FFT.
    >>> pa3 = patch.stft(time=1000, samples=True, nfft=4096)
    >>>
    >>> # Local f-k spectra: windows of 32 channels by 64 samples.
    >>> fk = dc.get_example_patch().stft(distance=32, time=64, samples=True)

    Notes
    -----
    - The output is scaled the same as [Patch.dft](`dascore.Patch.dft`).
      For a given sliding window, Parseval's theorem doesn't hold exactly
      (unless a boxcar window is used) because the taper window changes the time
      series signal before the transformation.
    - An array passed for taper_window must have as many samples as the
      window; one of another length is refused. To zero pad each window's
      FFT, give `nfft`.
    - Real data is transformed one-sided along the last windowed dimension
      and centred along the others, as [Patch.dft](`dascore.Patch.dft`) with
      ``real=True`` is; complex data is centred along every one.
    - Data that single precision holds (eg float32, complex64, int16) gives
      a complex64 output; wider data gives complex128.
    - The output is a stack of windows as
      [Patch.tile_apply](`dascore.Patch.tile_apply`) makes one, transformed
      along the window: the transformed dimension becomes the window
      centres, ``{dim}_start`` and ``{dim}_stop`` give physical cell edges
      in the dimension's units, the frequencies sit where the dimension was
      and the centres come last. Private ``_tile_index_{dim}`` coordinates
      retain sample indices, and the coordinates the stack carries for
      [Patch.reassemble](`dascore.Patch.reassemble`) are what
      [Patch.istft](`dascore.Patch.istft`) blends the windows back with.
      Non-dimensional coordinates along the transformed dimension travel with
      the stack and come back on the inverse.

    See Also
    --------
    [Patch.dft](`dascore.Patch.dft`), [Patch.istft](`dascore.Patch.istft`)
    """

    __version__ = "2.2"

    taper_window: Any = "hann"
    overlap: Any = Field(default_factory=lambda: _HALF_OVERLAP)
    samples: Any = False
    detrend: Any = False
    nfft: Any = None

    model_config = ConfigDict(extra="allow")
    data_type = "fourier_transform"

    def _tiler(self, **kwargs) -> TileApply:
        """Return the tile_apply which cuts the windows, bare or under the taper."""
        # A detrended window is tapered after the trend is removed, so the
        # stack is cut bare and the taper goes on in the kernel; an invertible
        # one is cut under the taper, which the stack then carries for istft.
        analysis = None if self.detrend else self.taper_window
        return TileApply(function=_as_is, mode="stack", analysis=analysis, **kwargs)

    def get_metadata(self, meta):
        """Return the stack's spectra metadata, and the windows and frequencies."""
        nfft = self.nfft
        resolved = resolve_window(
            meta,
            self.model_extra or {},
            samples=self.samples,
            overlap=self.overlap,
            enforce_lt_coord=True,
        )
        # In the patch's axis order, whatever order they were named in: the
        # stack's window axes come out in that order.
        order = np.argsort(resolved.axes)
        dims = tuple(resolved.dims[i] for i in order)
        sizes = tuple(resolved.size[i] for i in order)
        coords = [meta.get_coord(dim) for dim in dims]
        # No overlap given means none: the windows abut.
        strides = resolved.stride
        hops = sizes if strides is None else tuple(strides[i] for i in order)
        if isinstance(nfft, Mapping) and (extra := set(nfft) - set(dims)):
            names = sorted(map(str, extra))
            msg = f"nfft names dimensions which are not windowed: {names}."
            raise ParameterError(msg)
        nffts = tuple(
            _resolve_nfft(nfft.get(d) if isinstance(nfft, Mapping) else nfft, c, z)
            for d, c, z in zip(dims, coords, sizes)
        )
        real = not _is_complex(meta.dtype)
        overlap = {d: z - h for d, z, h in zip(dims, sizes, hops)}
        tiler = self._tiler(overlap=overlap, samples=True, **dict(zip(dims, sizes)))
        stack, plan = tiler.get_metadata(meta)
        steps = [to_float(coord.step) for coord in coords]
        # Real data is transformed one-sided along the last windowed dimension
        # and centred along the others, as dft does; complex data centred along
        # every one.
        freqs = [nft.fftshift(nft.fftfreq(n, d=step)) for n, step in zip(nffts, steps)]
        if real:
            freqs[-1] = nft.rfftfreq(nffts[-1], d=steps[-1])
        ft_dims = FourierTransformatter().rename_dims(dims)
        new_dims = (
            *(ft_dims[dims.index(d)] if d in dims else d for d in meta.dims),
            *dims,
        )
        coord_map = stack.coords.get_coord_tuple_map()
        for dim, ft_dim, values, coord in zip(dims, ft_dims, freqs, coords):
            coord_map.pop(f"{dim}_offset")
            freq_coord = get_coord(data=values, units=invert_quantity(coord.units))
            coord_map[ft_dim] = ((ft_dim,), freq_coord)
        cm = get_coord_manager(coords=coord_map, dims=new_dims)
        attrs = stack.attrs.update(
            _stft_detrended=self.detrend,
            _stft_real=dims[-1] if real else None,
            _pre_stft_data_type=meta.attrs.get("data_type"),
            data_units=_get_data_units_from_dims(meta, dims, mul),
            **{f"_stft_mfft_{dim}": n for dim, n in zip(dims, nffts)},
        )
        cycles = [values * step for values, step in zip(freqs, steps)]
        plan |= {"nffts": nffts, "steps": steps, "cycles": cycles, "real": real}
        # scipy's detrend computes in double unless the data are single or double.
        tiles = meta.dtype
        if self.detrend and _as_numpy_dtype(tiles).char not in "fdFD":
            tiles = np.dtype(np.float64)
        dtype = _result_dtype(tiles, np.complex64, like=meta.dtype)
        return meta.new(coords=cm, attrs=attrs, dtype=dtype), plan

    def numpy_kernel(self, data, *, axes, size, stride, nffts, steps, cycles, real):
        """Return each window's spectrum, its phase referred to the centre."""
        cut = TileApply.kernel_for("numpy")
        tiles = cut(self._tiler(), data, axes=axes, size=size, stride=stride)
        tail = tuple(range(-len(axes), 0))
        if self.detrend:
            for axis in tail:
                tiles = sp_detrend(tiles, axis=axis, type="linear")
            tiles = tiles * _taper_like(get_window_nd(self.taper_window, size), tiles)
        # scipy's transforms compute single precision data in single
        # precision; numpy's use double internally, several times the memory.
        fft = sft.rfftn if real else sft.fftn
        spectra = fft(tiles, s=nffts, axes=tail)
        centred = tail[:-1] if real else tail
        if centred:
            spectra = nft.fftshift(spectra, axes=centred)
        spectra *= _centre_phase(cycles, size, steps, spectra.dtype)
        return _swap_window_axes(spectra, axes)


class Istft(PatchProcessor):
    """
    Invert a short-time fourier transform.

    The patch must be one returned by [stft](`dascore.Patch.stft`).

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.units import second
    >>> patch = dc.get_example_patch("chirp")
    >>>
    >>> # Simple stft with 10 second window and 4 seconds overlap
    >>> pa1 = patch.stft(time=10*second, overlap=4*second)
    >>> pa2 = pa1.istft()
    >>> assert pa2.equals(patch, close=True)

    Notes
    -----
    - Each window's spectrum is inverted and the windows are blended back
      by [Patch.reassemble](`dascore.Patch.reassemble`), under the dual of
      the taper they were cut with, so the coordinates the stft carried
      come back with them -- those along the transformed dimension included.
    - Coordinates associated with the frequency dimension the stft created
      are dropped, since it does not survive the inverse.
      [idft](`dascore.Patch.idft`) behaves the same way.

    See Also
    --------
    [Patch.stft](`dascore.Patch.stft`), [Patch.idft](`dascore.Patch.idft`)
    """

    __version__ = "2.1"

    def check(self, patch):
        """Refuse a patch stft did not make, or made past inverting."""
        out = super().check(patch)
        _stft_dims(patch)
        return out

    def get_metadata(self, meta):
        """Return the reassembled metadata, and how to invert each window."""
        coord_map = meta.coords.get_coord_tuple_map()
        dims = _stft_dims(meta)
        # The dimension transformed one-sided, if any, is last, as it was cut.
        real = meta.attrs["_stft_real"]
        dims = [d for d in dims if d != real] + ([real] if real else [])
        windows = [f"_tile_analysis_{dim}" for dim in dims]
        ft_dims = FourierTransformatter().rename_dims(dims)
        sizes = [len(coord_map[name][1]) for name in windows]
        nffts = [int(meta.attrs[f"_stft_mfft_{dim}"]) for dim in dims]
        steps = [to_float(coord_map[f"_tile_source_{dim}"][1].step) for dim in dims]
        cycles = [
            meta.get_coord(ft_dim).values * step for ft_dim, step in zip(ft_dims, steps)
        ]
        # The window centres go last, then the frequencies swap with them, so
        # the windows' samples are the last axes whatever order the patch is in.
        base_dims = [d for d in meta.dims if d not in dims]
        offsets = [f"{dim}_offset" for dim in dims]
        # The frequency axes do not survive, nor does anything riding on them.
        for name, cdims in meta.coords.dim_map.items():
            if set(cdims) & set(ft_dims):
                coord_map.pop(name)
        for offset, size in zip(offsets, sizes):
            coord_map[offset] = ((offset,), get_coord(data=np.arange(size)))
        renamed = (dims[ft_dims.index(d)] if d in ft_dims else d for d in base_dims)
        stack_dims = (*renamed, *offsets)
        data_type = meta.attrs.get("_pre_stft_data_type")
        private = ("_stft", "_pre_stft")
        attrs = {k: v for k, v in dict(meta.attrs).items() if not k.startswith(private)}
        stack = meta.new(
            coords=get_coord_manager(coords=coord_map, dims=stack_dims),
            attrs=dc.PatchAttrs(**attrs),
        )
        out, plan = Reassemble().get_metadata(stack)
        out = out.update_attrs(
            data_type=data_type or "",
            data_units=_get_data_units_from_dims(meta, dims, truediv),
        )
        dtype = _result_dtype(meta.dtype, np.complex64)
        out = out.new(dtype=_result_dtype(dtype, real=bool(real)))
        return out, plan | {
            "centres": tuple(meta.get_axis(dim) for dim in dims),
            "swap": tuple(base_dims.index(ft_dim) for ft_dim in ft_dims),
            "sizes": sizes,
            "nffts": nffts,
            "steps": steps,
            "cycles": cycles,
            "real": bool(real),
        }

    def numpy_kernel(
        self, data, *, centres, swap, sizes, nffts, steps, cycles, real, **tiles
    ):
        """Return each window's inverse, blended back by reassemble."""
        tail = tuple(range(-len(centres), 0))
        spectra = _swap_window_axes(np.moveaxis(data, centres, tail), swap)
        spectra = spectra / _centre_phase(cycles, sizes, steps, spectra.dtype)
        centred = tail[:-1] if real else tail
        if centred:
            spectra = nft.ifftshift(spectra, axes=centred)
        ifft = nft.irfftn if real else nft.ifftn
        stack = ifft(spectra, s=nffts, axes=tail)
        # The FFT was zero padded past the window; the window is its first samples.
        stack = stack[(..., *(slice(0, size) for size in sizes))]
        return Reassemble.kernel_for("numpy")(Reassemble(), stack, **tiles)


def _stft_dims(patch) -> list[str]:
    """Return the dimensions an invertible stft windowed, or say why there are none."""
    coord_map = patch.coords.coord_map
    dims = [d for d in patch.dims if f"_tile_source_{d}" in coord_map]
    if not dims or "_stft_real" not in dict(patch.attrs):
        msg = (
            "Inverse short time fourier transform requires a patch that has"
            " undergone stft but this patch is missing required attrs. "
        )
        raise PatchError(msg)
    windows = [f"_tile_analysis_{dim}" for dim in dims]
    if patch.attrs["_stft_detrended"] or any(w not in coord_map for w in windows):
        # The patch itself, as the message has always shown it.
        msg = f"Inverse stft not possible for patch {patch}."
        raise PatchError(msg)
    return dims


def _is_complex(dtype) -> bool:
    """Whether a dtype, numpy's or another backend's, is complex."""
    return str(dtype).rsplit(".", 1)[-1].startswith("complex")
