"""
Module for Fourier transforms.

See the [FFT note](/notes/dft_notes.qmd) for discussion on the
implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from operator import mul, truediv
from typing import Any, Literal

import numpy as np
import numpy.fft as nft

import dascore as dc
from dascore import units
from dascore.compat import ndarray
from dascore.constants import PatchType
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import ParameterError, PatchError
from dascore.units import Quantity, invert_quantity, percent
from dascore.utils.imports import lazy_import
from dascore.utils.misc import iterate
from dascore.utils.patch import (
    _get_data_units_from_dims,
    _get_dx_or_spacing_and_axes,
    patch_function,
)
from dascore.utils.signal import get_window
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


def _associated_prefix(dim: str) -> str:
    """Where dft parks the coordinates measured on a dimension it takes."""
    # Private, like the unpadded coordinate parked beside them, and named
    # after the dimension so idft knows what to put each one back on.
    return f"_{dim}_associated_"


def _get_dft_coord_units(units):
    """Get units for DFT coordinates."""
    new_units = invert_quantity(units)
    # This purposefully converts 1/s to Hz to be more conventional. See #693.
    if new_units == dc.get_quantity("1/s"):
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


def _convert_dft_spectral_amplitudes(patch, output, dims, real, db):
    """Convert the FFT output to spectral amplitude representations."""
    amp = patch.abs()
    extent = _get_transformed_domain_extent(amp, dims)
    data_units = _get_dft_data_units(patch, dims, output)
    if output == "AS":
        # Convert DASCore's dx-scaled Fourier coefficients to harmonic
        # amplitude.
        out = amp / extent
    elif output == "PS":
        # PS bins sum to mean square.
        out = amp * amp / (extent * extent)
    elif output == "PSD":
        # PSD bins integrate to mean square when multiplied by frequency-bin
        # volume, which is the reciprocal of the transformed-domain extent.
        out = amp * amp / extent
    if db:
        out = out + np.finfo(out.data.dtype).eps
        db_scale = 20 if output == "AS" else 10
        out = db_scale * out.log10()
        out = out.set_units(units.dB)
        data_units = out.attrs.data_units

    return out.update_attrs(
        data_type=DFT_OUTPUT_DATA_TYPE_MAP[output],
        data_units=data_units,
    )


@patch_function()
def dft(
    patch: PatchType,
    dim: str | Sequence[str] | None,
    *,
    real: str | bool | None = None,
    pad: bool = True,
    output: Literal["FFT", "PSD", "PS", "AS"] = "FFT",
    db: bool = False,
) -> PatchType:
    """
    Perform the discrete Fourier transform (dft) on specified dimension(s).

    Parameters
    ----------
    patch
        Patch to transform.
    dim
        A single, or multiple dimensions over which to perform dft. If
        None, perform dft over all dimensions.
    real
        Either 1) The name of the axis over which to perform a rfft, 2)
        True, which means the last (possibly only) dimension should have an
        rfft performed, or 3) None, meaning no rfft.
    pad
        If True, pad patch before performing dft along desired dimensions to
        the next fast length. This can avoid major slow-downs when dimension
        lengths are prime numbers.
    output
        Spectral representation to return for each frequency bin
        - ``'FFT'``: Complex Fourier coefficients scaled by sample spacing.
        - ``'AS'``: Amplitude spectrum in the original data units.
        - ``'PS'``: Power spectrum whose bin sum gives mean square.
        - ``'PSD'``: Spectral density whose bin-width-weighted sum gives
                     mean square.
    db
        If True, converts the output into decibel units, if output is not FFT.
        This applies ``20 * log10`` to ``'AS'`` and ``10 * log10`` to ``'PS'``
        or ``'PSD'`` without a reference value.


    Notes
    -----
    - Simply uses numpy's fft module but outputs are scaled by the sample
      spacing along each transformed dimension and coordinates corresponding
      to frequency bins are shifted so they remain ordered.

    - Each transformed dimension is renamed with a preceding `ft_`. e.g.,
      `time` becomes `ft_time` (ft stands for fourier transform).

    - Each transformed dimension has units of 1/original units.

    - A non-dimensional coordinate measured on a transformed dimension is
      not a coordinate of the output -- the frequency axis is not what it
      was measured on, and a real transform is not even the same length --
      but it is kept for [idft](`dascore.transform.fourier.idft`) to
      restore. One spanning more than one dimension is dropped.

    - For ``output='FFT'``, output data units are the original data units
      multiplied by the units of each transformed dimension. Other output
      types are normalized as described in the ``output`` parameter.

    - For ``output='AS'``, ``'PS'``, or ``'PSD'`` with ``real=True``, the
      non-DC and non-Nyquist bins have not been converted to one-sided spectra.
      Depending on your use case, you may need to multiply non-zero bins by 2.

    - If all requested dimensions are already transformed, ``dft`` returns
      the input patch unchanged, regardless of the requested ``output``.

    - Non-dimensional coordinates associated with transformed coordinates
      will be dropped in the output.

    - See the [FFT notes](`docs/notes/dft_notes.qmd`) for more details.

    See Also
    --------
    - [idft](`dascore.transform.fourier.idft`)
    - [stft](`dascore.transform.fourier.stft`)

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # perform dft (fft) on time axis
    >>> dft_time = patch.dft(dim="time")
    >>> # make it a real fft (no negative frequencies)
    >>> dft_time_real = patch.dft(dim="time", real=True)
    >>> # dft on specified dimensions, specify real dimension
    >>> dft_some_real = patch.dft(dim=("time", "distance"), real="time")
    >>> # calculate a power spectral density along time
    >>> psd = patch.dft(dim="time", real=True, output="PSD")
    """
    output_type = output.upper()
    if output_type not in DFT_OUTPUT_TYPES:
        msg = f"Unknown output={output!r}. Expected one of: {DFT_OUTPUT_TYPES}."
        raise ValueError(msg)
    if output_type == "FFT" and db:
        msg = "db=True is only supported for output='AS', 'PS', or 'PSD'."
        raise ParameterError(msg)

    dims = list(iterate(dim if dim is not None else patch.dims))
    patch.check_coords(coords=dims)
    real = dims[-1] if real is True else real  # if true grab last dim
    dims = _get_untransformed_dims(patch, dims)
    real = real if real in dims else None  # may need to reset real
    if not dims:  # no transformation needed.
        return patch
    # re-arrange list so real dim is last (if provided)
    if isinstance(real, str):
        assert real in dims, "real must be in provided dimensions."
        dims.append(dims.pop(dims.index(real)))
    original_cm = patch.coords if pad else None
    if pad:  # apply padding to avoid slow dft lengths.
        pad_kwargs = {x: "fft" for x in dims}
        patch = patch.pad.func(patch, **pad_kwargs)
    # get axes and spacing along desired dimensions.
    dxs, axes = _get_dx_or_spacing_and_axes(patch, dims, require_evenly_spaced=True)
    # get new coordinates (need before pad)
    new_coords = _get_dft_new_coords(
        patch, dxs, dims, axes, real, original_cm=original_cm
    )
    func = nft.rfftn if real is not None else nft.fftn
    # scale as explained above and in notes, then shift
    scale_factor = np.prod(dxs)
    fft_data = func(patch.data, axes=axes) * scale_factor
    shift_slice = slice(None) if real is None else slice(None, -1)
    data = nft.fftshift(fft_data, axes=axes[shift_slice])
    # get attributes
    attrs = _get_dft_attrs(patch, dims, new_coords, pad=pad, output=output_type)
    patch_out = patch.new(data=data, coords=new_coords, attrs=attrs)

    if output_type != "FFT":
        patch_out = _convert_dft_spectral_amplitudes(
            patch_out, output_type, dims, real, db
        )

    return patch_out


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


@patch_function()
def idft(patch: PatchType, dim: str | Sequence[str] | None = None) -> PatchType:
    """
    Perform the inverse discrete Fourier transform (idft) on specified dimension(s).

    Currently, only patches that have been transformed with
    [dft](`dascore.transform.fourier.dft`) can be used with this function.
    After transformation with dft, the transformed coordinates cannot change
    (e.g., with [select]('dascore.proc.basic.select`) otherwise idft won't
    work.

    Parameters
    ----------
    patch
        Patch to transform.
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
      [dft](`dascore.transform.fourier.dft`) parked for them. One
      spanning more than one dimension is not parked, so it does not come
      back.

    - See the [FFT note](dascore.org/notes/fft_notes.html) in Notes section
      of DASCore's documentation.

    See Also
    --------
    - [dft](`dascore.transform.fourier.dft`)
    - [istft](`dascore.transform.fourier.istft`)

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # perform dft (fft) on time axis
    >>> dft_time = patch.dft(dim="time")
    >>> # get inverse dft, transformed axis are ascertained automatically
    >>> idft = dft_time.idft()
    """
    _check_dft_output_invertible(patch)
    dims, _steps, axes, real = _get_idft_dims_steps_axis(patch, dim)
    new_dims = FourierTransformatter().rename_dims(dims, forward=False)
    func = nft.irfftn if real else nft.ifftn
    # Get new coords, fft sizes, and padding to remove.
    coords, sizes, padding = _get_idft_coords_and_sizes(
        patch, dims, new_dims, axes, real
    )
    # now unshift data and undo scaling
    ax_slice = slice(None, -1) if real else slice(None)
    scale_factor = np.prod([to_float(coords.coord_map[x].step) for x in new_dims])
    _prepped = nft.ifftshift(patch.data / scale_factor, axes=axes[ax_slice])
    data = func(_prepped, s=sizes, axes=axes)
    attrs = _get_idft_attrs(patch, dims, coords)
    out = patch.new(data=data, attrs=attrs, coords=coords)
    if padding:
        out = out.select(**padding, samples=True)
    return out


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


def _centre_phase(cycles: np.ndarray, size: int) -> np.ndarray:
    """
    Return the phase which refers each window's spectrum to its centre sample.

    An FFT refers phase to the window's first sample; the window's time
    coordinate is its centre, so the spectrum is rotated to say the phase
    there, as scipy's `ShortTimeFFT` does. `cycles` is each frequency in
    cycles per sample.
    """
    return np.exp(2j * np.pi * cycles * (size // 2)).astype(np.complex64)


def _as_is(tiles: np.ndarray) -> np.ndarray:
    """The stack, untouched: `stft` transforms it once it has coordinates."""
    return tiles


def _swap_window_axes(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Move a stack's last axis to `axis` and what was there to the end.

    A stack keeps the windows within a tile as its last axis; an stft keeps
    the frequencies where the transformed dimension was and the window
    centres last. The swap is its own inverse.
    """
    return np.moveaxis(data, (axis, -1), (-1, axis))


@patch_function(data_type="fourier_transform")
def stft(
    patch: PatchType,
    taper_window: str | ndarray | tuple[str | Any, ...] = "hann",
    overlap: Quantity | int | None = 50 * percent,
    samples: bool = False,
    detrend: bool = False,
    nfft: int | Quantity | np.timedelta64 | None = None,
    **kwargs,
):
    """
    Perform a short-time fourier transform.

    Parameters
    ----------
    patch
        The patch to transform.
    taper_window
        Parameter controlling the tapering of each time window before
        fourier transform. Can either be the name of the window to use,
        or an array, or a tuple of name and parameters passed to scipy.signal's
        get_window function.
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
        quantity or timedelta in the transformed dimension's units. None, the default,
        is the window length. A longer FFT zero pads each window, which
        samples the same spectrum at more, closer frequencies; it adds no
        resolution, since the window holds no more data. Must be at least
        the window length.
    **kwargs
        Used to specify window length in data units, percent, or samples.

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

    Notes
    -----
    - The output is scaled the same as [Patch.dft](`dascore.Patch.dft`).
      For a given sliding window, Parseval's theorem doesn't hold exactly
      (unless a boxcar window is used) because the taper window changes the time
      series signal before the transformation.
    - An array passed for taper_window must have as many samples as the
      window; one of another length is refused. To zero pad each window's
      FFT, give `nfft`.
    - The output is a stack of windows as
      [Patch.tile_apply](`dascore.Patch.tile_apply`) makes one, transformed
      along the window: the transformed dimension becomes the window
      centres, ``{dim}_start`` and ``{dim}_stop`` say where each window came
      from in samples, and the coordinates the stack carries for
      [Patch.reassemble](`dascore.Patch.reassemble`) are what
      [Patch.istft](`dascore.Patch.istft`) blends the windows back with.
      Non-dimensional coordinates along the transformed dimension travel with
      the stack and come back on the inverse.

    See Also
    --------
    [Patch.dft](`dascore.Patch.dft`), [Patch.istft](`dascore.Patch.istft`)
    """
    from dascore.proc.tile_apply import TileApply  # noqa: PLC0415

    resolved = resolve_window(
        patch,
        kwargs,
        samples=samples,
        overlap=overlap,
        allow_multiple=False,
        enforce_lt_coord=True,
    )
    dim, axis, size = resolved.dims[0], resolved.axes[0], resolved.size[0]
    coord = patch.get_coord(dim)
    # No overlap given means none: the windows abut.
    hop = size if resolved.stride is None else resolved.stride[0]
    window = get_window(taper_window, size)
    nfft = _resolve_nfft(nfft, coord, size)
    fft_mode = "onesided" if np.isrealobj(patch.data) else "centered"
    # A detrended window is tapered after the trend is removed, so the
    # stack is cut bare and the taper goes on here; an invertible one is
    # cut under the taper, which the stack then carries for istft.
    settings: dict[str, Any] = {
        "function": _as_is,
        "mode": "stack",
        "analysis": None if detrend else window,
        "overlap": size - hop,
        "samples": True,
        dim: size,
    }
    stack = TileApply(**settings)._apply(patch)
    tiles = stack.data
    if detrend:
        tiles = sp_detrend(tiles, axis=-1, type="linear") * window
    step = to_float(coord.step)
    if fft_mode == "onesided":
        spectra = nft.rfft(tiles, n=nfft, axis=-1)
        freqs = nft.rfftfreq(nfft, d=step)
    else:
        spectra = nft.fftshift(nft.fft(tiles, n=nfft, axis=-1), axes=-1)
        freqs = nft.fftshift(nft.fftfreq(nfft, d=step))
    # One pass: the phase and, for compatibility with dft, the scale by step.
    spectra *= _centre_phase(freqs * step, size) * spectra.dtype.type(step)
    ft_dim = FourierTransformatter().rename_dims(patch.dims, index=axis)[axis]
    new_dims = (*patch.dims[:axis], ft_dim, *patch.dims[axis + 1 :], dim)
    coord_map = stack.coords.get_coord_tuple_map()
    coord_map.pop(f"{dim}_offset")
    freq_coord = get_coord(data=freqs, units=invert_quantity(coord.units))
    coord_map[ft_dim] = ((ft_dim,), freq_coord)
    cm = get_coord_manager(coords=coord_map, dims=new_dims)
    attrs = stack.attrs.update(
        _stft_detrended=detrend,
        _stft_fft_mode=fft_mode,
        _stft_mfft=nfft,
        _pre_stft_data_type=patch.attrs.get("data_type"),
        data_units=_get_data_units_from_dims(patch, dim, mul),
    )
    return patch.new(data=_swap_window_axes(spectra, axis), coords=cm, attrs=attrs)


@patch_function()
def istft(patch) -> dc.Patch:
    """
    Invert a short-time fourier transform.

    Parameters
    ----------
    patch
        A patch return from [stft](`dascore.transform.fourier.stft`).

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
    from dascore.proc.tile_apply import reassemble  # noqa: PLC0415

    coord_map = patch.coords.get_coord_tuple_map()
    dims = [d for d in patch.dims if f"_tile_source_{d}" in coord_map]
    if len(dims) != 1 or "_stft_mfft" not in dict(patch.attrs):
        msg = (
            "Inverse short time fourier transform requires a patch that has"
            " undergone stft but this patch is missing required attrs. "
        )
        raise PatchError(msg)
    if patch.attrs["_stft_detrended"] or f"_tile_analysis_{dims[0]}" not in coord_map:
        msg = f"Inverse stft not possible for patch {patch}."
        raise PatchError(msg)
    dim = dims[0]
    ft_dim = FourierTransformatter().rename_dims([dim])[0]
    axis = patch.get_axis(ft_dim)
    source = coord_map[f"_tile_source_{dim}"][1]
    size = len(coord_map[f"_tile_analysis_{dim}"][1])
    nfft = int(patch.attrs["_stft_mfft"])
    step = to_float(source.step)
    cycles = patch.get_coord(ft_dim).values * step
    spectra = _swap_window_axes(patch.data, axis)
    spectra = spectra / (_centre_phase(cycles, size) * step)
    if patch.attrs["_stft_fft_mode"] == "onesided":
        tiles = nft.irfft(spectra, n=nfft, axis=-1)
    else:
        tiles = nft.ifft(nft.ifftshift(spectra, axes=-1), n=nfft, axis=-1)
    # The FFT was zero padded past the window; the window is its first samples.
    tiles = tiles[..., :size]
    offset = f"{dim}_offset"
    # The frequency axis does not survive, nor does anything riding on it.
    for name, cdims in patch.coords.dim_map.items():
        if ft_dim in cdims:
            coord_map.pop(name)
    coord_map[offset] = ((offset,), get_coord(data=np.arange(size)))
    others = [d for d in patch.dims[axis + 1 :] if d != dim]
    stack_dims = (*patch.dims[:axis], dim, *others, offset)
    attrs = {k: v for k, v in dict(patch.attrs).items() if not k.startswith("_stft")}
    stack = patch.new(
        data=tiles,
        coords=get_coord_manager(coords=coord_map, dims=stack_dims),
        attrs=dc.PatchAttrs(**attrs),
    )
    out = reassemble.func(stack)
    new_attrs = dict(out.attrs)
    if "_pre_stft_data_type" in new_attrs:
        new_attrs["data_type"] = new_attrs.pop("_pre_stft_data_type")
    new_attrs["data_units"] = _get_data_units_from_dims(patch, dim, truediv)
    return out.update_attrs(**new_attrs)
