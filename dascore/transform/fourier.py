"""
Module for Fourier transforms.

See the [FFT note](/notes/dft_notes.qmd) for discussion on the
implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from operator import mul, truediv
from typing import Any

import numpy as np
import numpy.fft as nft
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import get_window

import dascore as dc
from dascore.compat import ndarray
from dascore.constants import PatchType
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import PatchError
from dascore.units import Quantity, invert_quantity, percent
from dascore.utils.misc import broadcast_for_index, iterate
from dascore.utils.patch import (
    _get_data_units_from_dims,
    _get_dx_or_spacing_and_axes,
    get_dim_axis_value,
    patch_function,
)
from dascore.utils.time import is_datetime64, is_timedelta64, to_float
from dascore.utils.transformatter import FourierTransformatter


def _get_dft_new_coords(patch, dxs, dims, axes, real, original_cm=None):
    """
    Create coordinates based on dxs and patch shape.

    if original_cm is not none, it means the patch was padded.
    """
    # Note: We need original_cm and patch because patch may have undergone
    # padding.

    def _get_fft_coord(x_len, dx, units):
        """Get coord for normal fft coord."""
        new_dx = 1.0 / (x_len * dx)
        stop = ((x_len - 1) // 2 + 1) * new_dx
        start = -(x_len // 2) * new_dx
        units = invert_quantity(units)
        return get_coord(start=start, stop=stop, step=new_dx, units=units)

    def _get_rfft_coord(x_len, dx, units):
        """Get coord from real fft coord."""
        new_dx = 1.0 / (x_len * dx)
        start = 0
        stop = (x_len // 2 + 1) * new_dx
        units = invert_quantity(units)
        return get_coord(start=start, stop=stop, step=new_dx, units=units)

    # first disassociate old coordinates. We do this rather than drop them
    # so the idft can find them and exactly restore old coords.
    old_cm = patch.coords.disassociate_coord(*dims)
    new_coords = old_cm.get_coord_tuple_map()
    ft = FourierTransformatter()
    for i, dim in enumerate(dims):
        old_coord = patch.get_coord(dim)
        units = old_coord.units
        size = old_coord.shape[0]
        dx = dxs[i]
        new_name = ft.rename_dims(dim)[0]
        if dim == real:
            coord = _get_rfft_coord(size, dx, units)
        else:
            coord = _get_fft_coord(size, dx, units)
        new_coords[new_name] = (new_name, coord)
        # Add padded coordinates
        if original_cm is not None:
            new_coords[f"_{dim}_unpadded"] = (None, original_cm.get_coord(dim))
    new_dims = ft.rename_dims(patch.dims, index=axes)
    cm = get_coord_manager(new_coords, dims=new_dims)
    return cm


def _get_dft_attrs(patch, dims, new_coords, pad=False):
    """Get new attributes for transformed patch."""
    new = dict(patch.attrs)
    new["dims"] = new_coords.dims
    new["data_units"] = _get_data_units_from_dims(patch, dims, mul)
    # As per #390, we also want to remove data_type (eg the patch is no
    # longer in strain rate after the dft)
    new["_pre_dft_data_type"] = new.pop("data_type", None)
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


@patch_function()
def dft(
    patch: PatchType,
    dim: str | None | Sequence[str],
    *,
    real: str | bool | None = None,
    pad: bool = True,
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
        True, which means the last (possibly only) dimenson should have an
        rfft performed, or 3) None, meaning no rfft.
    pad
        If True, pad patch before performing dft along desired dimensions to
        the next fast length. This can avoid major slow-downs when dimension
        lengths are prime numbers.

    Notes
    -----
    - Simply uses numpy's fft module but outputs are scaled by the sample
      spacing along each transformed dimension and coordinates corresponding
      to frequency bins are shifted so they remain ordered.

    - Each transformed dimension is renamed with a preceding `ft_`. e.g.,
      `time` becomes `ft_time` (ft stands for fourier transform).

    - Each transformed dimension has units of 1/original units.

    - Output data units are the original data units multiplied by the units
      of each transformed dimension.

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
    """
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
    attrs = _get_dft_attrs(patch, dims, new_coords, pad=pad)
    return patch.new(data=data, coords=new_coords, attrs=attrs)


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
    new["dims"] = new_coords.dims
    new["data_units"] = _get_data_units_from_dims(patch, dims, mul)
    # Restore the pre-dft datatype.
    if "_pre_dft_data_type" in new:
        new["data_type"] = new.pop("_pre_dft_data_type", None)
    new.pop("_dft_padded", None)
    return PatchAttrs(**new)


@patch_function()
def idft(patch: PatchType, dim: str | None | Sequence[str] = None) -> PatchType:
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
    dims, steps, axes, real = _get_idft_dims_steps_axis(patch, dim)
    new_dims = FourierTransformatter().rename_dims(dims, forward=False)
    func = nft.irfftn if real else nft.ifftn
    # Get new coords, fft sizes, and padding to remove.
    coords, sizes, padding = _get_idft_coords_and_sizes(
        patch, dims, new_dims, axes, real
    )
    # now unshift data and undo scaling
    ax_slice = slice(None, -1) if real else slice(None)
    scale_factor = np.prod([to_float(coords.coord_map[x].step) for x in new_dims])
    _preped = nft.ifftshift(patch.data / scale_factor, axes=axes[ax_slice])
    data = func(_preped, s=sizes, axes=axes)
    attrs = _get_idft_attrs(patch, dims, coords)
    out = patch.new(data=data, attrs=attrs, coords=coords)
    if padding:
        out = out.select(**padding, samples=True)
    return out


def _get_stft_coords(patch, dim, axis, coord, stft, window):
    """Get the new coordinate manager following stft."""

    def _get_stft_dims(dim, dims, axis):
        """Get the output dimensions after stft."""
        ft = FourierTransformatter()
        new_dims = list(ft.rename_dims(dims, index=axis, forward=True))
        new_dims.append(dim)
        return tuple(new_dims)

    # Get new coordinate. Just called time here because that is most common.
    time = stft.t(len(coord))
    if is_datetime64(coord.dtype) or is_timedelta64(coord.dtype):
        time = dc.to_timedelta64(time)
    # Get new dimensions
    new_dims = list(_get_stft_dims(dim, patch.dims, axis))
    # Make dict of coordinates nd return coord manager.
    coord_map = dict(patch.coords.coord_map)
    new_units = invert_quantity(coord.units)
    coord_map.update(
        {
            dim: get_coord(values=time + coord.min(), units=coord.units),
            new_dims[axis]: get_coord(values=stft.f, units=new_units),
            # Add window array for inverse stft.
            "_stft_window": (None, window),
            "_stft_old_coord": (None, patch.get_coord(dim)),
        }
    )
    out = get_coord_manager(coords=coord_map, dims=tuple(new_dims))
    return out


@patch_function()
def stft(
    patch: PatchType,
    taper_window: str | ndarray | tuple[str, Any, ...] = "hann",
    overlap: Quantity | int | None = 50 * percent,
    samples: bool = False,
    detrend: bool = False,
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

    Notes
    -----
    - The output is scaled the same as [Patch.dft](`dascore.Patch.dft`).
      For a given sliding window, Parseval's theorem doesn't hold exactly
      (unless a boxcar window is used) because the taper window changes the time
      series signal before the transformation.
    - If an array is passed for taper_window that has a different length
      than specified in kwargs, artificial enriching of frequency resolution
      (equivalent to zero padding in time domain) can occur.

    See Also
    --------
    [Patch.dft](`dascore.Patch.dft`), [Patch.istft](`dascore.Patch.istft`)
    """
    # Get coordinate information.
    (dim, axis, val) = get_dim_axis_value(patch, kwargs=kwargs)[0]
    coord = patch.get_coord(dim, require_evenly_sampled=True)
    window_samples = coord.get_sample_count(val, samples=samples, enforce_lt_coord=True)
    step = dc.to_float(coord.step)
    sampling_rate = 1 / abs(step)
    # Create window and calculate hop.
    if isinstance(taper_window, ndarray):
        window = taper_window
    else:
        window = get_window(taper_window, window_samples, fftbins=False)
    # By using a coord and enforce_lt_coord, we guarantee the overlap is lt window.
    if overlap is not None:
        overlap = coord[:window_samples].get_sample_count(
            overlap,
            samples=samples,
            enforce_lt_coord=True,
        )
    else:
        overlap = 0
    hop = window_samples - overlap
    # Perform stft
    fft_mode = "onesided" if np.isrealobj(patch.data) else "centered"
    stft = ShortTimeFFT(
        win=window,
        hop=hop,
        fs=sampling_rate,
        fft_mode=fft_mode,
        mfft=window_samples,
    )
    func = stft.stft if not detrend else partial(stft.stft_detrend, detr="linear")
    # For compatibility with dft, we scale by step. See the DFT note for why.
    new_data = func(patch.data, axis=axis) * step
    # Get new coordinate manager
    cm = _get_stft_coords(patch, dim, axis, coord, stft, window)
    # Update attrs with metadata needed to invert stft
    new_attrs = {
        "_stft_time_dimension": dim,
        "_stft_frequency_dimension": cm.dims[axis],
        "_stft_hop": hop,
        "_stft_sampling_rate": sampling_rate,
        "_stft_detrended": detrend,
        "_stft_fft_mode": fft_mode,
        "_stft_mfft": window_samples,
        "_stft_performed": True,
        "data_units": _get_data_units_from_dims(patch, dim, mul),
    }
    attrs = patch.attrs.drop("coords").update(**new_attrs)
    return patch.new(data=new_data, coords=cm, attrs=attrs)


def _get_inverse_axes(patch):
    """Get the inverse dimension and axes."""
    time_dimension = patch.attrs.get("_stft_time_dimension")
    frequency_dimension = patch.attrs.get("_stft_frequency_dimension")
    if time_dimension is None or frequency_dimension is None:
        msg = (
            "Inverse short time fourier transform requires a patch that has"
            " undergone stft but this patch is missing required attrs. "
        )
        raise PatchError(msg)
    time_axis = patch.get_axis(time_dimension)
    frequency_axis = patch.get_axis(frequency_dimension)
    return time_axis, frequency_axis


def _get_istft_coord(coords, frequency_axis, time_axis):
    """
    Get the coordinate manager for the inverse of the short time fourier transform.
    """
    dims = coords.dims
    # Create new time coordinate.
    coord_map = dict(coords.coord_map)
    coord_map.pop("_stft_window")
    time = coord_map.pop("_stft_old_coord")
    coord_map.pop(coords.dims[frequency_axis])
    coord_map[dims[time_axis]] = time
    # Get new dimensions
    new_dims = list(dims)
    new_dims[frequency_axis] = dims[time_axis]
    new_dims.pop(time_axis)
    return get_coord_manager(coords=coord_map, dims=tuple(new_dims)), time


def _get_short_time_fft(patch) -> ShortTimeFFT:
    """Reconstruct the short time fft from the attrs/coords in patch."""
    sr = patch.attrs.get("_stft_sampling_rate")
    # Recreate STFFT class based on saved coords/attrs.
    stft = ShortTimeFFT(
        win=patch.get_coord("_stft_window").values,
        hop=patch.attrs.get("_stft_hop"),
        fs=sr,
        fft_mode=patch.attrs.get("_stft_fft_mode"),
        mfft=patch.attrs.get("_stft_mfft"),
    )
    return stft


@patch_function()
def istft(patch) -> PatchType:
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
    """
    time_axis, frequency_axis = _get_inverse_axes(patch)
    detrended = patch.attrs.get("_stft_detrended")
    # Instantiate the transformer.
    stft = _get_short_time_fft(patch)
    # Raise if inverse not possible.
    if detrended or not stft.invertible:
        msg = f"Inverse stft not possible for patch {patch}."
        raise PatchError(msg)
    # Get coord manager and perform inverse transform.
    cm, coord = _get_istft_coord(patch.coords, frequency_axis, time_axis)
    data_untrimmed = stft.istft(
        patch.data / dc.to_float(coord.step), t_axis=time_axis, f_axis=frequency_axis
    )
    # Trim data array to remove effect of padding.
    # Note: after ISTFT, `frequency_axis` now indexes the restored time axis.
    index = broadcast_for_index(
        data_untrimmed.ndim, frequency_axis, slice(0, len(coord))
    )
    new_data = data_untrimmed[index]
    assert new_data.shape == cm.shape
    # Re-assemble and return new patch.
    new_attrs = {i: v for i, v in patch.attrs.items() if not i.startswith("_stft")}
    dim = patch.dims[time_axis]
    new_attrs["data_units"] = _get_data_units_from_dims(patch, dim, truediv)
    attrs = dc.PatchAttrs(**new_attrs).drop("coords")
    return patch.new(data=new_data, coords=cm, attrs=attrs)
