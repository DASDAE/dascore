"""
Adaptive spectral filtering for DASCore patches.

The filter walks a patch in overlapping windows along one or two dimensions.
Each window is transformed to the spectral domain, every coefficient is
weighted by a power of its own magnitude, and the window is transformed back
and added into the output under a tapered overlap-add. Energy which is
coherent within a window concentrates in a few large coefficients, which the
weighting keeps; energy spread across the spectrum is suppressed.

Over two dimensions this is the adaptive frequency-wavenumber filter of
@isken2022denoising as implemented by Pyrocko
[Lightguide](https://github.com/pyrocko/lightguide), which the SciPy engine
here matches to floating-point precision. Over one dimension it is the same
weighting applied to each trace on its own.

The public patch function converts dimension names and coordinate units to
axes and sample counts and batches over every dimension not selected. The
engines work on raw one- or two-dimensional arrays; the optional
Numba/rocket-fft engine in `_adaptive_spectral_filter_numba` handles the
two-dimensional case and shares this module's validation, padding, and taper
so the two produce the same output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import product
from math import prod
from typing import Any, Literal

import numpy as np
from scipy import fft as sp_fft

from dascore.constants import PatchType
from dascore.exceptions import MissingOptionalDependencyError, ParameterError
from dascore.utils.misc import is_power_of_two
from dascore.utils.patch import get_dim_axis_value, patch_function
from dascore.utils.signal import _triangular_taper

_AdaptiveSpectralEngine = Literal["auto", "numba", "scipy"]
__all__ = ("adaptive_spectral_filter",)


def _check_window(window: Any, overlap: Any, label: str) -> None:
    """Raise ValueError unless a window and its overlap can tile an axis."""
    if not isinstance(window, int | np.integer):
        msg = f"window for {label} must be an integer; got {window!r}."
        raise ValueError(msg)
    if not isinstance(overlap, int | np.integer):
        msg = f"overlap for {label} must be an integer; got {overlap!r}."
        raise ValueError(msg)
    if window <= 4 or not is_power_of_two(window):
        msg = (
            f"window for {label} must be a power of two greater than 4; got {window!r}."
        )
        raise ValueError(msg)
    if overlap < 0:
        msg = f"overlap for {label} must be non-negative; got {overlap!r}."
        raise ValueError(msg)
    if overlap >= window / 2:
        msg = f"overlap for {label} is too large; maximum is {window // 2 - 1} samples."
        raise ValueError(msg)


def _check_exponent(exponent: float) -> None:
    """Raise ValueError unless the exponent is finite."""
    if not np.isfinite(exponent):
        msg = "exponent must be finite."
        raise ValueError(msg)


def _validate_filter_inputs(
    data: np.ndarray,
    *,
    window_size: tuple[int, ...],
    overlap: tuple[int, ...],
    exponent: float,
) -> None:
    """Validate direct array-filter inputs before entering FFT kernels."""
    if data.ndim not in {1, 2}:
        msg = (
            f"adaptive spectral array filters require 1D or 2D input; got {data.ndim}D."
        )
        raise ValueError(msg)
    if len(window_size) != data.ndim or len(overlap) != data.ndim:
        msg = "window_size and overlap must match the input dimensionality."
        raise ValueError(msg)
    _check_exponent(exponent)
    for axis, (window, axis_overlap) in enumerate(zip(window_size, overlap)):
        _check_window(window, axis_overlap, f"axis {axis}")


def _validate_window_and_overlap(
    dims: tuple[str, ...],
    windows: tuple[int, ...],
    overlaps: tuple[int, ...],
    exponent: float,
) -> None:
    """Validate the patch-level settings, naming dimensions rather than axes."""
    try:
        _check_exponent(exponent)
        for dim, window, overlap in zip(dims, windows, overlaps):
            _check_window(window, overlap, repr(dim))
    except ValueError as exc:
        raise ParameterError(str(exc)) from exc


def _prepare_work_arrays(
    data: np.ndarray,
    *,
    window_size: tuple[int, ...],
    overlap: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """
    Return the padded float32 input, the taper, the stride, and the tile grid.

    The input is padded by one stride of zeros on every side so the tiles
    which straddle its edges see a full taper ramp.
    """
    working = np.ascontiguousarray(data, dtype=np.float32)
    stride = tuple(win - over for win, over in zip(window_size, overlap))
    plateau = tuple(win - 2 * over for win, over in zip(window_size, overlap))
    taper = _triangular_taper(window_size, plateau)
    padded_shape = tuple(
        length + 2 * step for length, step in zip(working.shape, stride)
    )
    padded = np.zeros(padded_shape, dtype=np.float32)
    inner = tuple(
        slice(step, length + step) for length, step in zip(working.shape, stride)
    )
    padded[inner] = working
    n_tiles = tuple(pad_len // step for pad_len, step in zip(padded.shape, stride))
    return padded, taper, stride, n_tiles


def _finalize_output(
    filtered: np.ndarray,
    shape: tuple[int, ...],
    dtype: np.dtype,
    stride: tuple[int, ...],
) -> np.ndarray:
    """Crop the padding away and restore a floating input dtype."""
    inner = tuple(slice(step, length + step) for length, step in zip(shape, stride))
    out = filtered[inner]
    if np.issubdtype(dtype, np.floating):
        return out.astype(dtype, copy=False)
    return out


def _extract_tiles(
    padded: np.ndarray,
    window_size: tuple[int, ...],
    stride: tuple[int, ...],
    n_tiles: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy every window into a dense tile stack for batched FFTs."""
    ndim = len(window_size)
    tiles = np.zeros((prod(n_tiles), *window_size), dtype=np.float32)
    begins = np.zeros((*n_tiles, ndim), dtype=np.int64)
    sizes = np.zeros((*n_tiles, ndim), dtype=np.int64)
    for tile_index, tile_inds in enumerate(product(*(range(num) for num in n_tiles))):
        beg = tuple(ind * step for ind, step in zip(tile_inds, stride))
        end = tuple(
            min(start + win, size)
            for start, win, size in zip(beg, window_size, padded.shape)
        )
        valid_shape = tuple(stop - start for start, stop in zip(beg, end))
        begins[tile_inds] = beg
        sizes[tile_inds] = valid_shape
        data_slices = tuple(slice(start, stop) for start, stop in zip(beg, end))
        tile_slices = tuple(slice(0, size) for size in valid_shape)
        tiles[(tile_index, *tile_slices)] = padded[data_slices]
    return tiles, begins, sizes


def _overlap_add_tiles(
    out: np.ndarray,
    tiles: np.ndarray,
    taper: np.ndarray,
    begins: np.ndarray,
    sizes: np.ndarray,
) -> None:
    """Add every tapered tile back into the padded output."""
    grid_shape = begins.shape[:-1]
    for tile_index, tile_inds in enumerate(
        product(*(range(num) for num in grid_shape))
    ):
        beg = tuple(begins[tile_inds])
        valid_shape = tuple(sizes[tile_inds])
        out_slices = tuple(
            slice(start, start + size) for start, size in zip(beg, valid_shape)
        )
        tile_slices = tuple(slice(0, size) for size in valid_shape)
        out[out_slices] += tiles[(tile_index, *tile_slices)] * taper[tile_slices]


def _adaptive_spectral_filter_scipy(
    data: np.ndarray,
    *,
    window_size: tuple[int, ...],
    overlap: tuple[int, ...],
    exponent: float = 0.8,
    normalize_power: bool = False,
) -> np.ndarray:
    """
    Filter a 1D or 2D array with the SciPy adaptive spectral implementation.

    Parameters
    ----------
    data
        One- or two-dimensional input array. The filter computes in ``float32``.
    window_size
        Power-of-two window lengths, one per array axis. Values must be greater
        than 4.
    overlap
        Number of samples each neighboring window overlaps on each axis. Values
        must be non-negative and smaller than half the matching window.
    exponent
        Spectral magnitude exponent used as the adaptive weighting power. ``0``
        leaves the spectrum unweighted before overlap-add reconstruction.
    normalize_power
        If ``True``, normalize each tile's spectral magnitudes by that tile's
        maximum magnitude before applying ``exponent``.

    Returns
    -------
    numpy.ndarray
        The filtered array with the same shape as ``data``. Floating input
        dtypes are restored; non-floating inputs return ``float32`` output.

    Raises
    ------
    ValueError
        If ``data`` is not one- or two-dimensional, ``exponent`` is not finite,
        ``window_size`` and ``overlap`` do not match ``data.ndim``, any window
        size is not a power of two greater than 4, or any overlap is negative or
        at least half the matching window size.
    """
    data = np.asarray(data)
    _validate_filter_inputs(
        data, window_size=window_size, overlap=overlap, exponent=float(exponent)
    )
    padded, taper, stride, n_tiles = _prepare_work_arrays(
        data, window_size=window_size, overlap=overlap
    )
    tiles, begins, sizes = _extract_tiles(padded, window_size, stride, n_tiles)
    axes = tuple(range(-data.ndim, 0))

    spec = sp_fft.rfftn(tiles, s=window_size, axes=axes, workers=-1)
    if exponent != 0.0:
        power = np.abs(spec).astype(np.float32, copy=False)
        if normalize_power:
            max_power = power.max(axis=axes, keepdims=True)
            power = np.divide(
                power, max_power, out=np.zeros_like(power), where=max_power != 0
            )
        spec *= power**exponent
    tiles = sp_fft.irfftn(spec, s=window_size, axes=axes, workers=-1).astype(
        np.float32, copy=False
    )
    filtered = np.zeros_like(padded)
    _overlap_add_tiles(filtered, tiles, taper, begins, sizes)
    return _finalize_output(filtered, data.shape, data.dtype, stride)


def _dim_values_to_samples(
    patch: PatchType,
    dim_axis_values,
    *,
    samples: bool,
    name: str,
    force_sample_dims: frozenset[str] = frozenset(),
) -> tuple[int, ...]:
    """Convert DASCore dimension values from units or samples into sample counts."""
    out: list[int] = []
    for dim, _, value in dim_axis_values:
        if samples or dim in force_sample_dims:
            count = int(value)
        else:
            coord = patch.get_coord(dim, require_evenly_sampled=True)
            count = coord.get_sample_count(value, samples=False)
        invalid = count < 0 if name == "overlap" else count <= 0
        if invalid:
            requirement = "non-negative" if name == "overlap" else "positive"
            msg = f"{name} for dimension {dim!r} must be {requirement}."
            raise ParameterError(msg)
        out.append(count)
    return tuple(out)


def _normalize_overlap(
    overlap: Any,
    dims: tuple[str, ...],
    windows: tuple[int, ...],
) -> tuple[dict[str, Any], frozenset[str]]:
    """Return per-dimension overlap values and internally defaulted dimensions."""
    # The largest overlap the window allows, which is what Lightguide uses.
    defaults = {dim: window // 2 - 1 for dim, window in zip(dims, windows)}
    if overlap is None:
        return defaults, frozenset(dims)
    if isinstance(overlap, Mapping):
        extra = set(overlap) - set(dims)
        if extra:
            names = sorted(map(str, extra))
            msg = f"overlap contains dimensions not being filtered: {names}"
            raise ParameterError(msg)
        return defaults | dict(overlap), frozenset(set(dims) - set(overlap))
    # Uncoerced: the value the caller gave is read in coordinate units
    # unless samples says otherwise, and int() of a timedelta64 or a
    # fractional second is not the overlap they asked for. Only the
    # defaults above are sample counts, which is what the returned set says.
    return dict.fromkeys(dims, overlap), frozenset()


def _get_engine(engine: _AdaptiveSpectralEngine, selected_ndim: int) -> Callable:
    """Return the requested adaptive spectral array filter implementation."""
    if engine == "scipy" or (engine == "auto" and selected_ndim == 1):
        return _adaptive_spectral_filter_scipy
    if engine not in {"auto", "numba"}:
        msg = "engine must be one of 'auto', 'numba', or 'scipy'."
        raise ParameterError(msg)
    if selected_ndim != 2:
        msg = "engine='numba' currently supports exactly two selected dimensions."
        raise ParameterError(msg)
    # Deferred: the numba engine is optional, and importing it eagerly
    # would compile it whenever dascore is imported.
    from dascore.proc._adaptive_spectral_filter_numba import (  # noqa: PLC0415
        _NUMBA_ENGINE_AVAILABLE,
        _adaptive_spectral_filter_numba,
    )

    if _NUMBA_ENGINE_AVAILABLE:
        return _adaptive_spectral_filter_numba
    if engine == "numba":
        msg = (
            "engine='numba' requires optional dependencies numba and "
            "rocket-fft to be installed."
        )
        raise MissingOptionalDependencyError(msg)
    return _adaptive_spectral_filter_scipy


@patch_function()
def adaptive_spectral_filter(
    patch: PatchType,
    *,
    overlap: Any = None,
    exponent: float = 0.8,
    normalize_power: bool = False,
    samples: bool = False,
    engine: _AdaptiveSpectralEngine = "auto",
    **kwargs: Any,
) -> PatchType:
    """
    Apply adaptive spectral filtering over one or two patch dimensions.

    Parameters
    ----------
    patch
        DASCore patch whose data should be filtered.
    overlap
        Window overlap in samples when ``samples=True`` or in coordinate units
        otherwise. A single value applies to all selected dimensions; a mapping
        can specify dimensions independently. When omitted, each dimension
        defaults to ``window // 2 - 1`` samples, the largest overlap allowed.
    exponent
        Spectral magnitude exponent used as the adaptive weighting power.
        Larger values suppress incoherent energy harder; ``0`` leaves the
        spectrum unweighted, and values above 1 begin to remove weak coherent
        arrivals along with the noise.
    normalize_power
        If ``True``, normalize each tile's spectral magnitudes by that tile's
        maximum magnitude before applying ``exponent``. This keeps the
        amplitude of every window near its input level, at the cost of
        suppressing much less noise in windows which hold no signal.
    samples
        If ``True``, dimension kwargs and overlap values are interpreted as
        sample counts. If ``False``, values are converted through evenly sampled
        patch coordinates.
    engine
        ``"auto"`` uses SciPy for one selected dimension and the optional
        Numba/rocket-fft implementation for two selected dimensions when
        available. ``"numba"`` requires two selected dimensions and the optional
        fast engine. ``"scipy"`` always uses the SciPy FFT implementation.
    **kwargs
        One or two dimension names and their window sizes, such as ``time=32``
        or ``time=32, distance=32``.

    Returns
    -------
    Patch
        A new patch with filtered data and original dimensions and coordinates.

    Raises
    ------
    ParameterError
        If one or two dimensions are not selected, if selected window or overlap
        values are invalid, if ``exponent`` is not finite, or if an invalid
        engine name is requested.
    MissingOptionalDependencyError
        If ``engine="numba"`` is requested for two selected dimensions but the
        optional fast-engine dependencies are not installed.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch("example_event_2").pass_filter(time=(1, 300))
    >>> # Suppress energy which is not coherent across both time and distance,
    >>> # in windows of 16 samples along each.
    >>> filtered = patch.adaptive_spectral_filter(
    ...     time=16, distance=16, samples=True
    ... )
    >>> # Or weight each trace's spectrum on its own.
    >>> per_trace = patch.adaptive_spectral_filter(time=32, samples=True)

    Notes
    -----
    - With two selected dimensions this is the adaptive frequency-wavenumber
      (AFK) filter of @isken2022denoising, and matches Pyrocko
      [Lightguide](https://github.com/pyrocko/lightguide)'s `afk_filter`,
      whose defaults are ``window_size=16, overlap=7, exponent=0.8``.
    - The filter is not amplitude preserving. Each coefficient is scaled by
      its own magnitude to the power of ``exponent``, so the output's units
      are not the input's and its amplitudes grow with the input's; compare
      arrivals within one output rather than across inputs.
    - Windows must be powers of two greater than 4 samples. A window should
      hold a few cycles of the arrivals to keep and be short against the
      distance over which their moveout changes.
    """
    if len(kwargs) not in {1, 2}:
        msg = (
            "adaptive_spectral_filter requires one or two dimension window kwargs, "
            "e.g. patch.adaptive_spectral_filter(time=32, samples=True)."
        )
        raise ParameterError(msg)
    dim_axis_values = get_dim_axis_value(patch, kwargs=kwargs, allow_multiple=True)
    dims = tuple(x.dim for x in dim_axis_values)
    axes = tuple(x.axis for x in dim_axis_values)
    windows = _dim_values_to_samples(
        patch, dim_axis_values, samples=samples, name="window"
    )
    overlap_values, default_overlap_dims = _normalize_overlap(overlap, dims, windows)
    overlap_dim_axis_values = get_dim_axis_value(
        patch, kwargs=overlap_values, allow_multiple=True
    )
    overlaps = _dim_values_to_samples(
        patch,
        samples=samples,
        dim_axis_values=overlap_dim_axis_values,
        name="overlap",
        force_sample_dims=default_overlap_dims,
    )
    _validate_window_and_overlap(dims, windows, overlaps, float(exponent))

    data = np.asarray(patch.data)
    selected_ndim = len(axes)
    moved = np.moveaxis(data, axes, tuple(range(-selected_ndim, 0)))
    batch_shape = moved.shape[:-selected_ndim]
    selected_shape = moved.shape[-selected_ndim:]
    working = moved.reshape((-1, *selected_shape))
    filtered = np.empty_like(working, dtype=np.float32)
    engine_func = _get_engine(engine, selected_ndim)
    for ind, array in enumerate(working):
        filtered[ind] = engine_func(
            array,
            window_size=windows,
            overlap=overlaps,
            exponent=float(exponent),
            normalize_power=bool(normalize_power),
        )
    filtered = filtered.reshape((*batch_shape, *selected_shape))
    filtered = np.moveaxis(filtered, tuple(range(-selected_ndim, 0)), axes)
    if np.issubdtype(data.dtype, np.floating):
        filtered = filtered.astype(data.dtype, copy=False)
    return patch.update(data=filtered)
