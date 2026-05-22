"""Optional Numba/rocket-fft engine for adaptive spectral filtering."""

from __future__ import annotations

import numba as nb
import numpy as np
import rocket_fft  # noqa: F401  # registers FFT overloads with numba.

from dascore.proc.adaptive_spectral_filter import (
    _finalize_output,
    _prepare_work_arrays,
    _validate_filter_inputs,
)


def _tile_indices_from_parity_index(
    ind: int,
    count1: int,
    parity0: int,
    parity1: int,
) -> tuple[int, int]:
    """Map a flattened parity-local index back to full-grid tile indices."""
    ix = parity0 + 2 * (ind // count1)
    iy = parity1 + 2 * (ind % count1)
    return ix, iy


def _tile_bounds(
    ix: int,
    iy: int,
    wx: int,
    wy: int,
    stride0: int,
    stride1: int,
    shape0: int,
    shape1: int,
) -> tuple[int, int, int, int]:
    """Return padded-array origin and valid tile shape for one window."""
    beg0 = ix * stride0
    beg1 = iy * stride1
    end0 = min(beg0 + wx, shape0)
    end1 = min(beg1 + wy, shape1)
    return beg0, beg1, end0 - beg0, end1 - beg1


def _copy_padded_tile(
    padded: np.ndarray,
    tile: np.ndarray,
    beg0: int,
    beg1: int,
    n0: int,
    n1: int,
) -> None:
    """Copy the valid padded-array region into a fixed-shape zeroed tile."""
    for i in range(n0):
        for j in range(n1):
            tile[i, j] = padded[beg0 + i, beg1 + j]


def _complex_power(value: complex) -> np.float32:
    """Return the magnitude of a complex FFT coefficient as ``float32``."""
    return np.float32((value.real * value.real + value.imag * value.imag) ** 0.5)


def _max_spectral_power(spec: np.ndarray) -> np.float32:
    """Return the maximum spectral magnitude in one tile."""
    max_power = np.float32(0.0)
    for i in range(spec.shape[0]):
        for j in range(spec.shape[1]):
            power = _complex_power(spec[i, j])
            if power > max_power:
                max_power = power
    return max_power


def _apply_spectral_weight(
    spec: np.ndarray,
    exponent: float,
    normalize_power: bool,
) -> None:
    """Apply adaptive magnitude weighting to one tile spectrum in place."""
    max_power = np.float32(0.0)
    if normalize_power:
        max_power = _max_spectral_power(spec)

    for i in range(spec.shape[0]):
        for j in range(spec.shape[1]):
            power = _complex_power(spec[i, j])
            if normalize_power:
                if max_power != 0.0:
                    power = power / max_power
                else:
                    power = np.float32(0.0)
            weight = np.float32(power**exponent)
            spec[i, j] *= weight


def _overlap_add_tile(
    filtered: np.ndarray,
    tile: np.ndarray,
    taper: np.ndarray,
    beg0: int,
    beg1: int,
    n0: int,
    n1: int,
) -> None:
    """Accumulate the valid region of one filtered tile into the output."""
    for i in range(n0):
        for j in range(n1):
            filtered[beg0 + i, beg1 + j] += tile[i, j] * taper[i, j]


_tile_indices_from_parity_index_numba = nb.njit(cache=True, inline="always")(
    _tile_indices_from_parity_index
)
_tile_bounds_numba = nb.njit(cache=True, inline="always")(_tile_bounds)
_copy_padded_tile_numba = nb.njit(cache=True, inline="always")(_copy_padded_tile)
_complex_power_numba = nb.njit(cache=True, inline="always")(_complex_power)


def _max_spectral_power_numba_impl(spec: np.ndarray) -> np.float32:
    """Return the maximum spectral magnitude using compiled helpers."""
    max_power = np.float32(0.0)
    for i in range(spec.shape[0]):
        for j in range(spec.shape[1]):
            power = _complex_power_numba(spec[i, j])
            if power > max_power:
                max_power = power
    return max_power


def _apply_spectral_weight_numba_impl(
    spec: np.ndarray,
    exponent: float,
    normalize_power: bool,
) -> None:
    """Apply adaptive magnitude weighting using compiled helpers."""
    max_power = np.float32(0.0)
    if normalize_power:
        max_power = _max_spectral_power_numba(spec)

    for i in range(spec.shape[0]):
        for j in range(spec.shape[1]):
            power = _complex_power_numba(spec[i, j])
            if normalize_power:
                if max_power != 0.0:
                    power = power / max_power
                else:
                    power = np.float32(0.0)
            weight = np.float32(power**exponent)
            spec[i, j] *= weight


_max_spectral_power_numba = nb.njit(cache=True, inline="always")(
    _max_spectral_power_numba_impl
)
_apply_spectral_weight_numba = nb.njit(cache=True, inline="always")(
    _apply_spectral_weight_numba_impl
)
_overlap_add_tile_numba = nb.njit(cache=True, inline="always")(_overlap_add_tile)


def _process_tile_group_python(
    padded: np.ndarray,
    filtered: np.ndarray,
    taper: np.ndarray,
    wx: int,
    wy: int,
    stride0: int,
    stride1: int,
    nx: int,
    ny: int,
    parity0: int,
    parity1: int,
    exponent: float,
    normalize_power: bool,
) -> None:
    """Process one non-overlapping tile parity group in pure Python."""
    count0 = (nx - parity0 + 1) // 2
    count1 = (ny - parity1 + 1) // 2
    count = count0 * count1
    for ind in range(count):
        ix, iy = _tile_indices_from_parity_index(ind, count1, parity0, parity1)
        beg0, beg1, n0, n1 = _tile_bounds(
            ix, iy, wx, wy, stride0, stride1, padded.shape[0], padded.shape[1]
        )

        tile = np.zeros((wx, wy), dtype=np.float32)
        _copy_padded_tile(padded, tile, beg0, beg1, n0, n1)

        spec = np.fft.rfft2(tile)
        if exponent != 0.0:
            _apply_spectral_weight(spec, exponent, normalize_power)

        tile = np.fft.irfft2(spec, s=(wx, wy))
        _overlap_add_tile(filtered, tile, taper, beg0, beg1, n0, n1)


def _process_tile_group_numba_impl(
    padded: np.ndarray,
    filtered: np.ndarray,
    taper: np.ndarray,
    wx: int,
    wy: int,
    stride0: int,
    stride1: int,
    nx: int,
    ny: int,
    parity0: int,
    parity1: int,
    exponent: float,
    normalize_power: bool,
) -> None:
    """Process one non-overlapping tile parity group with compiled helpers."""
    count0 = (nx - parity0 + 1) // 2
    count1 = (ny - parity1 + 1) // 2
    count = count0 * count1
    for ind in nb.prange(count):  # type: ignore[not-iterable]
        ix, iy = _tile_indices_from_parity_index_numba(ind, count1, parity0, parity1)
        beg0, beg1, n0, n1 = _tile_bounds_numba(
            ix, iy, wx, wy, stride0, stride1, padded.shape[0], padded.shape[1]
        )

        tile = np.zeros((wx, wy), dtype=np.float32)
        _copy_padded_tile_numba(padded, tile, beg0, beg1, n0, n1)

        spec = np.fft.rfft2(tile)
        if exponent != 0.0:
            _apply_spectral_weight_numba(spec, exponent, normalize_power)

        tile = np.fft.irfft2(spec, s=(wx, wy))
        _overlap_add_tile_numba(filtered, tile, taper, beg0, beg1, n0, n1)


# fastmath is intentional here: the weighting is approximate and tests allow
# small SciPy/Numba differences from parallel floating-point evaluation.
_process_tile_group_numba = nb.njit(cache=True, fastmath=True, parallel=True)(
    _process_tile_group_numba_impl
)


def _adaptive_spectral_filter_numba(
    data: np.ndarray,
    *,
    window_size: tuple[int, int],
    overlap: tuple[int, int],
    exponent: float = 0.3,
    normalize_power: bool = False,
) -> np.ndarray:
    """
    Filter a 2D array with the optional Numba/rocket-fft implementation.

    Parameters
    ----------
    data
        Two-dimensional input array. The filter computes in ``float32``.
    window_size
        Two power-of-two window lengths, one per array axis. Values must be
        greater than 4.
    overlap
        Number of samples each neighboring window overlaps on each axis. Each
        value must be non-negative and smaller than half the matching window.
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
        If ``data`` is not two-dimensional, ``exponent`` is not finite,
        ``window_size`` and ``overlap`` do not contain exactly two integer
        values, any window size is not a power of two greater than 4, or any
        overlap is negative or at least half the matching window size.

    Notes
    -----
    This implementation uses Numba-compiled loops and rocket-fft-backed NumPy
    FFT calls. It is selected by
    :func:`dascore.proc.adaptive_spectral_filter.adaptive_spectral_filter` for
    two selected dimensions when ``engine="numba"`` or when ``engine="auto"``
    and optional dependencies are installed.
    """
    data = np.asarray(data)
    _validate_filter_inputs(
        data, window_size=window_size, overlap=overlap, exponent=float(exponent)
    )
    wx, wy = window_size
    working, original_dtype, stride, taper, padded, filtered, n_tiles = (
        _prepare_work_arrays(data, window_size=window_size, overlap=overlap)
    )
    for parity0 in range(2):
        for parity1 in range(2):
            _process_tile_group_numba(
                padded,
                filtered,
                taper,
                wx,
                wy,
                stride[0],
                stride[1],
                n_tiles[0],
                n_tiles[1],
                parity0,
                parity1,
                float(exponent),
                bool(normalize_power),
            )
    return _finalize_output(filtered, working, original_dtype, stride)
