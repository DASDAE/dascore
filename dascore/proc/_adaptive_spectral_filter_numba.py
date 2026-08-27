"""
Optional Numba/rocket-fft engine for the two-dimensional adaptive spectral filter.

The module imports whether or not numba and rocket-fft are installed; only
``_NUMBA_ENGINE_AVAILABLE`` says whether the kernel can actually compile.
"""

from __future__ import annotations

import numpy as np

from dascore.proc.adaptive_spectral_filter import (
    _finalize_output,
    _prepare_work_arrays,
    _validate_filter_inputs,
)
from dascore.utils.jit import maybe_numba_jit


# fastmath is intentional: the weighting is approximate, and tests allow small
# SciPy/Numba differences from parallel floating-point evaluation.
@maybe_numba_jit(
    required=True,
    deps="rocket_fft",
    nopython=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
def _filter_tile_group(
    padded: np.ndarray,
    filtered: np.ndarray,
    taper: np.ndarray,
    window0: int,
    window1: int,
    stride0: int,
    stride1: int,
    n_tiles0: int,
    n_tiles1: int,
    parity0: int,
    parity1: int,
    exponent: float,
    normalize_power: bool,
) -> None:
    """
    Filter every tile whose grid indices share a parity, adding into filtered.

    Same-parity tiles start two strides apart, and an overlap under half the
    window keeps a window shorter than two strides, so no two iterations of
    the parallel loop write to the same output sample.
    """
    count0 = (n_tiles0 - parity0 + 1) // 2
    count1 = (n_tiles1 - parity1 + 1) // 2
    for ind in numba.prange(count0 * count1):  # noqa: F821  # ty: ignore[unresolved-reference]
        beg0 = (parity0 + 2 * (ind // count1)) * stride0
        beg1 = (parity1 + 2 * (ind % count1)) * stride1
        n0 = min(window0, padded.shape[0] - beg0)
        n1 = min(window1, padded.shape[1] - beg1)
        tile = np.zeros((window0, window1), dtype=np.float32)
        tile[:n0, :n1] = padded[beg0 : beg0 + n0, beg1 : beg1 + n1]
        spec = np.fft.rfft2(tile)
        if exponent != 0.0:
            power = np.abs(spec)
            if normalize_power:
                # A silent tile weights to zero rather than dividing by it.
                max_power = power.max()
                power = power / max_power if max_power > 0.0 else power
            # float32 so the weighted spectrum keeps rfft2's complex64 type.
            spec = spec * (power**exponent).astype(np.float32)
        tile = np.fft.irfft2(spec, s=(window0, window1))
        filtered[beg0 : beg0 + n0, beg1 : beg1 + n1] += tile[:n0, :n1] * taper[:n0, :n1]


_NUMBA_ENGINE_AVAILABLE = _filter_tile_group.jit_available


def _adaptive_spectral_filter_numba(
    data: np.ndarray,
    *,
    window_size: tuple[int, int],
    overlap: tuple[int, int],
    exponent: float = 0.8,
    normalize_power: bool = False,
) -> np.ndarray:
    """
    Filter a 2D array with the optional Numba/rocket-fft implementation.

    Takes the arguments :func:`_adaptive_spectral_filter_scipy` takes and
    returns what it returns, to within floating-point evaluation order; the
    two-dimensional restriction is the only difference.
    """
    data = np.asarray(data)
    _validate_filter_inputs(
        data, window_size=window_size, overlap=overlap, exponent=float(exponent)
    )
    if data.ndim != 2:
        msg = "The numba engine filters two-dimensional arrays only."
        raise ValueError(msg)
    padded, taper, stride, n_tiles = _prepare_work_arrays(
        data, window_size=window_size, overlap=overlap
    )
    filtered = np.zeros_like(padded)
    for parity0 in range(2):
        for parity1 in range(2):
            _filter_tile_group(
                padded,
                filtered,
                taper,
                *window_size,
                *stride,
                *n_tiles,
                parity0,
                parity1,
                float(exponent),
                bool(normalize_power),
            )
    return _finalize_output(filtered, data.shape, data.dtype, stride)
