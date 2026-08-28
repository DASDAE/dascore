"""
Optional Numba/rocket-fft engine for the two-dimensional adaptive spectral filter.

The module imports whether or not numba and rocket-fft are installed; only
``_NUMBA_ENGINE_AVAILABLE`` says whether the kernel can actually compile.

The kernel is the filter with its tile loop written out, rather than the
plan's vectorized one, because it can then be compiled once and cached on
disk; a kernel taking the filter as an argument would compile again in
every process. It takes its geometry -- padding, grid, colour classes --
from the same `TilePlan` the numpy engine uses.
"""

from __future__ import annotations

import numpy as np

from dascore.proc.adaptive_spectral_filter import (
    _plan_and_taper,
    _restore_dtype,
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
def _filter_colour_class(
    padded: np.ndarray,
    filtered: np.ndarray,
    taper: np.ndarray,
    window0: int,
    window1: int,
    stride0: int,
    stride1: int,
    n_tiles0: int,
    n_tiles1: int,
    colours0: int,
    colours1: int,
    colour0: int,
    colour1: int,
    exponent: float,
    normalize_power: bool,
) -> None:
    """
    Filter every tile of one colour class, adding into filtered.

    Tiles of one class are `colours` strides apart, further than a tile
    reaches, so no two iterations of the parallel loop write to the same
    output sample.
    """
    count0 = (n_tiles0 - colour0 + colours0 - 1) // colours0
    count1 = (n_tiles1 - colour1 + colours1 - 1) // colours1
    for ind in numba.prange(count0 * count1):  # noqa: F821  # ty: ignore[unresolved-reference]
        beg0 = (colour0 + colours0 * (ind // count1)) * stride0
        beg1 = (colour1 + colours1 * (ind % count1)) * stride1
        tile = padded[beg0 : beg0 + window0, beg1 : beg1 + window1]
        spec = np.fft.rfft2(tile)
        if exponent != 0.0:
            power = np.abs(spec)
            if normalize_power:
                # A silent tile weights to zero rather than dividing by it.
                max_power = power.max()
                power = power / max_power if max_power > 0.0 else power
            # float32 so the weighted spectrum keeps rfft2's complex64 type.
            spec = spec * (power**exponent).astype(np.float32)
        out = np.fft.irfft2(spec, s=(window0, window1))
        filtered[beg0 : beg0 + window0, beg1 : beg1 + window1] += out * taper


_NUMBA_ENGINE_AVAILABLE = _filter_colour_class.jit_available


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
    plan, taper = _plan_and_taper(data, window_size, overlap)
    # The buffer is long enough for every tile to be whole, so the kernel
    # slices without checking the edges.
    padded = plan.pad(data, dtype=np.float32)
    filtered = np.zeros_like(padded)
    for colour0 in range(plan.colours[0]):
        for colour1 in range(plan.colours[1]):
            _filter_colour_class(
                padded,
                filtered,
                taper,
                *window_size,
                *plan.stride,
                *plan.grid,
                *plan.colours,
                colour0,
                colour1,
                float(exponent),
                bool(normalize_power),
            )
    return _restore_dtype(plan.crop(filtered), data.dtype)
