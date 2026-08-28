"""
Optional numba driver which gives a compiled function one tile at a time.

The module imports whether or not numba is installed; only `_JIT_AVAILABLE`
says whether the driver can compile. The driver is specialized on the
function it is handed, and numba cannot reuse such a specialization from
disk, so it compiles the first time each function is used in a process.
"""

from __future__ import annotations

import numpy as np

from dascore.utils.jit import maybe_numba_jit
from dascore.utils.tiles import TilePlan


@maybe_numba_jit(required=True, nopython=True, parallel=True)
def _apply_colour_class(
    padded: np.ndarray,
    out: np.ndarray,
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
    func,
    analysis: np.ndarray,
) -> None:
    """
    Apply `func` to every tile of one colour class, adding into `out`.

    Tiles of one class are `colours` strides apart, further than a tile
    reaches, so no two iterations of the parallel loop write to the same
    output sample.
    """
    count0 = (n_tiles0 - colour0 + colours0 - 1) // colours0
    count1 = (n_tiles1 - colour1 + colours1 - 1) // colours1
    for ind in numba.prange(count0 * count1):  # noqa: F821  # ty: ignore[unresolved-reference]
        beg0 = (colour0 + colours0 * (ind // count1)) * stride0
        beg1 = (colour1 + colours1 * (ind % count1)) * stride1
        tile = func(padded[beg0 : beg0 + window0, beg1 : beg1 + window1] * analysis)
        out[beg0 : beg0 + window0, beg1 : beg1 + window1] += tile * taper


_JIT_AVAILABLE = _apply_colour_class.jit_available


def apply_jit(
    plan: TilePlan,
    array: np.ndarray,
    func,
    taper: np.ndarray,
    analysis: np.ndarray,
) -> np.ndarray:
    """
    Return the array with `func` applied to every tile and the tiles blended.

    `func` is a numba-compiled function of one tile returning a tile of the
    same shape. Two dimensions only. `analysis` multiplies each tile before
    `func` sees it.
    """
    padded = plan.pad(array, dtype=np.result_type(array, np.float32))
    analysis = analysis.astype(padded.dtype)
    # One tile through the function first, to learn what it returns: the
    # output takes that dtype, so a real tile made complex is kept complex.
    probe = func(padded[tuple(slice(0, z) for z in plan.size)] * analysis)
    out = np.zeros(plan.extended, dtype=np.result_type(probe, padded, taper))
    for colour0 in range(plan.colours[0]):
        for colour1 in range(plan.colours[1]):
            _apply_colour_class(
                padded,
                out,
                taper.astype(padded.dtype),
                *plan.size,
                *plan.stride,
                *plan.grid,
                *plan.colours,
                colour0,
                colour1,
                func,
                analysis,
            )
    return plan.crop(out)
