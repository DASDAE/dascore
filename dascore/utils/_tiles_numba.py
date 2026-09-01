"""
Optional numba drivers which give a compiled function one tile at a time.

The module imports whether or not numba is installed; only `_JIT_AVAILABLE`
says whether the drivers can compile. A driver is specialized on the
function it is handed, and numba cannot reuse such a specialization from
disk, so it compiles the first time each function is used in a process.
"""

from __future__ import annotations

import itertools

import numpy as np

from dascore.utils.jit import maybe_numba_jit
from dascore.utils.tiles import TilePlan


@maybe_numba_jit(required=True, nopython=True, parallel=True)
def _blend_class(tiles, out, taper, rest, func, analysis) -> None:
    """
    Apply `func` to every tile of one colour class, adding into `out`.

    `tiles` and `out` are views of that class, ``[*counts, *size]``, and
    `rest` is ``counts[1:]``. Tiles of one class are further apart than a
    tile reaches, so no two iterations of the parallel loop write to the
    same output sample.
    """
    for first in numba.prange(tiles.shape[0]):  # noqa: F821  # ty: ignore[unresolved-reference]
        for other in np.ndindex(rest):
            index = (first,) + other  # noqa: RUF005  # numba joins tuples this way
            out[index] += func(tiles[index] * analysis) * taper


@maybe_numba_jit(required=True, nopython=True, parallel=True)
def _stack_each(tiles, out, func) -> None:
    """Apply `func` to every tile of a stack, in parallel."""
    for ind in numba.prange(tiles.shape[0]):  # noqa: F821  # ty: ignore[unresolved-reference]
        out[ind] = func(tiles[ind])


_JIT_AVAILABLE = _blend_class.jit_available


def apply_jit(
    plan: TilePlan,
    array: np.ndarray,
    func,
    taper: np.ndarray,
    analysis: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return the array with `func` applied to every tile and the tiles blended.

    `func` is a numba-compiled function of one tile returning a tile of the
    same shape. `analysis` multiplies each tile before `func` sees it; None
    is a window of ones.
    """
    padded = plan.pad(array, dtype=np.result_type(array, np.float32))
    if analysis is None:
        analysis = np.ones(plan.size, dtype=padded.dtype)
    analysis = analysis.astype(padded.dtype)
    # One tile through the function first, to learn what it returns: the
    # output takes that dtype, so a real tile made complex is kept complex.
    probe = func(padded[tuple(slice(0, z) for z in plan.size)] * analysis)
    out = np.zeros(plan.extended, dtype=np.result_type(probe, padded, taper))
    tiles_in = plan._tile_view(padded)
    tiles_out = plan._tile_view(out, writeable=True)
    ndim = len(plan.size)
    for colour in itertools.product(*(range(c) for c in plan.colours)):
        sub = tuple(slice(c, None, k) for c, k in zip(colour, plan.colours))
        tiles = tiles_in[sub]
        _blend_class(
            tiles,
            tiles_out[sub],
            taper.astype(out.dtype),
            tiles.shape[1:ndim],
            func,
            analysis,
        )
    return plan.crop(out)


def stack_jit(tiles: np.ndarray, func) -> np.ndarray:
    """Return a stack of tiles with `func` applied to each, in parallel."""
    probe = func(tiles[0])
    out = np.empty(tiles.shape, dtype=np.result_type(probe, tiles))
    _stack_each(tiles, out, func)
    return out
