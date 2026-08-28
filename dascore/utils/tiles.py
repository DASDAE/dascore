"""
Cut an array into overlapping tiles, and blend them back.

A :class:`TilePlan` is the geometry: where every tile of a given size and
stride sits over an array of a given shape, with one stride of zeros padded
on every side so the tiles at the edges see a full taper ramp. It cuts the
array into a dense stack of tiles, ``[n_tiles, *size]``, which one call to a
vectorized function can transform, and adds a stack back under a taper so
that, with a taper whose ramps are complementary, tiles of an untouched
stack sum to the array they were cut from.

The plan is the part of a windowed operation which has nothing to do with
the operation. The adaptive spectral filter is the first thing written on
it; anything which transforms, edits, and reassembles windows is the next.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np
from numpy.lib.stride_tricks import as_strided

from dascore.exceptions import ParameterError


@dataclass(frozen=True)
class TilePlan:
    """
    Where every tile sits over an array.

    Parameters
    ----------
    shape
        The array's shape along the tiled axes.
    size
        The tile's shape.
    stride
        How far each tile advances along each axis. Never longer than the
        tile: a gap between tiles is nothing to blend.

    Notes
    -----
    Build one with :func:`get_tile_plan`, which caches, rather than directly.
    """

    shape: tuple[int, ...]
    size: tuple[int, ...]
    stride: tuple[int, ...]

    def __post_init__(self):
        """Refuse a geometry which cannot tile."""
        if not len(self.shape) == len(self.size) == len(self.stride):
            msg = "shape, size, and stride must have the same length."
            raise ParameterError(msg)
        if any(step < 1 for step in self.stride) or any(z < 1 for z in self.size):
            msg = "tile size and stride must be positive."
            raise ParameterError(msg)
        if any(step > z for step, z in zip(self.stride, self.size)):
            msg = "a stride longer than the tile leaves gaps no tile covers."
            raise ParameterError(msg)

    @property
    def margin(self) -> tuple[int, ...]:
        """
        Zeros added at each end of every axis, in whole strides.

        Enough that every sample of the array is reached by every tile which
        would reach it in an unbounded grid: one stride when tiles overlap
        by no more than half, more when they overlap deeper, which a
        synthesis window computed as a dual relies on.
        """
        return tuple(
            max(step, math.ceil((z - step) / step) * step)
            for z, step in zip(self.size, self.stride)
        )

    @property
    def grid(self) -> tuple[int, ...]:
        """How many tiles along each axis."""
        return tuple(
            (length + 2 * m) // step
            for length, m, step in zip(self.shape, self.margin, self.stride)
        )

    @property
    def n_tiles(self) -> int:
        """How many tiles there are."""
        return math.prod(self.grid)

    @property
    def extended(self) -> tuple[int, ...]:
        """The padded buffer's shape, long enough for the last tile to fit whole."""
        return tuple(
            max(length + 2 * m, (count - 1) * step + z)
            for length, m, step, count, z in zip(
                self.shape, self.margin, self.stride, self.grid, self.size
            )
        )

    @property
    def colours(self) -> tuple[int, ...]:
        """
        Tiles per axis between two which cannot overlap.

        Tiles whose grid indices agree modulo this never share a sample, so
        each colour class of tiles can be added into the output at once.
        """
        return tuple(math.ceil(z / step) for z, step in zip(self.size, self.stride))

    def _inner(self) -> tuple[slice, ...]:
        """The array's place inside the padded buffer."""
        return tuple(
            slice(step, step + length) for step, length in zip(self.margin, self.shape)
        )

    def _tile_view(self, buffer: np.ndarray, writeable: bool = False) -> np.ndarray:
        """
        A view of the buffer as the tile grid, ``[*grid, *size]``.

        Strided directly rather than through `sliding_window_view`, whose
        view spans every window position: on a 32-bit build that logical
        size overflows for an ordinary patch, though nothing is allocated.
        """
        strides = (
            *(step * s for step, s in zip(self.stride, buffer.strides)),
            *buffer.strides,
        )
        return as_strided(
            buffer, shape=(*self.grid, *self.size), strides=strides, writeable=writeable
        )

    def pad(self, array: np.ndarray, dtype=None) -> np.ndarray:
        """
        Return the array inside a zeroed buffer every tile fits in whole.

        The buffer takes the array's dtype unless one is given.
        """
        buffer = np.zeros(self.extended, dtype=dtype or np.asarray(array).dtype)
        buffer[self._inner()] = array
        return buffer

    def crop(self, buffer: np.ndarray) -> np.ndarray:
        """Return the array's part of a buffer `pad` made."""
        return buffer[self._inner()]

    def extract(self, array: np.ndarray, dtype=None) -> np.ndarray:
        """Return the tiles as a stack, ``[n_tiles, *size]``, in the array's dtype."""
        buffer = self.pad(array, dtype)
        tiles = self._tile_view(buffer).reshape((self.n_tiles, *self.size))
        # A one-dimensional plan reshapes without copying, which would hand
        # out the read-only window view; a stack is for writing to.
        return tiles if tiles.flags.writeable else tiles.copy()

    def overlap_add(self, tiles: np.ndarray, taper: np.ndarray) -> np.ndarray:
        """
        Return the array the tapered tiles sum to, cropped to the array's shape.

        Parameters
        ----------
        tiles
            A stack as `extract` returns, ``[n_tiles, *size]``.
        taper
            The weights each tile is multiplied by, of the tile's shape; see
            `dascore.utils.signal.get_taper`.
        """
        dtype = np.result_type(tiles, taper)
        buffer = np.zeros(self.extended, dtype=dtype)
        grid = tiles.reshape((*self.grid, *self.size))
        view = self._tile_view(buffer, writeable=True)
        # One colour class at a time: its tiles never overlap, so adding
        # them through the strided view touches each sample once. The
        # taper goes on here, a class at a time, rather than on the stack.
        for colour in product(*(range(k) for k in self.colours)):
            pick = tuple(
                slice(c, count, k)
                for c, count, k in zip(colour, self.grid, self.colours)
            )
            view[pick] += grid[pick] * taper
        return self.crop(buffer)

    def apply(self, array: np.ndarray, func, taper: np.ndarray) -> np.ndarray:
        """
        Return the array with `func` applied to every tile and the tiles blended.

        Parameters
        ----------
        array
            The array to tile, of this plan's shape.
        func
            Takes the whole stack, ``[n_tiles, *size]``, and returns one of the
            same shape: one vectorized call, not one per tile.
        taper
            The weights the tiles are blended under.
        """
        return self.overlap_add(func(self.extract(array)), taper)


@lru_cache(maxsize=128)
def get_tile_plan(
    shape: tuple[int, ...], size: tuple[int, ...], stride: tuple[int, ...]
) -> TilePlan:
    """
    Return the plan for tiles of `size` at `stride` over an array of `shape`.

    Cached: a spool of patches of one shape shares a plan.
    """
    return TilePlan(tuple(shape), tuple(size), tuple(stride))
