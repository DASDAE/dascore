"""
Running a lazy array: which chunks, what each one reads, and computing it.

A chunk is a box of the final array, and the work is three separate steps:
[`chunk_boxes`](`dascore.lazy.execute.chunk_boxes`) cuts the array into
boxes, [`read_schedule`](`dascore.lazy.execute.read_schedule`) says which
box of the stored array each one needs, and
[`compute`](`dascore.lazy.execute.compute`) reads that box and runs each
step on it. [`load`](`dascore.lazy.execute.load`) cuts and computes, and
puts the boxes together in memory.

Examples
--------
>>> import numpy as np
>>> import dascore as dc
>>> from dascore.core.source import ArraySource
>>> from dascore.lazy.execute import chunk_boxes, compute, read_schedule
>>> from dascore.lazy.recipe import Recipe
>>>
>>> recipe = Recipe.from_source(ArraySource.full((100, 8), 1.0))
>>> coords = {"time": np.arange(100), "distance": np.arange(8)}
>>> patch = dc.Patch(data=recipe, coords=coords, dims=("time", "distance"))
>>> lazy = patch.select(time=(10, 90), samples=True)
>>> lazy = lazy.median_filter(time=3, samples=True).data
>>>
>>> # Four boxes of 20 samples; the second is read with a sample either side.
>>> boxes = chunk_boxes(lazy, block_size=22 * 8 * 8)
>>> assert [x[0] for x in boxes[:2]] == [slice(0, 20), slice(20, 40)]
>>> assert read_schedule(lazy, boxes)[1][0] == slice(29, 51)
>>> assert compute(lazy, boxes[1]).shape == (20, 8)
"""

from __future__ import annotations

import math
from functools import partial
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from dascore.config import get_config
from dascore.core.processor import running_under
from dascore.exceptions import PatchError
from dascore.lazy.contract import Box, ChunkContract, Window
from dascore.utils.array_api import (
    NUMPY_BACKEND,
    _as_numpy_dtype,
    array_namespace,
    fallbacks_decided,
    is_numpy,
)

if TYPE_CHECKING:
    from dascore.backends import ArrayBackend
    from dascore.lazy.array import LazyArray


def _converted(processor, backend: ArrayBackend) -> str | None:
    """Return the operation a step would take chunks through NumPy for, if any."""
    native = backend._native
    if native == NUMPY_BACKEND:
        return None
    kernel = type(processor).kernel_for(native, backend.name)
    if hasattr(kernel, "_converts"):
        return processor.name or type(processor).__name__
    # A step which runs another operation, as one in a window does, has a
    # kernel for any library and converts for the operation it runs.
    inner = getattr(processor, "operation", None)
    return _converted(inner, backend) if hasattr(inner, "kernel_for") else None


class _Everything(ChunkContract):
    """What a step which recorded no contract needs: all of its input."""

    def needs(self, box: Box, shape: tuple[int, ...]) -> Box:
        """Return the whole input."""
        return tuple(slice(0, size) for size in shape)

    def apply(self, run, data, box: Box, needed: Box):
        """Return the whole result, which `compute` checks and then cuts."""
        return run(data)


_EVERYTHING = _Everything()
# The most of its size a halved box may keep for the axis to be worth
# cutting. Half is what is left when nothing else is needed of the axis; an
# axis needed whole keeps it all, and one read with a halo keeps more the
# shorter its pieces get.
_WORTH_CUTTING = 0.75


def _needs(array: LazyArray, box: Box) -> list[Box]:
    """Return the box each step is given, the recipe's first and `box` last."""
    nodes = array.chain.nodes
    # The shape each step is given: the recipe's, then the step before's.
    shapes = [array.recipe.shape, *(x.made()[0] for x in nodes)][: len(nodes)]
    out = [box]
    for node, shape in zip(reversed(nodes), reversed(shapes), strict=True):
        out.append((node.contract or _EVERYTHING).needs(out[-1], shape))
    return out[::-1]


def _sizes(box: Box) -> tuple[int, ...]:
    """Return the shape of a box."""
    return tuple(x.stop - x.start for x in box)


def _largest(array: LazyArray, box: Box) -> int:
    """Return the bytes of the largest array computing a box holds."""
    dtypes = [array.recipe.dtype, *(x.made()[1] for x in array.chain)]
    needed = _needs(array, box)
    pairs = zip(needed, dtypes, strict=True)
    return max(math.prod(_sizes(x)) * dtype.itemsize for x, dtype in pairs)


def _middle(shape: tuple[int, ...], sizes: list[int]) -> Box:
    """Return a box of the given sizes in the middle of an array."""
    # The middle, because a box at an end reads no halo beyond it.
    starts = [(n - size) // 2 for n, size in zip(shape, sizes, strict=True)]
    return tuple(slice(a, a + size) for a, size in zip(starts, sizes, strict=True))


def chunk_boxes(array: LazyArray, block_size: int | None = None) -> list[Box]:
    """
    Cut a lazy array into the boxes it is computed in.

    There is one box if none of the arrays needed to compute it is larger
    than the block size, or if no step computes anything: reading in
    pieces would only open each file again. Otherwise the box is halved
    along the first axis worth cutting until its arrays fit. An axis a step
    needs whole is never worth cutting, so the block size is aimed for
    rather than promised.

    Parameters
    ----------
    array
        The lazy array.
    block_size
        The most bytes any one array of a box should hold; the
        `lazy_block_size` setting by default, and 0 for one box.
    """
    shape = array.shape
    block_size = get_config().lazy_block_size if block_size is None else block_size
    sizes = list(shape)
    cost = _largest(array, _middle(shape, sizes))
    # With nothing but windows there is nothing to compute in parts.
    cut = not all(isinstance(x.contract, Window) for x in array.chain)
    while cut and block_size and cost > block_size:
        for axis, size in enumerate(sizes):
            trial = [*sizes[:axis], math.ceil(size / 2), *sizes[axis + 1 :]]
            halved = _largest(array, _middle(shape, trial))
            if halved <= cost * _WORTH_CUTTING:
                cost, sizes = halved, trial
                break
        else:
            break
    edges = [range(0, n, size or 1) for n, size in zip(shape, sizes, strict=True)]
    spans = [
        [slice(a, min(a + size, n)) for a in starts] or [slice(0, 0)]
        for starts, size, n in zip(edges, sizes, shape, strict=True)
    ]
    return list(product(*spans))


def read_schedule(array: LazyArray, boxes: list[Box]) -> list[Box]:
    """
    Return the box of the stored array each box of the result reads.

    Nothing is read to work it out, so the reads are known before any of
    them happens, in the order the boxes are computed.
    """
    return [_needs(array, box)[0] for box in boxes]


def compute(array: LazyArray, box: Box) -> np.ndarray:
    """
    Read what one box of a lazy array needs, and compute it.

    Each step runs the kernel its processor has for the array's
    backend, and what it makes is
    checked against the dtype its metadata stated and the shape of the box.
    An empty box is returned without running anything.

    Raises
    ------
    PatchError
        If a step is still pending, or computed an array of another shape
        or dtype than its metadata stated.
    """
    backend = array._backend
    if 0 in _sizes(box):
        return backend.from_array(np.empty(_sizes(box), array.dtype))
    needed = _needs(array, box)
    stored = array.recipe[needed[0]]
    data = backend.from_recipe(stored)
    read = (tuple(data.shape), _as_numpy_dtype(data.dtype))
    if read != (stored.shape, stored.dtype):
        msg = (
            f"The {backend.name} backend read an array of shape {read[0]} and "
            f"dtype {read[1]} where the stored one has {stored.shape} and "
            f"{stored.dtype}. Its from_recipe must make the array the recipe "
            "describes."
        )
        raise PatchError(msg)
    # What the backend's arrays are, and its name: the same for every box,
    # so each step runs one kernel throughout.
    native, under = backend._native, backend.name
    steps = zip(array.chain, needed[:-1], needed[1:], strict=True)
    for node, before, after in steps:
        stated_shape, stated_dtype, _ = node.made()
        processor = node.processor
        kernel = type(processor).kernel_for(native, under)
        run = partial(kernel, processor, **node.plan)
        # A step with no kernel for the library was let through, or
        # refused, when it was chained: it is not asked again for each box.
        with fallbacks_decided(_converted(processor, backend)), running_under(under):
            data = (node.contract or _EVERYTHING).apply(run, data, after, before)
        # A step with no contract made the whole of its result, which is
        # checked as a whole before the box is taken from it.
        expected = _sizes(after) if node.contract else stated_shape
        shape, dtype = tuple(data.shape), _as_numpy_dtype(data.dtype)
        if shape != expected or dtype != stated_dtype:
            name = type(processor).__name__
            msg = (
                f"{name} computed an array of shape {shape} and dtype "
                f"{dtype} where its metadata stated one of {expected} "
                f"and {stated_dtype}. Its get_metadata must state what its "
                "kernel returns."
            )
            raise PatchError(msg)
        if node.contract is None:
            data = data[after]
        # One of the backend's arrays for the next step, as a patch in
        # memory holds between two operations.
        data = backend.from_array(data)
    return data


def load(array: LazyArray) -> np.ndarray:
    """Compute a lazy array a box at a time, and return the whole of it."""
    boxes = chunk_boxes(array)
    out = compute(array, boxes[0])
    if len(boxes) > 1 and is_numpy(out):
        # Written in place, so only one box is held beside the whole.
        first, out = out, np.empty(array.shape, array.dtype)
        out[boxes[0]] = first
        for box in boxes[1:]:
            out[box] = compute(array, box)
    elif len(boxes) > 1:
        out = _joined([out, *(compute(array, box) for box in boxes[1:])], boxes)
    # One of the backend's own arrays however many boxes it took: arrays of
    # no array library are joined as the `ndarray` NumPy makes of them.
    return array._backend.from_array(out)


def _joined(pieces: list, boxes: list[Box]):
    """
    Return the boxes of an array put together on their own array library.

    Nothing is written in place, which not every library allows, and
    nothing is taken through NumPy: the boxes, which come an axis at a
    time with the last changing fastest, are joined along each axis in
    turn. The boxes are all held until they are joined.
    """
    xp = array_namespace(pieces[0])
    for axis in reversed(range(len(boxes[0]))):
        count = len({(box[axis].start, box[axis].stop) for box in boxes})
        runs = range(0, len(pieces), count)
        pieces = [xp.concat(pieces[n : n + count], axis=axis) for n in runs]
    return pieces[0]
