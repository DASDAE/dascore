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
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from functools import cached_property, partial
from itertools import groupby, product
from typing import TYPE_CHECKING, Any

import numpy as np

from dascore.compat import is_array_like
from dascore.config import get_config
from dascore.core.processor import _REGISTERED, running_under
from dascore.exceptions import PatchError
from dascore.lazy.contract import Box, ChunkContract, Window, _sizes
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


@dataclass(frozen=True)
class _Slot:
    """Where an array of a plan was: its place in the arrays a step holds."""

    index: int


def _parts(value: Any) -> list | None:
    """Return what a plan's container holds, or None for anything else."""
    if type(value) is dict:
        return list(value.values())
    if type(value) in (tuple, list) or hasattr(type(value), "_fields"):
        return list(value)
    return None


def _rebuilt(value: Any, parts: list) -> Any:
    """Return a container of a plan like `value`, holding `parts`."""
    if type(value) is dict:
        return dict(zip(value, parts, strict=True))
    if hasattr(type(value), "_fields"):
        # Its constructor is not run again on what it made.
        return type(value)._make(parts)
    return type(value)(parts)


def _split(value: Any, arrays: list, traced: bool = False) -> Any:
    """Return a plan value with each array in it moved to `arrays`."""
    if traced or is_array_like(value):
        arrays.append(value)
        return _Slot(len(arrays) - 1)
    parts = _parts(value)
    if parts is None:
        return value
    return _rebuilt(value, [_split(x, arrays) for x in parts])


def _filled(value: Any, arrays: Sequence) -> Any:
    """Return a plan value with the arrays put back in their slots."""
    if isinstance(value, _Slot):
        return arrays[value.index]
    parts = _parts(value)
    if parts is None:
        return value
    return _rebuilt(value, [_filled(x, arrays) for x in parts])


def _frozen(value: Any) -> Hashable:
    """
    Return a hashable stand-in for a value, equal only for equal values.

    For what a contract holds, and what a plan holds without its arrays,
    which is numbers, slices, strings and containers of them. A float
    stands for its exact spelling, so 0.0 and -0.0 differ.

    Raises
    ------
    TypeError
        If the value, or something in it, cannot be hashed.
    """
    if isinstance(value, float | np.floating):
        return type(value), float(value).hex()
    if is_dataclass(value) and not isinstance(value, type):
        items = ((x.name, getattr(value, x.name)) for x in fields(value))
        return type(value), tuple((k, _frozen(v)) for k, v in items)
    if type(value) is dict:
        return dict, tuple((k, _frozen(v)) for k, v in value.items())
    if (parts := _parts(value)) is not None:
        return type(value), tuple(_frozen(x) for x in parts)
    if isinstance(value, Mapping | list | tuple):
        # A container of a kind of its own may hold more than its items.
        raise TypeError(f"A {type(value).__name__} is keyed by more than its items.")
    hash(value)
    # By type too: True and 1 are equal, and a kernel may tell them apart.
    return type(value), value


def _signature(value: Any) -> Hashable:
    """Return what a compiled chain is specialized to of a value it is handed."""
    if is_array_like(value):
        return tuple(value.shape), _as_numpy_dtype(value.dtype)
    parts = _parts(value)
    if parts is None:
        return type(value)
    names = tuple(value) if type(value) is dict else None
    return type(value), names, tuple(_signature(x) for x in parts)


@dataclass(frozen=True)
class Step:
    """
    One step of a lazy array's chain, for one box, as a backend runs it.

    [`ArrayBackend.run_chain`](`dascore.backends.ArrayBackend.run_chain`)
    is handed the steps, so a backend which compiles a chain needs nothing
    else of DASCore: [`run`](`dascore.lazy.execute.Step.run`) runs one,
    and [`arrays`](`dascore.lazy.execute.Step.arrays`) and
    [`key`](`dascore.lazy.execute.Step.key`) say what a compiled chain is
    given and what it may be reused for.

    Parameters
    ----------
    processor
        The operation.
    kernel
        Its kernel for the backend, already resolved.
    plan
        The keywords the kernel takes, as `get_metadata` worked them out.
    contract
        What a box of the step's output needs of its input, or None, when
        the step is given all of its input.
    given
        The box of the step's input it is given.
    made
        The box of its output it makes.
    shape
        The shape of the step's whole output, as its metadata stated.
    dtype
        The dtype its metadata stated.
    """

    processor: Any
    kernel: Callable[..., Any]
    plan: Mapping[str, Any]
    contract: ChunkContract | None
    given: Box
    made: Box
    shape: tuple[int, ...]
    dtype: np.dtype

    @cached_property
    def _parts(self) -> tuple[dict[str, Any], tuple]:
        """Return the plan with its arrays taken out, and the arrays."""
        arrays: list = []
        named = type(self.processor).plan_arrays
        static = {k: _split(v, arrays, k in named) for k, v in self.plan.items()}
        return static, tuple(arrays)

    @property
    def arrays(self) -> tuple:
        """
        The arrays of the plan, wherever in it they are, in a fixed order.

        A backend which compiles a chain hands these to it as arguments, so
        a patch whose plan holds others of the same shapes, as filter
        coefficients for another sampling rate, runs what was compiled.
        Entries the processor names in `plan_arrays` are among them, though
        they are not arrays.
        """
        return self._parts[1]

    @property
    def static(self) -> dict[str, Any]:
        """The plan without its arrays, which a compiled chain holds fixed."""
        return self._parts[0]

    @cached_property
    def key(self) -> Hashable | None:
        """
        What a chain compiled with this step is specialized to.

        The kernel, and how many kernels had been registered when it was
        resolved, since one run inside it, as a window's operation is, may
        be another since; the operation, by its id; the plan without its
        arrays, and the shape and dtype of each array; the contract, and
        what it does with the boxes, which for most is where they are
        relative to each other and not in the array, so boxes of one shape
        cut alike along a patch share a key, and for a tiled step the boxes
        themselves; and the shape and dtype the metadata stated. A backend adds the
        shape and dtype of the data it is handed. None for a step whose
        operation has no id, as one holding a patch, or whose plan holds
        something which is not hashed by its value; its chain is not to be
        reused.
        """
        contract = self.contract or _EVERYTHING
        try:
            out = (
                self.kernel,
                _REGISTERED[0],
                type(self.processor),
                self.processor._identity(),
                _frozen(self.static),
                tuple(_signature(x) for x in self.arrays),
                _frozen(self.contract),
                contract.key(self.made, self.given),
                self.shape,
                self.dtype,
            )
            hash(out)
        # An operation's id is refused for whatever it holds, as comparing
        # two refuses them.
        except Exception:
            return None
        return out

    def run(self, data, arrays: Sequence | None = None):
        """
        Return the step's output box, computed from its input box.

        Parameters
        ----------
        data
            The input over `given`.
        arrays
            The arrays to run the plan with in place of its own, in the
            order of `arrays`, as a compiled chain is handed them.

        Raises
        ------
        PatchError
            If the kernel made an array of another shape or dtype than the
            step's metadata stated; see `check`.
        """
        plan = self.plan if arrays is None else _filled(self.static, arrays)
        run = partial(self.kernel, self.processor, **plan)
        out = (self.contract or _EVERYTHING).apply(run, data, self.made, self.given)
        self.check(out)
        # A step with no contract made the whole of its result.
        return out if self.contract else out[self.made]

    def check(self, out) -> None:
        """
        Raise if what the kernel returned is not what the metadata stated.

        Shapes and dtypes are known while a chain is traced, so a backend
        which compiles one can check each step as it is traced.

        Raises
        ------
        PatchError
            If the array is of another shape or dtype.
        """
        # A step with no contract is checked as a whole, before the box is
        # taken from it.
        expected = _sizes(self.made) if self.contract else self.shape
        shape, dtype = tuple(out.shape), _as_numpy_dtype(out.dtype)
        if shape != expected or dtype != self.dtype:
            name = type(self.processor).__name__
            msg = (
                f"{name} computed an array of shape {shape} and dtype "
                f"{dtype} where its metadata stated one of {expected} "
                f"and {self.dtype}. Its get_metadata must state what its "
                "kernel returns."
            )
            raise PatchError(msg)


def _needs(array: LazyArray, box: Box) -> list[Box]:
    """Return the box each step is given, the recipe's first and `box` last."""
    nodes = array.chain.nodes
    # The shape each step is given: the recipe's, then the step before's.
    shapes = [array.recipe.shape, *(x.made()[0] for x in nodes)][: len(nodes)]
    out = [box]
    for node, shape in zip(reversed(nodes), reversed(shapes), strict=True):
        out.append((node.contract or _EVERYTHING).needs(out[-1], shape))
    return out[::-1]


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

    Each step runs the kernel its processor has for the array's backend,
    and what it makes is checked against the dtype its metadata stated and
    the shape of the box. Each run of steps with a kernel for the library is
    handed to the backend's
    [`run_chain`](`dascore.backends.ArrayBackend.run_chain`), and a step
    which goes through NumPy is run between them. An empty box is returned
    without running anything.

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
    steps = [
        Step(
            node.processor,
            type(node.processor).kernel_for(native, under),
            node.plan,
            node.contract,
            before,
            after,
            *node.made()[:2],
        )
        for node, before, after in zip(
            array.chain, needed[:-1], needed[1:], strict=True
        )
    ]
    # A step with no kernel for the library is run by itself, with the
    # backend's chain run up to it and on from it: a backend never sees a
    # step which goes through NumPy. It was let through, or refused, when
    # it was chained, so it is not asked again for each box.
    names = [_converted(x.processor, backend) for x in steps]
    for name, group in groupby(zip(names, steps, strict=True), key=lambda x: x[0]):
        section = [x for _, x in group]
        if name is None:
            with running_under(backend):
                data = backend.run_chain(section, data)
            continue
        for step in section:
            with fallbacks_decided(name), running_under(backend):
                out = step.run(data)
            # One of the backend's arrays for the next step, as a patch in
            # memory holds between two operations.
            data = backend.from_array(out)
    return data


def load(array: LazyArray) -> np.ndarray:
    """Compute a lazy array a box at a time, and return the whole of it."""
    boxes = chunk_boxes(array)
    out = compute(array, boxes[0])
    if len(boxes) > 1 and is_numpy(out):
        # Written in place, so only one box is held beside the whole.
        first, out = out, np.empty(array.shape, array.dtype)
        out[boxes[0]] = first
        del first
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
