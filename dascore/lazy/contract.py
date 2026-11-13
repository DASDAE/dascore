"""
Chunk contracts: what one chunk of an operation's output needs of its input.

An operation which can run on a lazy array says so by returning a contract
from `PatchProcessor.chunk_contract`, for the input's dtype and the plan
`get_metadata` worked out. The contract is what lets the work be split
into chunks without changing the result: each output chunk is computed
from the input it names, and nothing else. Returning `None` says the
operation cannot run on a lazy array.

A chunk is a box: one slice for each axis, with a start and a stop and a
step of one. A contract answers two questions about one.
[`needs`](`dascore.lazy.contract.ChunkContract.needs`) says which box of
the input an output box is computed from, and
[`apply`](`dascore.lazy.contract.ChunkContract.apply`) computes the output
box from that input. Neither chooses a chunk size or loops over chunks:
[`dascore.lazy.execute`](`dascore.lazy.execute`) does both.

This module imports nothing from the rest of DASCore, because the
processors in `dascore.proc` and `dascore.transform` import it.

Examples
--------
>>> import numpy as np
>>> import dascore as dc
>>> from dascore.lazy.contract import Pointwise, Unchanged
>>>
>>> class Double(dc.PatchProcessor):
...     '''Double every sample.'''
...
...     name = None  # so this example claims no name of its own
...
...     def chunk_contract(self, dtype, plan):
...         return Pointwise()
...
...     def kernel(self, data):
...         return data * 2
>>>
>>> assert isinstance(Double().chunk_contract(np.dtype("f8"), {}), Pointwise)
>>> assert dc.PatchProcessor().chunk_contract(np.dtype("f8"), {}) is None
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

# One slice for each axis, each with an int start and stop and a step of one.
Box = tuple[slice, ...]


@dataclass(frozen=True)
class ChunkContract:
    """What one chunk of an operation's output needs of its input."""

    def needs(self, box: Box, shape: tuple[int, ...]) -> Box:
        """
        Return the box of the input an output box is computed from.

        Parameters
        ----------
        box
            The box of the output wanted.
        shape
            The shape of the whole input.
        """
        return box

    def apply(self, run: Callable[[Any], Any], data, box: Box, needed: Box):
        """
        Return the output box, computed from the input it needs.

        Parameters
        ----------
        run
            The operation's kernel, taking the data alone.
        data
            The input over `needed`.
        box
            The box of the output wanted.
        needed
            What [`needs`](`dascore.lazy.contract.ChunkContract.needs`)
            returned for `box`.
        """
        out = run(data)
        # A result which is not the shape of what went in is handed back as
        # it is, for the caller to refuse: trimming it would hide the fault.
        if tuple(out.shape) != tuple(data.shape):
            return out
        # The result starts where its input did, so the box sits in it at
        # the distance the input was grown by.
        spans = zip(box, needed, strict=True)
        return out[tuple(slice(b.start - n.start, b.stop - n.start) for b, n in spans)]


@dataclass(frozen=True)
class Pointwise(ChunkContract):
    """
    An output chunk needs only the input chunk at the same place.

    The output has the input's shape, and each sample is computed from the
    input sample at its own position alone.
    """


@dataclass(frozen=True)
class Unchanged(ChunkContract):
    """
    The kernel hands its input back, for this dtype and plan.

    The step does nothing, so it adds nothing to the chain; the patch is
    handed back as it is, with its own id, as the eager no-op does. Only
    for a no-op decided by the dtype or the plan, never by the values.
    """


@dataclass(frozen=True)
class Halo(ChunkContract):
    """
    An output chunk needs the input chunk and some samples around it.

    The output has the input's shape, and each sample is computed from the
    input samples within a fixed distance of its own position. A chunk is
    grown by that distance, stopping at the ends of the array, so the
    kernel treats the array's own ends as it does eagerly and what it
    makes of a cut end is trimmed off.

    Parameters
    ----------
    widths
        For each axis the kernel reaches along, how many samples before
        and after an output sample it reads.

    Examples
    --------
    >>> from dascore.lazy.contract import Halo
    >>> halo = Halo({0: (2, 3)})
    >>> box = (slice(10, 20), slice(0, 5))
    >>> assert halo.needs(box, (100, 5)) == (slice(8, 23), slice(0, 5))
    >>> assert halo.needs(box, (21, 5)) == (slice(8, 21), slice(0, 5))
    """

    widths: Mapping[int, tuple[int, int]]

    def needs(self, box: Box, shape: tuple[int, ...]) -> Box:
        """Return the box grown by the widths, inside the array."""
        out = list(box)
        for axis, (before, after) in self.widths.items():
            start = max(box[axis].start - before, 0)
            out[axis] = slice(start, min(box[axis].stop + after, shape[axis]))
        return tuple(out)


@dataclass(frozen=True)
class WholeAxis(ChunkContract):
    """
    An output chunk needs the whole of the input along some axes.

    Along every other axis it needs the input at the same place, so the
    work can still be split along those.

    Parameters
    ----------
    axes
        The axes the kernel needs whole.

    Examples
    --------
    >>> from dascore.lazy.contract import WholeAxis
    >>> box = (slice(10, 20), slice(2, 3))
    >>> assert WholeAxis((0,)).needs(box, (100, 5)) == (slice(0, 100), slice(2, 3))
    """

    axes: tuple[int, ...]

    def needs(self, box: Box, shape: tuple[int, ...]) -> Box:
        """Return the box with each of the axes whole."""
        out = list(box)
        for axis in self.axes:
            out[axis] = slice(0, shape[axis])
        return tuple(out)


@dataclass(frozen=True)
class Window(ChunkContract):
    """
    The output is a window of the input: a slice with a step of one per axis.

    Nothing is computed. An output chunk is the input chunk moved by where
    the window starts, so what is read is only what the window keeps.

    Parameters
    ----------
    index
        The slice of each axis the window keeps, as given to the array.

    Examples
    --------
    >>> from dascore.lazy.contract import Window
    >>> window = Window((slice(5, None), slice(None)))
    >>> box = (slice(0, 10), slice(0, 3))
    >>> assert window.needs(box, (100, 3)) == (slice(5, 15), slice(0, 3))
    """

    index: tuple[slice, ...]

    def needs(self, box: Box, shape: tuple[int, ...]) -> Box:
        """Return the box moved to where the window puts it in the input."""
        spans = zip(box, self.index, shape, strict=True)
        starts = [(b, item.indices(size)[0]) for b, item, size in spans]
        return tuple(slice(b.start + at, b.stop + at) for b, at in starts)

    def apply(self, run: Callable[[Any], Any], data, box: Box, needed: Box):
        """Return the input, which is already the window's samples."""
        return data
