"""
One array's members and where they are placed.

A block is the header and members of one array before they are stacked in a
table, and a view is a window of a block with the order its axes are shown
in. Clipping, transposing and canonical ordering work on blocks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

from dascore._lazy.columns import (
    _LAYOUT_DEFAULTS,
    SOURCE_FIELDS,
    _broadcast,
    _Column,
    _dtype_text,
    _Members,
    _sources_of_columns,
)
from dascore.core.source import ArraySource, _offsets_text
from dascore.exceptions import ParameterError

# The src_axis of an output axis which no stored axis feeds.
NEW_AXIS = -1

# The per member per axis matrices, in the order the digest takes them.
AXIS_FIELDS = ("out_start", "out_stop", "src_axis", "src_start", "src_extent")


@dataclass(frozen=True, eq=False)
class _Block:
    """One array's header and members, before they are stacked in a table."""

    shape: tuple[int, ...]
    dtype: np.dtype
    concat_axis: int
    members: _Members
    axes: dict[str, np.ndarray]

    @property
    def ndim(self) -> int:
        """The number of output axes."""
        return len(self.shape)

    def __len__(self) -> int:
        return len(self.members)

    def take(self, rows) -> _Block:
        """Return the block holding a selection of members."""
        axes = {name: matrix[rows] for name, matrix in self.axes.items()}
        return replace(self, members=self.members.take(rows), axes=axes)


class _View(NamedTuple):
    """A window, in a base block's samples, and the order its axes are shown in."""

    block: _Block
    starts: tuple[int, ...]
    stops: tuple[int, ...]
    order: tuple[int, ...]
    rows: slice  # the run of members the window can touch


def _is_canonical(out_start: np.ndarray) -> bool:
    """Whether the corners are in lexicographic order; nothing is sorted."""
    if len(out_start) < 2:
        return True
    difference = np.diff(out_start, axis=0)
    first = np.argmax(difference != 0, axis=1)
    lead = np.take_along_axis(difference, first[:, None], axis=1)[:, 0]
    return bool(np.all(lead >= 0))


def _canonical(block: _Block) -> _Block:
    """Return the block in canonical placement order."""
    out_start = block.axes["out_start"]
    if _is_canonical(out_start):
        return block
    return block.take(np.lexsort(out_start.T[::-1]))


def _clip(block: _Block, starts: np.ndarray, stops: np.ndarray) -> _Block:
    """Return the members a request selects, each clipped to it."""
    if _trims_ends(block, starts, stops):
        return _clip_ends(block, starts, stops)
    start = np.maximum(block.axes["out_start"], starts)
    stop = np.minimum(block.axes["out_stop"], stops)
    keep = np.flatnonzero(np.all(stop > start, axis=1))
    start, stop = start[keep], stop[keep]
    out = block.take(keep)
    axes = out.axes
    axes["src_start"] += np.where(axes["src_axis"] >= 0, start - axes["out_start"], 0)
    axes["out_start"] = start - starts
    axes["out_stop"] = stop - starts
    _flatten_constants(axes, out.members.filled)
    out = replace(out, shape=tuple((stops - starts).tolist()))
    # Stacked along the first axis, only one corner can be clipped, so the
    # members stay in order; any other placement may tie two of them.
    return out if block.concat_axis == 0 else _canonical(out)


def _trims_ends(block: _Block, starts: np.ndarray, stops: np.ndarray) -> bool:
    """Whether a window cuts only the first and last of the members stacked."""
    axis = block.concat_axis
    if axis < 0 or not len(block) or starts[axis] >= stops[axis]:
        return False
    shape = np.array(block.shape)
    rest = np.arange(block.ndim) != axis
    if starts[rest].any() or np.any(stops[rest] != shape[rest]):
        return False
    start, stop = block.axes["out_start"], block.axes["out_stop"]
    ends = stop[0, axis] > starts[axis] and start[-1, axis] < stops[axis]
    # Boxes of a valid array are inside it, never empty, and follow one
    # another along the axis they are stacked on.
    inside = start.min() >= 0 and (stop <= shape).all() and (stop > start).all()
    ordered = (start[1:, axis] >= stop[:-1, axis]).all()
    return bool(ends and inside and ordered)


def _clip_ends(block: _Block, starts: np.ndarray, stops: np.ndarray) -> _Block:
    """Clip the first and last member; the rest are copied, so the base can go."""
    axis, members, filled = block.concat_axis, block.members, block.members.filled
    start, stop, src_axis = (block.axes[x] for x in AXIS_FIELDS[:3])
    moved = max(int(starts[axis] - start[0, axis]), 0)
    flat = bool(filled.any())
    axes = {
        "out_start": start - starts,
        "out_stop": stop - starts,
        "src_axis": _owned(src_axis),
        "src_start": _owned(block.axes["src_start"], flat or bool(moved)),
        "src_extent": _owned(block.axes["src_extent"], flat),
    }
    axes["out_start"][0, axis] += moved
    axes["out_stop"][-1, axis] = min(stop[-1, axis], stops[axis]) - starts[axis]
    if moved and src_axis[0, axis] >= 0:
        axes["src_start"][0, axis] += moved
    _flatten_constants(axes, filled)
    cast = _Column(members.cast.values, _owned(members.cast.codes))
    members = _Members(members.sources, _owned(members.source_row), cast)
    shape = tuple((stops - starts).tolist())
    out = replace(block, shape=shape, members=members, axes=axes)
    return out if axis == 0 else _canonical(out)


def _owned(array: np.ndarray, write: bool = False) -> np.ndarray:
    """Return a copy of an array; one value broadcast is shared unless written."""
    return array if not (write or any(array.strides)) else array.copy()


def _transposed(block: _Block, order: tuple[int, ...]) -> _Block:
    """Return the block with its output axes permuted."""
    shape = tuple(block.shape[x] for x in order)
    concat_axis = block.concat_axis
    concat_axis = order.index(concat_axis) if concat_axis >= 0 else NEW_AXIS
    axes = {name: matrix[:, order] for name, matrix in block.axes.items()}
    moved = replace(block, shape=shape, concat_axis=concat_axis, axes=axes)
    return _canonical(moved)


def _candidates(block: _Block, starts, stops, rows: slice) -> slice:
    """
    Return the members of a run which a window can touch.

    The run is narrowed by binary search on the axis the members are laid
    along; a block laid along no one axis keeps the whole run.
    """
    axis = block.concat_axis
    if axis < 0:
        return rows
    low, high = rows.start, rows.stop
    first = low + np.searchsorted(
        block.axes["out_stop"][low:high, axis], starts[axis], "right"
    )
    last = low + np.searchsorted(
        block.axes["out_start"][low:high, axis], stops[axis], "left"
    )
    return slice(int(first), int(max(first, last)))


def _block_of_sources(
    sources: Sequence[ArraySource], base_uri: str = "", casts: Sequence | None = None
) -> _Block:
    """Return a block which reads each source whole, placed at the origin."""
    ndim = sources[0].ndim
    for source in sources:
        if not source.loadable:
            msg = f"{source} does not say enough to load an array."
            raise ParameterError(msg)
        if source.ndim != ndim:
            msg = "Every source of one lazy array must have the same ndim."
            raise ParameterError(msg)
        if len(source.extent) != ndim:
            msg = f"{source} does not state the extent of its whole array."
            raise ParameterError(msg)
    fields = ("path", "format", "version", "key", "origin_id", "filled", "value")
    fields += tuple(_LAYOUT_DEFAULTS)
    columns: dict[str, Any] = {x: [getattr(y, x) for y in sources] for x in fields}
    columns["offsets"] = [_offsets_text(x.offsets) for x in sources]
    columns["source_dtype"] = [np.dtype(x.dtype) for x in sources]
    if casts is not None and is_list_like(casts):
        # Normalized first, since a structured spec is a list, which no key is.
        casts = [None if x is None else np.dtype(x) for x in casts] or None
    columns["cast_via"] = casts
    lengths = np.array([x.shape for x in sources], np.int64)
    start = np.array([[w[0] for w in x.windows] for x in sources], np.int64)
    extent = np.array([x.extent for x in sources], np.int64)
    return _block_of_columns(columns, lengths, start, extent, base_uri)


def _block_of_columns(
    columns: Mapping[str, Any],
    lengths: np.ndarray,
    start: np.ndarray,
    extent: np.ndarray,
    base_uri: str,
) -> _Block:
    """Return a block whose members read one row of each column, at the origin."""
    count, ndim = lengths.shape
    if ndim == 0:
        msg = "A lazy array takes sources of at least one dimension."
        raise ParameterError(msg)
    if ((lengths < 0) | (start < 0) | (start + lengths > extent)).any():
        msg = "Some windows have a negative length or fall outside the array they read."
        raise ParameterError(msg)
    path = _broadcast(columns["path"], count, "path")
    filled = np.zeros(count, bool) | columns.get("filled", False)
    split = [_split_path(x, base_uri) for x in path.values]
    base = _Column.of([x[0] for x in split])
    dtypes = _broadcast(columns["source_dtype"], count, "source_dtype")
    if any(x is None or x is pd.NA or x != x for x in dtypes.values):
        msg = "Every member needs a source_dtype."
        raise ParameterError(msg)
    axes = {
        "out_start": np.zeros((count, ndim), np.int64),
        "out_stop": lengths,
        "src_axis": np.tile(np.arange(ndim, dtype=np.int64), (count, 1)),
        "src_start": start,
        "src_extent": extent,
    }
    fields = {
        "base_uri": _Column(base.values, base.codes[path.codes]),
        "path": _Column([x[1] for x in split], path.codes),
        **{x: _broadcast(columns[x], count, x) for x in SOURCE_FIELDS[2:6]},
        "dtype": dtypes.map(_dtype_text),
        **{
            x: _broadcast(columns.get(x, default), count, x)
            for x, default in _LAYOUT_DEFAULTS.items()
        },
        "value": _broadcast(columns.get("value"), count, "value"),
    }
    cast = _broadcast(columns["cast_via"], count, "cast_via").map(
        lambda x: "" if x is None else _dtype_text(x)
    )
    members = _Members(*_sources_of_columns(fields, filled), cast)
    _flatten_constants(axes, filled)
    dtype = np.result_type(*[np.dtype(x) for x in dtypes.values])
    return _Block((0,) * ndim, dtype, NEW_AXIS, members, axes)


def _flatten_constants(axes: dict[str, np.ndarray], filled: np.ndarray) -> None:
    """Set the source fields of every constant member to its box."""
    rows = np.flatnonzero(filled)
    if not len(rows):
        return
    lengths = axes["out_stop"][rows] - axes["out_start"][rows]
    axes["src_start"][rows] = 0
    axes["src_extent"][rows] = np.where(axes["src_axis"][rows] >= 0, lengths, 0)


def _split_path(path: str, base_uri: str) -> tuple[str, str]:
    """Split a path into the prefix it is stored under and the rest."""
    if base_uri and path.startswith(base_uri):
        return base_uri, path[len(base_uri) :]
    return "", path
