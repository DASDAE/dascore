"""
Region loads: abutting windows are merged and each resource is opened once.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from dascore._lazy.block import _Block
from dascore._lazy.columns import SOURCE_FIELDS, _dtype_of
from dascore.core.source import ArraySource, _text_offsets

# Merged reads of abutting windows are cut where a member starts this many
# bytes into its run, so one holds at most this and one member more.
_RUN_BYTES = 1 << 24

# The dtype kinds a merged read may convert between.
_NUMERIC = "biufc"


def _sources_of(block: _Block) -> list[ArraySource]:
    """Return the source each member of a block reads, in placement order."""
    sources, rows = block.members.sources, block.members.source_row
    at = {x: getattr(sources, x).at(rows) for x in SOURCE_FIELDS}
    src_axis = block.axes["src_axis"].tolist()
    src_start = block.axes["src_start"].tolist()
    lengths = (block.axes["out_stop"] - block.axes["out_start"]).tolist()
    extent = block.axes["src_extent"].tolist()
    filled = sources.filled[rows].tolist()
    out = []
    for member in range(len(block)):
        axes, starts = src_axis[member], src_start[member]
        spans, sizes = lengths[member], extent[member]
        windows: list = [()] * sum(1 for axis in axes if axis >= 0)
        stored: list = [0] * len(windows)
        for out_axis, axis in enumerate(axes):
            if axis < 0:
                continue
            start = starts[out_axis]
            windows[axis] = (start, start + spans[out_axis])
            stored[axis] = sizes[out_axis]
        out.append(
            ArraySource(
                path=at["base_uri"][member] + at["path"][member],
                format=at["format"][member],
                version=at["version"][member],
                key=at["key"][member],
                windows=tuple(windows),
                shape=tuple(stop - start for start, stop in windows),
                dtype=_dtype_of(at["dtype"][member]),
                origin_id=at["origin_id"][member],
                extent=tuple(stored),
                filled=filled[member],
                value=at["value"][member],
                offsets=_text_offsets(at["offsets"][member]),
                chunk_rows=at["chunk_rows"][member],
                mtime_ns=at["mtime_ns"][member],
                size_bytes=at["size_bytes"][member],
            )
        )
    return out


def _coalesced(block: _Block) -> _Block:
    """
    Return the block with abutting windows of one source merged into one member.

    Neighbours in placement order merge when they read one stored source
    through one cast and axis map, abut along one axis in both output and
    source, and match on every other axis. Only numeric conversions of one
    kind or wider merge. A run grows along one
    axis and is cut every `_RUN_BYTES` by where its members start.
    """
    count = len(block)
    if count < 2:
        return block
    axes, members = block.axes, block.members
    start, stop, src_axis = axes["out_start"], axes["out_stop"], axes["src_axis"]
    rows, dtypes = members.source_row, members.sources.dtype
    offset = start - axes["src_start"]
    casts, target = members.cast, np.dtype(block.dtype)
    # A conversion whose result may depend on how many samples it sees, as a
    # unit taken from text or a float out of an integer's range, must see
    # each window alone.
    via = [_dtype_of(x) if x else target for x in casts.values]
    smooth = np.array(
        [
            [_smooth(x, y) and _smooth(y, target) for y in via]
            for x in map(_dtype_of, dtypes.values)
        ]
    )
    alike = (rows[1:] == rows[:-1]) & smooth[dtypes.codes[rows[1:]], casts.codes[1:]]
    alike &= ~members.filled[1:] & (casts.codes[1:] == casts.codes[:-1])
    alike &= np.all(src_axis[1:] == src_axis[:-1], axis=1)
    alike &= np.all(offset[1:] == offset[:-1], axis=1)
    alike &= np.all(axes["src_extent"][1:] == axes["src_extent"][:-1], axis=1)
    if not alike.any():
        return block
    # Boxes are never empty, so neighbours which meet differ on that axis alone.
    differs = (start[1:] != start[:-1]) | (stop[1:] != stop[:-1])
    along, pick = np.argmax(differs, axis=1), np.arange(count - 1)
    joins = alike & (differs.sum(axis=1) == 1) & (src_axis[1:][pick, along] >= 0)
    joins &= (start[1:] == stop[:-1])[pick, along]
    # A run grows along one axis; turning to another starts a new one.
    joins[1:] &= ~(joins[:-1] & (along[1:] != along[:-1]))
    itemsize = max(
        np.dtype(block.dtype).itemsize,
        *[_dtype_of(x).itemsize for x in (*dtypes.values, *casts.values) if x],
    )
    # Where each member starts in its run, so a read holds at most the budget
    # and one member more.
    size = np.prod(stop - start, axis=1) * itemsize
    before = np.cumsum(size) - size
    new = np.r_[True, ~joins]
    offset = before - before[np.flatnonzero(new)][np.cumsum(new) - 1]
    piece = offset // _RUN_BYTES
    joins &= piece[1:] == piece[:-1]
    firsts = np.flatnonzero(np.r_[True, ~joins])
    if len(firsts) == count:
        return block
    lasts = np.r_[firsts[1:], count] - 1
    out = block.take(firsts)
    return replace(out, axes={**out.axes, "out_stop": stop[lasts]})


def _smooth(source: np.dtype, target: np.dtype) -> bool:
    """Whether a numeric conversion treats every sample alike, however many."""
    kinds = source.kind in _NUMERIC and target.kind in _NUMERIC
    return kinds and bool(np.can_cast(source, target, "same_kind"))


def _source_groups(block: _Block) -> list[list[int]]:
    """Return the stored members of each resource, each in placement order."""
    members = block.members
    rows = np.flatnonzero(~members.filled)
    if not len(rows):
        return []
    used, inverse = np.unique(members.source_row[rows], return_inverse=True)
    at = [getattr(members.sources, x).at(used) for x in SOURCE_FIELDS[:4]]
    # A path split under two base uris is one resource, opened once.
    names: dict[tuple[str, str, str], int] = {}
    codes = [names.setdefault((b + p, f, v), len(names)) for b, p, f, v in zip(*at)]
    group = np.array(codes, np.int64)[inverse.reshape(-1)]
    order = np.argsort(group, kind="stable")
    cuts = np.flatnonzero(np.diff(group[order])) + 1
    return [x.tolist() for x in np.split(rows[order], cuts)]


def _member_source(block: _Block, member: int) -> ArraySource:
    """Return the source one member of a block reads."""
    return _sources_of(block.take([member]))[0]


def _to_output(data: np.ndarray, axes: list[int]) -> np.ndarray:
    """Return a member's array with its stored axes on the output's."""
    order = [x for x in axes if x >= 0]
    if order != sorted(order):
        data = np.transpose(data, order)
    new = [index for index, axis in enumerate(axes) if axis < 0]
    return np.expand_dims(data, tuple(new)) if new else data
