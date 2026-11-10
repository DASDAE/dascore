"""
The id of a recipe, which no way of cutting it into members changes.

The members are grouped by what they read, each group's region is reduced to
a form unique to it, and the digest is taken over those groups.
"""

from __future__ import annotations

import hashlib
import itertools
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dascore._lazy.block import NEW_AXIS, _Block
from dascore._lazy.columns import SOURCE_FIELDS, _dtype_of
from dascore.core.source import ArraySource
from dascore.exceptions import ParameterError
from dascore.utils.identity import DIGEST_SIZE, H

# The version of the byte layout the data_id digest is taken over. Bump it
# when the layout changes, so ids from two layouts cannot meet.
DIGEST_LAYOUT = 3

# The most axes one array may be cut on; each one doubles the corners.
MAX_CUT_AXES = 16

# The bytes the digest starts with, so no other payload can read alike.
_DIGEST_TAG = b"dascore-lazy-blocks\0"


@dataclass(frozen=True, eq=False)
class _Signature:
    """
    What an array is, whatever the cuts which made it.

    `names` is the kind and id of each distinct source and `name_code` says
    which one each group reads. The rest hold one row per group, in
    canonical order: a member of it, the rest of its key, the start and the
    stop of each output axis it spans whole, and how many corners it kept.
    An axis whose start and stop are equal is one the region does not span,
    and the corners, lexicographic within the group, hold those axes alone.
    """

    names: list[tuple[int, str]]
    name_code: np.ndarray
    rows: np.ndarray
    fields: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    counts: np.ndarray
    corners: np.ndarray
    signs: np.ndarray


def _group_names(block: _Block) -> tuple[list[tuple[int, str]], np.ndarray]:
    """Return the id of each distinct source, and which one each member reads."""
    members = block.members
    sources, casts = members.sources, members.cast
    width = len(casts.values)
    # Each source row read is named once for each cast it is read through.
    pairs = members.source_row.astype(np.int64) * width + casts.codes
    pairs, code = np.unique(pairs, return_inverse=True)
    rows, vias = np.divmod(pairs, width)
    at = {x: getattr(sources, x).at(rows) for x in SOURCE_FIELDS}
    filled = sources.filled[rows].tolist()
    names: list[tuple[int, str]] = []
    for index, via in enumerate([casts.values[x] for x in vias.tolist()]):
        if filled[index]:
            # The shape is the region, which the corners hold, so it is not here.
            content = {"value": at["value"][index], "dtype": at["dtype"][index]}
            name = H("constant", content)
        else:
            location = {
                "path": at["base_uri"][index] + at["path"][index],
                "format": at["format"][index],
                "version": at["version"][index],
                "key": at["key"][index],
            }
            name = at["origin_id"][index] or H("location", location)
        # samples put through another dtype on the way are other samples
        if via:
            name = H("cast", [name, via])
        names.append((int(filled[index]), name))
    # One id reached two ways is one source, and the order is the ids' own.
    ranks = {name: rank for rank, name in enumerate(sorted(set(names)))}
    lookup = np.array([ranks[x] for x in names], np.int64)
    return list(ranks), lookup[code.reshape(-1)]


def _placement(block: _Block) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the axis map, offset and extent each member is laid by."""
    axes = block.axes
    # A constant is its value, however it is laid; a broadcast axis reads
    # nothing, so where it sits in the output says nothing either.
    reads = (axes["src_axis"] >= 0) & ~block.members.filled[:, None]
    src_axis = np.where(reads, axes["src_axis"], NEW_AXIS)
    offset = np.where(reads, axes["out_start"] - axes["src_start"], 0)
    extent = np.where(reads, axes["src_extent"], 0)
    return src_axis, offset, extent


def _grouped(keys: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the rows in key order, and where each run of equal keys starts.

    One group is one run of every row, so the rows are in order already and
    the caller is handed the order it would put them in.
    """
    count = len(keys[-1])
    # A key which is one value throughout can neither sort nor split.
    active = [x for x in keys if x.max() != x.min()]
    if not active:
        return np.arange(count), np.zeros(1, np.int64)
    order = np.lexsort(active)
    new = np.zeros(count, bool)
    new[0] = True
    for key in active:
        values = key[order]
        new[1:] |= values[1:] != values[:-1]
    return order, np.flatnonzero(new)


def _check_cuts(cut: int, ndim: int) -> None:
    """Refuse an array cut on more axes than the corners can be held for."""
    if cut > MAX_CUT_AXES:
        msg = (
            f"An array of {ndim} dimensions is cut on {cut} axes, and at "
            f"most {MAX_CUT_AXES} may be, since every one of them doubles "
            "the corners its id is taken over."
        )
        raise ParameterError(msg)


def _cut_axes(
    low: np.ndarray, high: np.ndarray, inner: np.ndarray, ndim: int
) -> np.ndarray:
    """Return the axes some group of boxes does not all span alike."""
    out = []
    for axis in range(ndim):
        moved = low[1:, axis] != low[:-1, axis]
        moved |= high[1:, axis] != high[:-1, axis]
        moved &= inner
        if moved.any():
            out.append(axis)
    return np.array(out, np.intp)


def _corners(low: np.ndarray, high: np.ndarray, owner: np.ndarray) -> tuple:
    """Return the signed corners over the cut axes, summed where they meet."""
    width = low.shape[1]
    count = len(owner)
    if width == 1:
        # One cut axis alternates its bounds, which a stack lays in order.
        points = np.empty((2 * count, 1), np.int64)
        points[0::2], points[1::2] = low, high
        signs = np.empty(2 * count, np.int64)
        signs[0::2], signs[1::2] = 1, -1
        group = np.repeat(owner, 2)
        steps = points[1:, 0] >= points[:-1, 0]
        ordered = bool(np.all(steps | (group[1:] != group[:-1])))
    else:
        bits = _patterns(width)
        points = np.empty((count * len(bits), width), np.int64)
        signs = np.empty(count * len(bits), np.int64)
        for slot, row in enumerate(bits):
            piece = slice(slot * count, (slot + 1) * count)
            points[piece] = np.where(row, high, low)
            signs[piece] = 1 - 2 * (int(row.sum()) & 1)
        group = np.tile(owner, len(bits))
        ordered = False
    if not ordered:
        place = np.lexsort([*[points[:, x] for x in reversed(range(width))], group])
        points, signs, group = points[place], signs[place], group[place]
    new = np.ones(len(group), bool)
    new[1:] = (points[1:] != points[:-1]).any(axis=1) | (group[1:] != group[:-1])
    index = np.flatnonzero(new)
    total = np.add.reduceat(signs, index)
    # Only the corners which are kept are worth gathering.
    alive = total != 0
    keep = index[alive]
    return points[keep], total[alive], group[keep]


def _factors(column: np.ndarray, holder: np.ndarray, groups: int) -> tuple:
    """Return the groups whose corners hold two coordinates of an axis."""
    sizes = np.bincount(holder, minlength=groups).astype(np.intp)
    present = np.flatnonzero(sizes)
    heads = np.zeros(groups + 1, np.intp)
    np.cumsum(sizes, out=heads[1:])
    low = np.zeros(groups, np.int64)
    high = np.zeros(groups, np.int64)
    low[present] = np.minimum.reduceat(column, heads[present])
    high[present] = np.maximum.reduceat(column, heads[present])
    edge = (column == low[holder]) | (column == high[holder])
    whole = np.bincount(holder[edge], minlength=groups) == sizes
    return (low < high) & whole, low, high


def _regions(
    block: _Block, order: np.ndarray, starts: np.ndarray, groups: int
) -> tuple:
    """
    Return what each group covers, in a form unique to the region.

    A box gives each of its corners the sign `(-1) ** upper bounds`, the
    signs of corners which fall together are summed, and a corner which
    sums to zero drops out. That is the mixed difference of how often the
    boxes cover each sample, and summing it back gives the region, so the
    corners say what is covered and never how it was cut up.

    Only the axes some group was cut on are expanded, so the corners cost
    two to the power of the cuts rather than of the dimensions. An axis
    whose corners hold two coordinates is then pulled back out as a plain
    interval, since a region whose mixed difference stands on two
    coordinates of an axis is that interval of it times the rest. What is
    left is the region's own, however it was cut and whichever axes were
    expanded to reach it.
    """
    ndim = block.ndim
    # One group is one run of every member, which is in order already.
    low, high = block.axes["out_start"], block.axes["out_stop"]
    if groups > 1:
        low, high = low[order], high[order]
    sizes = np.diff(np.append(starts, len(low))).astype(np.intp)
    # An axis every box of a group spans alike is a plain interval of it.
    same = np.ones(len(low), bool)
    same[starts] = False
    cut = _cut_axes(low, high, same[1:], ndim)
    _check_cuts(len(cut), ndim)
    plain = np.ones(ndim, bool)
    plain[cut] = False
    # An axis the region does not span is written as no interval at all.
    first = np.where(plain, low[starts], -1)
    last = np.where(plain, high[starts], -1)
    if len(cut):
        owner = np.repeat(np.arange(groups, dtype=np.intp), sizes)
        points, signs, holder = _corners(low[:, cut], high[:, cut], owner)
    else:
        # Nothing cut leaves one corner a group: how many boxes it lays.
        points = np.zeros((groups, 0), np.int64)
        signs = sizes.astype(np.int64)
        holder = np.arange(groups, dtype=np.intp)
    dead = ((first == last) & plain).any(axis=1)
    if dead.any():
        alive = ~dead[holder]
        points, signs, holder = points[alive], signs[alive], holder[alive]
    for column, axis in enumerate(cut):
        found, low_of, high_of = _factors(points[:, column], holder, groups)
        if not found.any():
            continue
        first[found, axis] = low_of[found]
        last[found, axis] = high_of[found]
        keep = ~(found[holder] & (points[:, column] == high_of[holder]))
        points, signs, holder = points[keep], signs[keep], holder[keep]
    counts = np.bincount(holder, minlength=groups).astype(np.int64)
    if not counts.all():
        # A group which covers nothing is described by nothing at all.
        empty = counts == 0
        first[empty] = -1
        last[empty] = -1
    core = (first[:, cut] == last[:, cut])[holder]
    return first, last, counts, points[core], signs


def _patterns(count: int) -> np.ndarray:
    """Return every choice of a lower or an upper bound, in order."""
    return np.array(list(itertools.product((False, True), repeat=count)), bool)


def _signature(block: _Block) -> _Signature:
    """
    Return what an array is, whatever the cuts which made it.

    Members are grouped by everything except the region they cover: which
    source they read, how its axes lie on the output, how far the output is
    from the source on each of them, and how long the whole source is. Two
    members of one group read one sample wherever their boxes meet one
    output sample, so the region alone says what a group holds, and
    [`_regions`](`dascore._lazy.identity._regions`) says what that is.
    """
    ndim = block.ndim
    count = len(block)
    if not count:
        return _Signature(
            names=[],
            name_code=np.zeros(0, np.int64),
            rows=np.zeros(0, np.int64),
            fields=np.zeros((0, 3 * ndim + 1), np.int64),
            starts=np.zeros((0, ndim), np.int64),
            stops=np.zeros((0, ndim), np.int64),
            counts=np.zeros(0, np.int64),
            corners=np.zeros(0, np.int64),
            signs=np.zeros(0, np.int64),
        )
    names, code = _group_names(block)
    src_axis, offset, extent = _placement(block)
    keys = [extent[:, x] for x in reversed(range(ndim))]
    keys += [offset[:, x] for x in reversed(range(ndim))]
    keys += [src_axis[:, x] for x in reversed(range(ndim))]
    keys.append(code)
    order, starts = _grouped(keys)
    groups = len(starts)
    first, last, counts, corners, signs = _regions(block, order, starts, groups)
    rows = order[starts]
    name_code = code[rows]
    kinds = np.array([x[0] for x in names], np.int64)[name_code]
    fields = np.concatenate(
        [kinds[:, None], src_axis[rows], offset[rows], extent[rows]], axis=1
    )
    return _Signature(
        names=names,
        name_code=name_code,
        rows=rows,
        fields=fields,
        starts=first,
        stops=last,
        counts=counts,
        corners=corners,
        signs=signs,
    )


def _whole_source_id(block: _Block, signature: _Signature) -> str | None:
    """Return the id of the one source an array is, if that is what it is."""
    ndim = block.ndim
    shape = np.asarray(block.shape, np.int64)
    # One group which spans every axis whole, with one corner left over.
    if len(signature.counts) != 1 or signature.counts[0] != 1:
        return None
    # The sign says how often the boxes cover a sample, which must be once.
    if signature.signs[0] != 1 or not np.array_equal(signature.stops[0], shape):
        return None
    if signature.starts[0].any():
        return None
    row = int(signature.rows[0])
    members = block.members
    sources, source = members.sources, int(members.source_row[row])
    if block.dtype != _dtype_of(sources.dtype[source]):
        return None
    kind, name = signature.names[int(signature.name_code[0])]
    if kind:
        # A constant is the array of its value, and a cast makes it another.
        if members.cast[row]:
            return None
        value = sources.value[source]
        source = ArraySource(filled=True, value=value)
        return source.describe(block.shape, block.dtype).data_id
    src_axis, offset, extent = signature.fields[0, 1:].reshape(3, ndim)
    if not np.array_equal(src_axis, np.arange(ndim)):
        return None
    windows = tuple((-int(x), -int(x) + int(y)) for x, y in zip(offset, shape))
    whole = tuple((0, int(x)) for x in extent)
    return name if windows == whole else H("window", [name, windows])


def _fold(value: str) -> bytes:
    """Return 16 bytes for one id, hashed as written so no two ids fold alike."""
    return hashlib.blake2b(value.encode(), digest_size=DIGEST_SIZE).digest()


def _digest(block: _Block, signature: _Signature) -> str:
    r"""
    Return the digest of what an array is.

    The bytes hashed are, in order: the tag `dascore-lazy-blocks\0`; the
    layout version, the ndim, the number of groups and the number of
    corners, as four little-endian int64; then six tables in canonical
    group order, whose lengths those counts and the ndim give.
    The 16 bytes naming each group's source; the rest of each group's key,
    as `3 * ndim + 1` little-endian int64, which are its kind and then the
    `src_axis`, the offset and the `src_extent` of each output axis; the
    start and then the stop of every output axis the group spans whole,
    `ndim` int64 each, an axis it does not span being written as one
    number twice; how many corners each group kept; every corner, as many
    int64 as its group has axes it does not span; and the sign of every
    corner. Paths never appear in these bytes; a group's name may be
    derived from one.
    """
    out = hashlib.blake2b(digest_size=DIGEST_SIZE)
    out.update(_DIGEST_TAG)
    groups, corners = len(signature.counts), len(signature.signs)
    out.update(struct.pack("<4q", DIGEST_LAYOUT, block.ndim, groups, corners))
    raw = b"".join(_fold(x) for _, x in signature.names)
    ids = np.frombuffer(raw, np.uint8).reshape(len(signature.names), 16)
    out.update(np.ascontiguousarray(ids[signature.name_code]))
    for table in (
        signature.fields,
        signature.starts,
        signature.stops,
        signature.counts,
        signature.corners,
        signature.signs,
    ):
        out.update(np.ascontiguousarray(table, dtype="<i8"))
    return out.hexdigest()
