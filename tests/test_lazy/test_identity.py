"""Tests for the id of a lazy array, whatever cuts made it."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from dascore.core.source import ArraySource
from dascore.exceptions import ParameterError
from dascore.io import core as io_core
from dascore.lazy import identity as identity_module
from dascore.lazy.block import NEW_AXIS
from dascore.lazy.identity import MAX_CUT_AXES
from dascore.lazy.recipe import LazyArray, LazyTable, concat, stack
from dascore.utils.identity import H
from tests.test_lazy.conftest import (
    constant,
    grid,
    many_members,
    mixed,
    placed,
    random_boxes,
    stored,
)


@pytest.fixture()
def expanded(monkeypatch):
    """Record the axes a signature expands, and cap what it may ask for."""
    asked = []
    patterns, check = identity_module._patterns, identity_module._check_cuts

    def counted(cut, ndim):
        asked.append(cut)
        assert cut <= 4, f"the signature expanded {cut} axes"
        check(cut, ndim)

    def capped(count):
        assert count <= 4, f"the signature asked for {2**count} patterns"
        return patterns(count)

    monkeypatch.setattr(identity_module, "_patterns", capped)
    monkeypatch.setattr(identity_module, "_check_cuts", counted)
    return asked


def squares():
    """Return two whole square sources side by side in one array."""
    first = stored((4, 4), path="/a/one.h5", origin_id="a" * 32)
    second = stored((4, 4), path="/a/two.h5", origin_id="b" * 32)
    return placed([first, second], [[0, 0], [0, 4]], (4, 8))


def bent(array, changes):
    """Return the array with some placement entries changed."""
    frame = array.to_frame()
    for name, member, out_axis, value in changes:
        row = (frame["ordinal"] == member) & (frame["out_axis"] == out_axis)
        frame.loc[row, name] = value
    return LazyArray.from_frame(frame, array.shape, array.dtype)


def random_tiling(rng, shape):
    """Return boxes which tile a shape, each grown from a free cell at random."""
    free = np.ones(shape, bool)
    out = []
    while free.any():
        cells = np.argwhere(free)
        low = cells[int(rng.integers(len(cells)))]
        high = low + 1
        while rng.random() > 0.3:
            axis = int(rng.integers(len(shape)))
            grown_low, grown_high = low.copy(), high.copy()
            if rng.random() < 0.5:
                grown_low[axis] -= 1
            else:
                grown_high[axis] += 1
            if grown_low[axis] < 0 or grown_high[axis] > shape[axis]:
                continue
            if free[tuple(map(slice, grown_low.tolist(), grown_high.tolist()))].all():
                low, high = grown_low, grown_high
        out.append((low, high))
        free[tuple(map(slice, low.tolist(), high.tolist()))] = False
    return out


def guillotine(boxes, shape):
    """Whether a tiling is one box, or some axis cuts clean through it."""
    if len(boxes) < 2:
        return True
    for axis in range(len(shape)):
        edges = {int(low[axis]) for low, _ in boxes} - {0}
        for edge in edges:
            if not any(low[axis] < edge < high[axis] for low, high in boxes):
                return True
    return False


def filler_source(filler, low, high):
    """Return the source one box reads: a window of a file, or a constant."""
    if isinstance(filler, ArraySource):
        return filler[tuple(map(slice, low.tolist(), high.tolist()))]
    return ArraySource.full(tuple((high - low).tolist()), filler)


def region_array(regions, shape):
    """Return the array which fills each box from the filler it was given."""
    sources = [filler_source(f, low, high) for low, high, f in regions]
    return placed(sources, [low.tolist() for low, _, _ in regions], shape)


def refine(regions, boxes):
    """Return each box cut down to the region which covers it."""
    out = []
    for low, high, filler in regions:
        for other_low, other_high in boxes:
            lo = np.maximum(low, other_low)
            hi = np.minimum(high, other_high)
            if np.all(hi > lo):
                out.append((lo, hi, filler))
    return out


def described(array):
    """Return the interval of each axis and the corner count of each group."""
    out = identity_module._signature(array._block())
    return [
        (list(zip(a, b)), n)
        for a, b, n in zip(out.starts.tolist(), out.stops.tolist(), out.counts.tolist())
    ]


def layouts(shape):
    """Return coarse layouts of a shape: one source, two, and one with a hole."""
    low = np.zeros(len(shape), np.int64)
    high = np.asarray(shape, np.int64)
    first = stored(shape, path="/a/one.h5", origin_id="a" * 32)
    second = stored(shape, path="/a/two.h5", origin_id="b" * 32)
    cut, start = high.copy(), low.copy()
    cut[0] = start[0] = shape[0] // 2
    return [
        [(low, high, first)],
        [(low, cut, first), (start, high, second)],
        [(low, cut, first), (start, high, 2.5)],
    ]


class TestIdentity:
    """The id says which array this is, and nothing about where it is."""

    def test_agrees_with_the_source(self, lazy, source):
        """One member in identity placement is the source's own array."""
        assert lazy.data_id == source.data_id
        assert lazy[10:20].data_id == source[10:20].data_id

    def test_split_windows_coalesce(self, lazy, source, patch):
        """Cutting one window into adjacent members does not rename it."""
        length = patch.shape[0]
        parts = [source[0:7], source[7:length]]
        array = concat([LazyArray.from_source(x) for x in parts], axis=0)
        assert array.data_id == lazy.data_id

    def test_member_size_is_not_in_the_id(self, joined):
        """How an array was cut up cannot reach its id."""
        first = joined.rechunk([0, 200, 601])
        second = joined.rechunk([0, 500, 601])
        assert concat(list(first), axis=0).data_id == joined.data_id
        assert concat(list(second), axis=0).data_id == joined.data_id

    def test_paths_are_not_in_the_id(self, source, path):
        """The same members under another base uri are the same array."""
        # A base is a literal prefix, so it ends as this platform's paths do.
        base = source.path[: -len(path.name)]
        under = LazyArray.from_source(source, base_uri=base)
        assert under.data_id == LazyArray.from_source(source).data_id
        assert under.to_frame()["base_uri"].iloc[0] == base
        assert under.source(0) == source

    def test_placement_is_in_the_id(self, joined):
        """Two arrays which place the same members differently differ."""
        assert joined.transpose().data_id != joined.data_id
        assert stack([joined]).data_id != joined.data_id

    def test_shape_and_dtype_are_in_the_id(self):
        """The header the members are placed in is part of the id."""
        pair = [constant((2, 3), 1.0), constant((2, 3), 2.0)]
        along_rows = concat(pair, axis=0)
        assert along_rows.data_id != concat(pair, axis=1).data_id
        assert (
            constant((2, 3), 1.0).data_id != constant((2, 3), 1.0, np.float32).data_id
        )

    def test_cached(self, joined):
        """An id is worked out once and kept."""
        assert joined.data_id == joined.data_id
        assert joined.table._ids[joined.row] == joined.data_id

    def test_empty_arrays(self, lazy):
        """An array of no members is named by its header."""
        assert lazy[0:0].data_id == lazy[0:0].data_id
        assert lazy[0:0].data_id != lazy[:, 0:0].data_id

    def test_location_names_an_unnamed_member(self):
        """A member with no origin is named by where it is, as a source is."""
        source = ArraySource(path="/a/b.h5", format="DASDAE", version="1").describe(
            (3, 2), np.float32
        )
        array = LazyArray.from_source(source)
        assert array.data_id == source.data_id
        assert array[0:2].data_id == source[0:2].data_id

    def test_odd_origin_id(self, lazy):
        """An origin which is not 32 hex characters is folded into the id."""
        frame = lazy.to_frame()
        frame["origin_id"] = "a stored name"
        array = LazyArray.from_frame(frame, lazy.shape, lazy.dtype)
        pair = concat([array[0:10], array[10:601]], axis=0)
        assert pair.data_id == array.data_id
        assert array.data_id == array.source(0).data_id

    def test_windows_of_many_files(self, joined, two_sources):
        """An array of several partial windows is named by all of them."""
        first, second = two_sources
        array = joined[50:150]
        assert len(array) == 2
        assert array.data_id == joined[50:150].data_id
        assert array.data_id != joined[51:151].data_id
        assert (
            array.data_id
            != concat(
                [
                    LazyArray.from_source(second[0:50]),
                    LazyArray.from_source(first[50:100]),
                ],
                axis=0,
            ).data_id
        )

    def test_staggered_members_do_not_merge(self):
        """Members which abut on one axis but are offset on another stay two."""
        whole = ArraySource(
            path="/a/b.h5", format="DASDAE", version="1", origin_id="b" * 32
        ).describe((4, 6), np.float32)
        parts = [whole[0:2, 0:3], whole[2:4, 3:6]]
        starts = np.array([[0, 0], [2, 3]])
        array = LazyArray.from_sources(parts, starts=starts, shape=(4, 6))
        assert array.data_id != LazyArray.from_source(whole).data_id
        assert (
            array.data_id
            == LazyArray.from_sources(parts, starts=starts, shape=(4, 6)).data_id
        )

    def test_constants_among_the_members(self, lazy, patch):
        """A constant which cannot merge is named by its own value."""
        width = patch.shape[1]
        first = concat([lazy, constant((5, width), 1.0)], axis=0)
        second = concat([lazy, constant((5, width), 2.0)], axis=0)
        assert first.data_id != second.data_id
        pair = concat([constant((2, 3), 1.0), constant((2, 3), 2.0)], axis=0)
        other = concat([constant((2, 3), 1.0), constant((2, 3), 3.0)], axis=0)
        assert pair.data_id != other.data_id

    @pytest.mark.parametrize("origin", ["", "a stored name", "abcd", "z" * 32])
    def test_folded_origin_ids(self, lazy, origin):
        """An origin which is not 32 hex characters still names its member."""
        frame = lazy.to_frame()
        frame["origin_id"] = origin
        array = LazyArray.from_frame(frame, lazy.shape, lazy.dtype)
        pair = concat([array[0:10], array[100:110]], axis=0)
        assert pair.data_id == concat([array[0:10], array[100:110]], axis=0).data_id
        assert pair.data_id != concat([array[0:10], array[101:111]], axis=0).data_id

    def test_is_an_operation_parameter(self, lazy):
        """An array carries its id where an operation hashes its arguments."""
        assert H("operation", {"array": lazy}) == H("operation", {"array": lazy})
        assert H("operation", {"array": lazy}) != H("operation", {"array": lazy[0:5]})

    def test_transposed_member_is_another_array(self):
        """A square source read the other way up is not the source's array."""
        source = stored((4, 4))
        array = LazyArray.from_source(source)
        assert array.data_id == source.data_id
        assert array.transpose().data_id != source.data_id
        assert len(array.transpose()) == 1

    @pytest.mark.parametrize(
        "changes",
        [
            [("src_axis", 0, 0, 1), ("src_axis", 0, 1, 0)],
            [("src_extent", 0, 0, 8)],
            [("out_stop", 0, 1, 3)],
            [("src_start", 0, 0, 1)],
        ],
    )
    def test_every_placement_matrix_is_in_the_id(self, changes):
        """Changing one entry of one placement matrix changes the id."""
        array = squares()
        assert array.data_id != bent(array, changes).data_id

    def test_swapped_axes_keep_the_member_ids(self):
        """The src_axis case differs in placement alone, not in its members."""
        array = squares()
        moved = bent(array, [("src_axis", 0, 0, 1), ("src_axis", 0, 1, 0)])
        assert [x.data_id for x in moved.sources] == [x.data_id for x in array.sources]

    def test_members_of_every_kind(self):
        """Named, unnamed, oddly named, windowed and constant members agree."""
        whole = stored((8, 3), path="/a/whole.h5", origin_id="c" * 32)
        unnamed = stored((8, 3), path="/a/unnamed.h5")
        odd = stored((8, 3), path="/a/odd.h5", origin_id="a stored name")
        members = [
            LazyArray.from_source(stored((2, 3), path="/a/one.h5", origin_id="d" * 32)),
            LazyArray.from_source(whole[0:2]),
            LazyArray.from_source(unnamed[3:5]),
            LazyArray.from_source(odd[6:8]),
            constant((2, 3), 5.0),
            constant((2, 3), 6, np.int16),
        ]
        array = concat(members, axis=0)
        assert array.data_id == concat(members, axis=0).data_id
        for index in range(len(members)):
            other = list(members)
            other[index] = constant((2, 3), 7.0, array.dtype)
            assert concat(other, axis=0).data_id != array.data_id

    def test_a_base_uri_does_not_rename_an_array(self):
        """A member stored under a prefix is the array its whole path names."""
        source = stored((8, 3), path="/data/alpha/f.h5")
        parts = [source[0:2], source[4:6]]
        starts = [[0, 0], [2, 0]]
        plain = placed(parts, starts, (4, 3))
        under = LazyArray.from_sources(
            parts, starts=np.array(starts), shape=(4, 3), base_uri="/data/alpha/"
        )
        assert len(under) == 2
        assert under.data_id == plain.data_id

    def test_matching_tails_under_two_bases(self):
        """Two files whose paths agree after the prefix are two arrays."""
        ids = []
        for base in ("/data/alpha/", "/data/beta/"):
            source = stored((8, 3), path=base + "f.h5")
            ids.append(
                LazyArray.from_sources(
                    [source[0:2], source[4:6]],
                    starts=np.array([[0, 0], [2, 0]]),
                    shape=(4, 3),
                    base_uri=base,
                ).data_id
            )
        assert ids[0] != ids[1]

    def test_the_arrays_dtype_is_in_the_id(self):
        """A member loaded at a promoted dtype is not the member's array."""
        big = constant((2, 2), 2**53 + 1, np.int64)
        other = constant((2, 2), 1.5)
        mixed = concat([big, other], axis=0)
        cut = mixed[0:2]
        assert cut.dtype == np.float64 and len(cut) == 1
        assert cut.data_id != big.data_id
        expected = np.concatenate([big.load(), other.load()])
        assert np.array_equal(mixed.load(), expected)

    def test_a_cast_member_is_not_its_source(self):
        """A stored array laid at another dtype is not the source's array."""
        source = stored((4, 3), origin_id="a" * 32)
        frame = LazyArray.from_source(source).to_frame()
        cast = LazyArray.from_frame(frame, (4, 3), np.float64)
        assert cast.dtype == np.float64
        assert cast.data_id != source.data_id
        assert LazyArray.from_source(source).data_id == source.data_id

    @pytest.mark.parametrize(
        "boxes",
        [
            [((0, 0), (2, 2)), ((0, 2), (1, 4)), ((1, 2), (2, 4))],
            [((0, 0), (1, 2)), ((1, 0), (2, 2)), ((0, 2), (2, 4))],
            [((0, 0), (2, 2)), ((0, 2), (2, 3)), ((0, 3), (2, 4))],
        ],
    )
    def test_pieces_cut_on_two_axes_coalesce(self, boxes):
        """A source cut on more than one axis and put back is that source."""
        whole = stored((2, 4), origin_id="e" * 32)
        parts = [whole[a[0] : b[0], a[1] : b[1]] for a, b in boxes]
        array = placed(parts, [list(a) for a, _ in boxes], (2, 4))
        assert array.data_id == LazyArray.from_source(whole).data_id

    def test_a_cube_cut_on_two_axes_coalesces(self):
        """The same holds for an array of three dimensions."""
        whole = stored((2, 2, 4), origin_id="f" * 32)
        parts = [whole[:, :, 0:2], whole[0:1, :, 2:4], whole[1:2, :, 2:4]]
        array = placed(parts, [[0, 0, 0], [0, 0, 2], [1, 0, 2]], (2, 2, 4))
        assert array.data_id == LazyArray.from_source(whole).data_id

    def test_one_file_two_keys(self):
        """Two patches of one file, abutting in both, stay two members."""
        first = stored((4, 3), key="patch_0")
        second = stored((4, 3), key="patch_1")
        array = concat(
            [LazyArray.from_source(first[0:2]), LazyArray.from_source(second[2:4])],
            axis=0,
        )
        assert array.data_id != LazyArray.from_source(first).data_id

    def test_one_file_two_origins(self):
        """Two origins of one location, abutting in both, stay two."""
        first = stored((4, 3), origin_id="a" * 32)
        second = stored((4, 3), origin_id="b" * 32)
        array = concat(
            [LazyArray.from_source(first[0:2]), LazyArray.from_source(second[2:4])],
            axis=0,
        )
        assert array.data_id != LazyArray.from_source(first).data_id

    def test_origins_name_their_own_members(self):
        """Members of one location with two origins are not one origin twice."""
        first = stored((8, 3), origin_id="a" * 32)
        second = stored((8, 3), origin_id="b" * 32)
        starts = [[0, 0], [2, 0]]
        mixed = placed([first[0:2], second[4:6]], starts, (4, 3))
        same = placed([first[0:2], first[4:6]], starts, (4, 3))
        assert mixed.data_id != same.data_id

    def test_a_member_is_named_by_its_whole_path(self):
        """A member under a base uri is named as the whole path names it."""
        source = stored((8, 3), path="/data/alpha/f.h5")
        array = LazyArray.from_source(source, base_uri="/data/alpha/")
        assert array.data_id == source.data_id

    def test_keys_name_their_own_members(self):
        """Members of one location with two keys are not one key twice."""
        first = stored((8, 3), key="patch_0")
        second = stored((8, 3), key="patch_1")
        starts = [[0, 0], [2, 0]]
        mixed = placed([first[0:2], second[4:6]], starts, (4, 3))
        same = placed([first[0:2], first[4:6]], starts, (4, 3))
        assert mixed.data_id != same.data_id

    def test_merged_members_go_back_in_order(self):
        """Members merged side by side are put back in placement order."""
        whole = [stored((4, 4), path=f"/a/{x}.h5", origin_id=x * 32) for x in "abcd"]
        a, b, c, d = whole
        cut = [a[0:1, 0:2], b[0:1, 0:2], b[1:2, 0:2], c[0:1, 0:2], c[1:2, 0:2]]
        starts = [[0, 0], [0, 2], [1, 2], [1, 0], [2, 0]]
        pieces = placed([*cut, d[0:1, 0:2]], [*starts, [2, 2]], (3, 4))
        merged = placed(
            [a[0:1, 0:2], b[0:2, 0:2], c[0:2, 0:2], d[0:1, 0:2]],
            [[0, 0], [0, 2], [1, 0], [2, 2]],
            (3, 4),
        )
        assert pieces.validate().data_id == merged.validate().data_id

    def test_odd_origins_are_folded_one_at_a_time(self, monkeypatch):
        """Two odd origins side by side cannot be read as two others."""
        pairs = [
            ("a", "a" * 30),
            ("b", "b" * 34),
            ("c", "a" * 30 + "bb"),
            ("d", "b" * 32),
        ]
        sources = [
            ArraySource(path=path, format="TEST", origin_id=origin).describe((1,), "i8")
            for path, origin in pairs
        ]
        values = {"a": 0, "b": 10, "c": 20, "d": 30}
        first = LazyArray.from_sources(sources[:2])
        second = LazyArray.from_sources(sources[2:])

        @contextmanager
        def fake(source):
            yield lambda x: np.array([values[x.path]], "i8")

        monkeypatch.setattr(io_core, "_open_array_reader", fake)
        assert first.load().tolist() == [0, 10]
        assert second.load().tolist() == [20, 30]
        assert first.data_id != second.data_id

    def test_sharing_a_table_does_not_rename(self):
        """An array is named by its own members, not by its neighbours'."""
        odd = stored((1,), path="/a/odd.h5", origin_id="a" * 30)
        other = stored((1,), path="/a/other.h5", origin_id="b" * 34)
        array = LazyArray.from_sources([odd, other])
        alone = LazyArray.from_source(other)
        assert LazyTable.from_arrays([alone, array])[1].data_id == array.data_id

    def test_record_dtypes_are_not_one_array(self):
        """Two record layouts of one width are two arrays."""
        frame = LazyArray.from_source(ArraySource.full((2,), 1)).to_frame()
        first = LazyArray.from_frame(frame, (2,), [("x", "<i4"), ("y", "<i4")])
        second = LazyArray.from_frame(frame, (2,), [("z", "<i8")])
        assert first.load().dtype != second.load().dtype
        assert first.data_id != second.data_id
        assert first[0:0].data_id != second[0:0].data_id

    def test_a_base_uri_does_not_stop_a_merge(self):
        """Two windows of one file merge however their paths were split."""
        source = stored((4, 3), path="/data/alpha/f.h5")
        parts = [source[0:2], source[2:4]]
        plain = concat([LazyArray.from_source(x) for x in parts], axis=0)
        mixed = concat(
            [
                LazyArray.from_source(parts[0], base_uri="/data/alpha/"),
                LazyArray.from_source(parts[1]),
            ],
            axis=0,
        )
        assert mixed.data_id == plain.data_id == LazyArray.from_source(source).data_id

    def test_equal_constants_are_one_group(self):
        """Two equal constants apart in the output are one source, not two."""
        values = [1.0, 2.0, 1.0]
        array = concat([constant((1, 2), x) for x in values], axis=0)
        assert array.data_id == concat([constant((1, 2), x) for x in values]).data_id
        other = concat([constant((1, 2), x) for x in (1.0, 2.0, 3.0)], axis=0)
        assert array.data_id != other.data_id

    def test_windows_apart_are_not_one_window(self):
        """Two windows laid side by side are not the window they look like."""
        whole = stored((8, 3), origin_id="a" * 32)
        parts = [whole[0:2], whole[4:6]]
        array = concat([LazyArray.from_source(x) for x in parts], axis=0)
        assert array.data_id != LazyArray.from_source(whole[0:4]).data_id
        assert array.data_id != LazyArray.from_source(whole[2:6]).data_id

    def test_the_same_samples_twice(self):
        """One window read twice is not two windows of one source."""
        whole = stored((8, 3), origin_id="a" * 32)
        twice = concat([LazyArray.from_source(whole[0:2])] * 2, axis=0)
        assert twice.data_id != LazyArray.from_source(whole[0:4]).data_id

    def test_the_whole_extent_is_in_the_id(self):
        """A window of a longer source is not the whole of a shorter one."""
        short = stored((4, 3), origin_id="a" * 32)
        long_source = stored((8, 3), origin_id="a" * 32)
        cut = []
        for source in (short, long_source):
            parts = [source[0:2], source[2:4]]
            cut.append(concat([LazyArray.from_source(x) for x in parts], axis=0))
        assert cut[0].data_id != cut[1].data_id
        assert cut[0].data_id == LazyArray.from_source(short).data_id
        assert cut[1].data_id == LazyArray.from_source(long_source[0:4]).data_id

    def test_a_shifted_window_is_another_array(self):
        """Two windows of one file a sample apart are two arrays."""
        whole = stored((8, 8), origin_id="a" * 32)
        assert (
            LazyArray.from_source(whole[0:4, 0:4]).data_id
            != LazyArray.from_source(whole[1:5, 0:4]).data_id
        )

    def test_a_signed_zero_is_its_own_constant(self):
        """A negative zero beside a positive one is two constants, not one."""
        mixed = concat([constant((1, 2), 0.0), constant((1, 2), -0.0)], axis=0)
        same = concat([constant((1, 2), 0.0)] * 2, axis=0)
        assert mixed.data_id != same.data_id

    def test_a_constant_value_keeps_its_type(self):
        """One value written as an int and as a float is two constants."""
        array = concat([constant((1, 2), 1.0), constant((1, 2), 2.0)], axis=0)
        frame = array.to_frame()
        ids = []
        for value in (1, 1.0):
            frame["value"] = np.array([1, 1, value, value], object)
            ids.append(LazyArray.from_frame(frame, array.shape, array.dtype).data_id)
        assert ids[0] != ids[1]

    def test_a_constant_dtype_is_in_its_group(self):
        """One value at two precisions is two constants, not one."""
        pair = [constant((2, 2), 1.0), constant((2, 2), 1.0, np.float32)]
        mixed = concat(pair, axis=0)
        same = concat([constant((2, 2), 1.0)] * 2, axis=0)
        assert mixed.dtype == same.dtype
        assert mixed.data_id != same.data_id

    def test_two_sources_read_the_same_way(self):
        """Two files, keys or origins read alike are two arrays."""
        for name, values in (
            ("path", ("/a/one.h5", "/a/two.h5")),
            ("key", ("patch_0", "patch_1")),
            ("origin_id", ("a" * 32, "b" * 32)),
        ):
            ids = set()
            for value in values:
                source = stored((4, 4), **{name: value})
                array = LazyArray.from_source(source[0:2])
                ids.add(concat([array, constant((2, 4), 1.0)], axis=0).data_id)
            assert len(ids) == 2

    def test_the_order_the_sources_were_given(self):
        """The same members laid in either order are one array."""
        first = stored((4, 4), path="/a/one.h5", origin_id="a" * 32)
        second = stored((4, 4), path="/a/two.h5", origin_id="b" * 32)
        parts = [first[0:4, 0:2], second[0:4, 2:4]]
        left = placed(parts, [[0, 0], [0, 2]], (4, 4))
        right = placed(parts[::-1], [[0, 2], [0, 0]], (4, 4))
        assert left.validate().data_id == right.validate().data_id

    def test_a_transposed_member_does_not_join_a_straight_one(self):
        """Two boxes of one square source read different ways up stay two."""
        whole = stored((4, 4), origin_id="a" * 32)
        turned = LazyArray.from_source(whole[0:4, 2:4]).transpose()
        array = concat([LazyArray.from_source(whole[0:2, 0:4]), turned], axis=0)
        assert array.shape == (4, 4)
        assert array.data_id != LazyArray.from_source(whole).data_id

    def test_groups_are_framed_by_their_size(self):
        """Which source fills how many boxes is part of the id."""
        first = stored((6,), path="/a/one.h5", origin_id="a" * 32)
        second = stored((6,), path="/a/two.h5", origin_id="b" * 32)
        holes = [(np.array([x]), np.array([x + 1]), 1.0) for x in (1, 3, 5)]
        boxes = [np.array([x]) for x in (0, 2, 4)]
        one = [(x, x + 1, y) for x, y in zip(boxes, (first, first, second))]
        two = [(x, x + 1, y) for x, y in zip(boxes, (first, second, second))]
        left = region_array(one + holes, (6,))
        right = region_array(two + holes, (6,))
        assert left.validate() is left and right.validate() is right
        assert left.data_id != right.data_id

    def test_corners_belong_to_their_group(self):
        """Swapping which source fills which box renames the array."""
        first = stored((4,), path="/a/one.h5", origin_id="a" * 32)
        second = stored((4,), path="/a/two.h5", origin_id="b" * 32)
        holes = [(np.array([x]), np.array([x + 1]), 1.0) for x in (1, 3)]
        boxes = [np.array([x]) for x in (0, 2)]
        one = [(x, x + 1, y) for x, y in zip(boxes, (first, second))]
        two = [(x, x + 1, y) for x, y in zip(boxes, (second, first))]
        assert region_array(one + holes, (4,)).data_id != (
            region_array(two + holes, (4,)).data_id
        )

    def test_a_hole_is_not_the_whole(self):
        """An array which leaves a hole is not the array which covers it."""
        source = stored((4, 3), origin_id="a" * 32)
        narrow = LazyArray.from_sources(
            [source[0:2, 0:2]], starts=np.zeros((1, 2)), shape=(2, 3)
        )
        stacked = concat([narrow, narrow], axis=0)
        with pytest.raises(ParameterError, match="leave a hole"):
            stacked.validate()
        assert narrow.data_id != LazyArray.from_source(source[0:2, 0:3]).data_id
        assert stacked.data_id != LazyArray.from_source(source).data_id

    def test_a_box_covered_twice(self):
        """A source laid twice in one place is not the source's own array."""
        whole = stored((4, 4), origin_id="a" * 32)
        twice = placed([whole, whole], [[0, 0], [0, 0]], (4, 4))
        with pytest.raises(ParameterError, match="overlap"):
            twice.validate()
        assert twice.data_id != LazyArray.from_source(whole).data_id

    def test_overlapping_boxes_are_not_a_tiling(self):
        """Boxes which overlap are named, and not as the samples they cover."""
        whole = stored((4, 3), origin_id="a" * 32)
        array = placed([whole[0:3], whole[2:4]], [[0, 0], [2, 0]], (4, 3))
        with pytest.raises(ParameterError, match="overlap"):
            array.validate()
        assert array.data_id != LazyArray.from_source(whole).data_id

    def test_a_hand_placed_partition_is_the_whole(self):
        """Five boxes with no mergeable pair are still the array they tile."""
        boxes = [
            ((0, 0), (1, 2)),
            ((0, 2), (2, 3)),
            ((2, 1), (3, 3)),
            ((1, 0), (3, 1)),
            ((1, 1), (2, 2)),
        ]
        sources = [
            ArraySource.full(tuple(b - a for a, b in zip(low, high)), 1.0)
            for low, high in boxes
        ]
        pinwheel = placed(sources, [list(x) for x, _ in boxes], (3, 3))
        whole = constant((3, 3), 1.0)
        pinwheel.validate()
        assert np.array_equal(pinwheel.load(), whole.load())
        assert pinwheel.data_id == whole.data_id

    def test_nested_cuts_on_two_axes_keep_the_id(self):
        """Slicing and joining alone build a partition which is renamed."""
        array = constant((3, 4), 1.0)
        top = concat(
            [array[:2, :1], concat([array[:1, 1:2], array[1:2, 1:2]], axis=0)],
            axis=1,
        )
        left = concat([top, array[2:, :2]], axis=0)
        right = concat([array[:1, 2:], array[1:, 2:]], axis=0)
        joined = concat([left, right], axis=1)
        joined.validate()
        assert np.array_equal(joined.load(), array.load())
        assert joined.data_id == array.data_id


class TestCutAxes:
    """An id costs what the cuts did, not how many axes the array has."""

    def test_a_high_dimensional_constant(self, expanded):
        """A constant of thirty axes is named without expanding one."""
        source = ArraySource.full((1,) * 30, 1)
        assert LazyArray.from_source(source).data_id == source.data_id
        assert max(expanded) == 0

    def test_a_high_dimensional_stored_source(self, expanded):
        """A stored array of thirty axes is named without expanding one."""
        source = stored((1,) * 30, path="/a/b.h5")
        array = LazyArray.from_source(source)
        assert array.data_id == source.data_id
        assert array[0:1].data_id == source[0:1].data_id
        assert max(expanded) == 0

    def test_a_long_concatenation(self, expanded):
        """A thousand members of a twelve axis array expand one axis."""
        whole = stored((1000, *[1] * 11), origin_id="a" * 32)
        parts = [whole[x : x + 1] for x in range(1000)]
        array = LazyArray.from_sources(parts)
        assert len(array) == 1000
        assert array.data_id == LazyArray.from_source(whole).data_id
        assert max(expanded) == 1

    def test_too_many_cut_axes(self):
        """An array cut on more axes than the bound is refused."""
        ndim = MAX_CUT_AXES + 1
        source = ArraySource.full((1,) * ndim, 1.0)
        array = placed([source, source], [[0] * ndim, [1] * ndim], (2,) * ndim)
        with pytest.raises(ParameterError, match=f"cut on {ndim} axes"):
            array.data_id

    @pytest.mark.parametrize(
        ("shape", "cuts"),
        [
            ((4, 6), [[[], []], [[1, 3], []], [[1, 3], [2]], [[], [2, 4]]]),
            ((2, 3, 4), [[[], [], []], [[1], [], []], [[1], [2], [1, 3]]]),
        ],
    )
    def test_cutting_a_product_axis(self, shape, cuts):
        """Cutting an axis a region spans whole does not rename it."""
        source = stored(shape, origin_id="a" * 32)
        low = np.zeros(len(shape), np.int64)
        middle, start = np.array(shape), low.copy()
        middle[0] = start[0] = shape[0] // 2
        layout = [(low, middle, source), (start, np.array(shape), 3.0)]
        whole = region_array(layout, shape)
        for edges in cuts:
            array = region_array(refine(layout, grid(shape, edges)), shape)
            assert array.validate().data_id == whole.validate().data_id

    def test_groups_cut_on_different_axes(self):
        """Two sources cut on axes of their own are the array they fill."""
        shape = (4, 4)
        first = stored(shape, path="/a/one.h5", origin_id="a" * 32)
        second = stored(shape, path="/a/two.h5", origin_id="b" * 32)
        layout = [
            (np.array([0, 0]), np.array([2, 4]), first),
            (np.array([2, 0]), np.array([4, 4]), second),
        ]
        whole = region_array(layout, shape)
        mixed = refine(layout[:1], grid(shape, [[1], []]))
        mixed += refine(layout[1:], grid(shape, [[], [2]]))
        both = refine(layout, grid(shape, [[1], [2]]))
        for regions in (mixed, both):
            array = region_array(regions, shape)
            assert array.validate().data_id == whole.validate().data_id

    def test_a_region_which_is_not_a_product(self):
        """An L keeps both axes in its core, and is its own array."""
        shape = (3, 3)
        source = stored(shape, origin_id="a" * 32)
        ell = [((0, 0), (2, 1)), ((0, 1), (1, 3))]
        turned = [((0, 2), (2, 3)), ((0, 0), (1, 3))]
        box = [((0, 0), (2, 3))]
        arrays = []
        for boxes in (ell, turned, box):
            regions = [(np.array(a), np.array(b), source) for a, b in boxes]
            arrays.append(region_array(regions, shape))
        # Neither L spans an axis whole, so both keep both in their corners.
        assert described(arrays[0]) == [([(-1, -1), (-1, -1)], 6)]
        assert described(arrays[1]) == [([(-1, -1), (-1, -1)], 7)]
        assert described(arrays[2]) == [([(0, 2), (0, 3)], 1)]
        assert len({x.data_id for x in arrays}) == 3

    def test_a_staircase_is_not_the_box_it_spans(self):
        """Boxes offset on both axes keep both, and are not their span."""
        shape = (2, 2)
        source = stored(shape, origin_id="a" * 32)
        boxes = [((0, 0), (1, 1)), ((1, 1), (2, 2))]
        regions = [(np.array(a), np.array(b), source) for a, b in boxes]
        steps = region_array(regions, shape)
        # Two coordinates on an axis would factor; a staircase has three.
        assert described(steps) == [([(-1, -1), (-1, -1)], 7)]
        assert steps.data_id != LazyArray.from_source(source).data_id

    def test_a_band_of_a_square_knows_its_axis(self):
        """A row band and a column band of one square are two arrays."""
        whole = stored((4, 4), origin_id="a" * 32)
        arrays = [
            LazyArray.from_sources(
                [window], starts=np.zeros((1, 2), np.int64), shape=(4, 4)
            )
            for window in (whole[0:2, 0:4], whole[0:4, 0:2])
        ]
        assert described(arrays[0]) == [([(0, 2), (0, 4)], 1)]
        assert described(arrays[1]) == [([(0, 4), (0, 2)], 1)]
        assert arrays[0].data_id != arrays[1].data_id

    def test_a_region_short_of_the_origin(self):
        """A box which reaches the far corner but not the origin is its own."""
        whole = stored((4, 4), origin_id="a" * 32)
        array = LazyArray.from_sources(
            [whole[1:4]], starts=np.array([[1, 0]]), shape=(4, 4)
        )
        assert described(array) == [([(1, 4), (0, 4)], 1)]
        assert array.data_id != LazyArray.from_source(whole).data_id

    def test_an_empty_region_cut_along_an_axis(self):
        """Boxes of no samples cover nothing, however they were cut up."""
        ids = []
        for pieces in (1, 2):
            array = concat([constant((2 // pieces, 1), 1.0)] * pieces, axis=0)
            frame = array.to_frame()
            frame.loc[frame["out_axis"] == 1, "out_stop"] = 0
            flat = LazyArray.from_frame(frame, (2, 0), array.dtype)
            assert described(flat) == [([(-1, -1), (-1, -1)], 0)]
            ids.append(flat.data_id)
        assert ids[0] == ids[1]

    def test_a_hole_keeps_its_axes(self):
        """A ring around another source keeps both axes in its core."""
        shape = (3, 3)
        source = stored(shape, origin_id="a" * 32)
        ring = [
            ((0, 0), (3, 1)),
            ((0, 1), (1, 2)),
            ((2, 1), (3, 2)),
            ((0, 2), (3, 3)),
        ]
        regions = [(np.array(a), np.array(b), source) for a, b in ring]
        regions += [(np.array([1, 1]), np.array([2, 2]), 4.0)]
        array = region_array(regions, shape)
        assert described(array) == [([(-1, -1), (-1, -1)], 8), ([(1, 2), (1, 2)], 1)]
        assert array.validate().data_id != LazyArray.from_source(source).data_id


class TestPartitions:
    """How an array was cut up never reaches its id."""

    shapes = (
        (7,),
        (4, 5),
        (3, 4, 2),
        (2, 3, 2, 2),
        (2, 2, 3, 2, 2),
        (2, 2, 2, 2, 2, 2),
    )

    @pytest.mark.parametrize("shape", shapes)
    def test_guillotine_partitions(self, shape):
        """Partitions cut from the array one split at a time are the array."""
        rng = np.random.default_rng(7)
        for layout in layouts(shape):
            whole = region_array(layout, shape)
            for _ in range(15):
                boxes = random_boxes(rng, shape, int(rng.integers(1, 9)))
                array = region_array(refine(layout, boxes), shape)
                assert array.validate().data_id == whole.data_id

    def test_staggered_partitions(self):
        """Tilings which no sequence of cuts can make are the array too."""
        rng = np.random.default_rng(8)
        staggered = 0
        for shape in self.shapes:
            for layout in layouts(shape):
                whole = region_array(layout, shape)
                for _ in range(30):
                    boxes = random_tiling(rng, shape)
                    staggered += not guillotine(boxes, shape)
                    array = region_array(refine(layout, boxes), shape)
                    assert array.validate().data_id == whole.data_id
        # Most tilings can be cut out; the id must hold for the rest as well.
        assert staggered > 20

    @pytest.mark.parametrize("shape", shapes)
    def test_cuts_along_product_axes(self, shape):
        """Cutting the axes a layout spans whole never renames it."""
        rng = np.random.default_rng(11)
        for layout in layouts(shape):
            whole = region_array(layout, shape)
            for _ in range(10):
                edges = [
                    [x for x in range(1, size) if rng.random() < 0.5] for size in shape
                ]
                array = region_array(refine(layout, grid(shape, edges)), shape)
                assert array.validate().data_id == whole.data_id

    def test_a_region_with_a_hole(self):
        """A ring of one source around a constant is one region, however cut."""
        source = stored((3, 3), origin_id="a" * 32)
        ring = [
            ((0, 0), (3, 1)),
            ((0, 1), (1, 2)),
            ((2, 1), (3, 2)),
            ((0, 2), (3, 3)),
        ]
        regions = [(np.array(a), np.array(b), source) for a, b in ring]
        regions += [(np.array([1, 1]), np.array([2, 2]), 4.0)]
        whole = region_array(regions, (3, 3))
        rng = np.random.default_rng(9)
        for _ in range(10):
            boxes = random_tiling(rng, (3, 3))
            array = region_array(refine(regions, boxes), (3, 3))
            assert array.validate().data_id == whole.validate().data_id

    def test_boxes_which_overlap_alike(self):
        """Two lists of overlapping boxes which cover alike are one array."""
        whole = stored((4, 3), origin_id="a" * 32)
        first = placed([whole[0:3], whole[2:4]], [[0, 0], [2, 0]], (4, 3))
        second = placed([whole[0:4], whole[2:3]], [[0, 0], [2, 0]], (4, 3))
        assert len(first) == len(second) == 2
        assert first.data_id == second.data_id

    def test_rechunk_and_join_again(self, joined):
        """Cutting an array into pieces and joining them gives it back."""
        for bounds in ([0, 7, 601], [0, 100, 200, 601], [0, 600, 601]):
            pieces = list(joined.rechunk(bounds))
            assert concat(pieces, axis=0).data_id == joined.data_id

    def test_a_slab_and_a_scattered_array_agree(self):
        """An array named as a stack is named as a scattered one is."""
        rng = np.random.default_rng(10)
        pair = [
            stored((16, 4), path=f"/a/{x}.h5", origin_id=x * 32) for x in ("a", "b")
        ]
        for _ in range(10):
            edges = np.unique(rng.integers(1, 16, 4)).tolist()
            bounds = [0, *edges, 16]
            spans = list(pairwise(bounds))
            parts = [pair[x % 2][low:high] for x, (low, high) in enumerate(spans)]
            array = LazyArray.from_sources(parts)
            assert array._block().concat_axis == 0
            scattered = replace(array.table, concat_axes=np.array([NEW_AXIS]), _ids={})
            assert scattered[0].data_id == array.data_id

    def test_a_member_of_an_empty_axis(self):
        """A box with no samples covers nothing, however the members are laid."""
        array = constant((2, 1), 1.0)
        frame = array.to_frame()
        frame.loc[frame["out_axis"] == 1, "out_stop"] = 0
        flat = LazyArray.from_frame(frame, (2, 0), array.dtype)
        assert flat._block().concat_axis == 0
        scattered = replace(flat.table, concat_axes=np.array([NEW_AXIS]), _ids={})
        assert scattered[0].data_id == flat.data_id

    def test_a_stack_is_the_boxes_it_lays(self):
        """A stack of constants is named as the same boxes placed by hand."""
        stacked = concat([constant((2, 3), 1.0), constant((2, 3), 2.0)], axis=0)
        sources = [ArraySource.full((2, 3), float(x)) for x in (1, 2)]
        by_hand = placed(sources, [[0, 0], [2, 0]], (4, 3))
        assert stacked._block().concat_axis == 0
        assert by_hand._block().concat_axis == NEW_AXIS
        assert by_hand.data_id == stacked.data_id


class TestOriginNames:
    """An origin id is taken as written, whatever it looks like."""

    @staticmethod
    def beside_a_constant(origin):
        """One named source beside a constant, so the digest is what names it."""
        source = ArraySource(path="a", format="TEST", origin_id=origin)
        source = source.describe((2,), "i8")
        blocks = [LazyArray.from_source(source)]
        blocks.append(LazyArray.from_source(ArraySource.full((1,), 0)))
        return concat(blocks)

    def test_case_is_kept(self):
        """Two ids differing only in case are two sources."""
        lower, upper = (
            self.beside_a_constant("a" * 32),
            self.beside_a_constant("A" * 32),
        )
        assert lower.data_id != upper.data_id

    def test_a_name_is_not_its_own_hash(self):
        """A name and the hex of its hash are two sources."""
        name = "source-name"
        hexed = hashlib.blake2b(name.encode(), digest_size=16).hexdigest()
        assert (
            self.beside_a_constant(name).data_id
            != self.beside_a_constant(hexed).data_id
        )

    def test_record_constants(self):
        """Two record layouts of one width are two constants."""
        ids = set()
        for dtype in ([("x", "<i4"), ("y", "<i4")], [("z", "<i8")]):
            source = ArraySource(filled=True, value=1).describe((2,), dtype)
            ids.add(LazyArray.from_source(source).data_id)
        assert len(ids) == 2


class TestPinnedIds:
    """The canonical bytes an id is taken over are a stored format."""

    def test_constant_members(self):
        """Two constants in one array hash to a known digest."""
        array = concat([constant((1, 2), 1.0), constant((1, 2), 2.0)], axis=0)
        assert array.data_id == "c29da8cfe43731e1bd827f0973925b77"

    def test_window_members(self):
        """Two windows of an unnamed file hash to a known digest."""
        source = stored((4, 4), path="/a/b.h5")
        array = placed([source[0:1], source[2:3]], [[0, 0], [1, 0]], (2, 4))
        assert array.data_id == "b250a3cc6ad346d3769967572f69f349"

    def test_a_transposed_window(self):
        """A member read the other way up hashes to a known digest."""
        source = stored((4, 4), path="/a/b.h5", origin_id="a" * 32)
        array = LazyArray.from_source(source[0:2, 0:4]).transpose()
        assert array.data_id == "af96513603df1a85c7d3d6b35f0996e4"


class TestPinnedSourceIds:
    """Ids of arrays of every kind of member, fixed before the sources table."""

    def test_mixed(self):
        """Files, windows, keys, origins, casts and constants in one array."""
        assert mixed().data_id == "0245b01f48f15b03f3b6668e4d368609"

    def test_rechunked(self):
        """A piece of many members rechunked, some reading one file."""
        piece = many_members().rechunk([0, 7, 333, 600])[1]
        assert piece.data_id == "693468130b8439f00c157a8151c94d8e"

    def test_joined_apart(self):
        """Arrays built apart, joined."""
        array = concat([many_members()[590:600, 0:2], mixed()[0:4]])
        assert array.data_id == "808b817683b4f79b94f3711b8d38a11d"

    def test_stacked_round_trip(self):
        """Two windows stacked, through a frame."""
        array = stack([mixed()[1:5], mixed()[6:10]], axis=1)
        back = LazyArray.from_frame(array.to_frame(), array.shape, array.dtype)
        assert back.data_id == "f3ddaf77e1e762dff63826dc6f4054cf"
