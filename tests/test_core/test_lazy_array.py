"""Tests for LazyArray, the recipe for an array read a member at a time."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from itertools import pairwise, product

import numpy as np
import pytest

import dascore as dc
from dascore.core import lazy_array as lazy_module
from dascore.core.lazy_array import (
    AXIS_FIELDS,
    MAX_CUT_AXES,
    MEMBER_FIELDS,
    NEW_AXIS,
    LazyArray,
    LazyTable,
    _tiles,
    concat,
    stack,
)
from dascore.core.source import ArraySource
from dascore.exceptions import ParameterError
from dascore.io import core as io_core
from dascore.utils.array_api import backend_name
from dascore.utils.downloader import fetch
from dascore.utils.identity import H


@pytest.fixture(scope="module")
def path():
    """A DASDAE file holding one two dimensional patch."""
    return fetch("example_dasdae_event_1.h5")


@pytest.fixture(scope="module")
def patch(path):
    """The patch read from the file; its axes have different lengths."""
    return dc.read(path)[0]


@pytest.fixture(scope="module")
def source(patch):
    """The source the framework gave the patch."""
    return patch._source


@pytest.fixture(scope="module")
def lazy(source):
    """The whole of the stored array, as one member."""
    return LazyArray.from_source(source)


@pytest.fixture(scope="module")
def two_files(patch, tmp_path_factory):
    """Two DASDAE files holding windows of different lengths."""
    directory = tmp_path_factory.mktemp("lazy_array")
    dim = patch.dims[0]
    out = []
    for index, span in enumerate(((0, 100), (100, None))):
        window = patch.select(**{dim: span}, samples=True)
        path = directory / f"part_{index}.h5"
        dc.write(window, path, "dasdae")
        out.append(path)
    return out


@pytest.fixture(scope="module")
def two_sources(two_files):
    """The source of each of the two files."""
    return [dc.read(x)[0]._source for x in two_files]


@pytest.fixture(scope="module")
def joined(two_sources):
    """The two files, joined along their first axis."""
    return LazyArray.from_sources(two_sources)


@pytest.fixture()
def reads(monkeypatch):
    """Count the arrays the loader reads."""
    calls = []
    original = io_core._load_array_source
    monkeypatch.setattr(
        io_core, "_load_array_source", lambda x: calls.append(x) or original(x)
    )
    return calls


@pytest.fixture()
def expanded(monkeypatch):
    """Record the axes a signature expands, and cap what it may ask for."""
    asked = []
    patterns, check = lazy_module._patterns, lazy_module._check_cuts

    def counted(cut, ndim):
        asked.append(cut)
        assert cut <= 4, f"the signature expanded {cut} axes"
        check(cut, ndim)

    def capped(count):
        assert count <= 4, f"the signature asked for {2**count} patterns"
        return patterns(count)

    monkeypatch.setattr(lazy_module, "_patterns", capped)
    monkeypatch.setattr(lazy_module, "_check_cuts", counted)
    return asked


@pytest.fixture(scope="module")
def parts():
    """Three arrays of one shape, each a different constant."""
    return [constant((2, 3, 4), float(x)) for x in range(3)]


def constant(shape, value=1.0, dtype=None):
    """Return a lazy array of one constant member."""
    return LazyArray.from_source(ArraySource.full(shape, value, dtype))


def broadcast_array(shape=(2, 3), length=4):
    """Return an array whose first axis is broadcast from a stored one."""
    array = constant(shape)
    frame = stack([array], axis=0).to_frame()
    new = frame["out_axis"] == 0
    frame.loc[new, "out_stop"] = length
    return LazyArray.from_frame(frame, (length, *shape), array.dtype)


def stored(shape, path="/a/b.h5", origin_id="", key="", dtype=np.float32):
    """Return a source for a whole stored array which is never opened."""
    source = ArraySource(
        path=path, format="DASDAE", version="1", origin_id=origin_id, key=key
    )
    return source.describe(shape, dtype)


def placed(sources, starts, shape):
    """Return the array which puts each source at a corner."""
    return LazyArray.from_sources(sources, starts=np.array(starts), shape=shape)


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


def random_boxes(rng, shape, splits):
    """Return boxes which tile a shape, cut from it one split at a time."""
    boxes = [(np.zeros(len(shape), np.int64), np.asarray(shape, np.int64))]
    for _ in range(splits):
        low, high = boxes.pop(int(rng.integers(len(boxes))))
        wide = [x for x in range(len(shape)) if high[x] - low[x] > 1]
        if not wide:
            boxes.append((low, high))
            continue
        axis = int(rng.choice(wide))
        cut = int(rng.integers(low[axis] + 1, high[axis]))
        first, second = high.copy(), low.copy()
        first[axis], second[axis] = cut, cut
        boxes += [(low, first), (second, high)]
    return boxes


def tiling_array(boxes, shape):
    """Return the array whose constant members fill each box."""
    sources = [
        ArraySource.full(tuple((high - low).tolist()), float(index))
        for index, (low, high) in enumerate(boxes)
    ]
    return placed(sources, [low.tolist() for low, _ in boxes], shape)


def boxed(boxes, shape):
    """Return the array whose constant members fill each corner pair."""
    pairs = [(np.array(low), np.array(high)) for low, high in boxes]
    return tiling_array(pairs, shape)


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


def grid(shape, edges):
    """Return the boxes of a grid cut at the given edges of each axis."""
    spans = [list(pairwise(sorted({0, *x, shape[a]}))) for a, x in enumerate(edges)]
    return [
        (np.array([a for a, _ in corner]), np.array([b for _, b in corner]))
        for corner in product(*spans)
    ]


def described(array):
    """Return the interval of each axis and the corner count of each group."""
    out = lazy_module._signature(array._block())
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


def storage(table):
    """Return every array a table holds."""
    return [
        table.member_offsets,
        table.axis_offsets,
        table.shape_offsets,
        table.shapes,
        table.concat_axes,
        table.dtypes.codes,
        table.members.filled,
        *table.axes.values(),
        *[getattr(table.members, x).codes for x in MEMBER_FIELDS],
    ]


def owned_storage(table):
    """Return every array a table holds and each ndarray backing one."""
    out = []
    for array in storage(table):
        while isinstance(array, np.ndarray):
            out.append(array)
            array = array.base
    return out


class TestDescription:
    """An array knows what it is without reading anything."""

    def test_header(self, lazy, patch):
        """Shape, ndim, dtype and size come from the header, not the members."""
        assert lazy.shape == patch.shape
        assert (lazy.ndim, lazy.size) == (patch.data.ndim, patch.data.size)
        assert lazy.dtype == patch.data.dtype
        assert len(lazy) == 1

    def test_repr(self, lazy):
        """The repr says the shape, dtype and how many members there are."""
        assert "shape" in repr(lazy) and "members" in repr(lazy)

    def test_nothing_read(self, joined, reads):
        """Building, slicing, joining and naming an array open no file."""
        array = concat([joined[0:50], joined[50:601]], axis=0)
        assert array.data_id and array[10:20].shape == (10, 1001)
        assert array.to_frame() is not None
        assert not reads

    def test_empty_slice_keeps_header(self, lazy):
        """A slice which selects nothing is still an array of its own."""
        empty = lazy[0:0]
        assert empty.shape == (0, lazy.shape[1])
        assert empty.dtype == lazy.dtype
        assert len(empty) == 0
        assert empty.load().shape == empty.shape
        assert empty.validate() is empty

    def test_needs_a_source(self):
        """An array is made of at least one source."""
        with pytest.raises(ParameterError, match="at least one source"):
            LazyArray.from_sources([])

    def test_refuses_unloadable(self):
        """A source which describes no array cannot be a member."""
        with pytest.raises(ParameterError, match="does not say enough"):
            LazyArray.from_source(ArraySource(key="a"))

    def test_refuses_scalar(self):
        """An array has at least one axis."""
        source = ArraySource.full((1,), 1.0).describe((), np.float64)
        with pytest.raises(ParameterError, match="at least one dimension"):
            LazyArray.from_source(source)

    def test_refuses_unknown_extent(self):
        """A member must know how big the whole stored array is."""
        source = ArraySource(
            path="a", format="b", windows=((0, 2),), shape=(2,), dtype=np.float64
        )
        with pytest.raises(ParameterError, match="extent"):
            LazyArray.from_source(source)

    def test_refuses_ragged_sources(self):
        """Sources of one array all have the same ndim."""
        sources = [ArraySource.full((2, 2), 1.0), ArraySource.full((2,), 1.0)]
        with pytest.raises(ParameterError, match="same ndim"):
            LazyArray.from_sources(sources)


class TestSlice:
    """Slicing clips the boxes and moves the windows; nothing is read."""

    def test_matches_numpy(self, lazy, patch):
        """A window of the array is the same window of the loaded data."""
        window = lazy[10:20, 5:9]
        assert window.shape == (10, 4)
        assert np.array_equal(window.load(), patch.data[10:20, 5:9])

    def test_twice_is_once(self, joined, two_sources):
        """Slicing twice gives the data and the id of slicing once."""
        twice = joined[100:400][50:100]
        once = joined[150:200]
        assert np.array_equal(twice.load(), once.load())
        assert twice.data_id == once.data_id

    def test_trailing_axes_are_whole(self, lazy):
        """An axis left out of the request is taken whole."""
        assert lazy[0:5].shape == (5, lazy.shape[1])
        assert lazy[0:5, :].shape == (5, lazy.shape[1])

    def test_whole_array_is_the_same_view(self, lazy):
        """A request which clips nothing returns the array itself."""
        assert lazy[:] is lazy
        assert lazy[0 : lazy.shape[0]] is lazy

    def test_negative_bounds(self, lazy, patch):
        """Negative bounds count back from the end, as numpy does."""
        assert np.array_equal(lazy[-5:].load(), patch.data[-5:])

    def test_only_touched_members(self, joined):
        """The result holds the members the request reaches, and no others."""
        assert len(joined) == 2
        assert len(joined[0:10]) == 1
        assert len(joined[90:110]) == 2
        assert len(joined[200:300]) == 1

    def test_scan_when_not_a_stack(self, lazy):
        """An array which is not a stack of slabs is scanned instead."""
        pieces = [constant((2, 3), float(x)) for x in range(4)]
        starts = np.array([[0, 0], [0, 3], [2, 0], [2, 3]])
        grid = LazyArray.from_sources([x.source(0) for x in pieces], starts=starts)
        assert grid.shape == (4, 6)
        assert np.array_equal(grid[0:2].load(), grid.load()[0:2])
        assert len(grid[0:2]) == 2

    @pytest.mark.parametrize("bad", [slice(None, None, 2), 3, [1, 2], Ellipsis, None])
    def test_refuses_other_indexes(self, lazy, bad):
        """Steps, integers, ellipses and index arrays are refused."""
        with pytest.raises(ParameterError, match="step of one"):
            lazy[bad]

    def test_refuses_too_many_axes(self, lazy):
        """An index cannot name more axes than the array has."""
        with pytest.raises(ParameterError, match="dimensional array"):
            lazy[0:1, 0:1, 0:1]

    def test_grid_windows_match_numpy(self):
        """Windows of a grid joined on two axes agree with numpy."""
        columns = [
            concat([constant((2, 3), float(2 * x)), constant((2, 3), float(2 * x + 1))])
            for x in range(3)
        ]
        grid = concat(columns, axis=1)
        assert grid._block().concat_axis == NEW_AXIS
        grid.validate()
        expected = grid.load()
        rng = np.random.default_rng(3)
        for _ in range(60):
            bounds = [sorted(rng.integers(0, size + 1, 2).tolist()) for size in (4, 9)]
            window = tuple(slice(low, high) for low, high in bounds)
            cut = grid[window]
            assert cut.shape == expected[window].shape
            assert np.array_equal(cut.load(), expected[window])

    def test_grid_of_stacks_matches_numpy(self):
        """A three dimensional grid is cut the same way."""
        pair = [constant((2, 2, 2), float(x)) for x in range(2)]
        planes = [concat(pair, axis=1) for _ in range(2)]
        cube = concat(planes, axis=2)
        cube.validate()
        expected = cube.load()
        window = (slice(0, 2), slice(1, 4), slice(1, 3))
        assert np.array_equal(cube[window].load(), expected[window])

    def test_staggered_tiling_keeps_its_order(self):
        """Clipping can tie two corners, which must not leave them out of order."""
        array = boxed([((0, 0), (1, 4)), ((0, 4), (2, 6)), ((1, 0), (2, 4))], (2, 6))
        expected = array.load()
        assert np.array_equal(array[1:2].load(), expected[1:2])

    @pytest.mark.parametrize("shape", [(4, 5), (3, 4, 2)])
    def test_random_windows_of_random_tilings(self, shape):
        """Any window of a legal tiling loads the window numpy loads."""
        rng = np.random.default_rng(2)
        for _ in range(40):
            boxes = random_boxes(rng, shape, int(rng.integers(1, 9)))
            array = tiling_array(boxes, shape)
            expected = array.load()
            bounds = [sorted(rng.integers(0, size + 1, 2).tolist()) for size in shape]
            window = tuple(slice(low, high) for low, high in bounds)
            cut = array[window]
            assert cut.validate() is cut
            assert np.array_equal(cut.load(), expected[window])


class TestCombine:
    """Arrays are joined in one pass, along old axes and new ones."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_concat_matches_numpy(self, parts, axis):
        """Joining agrees with numpy on every axis, the middle included."""
        joined = concat(parts, axis=axis)
        expected = np.concatenate([x.load() for x in parts], axis=axis)
        assert joined.shape == expected.shape
        assert np.array_equal(joined.load(), expected)
        joined.validate()

    @pytest.mark.parametrize("axis", [0, 1, 2, 3])
    def test_stack_matches_numpy(self, parts, axis):
        """Stacking adds an axis nothing feeds, and agrees with numpy."""
        stacked = stack(parts, axis=axis)
        expected = np.stack([x.load() for x in parts], axis=axis)
        assert stacked.shape == expected.shape
        assert np.array_equal(stacked.load(), expected)
        stacked.validate()
        assert (stacked._block().axes["src_axis"][:, axis] == NEW_AXIS).all()

    def test_concat_files(self, joined, patch, two_sources):
        """Two files join into the array they were cut from."""
        assert joined.shape == patch.shape
        assert np.array_equal(joined.load(), patch.data)
        assert len(joined.sources) == 2
        assert joined.sources[0].path == str(two_sources[0].path)

    def test_negative_axis(self, parts):
        """An axis may be counted from the end."""
        assert concat(parts, axis=-1).shape == (2, 3, 12)

    @pytest.mark.parametrize(
        "combine,expected", [(concat, np.concatenate), (stack, np.stack)]
    )
    def test_promotes_dtype(self, combine, expected):
        """The output takes the dtype every member fits in."""
        parts = [constant((2, 2), 1, np.int32), constant((2, 2), 1.5, np.float32)]
        out = combine(parts, axis=0)
        assert out.dtype == np.result_type(np.int32, np.float32)
        assert np.array_equal(out.load(), expected([x.load() for x in parts]))

    def test_sources_promote_dtype(self):
        """A directory whose files differ in dtype loads at the wider one."""
        sources = [
            ArraySource.full((2, 2), 1, np.int32),
            ArraySource.full((2, 2), 1.5, np.float32),
        ]
        array = LazyArray.from_sources(sources)
        assert array.dtype == np.result_type(np.int32, np.float32)
        expected = np.concatenate([x.load() for x in sources])
        assert np.array_equal(array.load(), expected)

    def test_refuses_mismatched_shapes(self, parts):
        """Arrays which disagree away from the joined axis are refused."""
        with pytest.raises(ParameterError, match="cannot be joined"):
            concat([parts[0], constant((2, 5, 4))], axis=0)

    def test_refuses_mismatched_stack(self, parts):
        """Stacking takes arrays of one shape."""
        with pytest.raises(ParameterError, match="cannot be stacked"):
            stack([parts[0], constant((2, 5, 4))])

    def test_refuses_mixed_ndim(self, parts):
        """Arrays of different ndim cannot be combined."""
        with pytest.raises(ParameterError, match="cannot be combined"):
            concat([parts[0], constant((2, 3))])

    def test_middle_axis_reorders(self):
        """Joining away from the first axis puts the members back in order."""
        pair = [constant((2, 3), 1.0), constant((2, 3), 2.0)]
        stacks = [concat(pair, axis=0), concat(pair, axis=0)]
        joined = concat(stacks, axis=1)
        joined.validate()
        assert np.array_equal(
            joined.load(), np.concatenate([x.load() for x in stacks], axis=1)
        )

    def test_refuses_nothing(self):
        """There must be something to combine."""
        with pytest.raises(ParameterError, match="At least one array"):
            concat([])


class TestTranspose:
    """A member whose axes are swapped still loads the right way up."""

    def test_matches_numpy(self, lazy, patch):
        """The whole array transposes."""
        moved = lazy.transpose()
        assert moved.shape == patch.data.T.shape
        assert np.array_equal(moved.load(), patch.data.T)

    def test_window_of_a_transposed_member(self, lazy, patch):
        """A window of a transposed array reads the right samples."""
        moved = lazy.transpose()[100:110, 3:9]
        assert np.array_equal(moved.load(), patch.data.T[100:110, 3:9])

    def test_explicit_order(self):
        """An order says where each old axis goes."""
        array = constant((2, 3, 4))
        moved = array.transpose((1, 2, 0))
        assert moved.shape == (3, 4, 2)
        assert np.array_equal(moved.load(), np.transpose(array.load(), (1, 2, 0)))

    def test_transposed_concat(self, joined, patch):
        """Transposing keeps the members in canonical order."""
        moved = joined.transpose()
        moved.validate()
        assert np.array_equal(moved.load(), patch.data.T)

    def test_refuses_bad_order(self, lazy):
        """The order must be a permutation of the axes."""
        with pytest.raises(ParameterError, match="permutation"):
            lazy.transpose((0, 0))
        with pytest.raises(ParameterError, match="outside an array"):
            lazy.transpose((0, 5))


class TestConstants:
    """A member which stores no data generates it instead."""

    def test_fills_its_box(self, lazy, patch):
        """A constant member fills the box it is placed in."""
        pad = constant((10, patch.shape[1]), 3.0)
        array = concat([lazy, pad], axis=0)
        loaded = array.load()
        assert np.array_equal(loaded[: patch.shape[0]], patch.data)
        assert (loaded[patch.shape[0] :] == 3.0).all()

    def test_nan(self):
        """A nan constant loads as nan, and two of them are one array."""
        array = concat([constant((2, 3), np.nan), constant((2, 3), np.nan)], axis=0)
        assert np.isnan(array.load()).all()
        assert array.data_id == constant((4, 3), np.nan).data_id

    def test_hole_is_a_member(self):
        """A hole is filled by a constant rather than left out."""
        array = concat([constant((2, 3), 1.0), constant((2, 3), np.nan)], axis=0)
        array.validate()
        assert np.isnan(array.load()[2:]).all()

    def test_window_of_a_constant(self):
        """A clipped constant is the constant of the smaller shape."""
        array = constant((10,), 2.0)[2:5]
        assert array.data_id == ArraySource.full((3,), 2.0).data_id
        assert np.array_equal(array.load(), np.full(3, 2.0))

    def test_integer_constant(self):
        """A constant keeps the dtype it was given."""
        array = constant((3, 2), 4, np.int16)
        assert array.dtype == np.int16
        assert array.load().dtype == np.int16

    def test_signed_zero(self):
        """A negative zero constant is not a positive one."""
        array = concat([constant((1, 2), 0.0), constant((1, 2), -0.0)], axis=0)
        loaded = array.load()
        assert not np.signbit(loaded[0]).any()
        assert np.signbit(loaded[1]).all()
        assert array.sources[0].data_id != array.sources[1].data_id

    def test_a_bool_an_int_and_a_float(self):
        """One value of three types is three constants, not one."""
        parts = [constant((1, 2), True), constant((1, 2), 1), constant((1, 2), 1.0)]
        array = concat(parts, axis=0)
        values = [x.value for x in array.sources]
        assert [type(x) for x in values] == [bool, int, float]
        assert len({x.data_id for x in array.sources}) == 3

    def test_a_big_integer_constant(self):
        """An integer constant beside a float keeps every digit it was given."""
        big = 2**53 + 1
        parts = [constant((1, 2), big, np.int64), constant((1, 2), 1.5)]
        array = concat(parts, axis=0)
        back = LazyArray.from_frame(array.to_frame(), array.shape, array.dtype)
        for out in (array, back):
            assert out.sources[0].value == big
            assert out.sources[0].dtype == np.dtype(np.int64)
            assert out.sources[1].value == 1.5
            assert out.data_id == array.data_id
        expected = np.concatenate([x.load() for x in parts])
        assert np.array_equal(array.load(), expected)

    def test_a_clipped_constant_is_a_constant(self):
        """A constant cut on every axis is the constant of the smaller shape."""
        array = constant((10, 4), 2.0)[2:5, 1:3]
        assert array.source(0) == ArraySource.full((3, 2), 2.0)

    def test_a_clipped_stacked_constant(self):
        """A constant under a new axis keeps no window when it is cut."""
        table = stack([constant((2, 3), 2.0)] * 3, axis=0).rechunk([0, 1, 3])
        assert table[0].source(0) == ArraySource.full((2, 3), 2.0)


class TestValidate:
    """The rules are checked on demand, not on construction."""

    def test_valid(self, joined):
        """A well formed array validates and returns itself."""
        assert joined.validate() is joined

    def test_hole(self):
        """A box left uncovered is refused."""
        array = LazyArray.from_sources(
            [ArraySource.full((2, 3), 1.0)], starts=np.zeros((1, 2)), shape=(4, 3)
        )
        with pytest.raises(ParameterError, match="leave a hole"):
            array.validate()

    def test_overlap_by_volume(self):
        """Boxes which cover too many samples are refused."""
        source = ArraySource.full((4, 3), 1.0)
        starts = np.array([[0, 0], [1, 0]])
        array = LazyArray.from_sources([source] * 2, starts=starts, shape=(5, 3))
        with pytest.raises(ParameterError, match="so they overlap"):
            array.validate()

    def test_overlap_with_a_hole(self):
        """Boxes which overlap and leave a hole are refused as well."""
        source = ArraySource.full((2, 2), 1.0)
        starts = np.array([[0, 0], [0, 1]])
        array = LazyArray.from_sources([source] * 2, starts=starts, shape=(2, 4))
        with pytest.raises(ParameterError, match="overlap"):
            array.validate()

    def test_box_outside_the_array(self):
        """A box must be inside the array it fills."""
        source = ArraySource.full((4, 3), 1.0)
        array = LazyArray.from_sources([source], starts=np.zeros((1, 2)), shape=(2, 3))
        with pytest.raises(ParameterError, match="outside an array"):
            array.validate()

    def test_negative_shape(self, lazy):
        """An array cannot have a negative shape."""
        frame = lazy.to_frame()
        array = LazyArray.from_frame(frame, (-1, 1001), lazy.dtype)
        with pytest.raises(ParameterError, match="cannot have shape"):
            array.validate()

    def test_window_outside_the_source(self, lazy):
        """A window must be inside the array it reads."""
        frame = lazy.to_frame()
        frame.loc[0, "src_start"] = 10
        array = LazyArray.from_frame(frame, lazy.shape, lazy.dtype)
        with pytest.raises(ParameterError, match="outside the source"):
            array.validate()

    def test_stored_axis_outside_the_array(self, lazy):
        """A stored axis cannot be read onto an axis which is not there."""
        frame = lazy.to_frame()
        frame.loc[0, "src_axis"] = 5
        array = LazyArray.from_frame(frame, lazy.shape, lazy.dtype)
        with pytest.raises(ParameterError, match="does not have"):
            array.validate()

    def test_repeated_stored_axis(self, lazy):
        """One stored axis cannot feed two output axes."""
        frame = lazy.to_frame()
        frame.loc[1, "src_axis"] = 0
        array = LazyArray.from_frame(frame, lazy.shape, lazy.dtype)
        with pytest.raises(ParameterError, match="two output axes"):
            array.validate()

    def test_out_of_order(self, joined):
        """Members which are not in placement order are refused."""
        frame = joined.to_frame()
        frame["ordinal"] = 1 - frame["ordinal"]
        array = LazyArray.from_frame(
            frame.sort_values(["ordinal", "out_axis"]), joined.shape, joined.dtype
        )
        with pytest.raises(ParameterError, match="canonical placement order"):
            array.validate()

    def test_members_of_an_empty_array(self, lazy):
        """An array of no samples has no box a member could fill."""
        array = LazyArray.from_frame(lazy.to_frame(), (0, 1001), lazy.dtype)
        with pytest.raises(ParameterError, match="outside an array"):
            array.validate()

    def test_grid_of_blocks(self):
        """Members which tile a two dimensional grid are accepted."""
        source = ArraySource.full((2, 3), 1.0)
        starts = np.array([[0, 0], [0, 3], [2, 0], [2, 3]])
        grid = LazyArray.from_sources([source] * 4, starts=starts, shape=(4, 6))
        assert grid.validate() is grid
        assert np.array_equal(grid.load(), np.ones((4, 6)))

    def test_slab_missing_from_a_grid(self):
        """Members which cover the right samples in the wrong place."""
        source = ArraySource.full((2, 2), 1.0)
        starts = np.array([[2, 0], [2, 0], [2, 2], [2, 2]])
        array = LazyArray.from_sources([source] * 4, starts=starts, shape=(4, 4))
        with pytest.raises(ParameterError, match="do not tile"):
            array.validate()

    def test_broadcast_fields(self):
        """A broadcast axis reads nothing, so it states no window."""
        array = broadcast_array()
        array.validate()
        frame = array.to_frame()
        frame.loc[0, "src_extent"] = 3
        with pytest.raises(ParameterError, match="outside the source"):
            LazyArray.from_frame(frame, array.shape, array.dtype).validate()

    def test_staircase_tiling(self):
        """Two differently cut segments joined on another axis are accepted."""
        left = concat([constant((1, 2), float(x)) for x in range(3)], axis=0)
        right = concat([constant((3, 1), 9.0) for _ in range(2)], axis=1)
        array = concat([left, right], axis=1)
        assert array.validate() is array
        assert array.shape == (3, 4)

    def test_l_shaped_tiling(self):
        """A column split on one axis beside a whole one is accepted."""
        boxes = [((0, 0), (1, 1)), ((1, 0), (2, 1)), ((0, 1), (2, 2))]
        array = tiling_array([(np.array(a), np.array(b)) for a, b in boxes], (2, 2))
        assert array.validate() is array

    @pytest.mark.parametrize("shape", [(4, 5), (3, 4, 2)])
    def test_random_tilings_are_accepted(self, shape):
        """Tilings cut from the array at random are legal, however deep."""
        rng = np.random.default_rng(0)
        for _ in range(40):
            boxes = random_boxes(rng, shape, int(rng.integers(1, 9)))
            tiling_array(boxes, shape).validate()

    @pytest.mark.parametrize("shape", [(4, 5), (3, 4, 2)])
    def test_shifted_tilings_are_refused(self, shape):
        """A box moved off its place leaves a hole and an overlap."""
        rng = np.random.default_rng(1)
        checked = 0
        for _ in range(40):
            boxes = random_boxes(rng, shape, int(rng.integers(2, 9)))
            movable = [index for index, (low, _) in enumerate(boxes) if np.any(low > 0)]
            if not movable:
                continue
            index = movable[int(rng.integers(len(movable)))]
            low, high = boxes[index]
            axis = int(np.flatnonzero(low > 0)[0])
            low, high = low.copy(), high.copy()
            low[axis] -= 1
            high[axis] -= 1
            start = np.array([x.tolist() for x, _ in boxes], np.int64)
            stop = np.array([x.tolist() for _, x in boxes], np.int64)
            start[index], stop[index] = low, high
            assert not _tiles(start, stop, shape)
            checked += 1
        assert checked > 20

    def test_stored_axes_must_be_a_permutation(self, lazy):
        """A member which reads its second stored axis but not its first."""
        frame = lazy.to_frame()
        frame.loc[frame["out_axis"] == 0, ["src_axis", "src_extent"]] = (NEW_AXIS, 0)
        array = LazyArray.from_frame(frame, lazy.shape, lazy.dtype)
        with pytest.raises(ParameterError, match="stored axes"):
            array.validate()

    def test_slice_of_a_stacked_constant(self):
        """A new axis keeps no window when a stack of constants is cut."""
        stacked = stack([constant((2, 3), 1.0)] * 3, axis=0)
        assert stacked.validate() is stacked
        assert stacked[0:2].validate() is not None
        for piece in stacked.rechunk([0, 1, 3]):
            piece.validate()

    def test_a_sliced_stack_is_a_smaller_stack(self):
        """Cutting a stack down to one gives the array stacking one gives."""
        array = constant((2, 3), 1.0)
        assert stack([array, array], axis=0)[0:1].data_id == stack([array]).data_id

    def test_a_wrong_stacking_hint_is_refused(self):
        """An array which says it is stacked on an axis it is not is refused."""
        source = ArraySource.full((2, 3), 1.0)
        starts = np.array([[0, 0], [0, 3], [2, 0], [2, 3]])
        grid = LazyArray.from_sources([source] * 4, starts=starts, shape=(4, 6))
        assert grid.validate() is grid
        bent_table = replace(grid.table, concat_axes=np.ones(1, np.int64))
        with pytest.raises(ParameterError, match="stacked along axis 1"):
            bent_table[0].validate()

    @pytest.mark.parametrize("hint", [2, 7, -2])
    def test_a_stacking_hint_outside_the_array_is_refused(self, hint):
        """A hint names an axis of the array, or no axis at all."""
        array = constant((3, 4), 1.0)
        bent_table = replace(array.table, concat_axes=np.array([hint], np.int64))
        with pytest.raises(ParameterError, match="outside an array"):
            bent_table[0].validate()

    @pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 4), (2, 3, 0, 5)])
    def test_a_source_of_no_samples(self, shape):
        """A source with an empty axis is the empty array it describes."""
        array = LazyArray.from_source(ArraySource.full(shape, 1))
        assert array.shape == shape and len(array) == 0
        assert array.validate() is array
        assert np.array_equal(array.load(), np.full(shape, 1))

    def test_operations_on_a_memberless_array(self):
        """Every operation works on an array which has no members."""
        array = LazyArray.from_source(ArraySource.full((0, 3), 1))
        back = LazyArray.from_frame(array.to_frame(), array.shape, array.dtype)
        assert back.data_id == array.data_id == array[:, 0:3].data_id
        assert array[:, 0:2].shape == (0, 2)
        assert array.transpose().shape == (3, 0)
        assert concat([array, array], axis=1).shape == (0, 6)
        assert stack([array, array]).shape == (2, 0, 3)
        assert array.validate().load().shape == (0, 3)

    def test_an_empty_source_among_others(self):
        """A source of no samples is left out, and the rest still cover."""
        sources = [
            ArraySource.full((2, 3), 1.0),
            ArraySource.full((0, 3), 1.0),
            ArraySource.full((1, 3), 2.0),
        ]
        array = LazyArray.from_sources(sources)
        assert array.shape == (3, 3) and len(array) == 2
        expected = np.concatenate([x.load() for x in sources])
        assert np.array_equal(array.validate().load(), expected)

    def test_an_empty_stored_window(self):
        """A stored window of no samples is left out as a constant one is."""
        whole = stored((4, 3), path="/a/b.h5")
        array = LazyArray.from_sources([whole[0:4], whole[2:2]])
        assert array.shape == (4, 3) and len(array) == 1
        assert array.validate().data_id == LazyArray.from_source(whole).data_id


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
        monkeypatch.setattr(
            io_core,
            "_load_array_source",
            lambda x: np.array([values[x.path]], "i8"),
        )
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


class TestSources:
    """A member and a source say the same thing."""

    def test_round_trip(self, source, lazy):
        """The member a source made gives that source back."""
        assert lazy.source(0) == source
        assert lazy.sources == (source,)

    def test_base_uri_is_joined(self, source, path):
        """A path stored in two parts is resolved as it is read out."""
        # A base is a literal prefix, so it ends as this platform's paths do.
        base = source.path[: -len(path.name)]
        array = LazyArray.from_source(source, base_uri=base)
        assert array.source(0).path == source.path

    def test_path_outside_the_base(self, source):
        """A path which does not start with the base uri is stored whole."""
        array = LazyArray.from_source(source, base_uri="/somewhere/else/")
        assert array.to_frame()["base_uri"].iloc[0] == ""
        assert array.source(0) == source

    def test_window_of_a_member(self, lazy, source):
        """A clipped member reads the window it was clipped to."""
        assert lazy[10:20].source(0) == source[10:20]

    def test_constant_source(self):
        """A constant member gives the constant source back."""
        source = ArraySource.full((3, 2), 5.0)
        assert constant((3, 2), 5.0).source(0) == source


class TestLoad:
    """Loading reads each member and puts it in its box."""

    def test_many_files(self, joined, patch, reads):
        """Each member is read once, through its own source."""
        assert np.array_equal(joined.load(), patch.data)
        assert len(reads) == 2

    def test_numpy_asks_for_it(self, lazy, patch):
        """Numpy can ask for the array, with or without a dtype."""
        assert np.array_equal(np.asarray(lazy), patch.data)
        assert np.asarray(lazy, dtype=np.float32).dtype == np.float32

    def test_broadcast_member(self):
        """A member with no stored axis fills the axis it was given."""
        array = broadcast_array(length=4)
        loaded = array.load()
        assert loaded.shape == (4, 2, 3)
        assert np.array_equal(loaded, np.broadcast_to(loaded[0], (4, 2, 3)))

    def test_a_hole_is_never_loaded(self):
        """An array which does not cover itself is refused, not made up."""
        array = LazyArray.from_sources(
            [ArraySource.full((2, 3), 1.0)], starts=np.zeros((1, 2)), shape=(4, 3)
        )
        with pytest.raises(ParameterError, match="leave a hole"):
            array.load()


class TestStatedDtype:
    """A caller which must match another promotion order states the dtype."""

    def _sources(self):
        """Two sources whose dtypes promote to something else together."""
        return [ArraySource.full((2, 3), np.int16(1)), ArraySource.full((2, 3), 2.0)]

    def test_the_default_promotes_the_members(self):
        """Without a dtype the members promote together, as before."""
        array = LazyArray.from_sources(self._sources(), axis=0)
        assert array.dtype == np.result_type(np.int16, np.float64)

    def test_load_gives_the_stated_dtype(self):
        """The array loads as what it says it loads as."""
        array = LazyArray.from_sources(self._sources(), axis=0, dtype="float32")
        assert array.dtype == np.dtype("float32")
        loaded = array.load()
        assert loaded.dtype == np.dtype("float32")
        assert np.array_equal(loaded, np.asarray(array))

    def test_the_stated_dtype_is_part_of_the_id(self):
        """Two arrays loading different dtypes are not the same array."""
        default = LazyArray.from_sources(self._sources(), axis=0)
        stated = LazyArray.from_sources(self._sources(), axis=0, dtype="float32")
        assert default.data_id != stated.data_id

    def test_one_whole_source_keeps_the_stated_dtype_in_its_id(self):
        """An array which is one source whole is not that source recast."""
        source = ArraySource.full((2, 3), np.int16(1))
        whole = LazyArray.from_source(source)
        recast = LazyArray.from_sources([source], dtype="float32")
        assert whole.data_id != recast.data_id
        assert recast.load().dtype == np.dtype("float32")

    def test_a_frame_round_trip_keeps_it(self):
        """The frame plus the array's own shape and dtype rebuild it."""
        array = LazyArray.from_sources(self._sources(), axis=0, dtype="float32")
        back = LazyArray.from_frame(array.to_frame(), array.shape, array.dtype)
        assert back.dtype == array.dtype
        assert back.data_id == array.data_id
        assert np.array_equal(back.load(), array.load())


class TestCastVia:
    """A member may be told to pass through another dtype on the way."""

    def test_a_frame_with_no_cast_column_reads_back(self):
        """A frame written before casts existed rebuilds with none."""
        source = ArraySource.full((3,), 1.5)
        array = LazyArray.from_source(source)
        frame = array.to_frame().drop(columns=["cast"])
        back = LazyArray.from_frame(frame, array.shape, array.dtype)
        assert back.data_id == array.data_id
        assert np.array_equal(back.load(), array.load())

    # the integer a float32 rounds and a float64 keeps
    _value = 2**24 + 1

    def _sources(self):
        """An integer beside a float it promotes with."""
        return [
            ArraySource.full((2, 3), self._value, "int32"),
            ArraySource.full((2, 3), 2.0, "float64"),
        ]

    def _via(self):
        """The same two, the first told to pass through float32."""
        return LazyArray.from_sources(
            self._sources(), axis=0, cast_via=["float32", None]
        )

    def test_the_member_rounds_where_it_is_told_to(self):
        """The stated intermediate is what the samples come back through."""
        straight = LazyArray.from_sources(self._sources(), axis=0)
        via = self._via()
        assert straight.load()[0, 0] == np.float64(self._value)
        assert via.load()[0, 0] == np.float64(np.float32(self._value))
        assert via.dtype == straight.dtype == np.dtype("float64")

    def test_the_cast_is_part_of_the_id(self):
        """Two arrays whose samples differ are not the same array."""
        straight = LazyArray.from_sources(self._sources(), axis=0)
        assert straight.data_id != self._via().data_id

    @pytest.mark.parametrize(("shape", "dtype"), [((2, 3), "float64"), ((3,), None)])
    def test_a_constant_states_its_cast_too(self, shape, dtype):
        """A constant put through another dtype is another constant.

        The whole-array shortcut names one at its own dtype as well.
        """
        source = ArraySource.full(shape, self._value, "int32")
        kwargs = {} if dtype is None else {"dtype": dtype}
        whole = LazyArray.from_sources([source], **kwargs)
        via = LazyArray.from_sources([source], cast_via=["float32"], **kwargs)
        assert whole.data_id != via.data_id
        assert not np.array_equal(whole.load(), via.load())
        assert np.ravel(via.load())[0] == np.float64(np.float32(self._value))

    def test_a_whole_constant_keeps_its_cast_through_a_round_trip(self):
        """The cast rides with the members, so the id does not move."""
        source = ArraySource.full((3,), self._value, "int32")
        via = LazyArray.from_sources([source], cast_via=["float32"])
        back = LazyArray.from_frame(via.to_frame(), via.shape, via.dtype)
        assert back.data_id == via.data_id
        assert LazyTable.from_arrays([via])[0].data_id == via.data_id

    @pytest.mark.parametrize(("count", "dtype"), [(1, None), (2, "float64")])
    def test_a_stored_member_states_its_cast_too(self, count, dtype):
        """A whole stored file put through another dtype is another array."""
        sources = [stored((4, 4), path=f"/a/{num}.h5") for num in range(count)]
        kwargs = {} if dtype is None else {"dtype": dtype}
        whole = LazyArray.from_sources(sources, **kwargs)
        casts = ["float16", *[None] * (count - 1)]
        via = LazyArray.from_sources(sources, cast_via=casts, **kwargs)
        assert whole.data_id != via.data_id

    def test_a_frame_round_trip_keeps_it(self):
        """The cast rides in the member frame with everything else."""
        array = self._via()
        back = LazyArray.from_frame(array.to_frame(), array.shape, array.dtype)
        assert back.data_id == array.data_id
        assert np.array_equal(back.load(), array.load())

    def test_one_cast_per_source(self):
        """A cast for some of the members says nothing about the rest."""
        with pytest.raises(ParameterError, match="every source"):
            LazyArray.from_sources(self._sources(), axis=0, cast_via=["float32"])


class TestTable:
    """One table holds many arrays and owns all the storage."""

    def test_views_share_storage(self, joined):
        """An array's matrices are views of the table's own arrays."""
        table = joined.table
        for name in AXIS_FIELDS:
            assert np.shares_memory(joined._block().axes[name], table.axes[name])

    def test_slicing_copies_no_more_than_it_selects(self, joined):
        """A window holds the members it selected, and no other rows."""
        window = joined[0:10]
        assert window.table is not joined.table
        assert window.table.n_members == 1
        assert not np.shares_memory(
            window._block().axes["out_start"], joined.table.axes["out_start"]
        )

    def test_mixed_ndim(self, joined):
        """Arrays of different ndim sit in one table."""
        table = LazyTable.from_arrays([joined, stack([joined, joined])])
        assert [x.ndim for x in table] == [2, 3]
        assert [x.shape for x in table] == [joined.shape, (2, *joined.shape)]
        assert table.n_members == len(joined) + 2 * len(joined)
        for array in table:
            assert np.shares_memory(
                array._block().axes["out_start"], table.axes["out_start"]
            )

    def test_indexing(self, joined):
        """A table is indexed by row, from either end."""
        table = LazyTable.from_arrays([joined, joined[0:10]])
        assert len(table) == 2
        assert table[-1].shape == (10, joined.shape[1])
        with pytest.raises(IndexError, match="outside a table"):
            table[5]


class TestFrames:
    """The frames are the database tables, and read back as they are."""

    def test_array_round_trip(self, joined, patch):
        """One array round trips through its long frame."""
        frame = joined.to_frame()
        back = LazyArray.from_frame(frame, joined.shape, joined.dtype)
        assert back.data_id == joined.data_id
        assert np.array_equal(back.load(), patch.data)
        assert len(frame) == len(joined) * joined.ndim

    def test_array_frame_columns(self, joined):
        """The columns are the placement and what each member reads."""
        columns = set(joined.to_frame().columns)
        assert set(AXIS_FIELDS) <= columns
        assert {"ordinal", "out_axis", "base_uri", "path", "origin_id"} <= columns

    def test_out_of_order_frame(self, joined):
        """A frame which was shuffled is put back in order."""
        frame = joined.to_frame()
        back = LazyArray.from_frame(frame.iloc[::-1], joined.shape, joined.dtype)
        assert back.data_id == joined.data_id

    def test_frame_keeps_the_concat_hint(self, joined):
        """A round tripped array still knows which axis it is stacked on."""
        pair = [constant((2, 3), 1.0), constant((2, 3), 2.0)]
        grid = concat([concat(pair, axis=0), concat(pair, axis=0)], axis=1)
        for array in (joined, stack([joined, joined]), concat(pair, axis=1), grid):
            back = LazyArray.from_frame(array.to_frame(), array.shape, array.dtype)
            assert back._block().concat_axis == array._block().concat_axis

    def test_empty_table(self):
        """A table of no arrays holds no members."""
        table = LazyTable.from_arrays([])
        assert len(table) == 0 and table.n_members == 0

    def test_constant_round_trip(self):
        """A constant member keeps its value and dtype through a frame."""
        array = concat([constant((2, 3), np.nan), constant((2, 3), 1.0)], axis=0)
        back = LazyArray.from_frame(array.to_frame(), array.shape, array.dtype)
        assert back.data_id == array.data_id
        assert np.array_equal(back.load(), array.load(), equal_nan=True)

    def test_a_record_member_keeps_its_fields(self):
        """A member of a record dtype is read back with every field."""
        dtype = np.dtype([("x", "<i4"), ("y", "<f4")])
        source = stored((3,), path="/a/record.h5", dtype=dtype)
        array = LazyArray.from_source(source)
        back = LazyArray.from_frame(array.to_frame(), array.shape, array.dtype)
        assert array.dtype == back.dtype == dtype
        assert array.source(0) == source
        assert back.source(0) == source
        assert back.data_id == array.data_id

    def test_frame_edits_do_not_reach_the_array(self):
        """The array holds its own copy of what the frame said."""
        original = LazyArray.from_sources(
            [ArraySource.full((2,), 1.0), ArraySource.full((2,), 2.0)]
        )
        frame = original.to_frame()
        array = LazyArray.from_frame(frame, original.shape, original.dtype)
        before = array.data_id
        frame.loc[0, "out_stop"] = 1
        frame.loc[0, "src_extent"] = 1
        frame.loc[1, "out_start"] = 1
        frame.loc[1, "src_extent"] = 3
        assert np.array_equal(array.load(), original.load())
        assert array._data_id() == before == original.data_id

    def test_frame_edits_do_not_reach_a_table(self):
        """Tables built from the array do not read the frame either."""
        whole = stored((10,), path="/a/w.h5", dtype=np.int64)
        array = LazyArray.from_source(whole[0:3])
        frame = array.to_frame()
        back = LazyArray.from_frame(frame, array.shape, array.dtype)
        table = LazyTable.from_arrays([back])
        frame.loc[0, "src_start"] = 4
        assert back.source(0) == whole[0:3]
        assert table[0].source(0) == whole[0:3]

    def test_storage_is_read_only(self, joined):
        """Nothing a table exposes can be written, so no view can be bent."""
        table = LazyTable.from_arrays([joined, joined[0:10]])
        assert not any(x.flags.writeable for x in storage(table))
        assert not any(x.flags.writeable for x in table[0]._block().axes.values())
        with pytest.raises(ValueError, match="read-only"):
            table.axes["out_start"][0] = 5

    def test_every_owner_of_the_storage_is_read_only(self, joined):
        """The arrays a table's storage is a view of cannot be written."""
        array = joined[0:10]
        tables = [
            LazyArray.from_source(ArraySource.full((4, 3), 1.0)).table,
            LazyArray.from_sources([ArraySource.full((4, 3), 1.0)]).table,
            LazyArray.from_frame(array.to_frame(), array.shape, array.dtype).table,
            array.table,
            concat([array, array], axis=0).table,
            stack([array, array], axis=0).table,
            array.transpose().table,
            array.rechunk([0, 4, 10]),
            LazyTable.from_arrays([array, joined]),
        ]
        for table in tables:
            assert not any(x.flags.writeable for x in owned_storage(table))

    def test_a_shared_owner_cannot_be_bent(self):
        """Writing through a view's owner would change two tables at once."""
        array = LazyArray.from_source(stored((6,), dtype=np.int64)[0:2])
        shared = LazyTable.from_arrays([array])[0]
        before = (array.data_id, shared.data_id)
        with pytest.raises(ValueError, match="read-only"):
            array.table.axes["src_start"].base[0, 0] = 2
        assert (array.data_id, shared.data_id) == before

    def test_axes_cannot_be_rebound(self):
        """A table's axes mapping refuses a new array under an old name."""
        table = constant((2, 3), 1.0).table
        with pytest.raises(TypeError):
            table.axes["src_start"] = np.zeros(2, np.int64)


class TestRechunk:
    """Cutting an array at new bounds gives many arrays in one table."""

    def test_pieces_tile_the_parent(self, joined, patch):
        """The pieces are the parent, cut where it was asked for."""
        table = joined.rechunk([0, 150, 400, 601])
        assert [x.shape[0] for x in table] == [150, 250, 201]
        loaded = [x.validate().load() for x in table]
        assert np.array_equal(np.concatenate(loaded), patch.data)

    def test_pieces_keep_their_ids(self, joined, two_sources):
        """A piece which is one window is named as that window is."""
        table = joined.rechunk([0, 50, 100])
        assert table[0].data_id == two_sources[0][0:50].data_id
        assert table[1].data_id == two_sources[0][50:100].data_id

    def test_merging(self, joined, patch):
        """Bounds coarser than the members merge them."""
        table = joined.rechunk([0, 601])
        assert len(table) == 1 and table[0].shape == patch.shape
        assert np.array_equal(table[0].load(), patch.data)

    def test_dropping_samples(self, joined, patch):
        """Samples outside the bounds are left out."""
        table = joined.rechunk([50, 150])
        assert len(table) == 1 and table[0].shape[0] == 100
        assert np.array_equal(table[0].load(), patch.data[50:150])

    def test_other_axis(self, patch):
        """Any axis can be cut along."""
        array = concat([constant((3, 4), 1.0), constant((3, 6), 2.0)], axis=1)
        table = array.rechunk([0, 5, 10], axis=1)
        assert [x.shape for x in table] == [(3, 5), (3, 5)]
        loaded = np.concatenate([x.load() for x in table], axis=1)
        assert np.array_equal(loaded, array.load())

    def test_constants(self):
        """A constant cut in two is two constants."""
        table = constant((10, 3), 2.0).rechunk([0, 4, 10])
        assert table[0].data_id == ArraySource.full((4, 3), 2.0).data_id
        assert np.array_equal(table[1].load(), np.full((6, 3), 2.0))

    @pytest.mark.parametrize(
        "shape,axis,bounds,expected",
        [
            ((0, 3), 1, [0, 1, 3], [(0, 1), (0, 2)]),
            ((2, 0, 4), 0, [0, 1, 2], [(1, 0, 4), (1, 0, 4)]),
            ((2, 0, 4), 2, [0, 1, 4], [(2, 0, 1), (2, 0, 3)]),
        ],
    )
    def test_a_memberless_array(self, shape, axis, bounds, expected):
        """An array with no members is cut into the empty pieces asked for."""
        array = LazyArray.from_source(ArraySource.full(shape, 1))
        table = array.rechunk(bounds, axis=axis)
        assert [x.shape for x in table] == expected
        for piece in table:
            assert len(piece) == 0 and piece.dtype == array.dtype
            assert np.array_equal(piece.validate().load(), np.empty(piece.shape))

    def test_refuses_a_grid(self):
        """An array which is not a stack of slabs is not rechunked yet."""
        source = ArraySource.full((2, 3), 1.0)
        starts = np.array([[0, 0], [0, 3], [2, 0], [2, 3]])
        grid = LazyArray.from_sources([source] * 4, starts=starts, shape=(4, 6))
        with pytest.raises(NotImplementedError, match="full width"):
            grid.rechunk([0, 2, 4])

    def test_refuses_bad_bounds(self, joined):
        """Bounds must ascend and stay inside the array."""
        with pytest.raises(ParameterError, match="at least two bounds"):
            joined.rechunk([10])
        with pytest.raises(NotImplementedError, match="overlapping or empty"):
            joined.rechunk([10, 10, 20])
        with pytest.raises(ParameterError, match="outside the array"):
            joined.rechunk([0, 10_000])


class TestArrayApi:
    """A lazy array says which backend owns it."""

    def test_backend_name(self, lazy):
        """The array api reports the lazy backend."""
        assert backend_name(lazy) == "lazy"

    def test_namespace(self, lazy):
        """The namespace is the one the backend name comes from."""
        assert lazy.__array_namespace__().__name__ == "lazy"
