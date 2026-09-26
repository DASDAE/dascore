"""Tests for LazyArray, the recipe for an array read a member at a time."""

from __future__ import annotations

import gc
import hashlib
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise, product
from types import MappingProxyType, SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core import lazy_array as lazy_module
from dascore.core.lazy_array import (
    AXIS_FIELDS,
    MAX_CUT_AXES,
    NEW_AXIS,
    SOURCE_FIELDS,
    LazyArray,
    LazyTable,
    _tiles,
    concat,
    stack,
)
from dascore.core.source import ArraySource
from dascore.exceptions import InvalidFiberIOError, ParameterError
from dascore.io import core as io_core
from dascore.utils.array_api import backend_name
from dascore.utils.downloader import fetch
from dascore.utils.hdf5 import H5Reader
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
def square_source(patch, tmp_path_factory):
    """The source of a DASDAE file holding a square array."""
    path = tmp_path_factory.mktemp("lazy_square") / "square.h5"
    square = patch.select(distance=(0, 20), time=(0, 20), samples=True)
    dc.write(square, path, "dasdae")
    return dc.read(path)[0]._source


@pytest.fixture(scope="module")
def two_keys(patch, tmp_path_factory):
    """Two patches of different data in one DASDAE file, read back."""
    path = tmp_path_factory.mktemp("lazy_keys") / "keys.h5"
    small = patch.select(distance=(0, 20), time=(0, 30), samples=True)
    dc.write(dc.spool([small, small.new(data=small.data * 2)]), path, "dasdae")
    return list(dc.read(path))


@pytest.fixture(scope="module")
def joined(two_sources):
    """The two files, joined along their first axis."""
    return LazyArray.from_sources(two_sources)


@pytest.fixture()
def reads(monkeypatch):
    """Record each source the loader reads, whatever reads it."""
    calls = []
    original = io_core._open_array_reader

    @contextmanager
    def recorded(source):
        with original(source) as load:
            yield lambda x: calls.append(x) or load(x)

    monkeypatch.setattr(io_core, "_open_array_reader", recorded)
    return calls


@pytest.fixture()
def handles(monkeypatch):
    """Record every HDF5 handle opened, and how many others were open then."""
    opened, busy = [], []
    original = H5Reader.get_handle.__func__

    def get_handle(cls, resource):
        busy.append(sum(x.id.valid for x in opened))
        opened.append(out := original(cls, resource))
        return out

    monkeypatch.setattr(H5Reader, "get_handle", classmethod(get_handle))
    return SimpleNamespace(opened=opened, busy=busy)


@pytest.fixture()
def lookups(monkeypatch):
    """Count the HDF5 objects looked up by name."""
    calls = []
    original = h5py.Group.__getitem__

    def getitem(group, name):
        calls.append(name)
        return original(group, name)

    monkeypatch.setattr(h5py.Group, "__getitem__", getitem)
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
    return table._storage()


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


def random_chain(rng, ndim, length):
    """Return random slices, some open, negative or empty, and transposes."""
    ops = []
    for _ in range(length):
        if rng.random() < 0.3:
            order = None if rng.random() < 0.3 else tuple(rng.permutation(ndim))
            ops.append(("transpose", order))
            continue
        bounds = [None, *range(-4, 12)]
        count = int(rng.integers(0, ndim + 1))
        items = [slice(*rng.choice(bounds, 2).tolist()) for _ in range(count)]
        ops.append(("slice", tuple(items)))
    return ops


def run_chain(array, ops, eager):
    """Apply a chain to an array, materializing after each step if eager."""
    for name, argument in ops:
        array = array.transpose(argument) if name == "transpose" else array[argument]
        if eager:
            array = lazy_module._array(array._block())
    return array


def assert_same_array(view, eager, loadable):
    """Assert a view is the array the eager chain made."""
    assert view.shape == eager.shape and view.ndim == eager.ndim
    assert view.dtype == eager.dtype and view.size == eager.size
    assert len(view) == len(eager)
    assert view.data_id == eager.data_id
    assert view.sources == eager.sources
    pd.testing.assert_frame_equal(view.to_frame(), eager.to_frame())
    assert view.validate() is view
    if loadable:
        assert np.array_equal(view.load(), eager.load())


class TestViews:
    """Slices and transposes wait to clip and permute the members."""

    @pytest.fixture()
    def cases(self, joined, lazy):
        """Arrays of several layouts, and whether each can be loaded."""
        pieces = [constant((3, 4), float(x)) for x in range(4)]
        mixed = [stored((3, 4), path=f"/a/{x}.h5") for x in range(3)]
        return [
            (concat(pieces, axis=0), True),
            (concat(pieces, axis=1), True),
            (LazyArray.from_sources([*mixed, ArraySource.full((3, 4), 2.0)]), False),
            (stack(pieces, axis=1), True),
            (concat([concat(pieces[:2]), concat(pieces[2:])], axis=1), True),
            (
                boxed([((0, 0), (2, 1)), ((0, 1), (1, 2)), ((1, 1), (2, 2))], (2, 2)),
                True,
            ),
            (
                tiling_array(random_boxes(np.random.default_rng(5), (5, 6), 7), (5, 6)),
                True,
            ),
            (concat([constant((5,), 1.0), constant((4,), 2.0)]), True),
            (concat([constant((2, 3, 4), 1.0), constant((3, 3, 4), 2.0)]), True),
            (stack([concat(pieces[:2], axis=1)] * 2, axis=2), True),
            (joined, True),
            (lazy, True),
            (concat([joined, constant(joined.shape, 3.0)], axis=1), True),
        ]

    def test_chains_match_eager(self, cases):
        """Random chains of views give the arrays eager steps give."""
        rng = np.random.default_rng(42)
        for array, loadable in cases:
            for _ in range(25):
                ops = random_chain(rng, array.ndim, int(rng.integers(1, 5)))
                view = run_chain(array, ops, eager=False)
                assert_same_array(view, run_chain(array, ops, eager=True), loadable)
                if loadable:
                    expected = run_chain(array.load(), ops, eager=False)
                    assert np.array_equal(view.load(), expected)

    def test_members_untouched(self, monkeypatch):
        """Views of a large array clip nothing until the members are needed."""
        paths = np.char.add("/a/f", np.arange(10_000).astype(str))
        array = LazyArray.from_columns(
            paths, (4, 3), format="DASDAE", version="1", source_dtype="f4"
        )

        def refuse(*args):
            raise AssertionError("the members were touched")

        monkeypatch.setattr(lazy_module, "_clip", refuse)
        monkeypatch.setattr(lazy_module, "_transposed", refuse)
        view = array[10:-10, 1:].transpose()[:, 5:9]
        assert view.shape == (2, 4) and view.dtype == np.float32
        assert view.ndim == 2 and view.size == 8
        with pytest.raises(AssertionError, match="touched"):
            view.data_id

    def test_clips_only_candidates(self, monkeypatch):
        """Resolving a view clips only the members its window can touch."""
        array = concat([constant((4, 3), float(x)) for x in range(100)])
        sizes = []
        clip = lazy_module._clip
        monkeypatch.setattr(
            lazy_module, "_clip", lambda x, *y: sizes.append(len(x)) or clip(x, *y)
        )
        assert len(array.transpose()[:, 2:][:, 3:13]) == 3
        assert len(array[:, 1:]) == 100
        # Edges on member boundaries, and a slice of a slice keeping both.
        assert len(array[4:8]) == 1
        assert len(array[2:30][2:6]) == 1
        assert sizes == [3, 100, 1, 1]

    def test_resolves_once(self, monkeypatch, joined):
        """A view clips its members once, however often they are asked for."""
        calls = []
        clip = lazy_module._clip
        monkeypatch.setattr(
            lazy_module, "_clip", lambda *x: calls.append(1) or clip(*x)
        )
        view = joined[90:110].transpose()
        assert not calls
        for _ in range(2):
            view.data_id, view.load(), len(view), view.table, view.sources
        assert len(calls) == 1
        # A view of a resolved view starts from its resolved members.
        assert view[1:].data_id == joined[90:110, 1:].transpose().data_id
        assert len(calls) == 3

    def test_table_holds_the_view(self, joined):
        """A view's table and row are those of the array it resolves to."""
        view = joined[90:110]
        assert view.table.n_members == 2 and view.row == 0
        assert view.table[view.row].data_id == view.data_id
        assert len(LazyTable.from_arrays([view, joined.transpose()])) == 2

    def test_views_compose(self, joined, patch):
        """Views of views agree with numpy."""
        view = joined.transpose()[2:9].transpose((1, 0))[95:120][:, 1:]
        assert np.array_equal(view.load(), patch.data.T[2:9].T[95:120][:, 1:])
        assert view.data_id == joined[95:120, 3:9].data_id
        assert view[:] is view

    def test_resolved_view_lets_its_base_go(self):
        """Once a view is resolved, the array it was cut from can be freed."""
        array = concat([constant((4, 3), float(x)) for x in range(50)])
        owner = array.table.axes["out_start"]
        while owner.base is not None:
            owner = owner.base
        base = weakref.ref(owner)
        window = array[5:9]
        assert window.load().shape == (4, 3)
        del array, owner
        gc.collect()
        assert base() is None

    def test_turned_back_lets_its_base_go(self):
        """A view which shows its whole base unmoved does not keep the base."""
        big = concat([constant((4, 3), float(x)) for x in range(50)])
        table = LazyTable.from_arrays([constant((4, 3)), big])
        owner = table.axes["out_start"]
        while owner.base is not None:
            owner = owner.base
        base = weakref.ref(owner)
        view = table[0].transpose().transpose()
        assert view.load().shape == (4, 3)
        del big, table, owner
        gc.collect()
        assert base() is None

    def test_threads_share_one_table(self, monkeypatch):
        """Threads resolving one view at once all get the one table."""
        clip, calls = lazy_module._clip, []

        def slow(*args):
            calls.append(1)
            time.sleep(0.05)
            return clip(*args)

        monkeypatch.setattr(lazy_module, "_clip", slow)
        view = concat([constant((4, 3), float(x)) for x in range(20)])[5:30]
        with ThreadPoolExecutor(8) as pool:
            tables = list(pool.map(lambda _: view.table, range(8)))
        assert all(x is tables[0] for x in tables) and len(calls) == 1
        assert view.ndim == 2 and view.dtype == np.float64

    def test_bare_transpose_keeps_members(self):
        """A transpose with no window keeps each member as it is stored."""
        frame = LazyArray.from_source(ArraySource.full((4, 3), 1.0)).to_frame()
        frame["src_start"] += 2
        frame["src_extent"] += 2
        array = LazyArray.from_frame(frame, (4, 3), "f8")
        # Windows are stated in the source's own axis order, so they stand.
        assert array.transpose().sources == array.sources

    def test_turned_back_is_in_order(self):
        """A transpose and its inverse put members out of order back in order."""
        frame = concat([constant((2, 3), 1.0), constant((2, 3), 2.0)]).to_frame()
        frame["ordinal"] = 1 - frame["ordinal"]
        array = LazyArray.from_frame(frame, (4, 3), "f8")
        expected = np.concatenate([np.full((2, 3), 1.0), np.full((2, 3), 2.0)])
        assert np.array_equal(array.transpose().transpose().load(), expected)

    def test_transpose_keeps_refusing(self):
        """A transpose does not hide a member outside its array."""
        array = LazyArray.from_sources([ArraySource.full((4, 3), 7.0)], shape=(2, 3))
        with pytest.raises(ParameterError):
            array.transpose().load()

    @pytest.mark.parametrize("bad", [slice(None, None, 2), 3])
    def test_view_refuses_other_indexes(self, joined, bad):
        """A view refuses what the array refuses."""
        with pytest.raises(ParameterError, match="step of one"):
            joined.transpose()[1:][bad]

    def test_view_refuses_bad_axes(self, joined):
        """A view refuses too many indexes and orders which are not a permutation."""
        view = joined.transpose()[1:]
        with pytest.raises(ParameterError, match="dimensional array"):
            view[0:1, 0:1, 0:1]
        with pytest.raises(ParameterError, match="permutation"):
            view.transpose((1, 1))


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

    def test_one_open_for_many_members(self, two_sources, patch, handles, lookups):
        """Members of one file share its handle and its dataset lookups."""
        source = two_sources[0]
        # Windows with gaps between them, so each is its own read.
        spans = [(x, x + 5) for x in range(0, 100, 10)]
        many = LazyArray.from_sources([source[a:b] for a, b in spans])
        LazyArray.from_source(source[0:10]).load()
        alone = len(handles.opened), len(lookups)
        expected = np.concatenate([patch.data[a:b] for a, b in spans])
        assert np.array_equal(many.load(), expected)
        assert (len(handles.opened), len(lookups)) == (2 * alone[0], 2 * alone[1])
        assert not any(x.id.valid for x in handles.opened)

    def test_interleaved_files(self, two_sources, patch, handles):
        """Members alternating between files are read a file at a time."""
        first, second = two_sources
        parts = [
            part
            for start in range(0, 100, 25)
            for part in (first[start : start + 25], second[start : start + 25])
        ]
        # The second file starts at row 100 of the patch.
        expected = np.concatenate(
            [
                patch.data[offset + start : offset + start + 25]
                for start in range(0, 100, 25)
                for offset in (0, 100)
            ]
        )
        assert np.array_equal(LazyArray.from_sources(parts).load(), expected)
        assert handles.busy == [0, 0]
        assert not any(x.id.valid for x in handles.opened)

    def test_keys_of_one_file(self, two_keys, handles, reads):
        """Several keys, logical or stored, are read through one handle."""
        first, second = (x._source for x in two_keys)
        one, two = (x.data for x in two_keys)
        raw = replace(first, key=f"/waveforms/{first.key}/data")
        parts = [first[0:4], second[4:8], raw[8:12], second[12:16], first[16:]]
        expected = [one[0:4], two[4:8], one[8:12], two[12:16], one[16:]]
        out = LazyArray.from_sources(parts).load()
        assert np.array_equal(out, np.concatenate(expected))
        assert len(handles.opened) == 1 and len(reads) == len(parts)

    def test_constants_transposes_and_casts(self, two_sources, patch, handles):
        """Stored members of one file are grouped around constants and casts."""
        source = two_sources[0][0:4, 0:6]
        data = patch.data[0:4, 0:6]
        pieces = [source[0:2], source[2:4]]
        rounded = LazyArray.from_sources(pieces, cast_via=["float16", None])
        array = concat(
            [
                LazyArray.from_source(source).transpose(),
                constant((6, 4), 7.0),
                rounded.transpose(),
            ],
            axis=0,
        )
        half = np.concatenate([data[0:2].astype(np.float16), data[2:4]])
        expected = np.concatenate([data.T, np.full((6, 4), 7.0), half.T])
        assert np.array_equal(array.load(), expected.astype(array.dtype))
        assert len(handles.opened) == 1

    def test_changed_resource_raises_and_closes(self, two_sources, handles):
        """A member which no longer matches its source is refused, not cast."""
        stale = replace(two_sources[0], dtype=np.dtype(np.float32))
        array = LazyArray.from_sources([stale[0:10], stale[20:30]])
        with pytest.raises(InvalidFiberIOError, match="may have changed"):
            array.load()
        assert len(handles.opened) == 1 and not handles.opened[0].id.valid

    @pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
    def test_placing_failure_closes(self, two_sources, handles, monkeypatch, error):
        """A failure between reads still releases the open resource."""

        def fail(data, axes):
            raise error("placing failed")

        monkeypatch.setattr(lazy_module, "_to_output", fail)
        array = LazyArray.from_sources([two_sources[0][0:10], two_sources[0][20:30]])
        with pytest.raises(error, match="placing failed"):
            array.load()
        assert len(handles.opened) == 1 and not handles.opened[0].id.valid

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


class TestCoalescedReads:
    """Abutting windows of one source are read as one window."""

    def test_windows_of_one_file(self, two_sources, patch, reads):
        """Abutting windows of one file take one read."""
        source = two_sources[0]
        many = LazyArray.from_sources([source[x : x + 10] for x in range(0, 100, 10)])
        assert np.array_equal(many.load(), patch.data[:100])
        assert len(reads) == 1 and reads[0].windows[0] == (0, 100)

    def test_breaks_split_runs(self, two_sources, patch, reads):
        """A gap, a reordering, a constant or a file change starts a new read."""
        first, second = two_sources
        width = patch.shape[1]
        parts = [first[0:10], first[10:20], first[30:40], first[20:30]]
        parts += [ArraySource.full((2, width), 0.0, dtype=patch.dtype)]
        parts += [first[40:50], second[0:10], second[10:20]]
        expected = [patch.data[a:b] for a, b in [(0, 20), (30, 40), (20, 30)]]
        expected += [np.zeros((2, width)), patch.data[40:50], patch.data[100:120]]
        out = LazyArray.from_sources(parts).load()
        assert np.array_equal(out, np.concatenate(expected))
        assert [x.windows[0] for x in reads] == [
            (0, 20),
            (30, 40),
            (20, 30),
            (40, 50),
            (0, 20),
        ]

    def test_casts_split_runs(self, two_sources, patch, reads):
        """Members cast through different dtypes are read apart."""
        source = two_sources[0]
        parts = [source[0:10], source[10:20], source[20:30]]
        array = LazyArray.from_sources(parts, dtype="f8", cast_via=[None, "f4", "f4"])
        expected = np.concatenate(
            [patch.data[0:10], patch.data[10:30].astype("f4")]
        ).astype("f8")
        assert np.array_equal(array.load(), expected)
        assert [x.windows[0] for x in reads] == [(0, 10), (10, 30)]

    def test_tiles_merge_along_one_axis(self, two_sources, patch, reads):
        """Tiles merge along one axis only, so each row of tiles takes one read."""
        source = two_sources[0]
        tile = LazyArray.from_source
        rows = [
            concat([tile(source[a:b, 0:5]), tile(source[a:b, 5:10])], axis=1)
            for a, b in ((0, 50), (50, 100))
        ]
        grid = concat(rows)
        assert np.array_equal(grid.load(), patch.data[:100, :10])
        assert [x.windows for x in reads] == [((0, 50), (0, 10)), ((50, 100), (0, 10))]

    def test_other_axes_must_match(self, two_sources, patch, reads):
        """Windows which meet on one axis but differ on another are read apart."""
        source = two_sources[0]
        array = LazyArray.from_sources([source[0:10, 0:5], source[10:20, 100:105]])
        expected = np.concatenate([patch.data[0:10, 0:5], patch.data[10:20, 100:105]])
        assert np.array_equal(array.load(), expected)
        assert len(reads) == 2

    def test_axis_maps_must_match(self, two_sources, patch, reads):
        """A transposed member is not merged with a plain one of its file."""
        source = two_sources[0]
        plain = LazyArray.from_source(source[0:10, 0:10])
        turned = LazyArray.from_source(source[0:10, 10:20]).transpose()
        expected = np.concatenate([patch.data[0:10, 0:10], patch.data[0:10, 10:20].T])
        assert np.array_equal(concat([plain, turned]).load(), expected)
        assert len(reads) == 2

    def test_square_transpose_is_not_merged(self, square_source, reads):
        """A transposed window of a square source is not merged with a plain one."""
        data = square_source.load()
        reads.clear()
        plain = LazyArray.from_source(square_source[0:5, 0:5])
        turned = LazyArray.from_source(square_source[0:5, 5:10]).transpose()
        expected = np.concatenate([data[0:5, 0:5], data[0:5, 5:10].T])
        assert np.array_equal(concat([plain, turned]).load(), expected)
        assert len(reads) == 2

    def test_a_run_turning_is_cut(self, two_sources, patch, reads):
        """Tiles which join along one axis, then another, are read as two runs."""
        whole = two_sources[0]
        boxes = [(0, 1, 0, 1), (0, 2, 1, 2), (1, 2, 0, 1), (2, 3, 0, 1), (2, 3, 1, 2)]
        array = LazyArray.from_sources(
            [whole[a:b, c:d] for a, b, c, d in boxes],
            starts=[(a, c) for a, _, c, _ in boxes],
        )
        assert np.array_equal(array.load(), patch.data[0:3, 0:2])
        windows = [x.windows for x in reads]
        assert windows == [
            ((0, 1), (0, 1)),
            ((0, 2), (1, 2)),
            ((1, 3), (0, 1)),
            ((2, 3), (1, 2)),
        ]

    @pytest.mark.parametrize(
        ("dtype", "cast", "count"),
        [("f8", None, 1), ("u4", None, 2), ("f8", "u4", 2), ("f8", "U", 2)],
    )
    def test_conversions_which_vary_are_read_apart(
        self, two_sources, reads, dtype, cast, count
    ):
        """Floats read into integers, or through text, are read a window at a time."""
        source = two_sources[0]
        array = LazyArray.from_sources(
            [source[0:1], source[1:2]], dtype=dtype, cast_via=[cast, cast]
        )
        with np.errstate(invalid="ignore"):
            array.load()
        assert len(reads) == count

    @pytest.mark.parametrize(("dtype", "merged"), [("f4", 1), ("S10", 2)])
    def test_only_numbers_merge(self, dtype, merged):
        """Text, whose casts may take their unit from the samples, is read apart."""
        array = LazyArray.from_columns(
            ["/a.h5", "/a.h5"],
            (1,),
            start=[[0], [1]],
            extent=2,
            **{**FORMAT, "source_dtype": dtype},
        )
        assert len(lazy_module._coalesced(array._block())) == merged

    def test_extents_must_match(self):
        """Windows of one source which state different extents are read apart."""
        array = LazyArray.from_columns(
            ["/a.h5", "/a.h5"], (4,), start=[[0], [4]], extent=[[8], [9]], **FORMAT
        )
        assert len(lazy_module._coalesced(array._block())) == 2

    def test_reads_stop_at_the_byte_budget(
        self, two_sources, patch, reads, monkeypatch
    ):
        """A run is cut where a member starts a budget or more into it."""
        source = two_sources[0]
        itemsize = np.dtype(patch.dtype).itemsize
        monkeypatch.setattr(lazy_module, "_RUN_BYTES", 30 * patch.shape[1] * itemsize)
        spans = [(0, 10), (10, 20), (20, 50), (50, 60), (60, 100)]
        array = LazyArray.from_sources([source[a:b] for a, b in spans])
        assert np.array_equal(array.load(), patch.data[:100])
        # Cut where members start in another 30 rows: 0-29, 30-59, 60-89.
        assert [x.windows[0] for x in reads] == [(0, 50), (50, 60), (60, 100)]

    def test_members_over_the_budget_stay_apart(self, two_sources, reads, monkeypatch):
        """Members each bigger than the budget are read one at a time."""
        monkeypatch.setattr(lazy_module, "_RUN_BYTES", 1)
        array = LazyArray.from_sources([two_sources[0][0:10], two_sources[0][10:20]])
        array.load()
        assert len(reads) == 2

    @pytest.mark.parametrize("seed", range(20))
    def test_random_cuts_load_alike(self, two_sources, monkeypatch, seed):
        """Random cuts load the same array with or without coalescing."""
        rng = np.random.default_rng(seed)
        parts = []
        for source in two_sources:
            size = source.shape[0]
            cuts = np.unique(rng.integers(1, size, rng.integers(0, 8)))
            pieces = list(pairwise([0, *cuts.tolist(), size]))
            if rng.random() < 0.3:
                rng.shuffle(pieces)
            parts += [source[a:b, 2:7] for a, b in pieces]
            if rng.random() < 0.3:
                parts.append(ArraySource.full((3, 5), 1.0, dtype=source.dtype))
        array = LazyArray.from_sources(parts)
        window = array[int(rng.integers(0, 20)) :]
        got = [array.load(), window.load()]
        monkeypatch.setattr(lazy_module, "_coalesced", lambda block: block)
        assert all(
            np.array_equal(x, y) for x, y in zip(got, [array.load(), window.load()])
        )


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


def column_case(rng):
    """Return random columns for from_columns and the sources they describe."""
    ndim = int(rng.integers(1, 4))
    count = int(rng.integers(1, 7))
    axis = int(rng.integers(-ndim, ndim))
    lengths = np.tile(rng.integers(1, 4, ndim), (count, 1))
    # Some members along the axis have no samples at all.
    lengths[:, axis] = rng.integers(0, 4, count)
    extra = rng.integers(0, 3, (count, ndim))
    start = rng.integers(0, 3, (count, ndim)) * (extra > 0)
    start = np.minimum(start, extra)
    extent = lengths + extra
    base = "/root/"
    pool = [base + "a.h5", base + "sub/b.h5", "/elsewhere/c.h5", base + "é.h5"]
    kwargs = {
        "path": [pool[x] for x in rng.integers(0, len(pool), count)],
        "shape": lengths,
        "start": start,
        "extent": extent,
        "axis": axis,
        "base_uri": base if rng.random() < 0.5 else "",
    }
    choices = {
        "format": ["DASDAE", "H5Simple"],
        "version": ["1", "2"],
        "key": ["", "k1", "k2"],
        "origin_id": ["", "a" * 32, "b" * 32],
        "source_dtype": [np.dtype(x) for x in ("float32", "int16", "uint8")],
    }
    for name, options in choices.items():
        picks = [options[x] for x in rng.integers(0, len(options), count)]
        kwargs[name] = picks if rng.random() < 0.5 else picks[0]
    if rng.random() < 0.3:
        kwargs["cast_via"] = [[None, "float32"][x] for x in rng.integers(0, 2, count)]
    if rng.random() < 0.3:
        kwargs["dtype"] = "float64"
    return kwargs


def sources_of_columns(kwargs):
    """Return the ArraySources the columns of one member each describe."""
    count = len(kwargs["shape"])
    out = []
    for row in range(count):

        def pick(name, row=row):
            value = kwargs[name]
            return value[row] if isinstance(value, list) else value

        start, length = kwargs["start"][row], kwargs["shape"][row]
        windows = tuple((int(a), int(a + b)) for a, b in zip(start, length))
        out.append(
            ArraySource(
                path=pick("path"),
                format=pick("format"),
                version=pick("version"),
                key=pick("key"),
                origin_id=pick("origin_id"),
                windows=windows,
                shape=tuple(int(x) for x in length),
                dtype=pick("source_dtype"),
                extent=tuple(int(x) for x in kwargs["extent"][row]),
            )
        )
    return out


# The format arguments every from_columns test shares.
FORMAT = MappingProxyType({"format": "DASDAE", "version": "1", "source_dtype": "f4"})


def reference_encoding(values):
    """Return the distinct values in first-seen order and each one's code."""
    index = {}
    codes = [index.setdefault(x, len(index)) for x in values]
    return list(index), codes


class TestStringColumns:
    """Strings are dictionary encoded in one vectorized pass."""

    values = ("b", "", "a", "b", "é", "", "日本", "a", "é")

    def test_matches_the_loop(self):
        """The distinct values and codes are those a dict would give."""
        column = lazy_module._Column.of(self.values)
        distinct, codes = reference_encoding(self.values)
        assert list(column.values) == distinct
        assert column.codes.tolist() == codes
        assert column.codes.dtype == np.int32

    @pytest.mark.parametrize("kind", ["numpy", "pandas"])
    def test_array_input(self, kind):
        """Numpy and pandas strings give plain python strings."""
        array = np.array(self.values)
        if kind == "pandas":
            array = pd.Series(list(self.values)).astype(str)
        column = lazy_module._Column.of(array)
        assert list(column.values) == reference_encoding(self.values)[0]
        assert all(type(x) is str for x in column.values)
        assert type(column.codes) is np.ndarray

    def test_scalars_keep_their_type(self):
        """Equal values of another type or sign stay distinct."""
        column = lazy_module._Column.of([1, True, 1.0, -0.0, 0.0, None, "1"])
        assert len(column.values) == 7

    @pytest.mark.parametrize("repeats", [1, 100])
    def test_missing_strings_stay_distinct(self, repeats):
        """A missing value among strings is a value of its own."""
        keys = pd.Series(["k1", None, "k2"] * repeats)
        array = LazyArray.from_columns("/a", (2,), key=keys, **FORMAT)
        members = array.table.members
        key = members.sources.key
        assert key.codes[members.source_row].tolist() == [0, 1, 2] * repeats
        assert key.values[0] == "k1" and key.values[2] == "k2"
        assert pd.isna(key.values[1])

    @pytest.mark.parametrize("count", [1, 5, 255, 256, 500])
    def test_any_length(self, count):
        """Short and long columns encode alike, in first-seen order."""
        values = [f"v{-x % 7}" for x in range(count)]
        column = lazy_module._Column.of(values)
        assert (list(column.values), column.codes.tolist()) == reference_encoding(
            values
        )
        assert all(type(x) is str for x in column.values)

    def test_nul_strings_stay_distinct(self):
        """Strings which differ after a NUL byte stay distinct at any length."""
        values = ["a\0b", "a\0c"] * 200
        column = lazy_module._Column.of(values)
        assert (list(column.values), column.codes.tolist()) == reference_encoding(
            values
        )

    @pytest.mark.parametrize("count", [3, 300])
    def test_series_index_ignored(self, count):
        """A series is read by position, whatever its index."""
        values = [f"k{x % 2}" for x in range(count)]
        series = pd.Series(values, index=np.arange(count) + 10)
        column = lazy_module._Column.of(series)
        assert (list(column.values), column.codes.tolist()) == reference_encoding(
            values
        )

    def test_all_none(self):
        """A column of None is one value."""
        column = lazy_module._Column.of([None] * 5)
        assert column.values == (None,)
        assert not column.codes.any()


class TestFromColumns:
    """Columns build the same array as the sources they describe."""

    @pytest.mark.parametrize("seed", range(60))
    def test_matches_sources(self, seed):
        """A random case gives the frame, id, shape and dtype of its sources."""
        kwargs = column_case(np.random.default_rng(seed))
        sources = sources_of_columns(kwargs)
        expected = LazyArray.from_sources(
            sources,
            axis=kwargs["axis"],
            base_uri=kwargs["base_uri"],
            dtype=kwargs.get("dtype"),
            cast_via=kwargs.get("cast_via"),
        )
        array = LazyArray.from_columns(**kwargs)
        pd.testing.assert_frame_equal(array.to_frame(), expected.to_frame())
        assert array.data_id == expected.data_id
        assert array.shape == expected.shape
        assert array.dtype == expected.dtype
        # Members with samples are laid end to end, in the order given.
        assert array.sources == tuple(x for x in sources if x.size)
        lengths = kwargs["shape"][kwargs["shape"].all(axis=1)]
        axis = kwargs["axis"] % lengths.shape[1]
        corners = array.table.axes["out_start"].reshape(lengths.shape)[:, axis]
        expected_corners = np.cumsum(lengths[:, axis]) - lengths[:, axis]
        assert corners.tolist() == expected_corners.tolist()

    def test_many_members(self):
        """Hundreds of members, past the dict loop, match their sources."""
        rng = np.random.default_rng(0)
        count, pool = 300, ["/root/a.h5", "/root/b.h5", "/c.h5"]
        kwargs = {
            "path": [pool[x] for x in rng.integers(0, 3, count)],
            "shape": np.tile([2, 3], (count, 1)),
            "start": np.zeros((count, 2), np.int64),
            "extent": np.tile([2, 3], (count, 1)),
            "format": [["DASDAE", "H5Simple"][x] for x in rng.integers(0, 2, count)],
            "version": "1",
            "key": [f"k{x}" for x in rng.integers(0, 4, count)],
            "origin_id": "",
            "source_dtype": np.dtype("f4"),
            "axis": 1,
            "base_uri": "/root/",
        }
        sources = sources_of_columns(kwargs)
        array = LazyArray.from_columns(**kwargs)
        expected = LazyArray.from_sources(sources, axis=1, base_uri="/root/")
        assert array.sources == tuple(sources)
        pd.testing.assert_frame_equal(array.to_frame(), expected.to_frame())
        assert array.data_id == expected.data_id

    def test_loads_files(self, two_sources, joined):
        """Windows of files load as the sources' array does."""
        sources = [x[5:50] for x in two_sources]
        array = LazyArray.from_columns(
            np.array([x.path for x in sources]),
            [x.shape for x in sources],
            format=sources[0].format,
            version=sources[0].version,
            source_dtype=sources[0].dtype,
            key=[x.key for x in sources],
            origin_id=[x.origin_id for x in sources],
            start=[[w[0] for w in x.windows] for x in sources],
            extent=[x.extent for x in sources],
        )
        expected = LazyArray.from_sources(sources)
        assert array.data_id == expected.data_id
        assert np.array_equal(array.load(), expected.load())

    def test_whole_files(self, two_sources, joined):
        """One shared shape and default windows read each file whole."""
        first = two_sources[0]
        array = LazyArray.from_columns(
            [first.path, first.path],
            first.shape,
            format=first.format,
            version=first.version,
            source_dtype=first.dtype,
            key=first.key,
            origin_id=first.origin_id,
            axis=1,
        )
        data = first.load()
        assert array.shape == (data.shape[0], 2 * data.shape[1])
        assert np.array_equal(array.load(), np.concatenate([data, data], axis=1))

    def test_one_path_many_rows(self):
        """A scalar path is repeated over the rows of the shape."""
        array = LazyArray.from_columns(
            "/a.h5", [[2, 3], [4, 3]], format="DASDAE", version="1", source_dtype="f4"
        )
        assert array.shape == (6, 3) and len(array) == 2
        sources = array.table.members.sources
        assert len(sources) == 1 and sources.path.at(np.arange(1)) == ["/a.h5"]
        assert (sources.base_uri[0], sources.format[0], sources.version[0]) == (
            "",
            "DASDAE",
            "1",
        )

    def test_paths_split_under_base(self):
        """A path under the base uri is stored relative to it; others whole."""
        array = LazyArray.from_columns(
            ["/root/a.h5", "/other/b.h5"],
            (2,),
            format="DASDAE",
            version="1",
            source_dtype="f4",
            base_uri="/root/",
        )
        sources = array.table.members.sources
        rows = np.arange(len(sources))
        fields = [getattr(sources, x).at(rows) for x in SOURCE_FIELDS[:4]]
        assert list(zip(*fields)) == [
            ("/root/", "a.h5", "DASDAE", "1"),
            ("", "/other/b.h5", "DASDAE", "1"),
        ]
        assert [x.path for x in array.sources] == ["/root/a.h5", "/other/b.h5"]

    def test_rows_from_any_column(self):
        """A per member key sets the member count when path and shape are shared."""
        array = LazyArray.from_columns("/a.h5", (2,), key=["a", "b"], **FORMAT)
        assert array.shape == (4,)
        assert [x.key for x in array.sources] == ["a", "b"]

    @pytest.mark.parametrize("name", ["start", "extent"])
    def test_rows_from_windows(self, name):
        """Per member windows of one resource set the member count too."""
        windows = {"start": [[0], [2]], "extent": 4}
        if name == "extent":
            windows = {"extent": [[4], [4]]}
        array = LazyArray.from_columns("/a.h5", (2,), **windows, **FORMAT)
        assert array.shape == (4,) and len(array) == 2

    def test_spellings_of_one_dtype(self):
        """Two spellings of one dtype or cast are one value."""
        array = LazyArray.from_columns(
            ["/a", "/b"],
            (2,),
            format="DASDAE",
            version="1",
            source_dtype=["f4", np.dtype("float32")],
            cast_via=["f8", np.float64],
        )
        members = array.table.members
        dtype = members.sources.dtype
        assert dtype.values == ("<f4",)
        assert members.cast.values == ("<f8",)
        codes = dtype.codes[members.source_row].tolist()
        assert codes == members.cast.codes.tolist() == [0, 0]

    def test_structured_source_dtype(self):
        """A source whose dtype is a field list names it as numpy does."""
        spec = [("x", "i4"), ("y", "f8")]
        listed, typed = (
            ArraySource(path="/a", format="DASDAE", dtype=x).describe((2,), x)
            for x in (spec, np.dtype(spec))
        )
        listed = replace(listed, dtype=spec)
        first, second = (LazyArray.from_sources([x]) for x in (listed, typed))
        pd.testing.assert_frame_equal(first.to_frame(), second.to_frame())
        assert first.data_id == second.data_id
        assert first.dtype == np.dtype(spec)

    def test_structured_cast(self):
        """A cast given as a field list is named as numpy names it."""
        spec = [("x", "i4"), ("y", "f8")]
        source = ArraySource(path="/a", format="DASDAE", dtype=spec).describe(
            (2,), spec
        )
        first, second = (
            LazyArray.from_sources([source], cast_via=[x])
            for x in (spec, np.dtype(spec))
        )
        pd.testing.assert_frame_equal(first.to_frame(), second.to_frame())
        assert first.data_id == second.data_id

    def test_cast_array(self):
        """Casts given as an array apply one per source."""
        sources = [ArraySource.full((2,), 1.0), ArraySource.full((3,), 2.0)]
        array = LazyArray.from_sources(sources, cast_via=np.array(["f4", "f4"]))
        assert array.to_frame()["cast"].tolist() == ["<f4", "<f4"]

    def test_caller_arrays_stay_writable(self):
        """The array keeps copies, never the caller's own arrays."""
        shape = np.array([[2, 3]])
        LazyArray.from_columns(
            ["/a.h5"], shape, format="DASDAE", version="1", source_dtype="f4"
        )
        shape[0, 0] = 5


class TestFromColumnsRefuses:
    """Columns which cannot describe an array are refused."""

    kwargs = FORMAT

    def test_no_members(self):
        """An array needs at least one member."""
        with pytest.raises(ParameterError, match="at least one member"):
            LazyArray.from_columns([], (2,), **self.kwargs)

    def test_no_axes(self):
        """An array has at least one axis."""
        with pytest.raises(ParameterError, match="at least one dimension"):
            LazyArray.from_columns("/a.h5", (), **self.kwargs)

    def test_bad_shape(self):
        """A shape is one row of lengths, or one row per member."""
        with pytest.raises(ParameterError, match="shape"):
            LazyArray.from_columns("/a.h5", [[[1]]], **self.kwargs)

    def test_too_many_axes(self):
        """A flat shape is one shape, so per member lengths are refused."""
        with pytest.raises(ParameterError, match=r"\(n, ndim\)"):
            LazyArray.from_columns(["/a"] * 65, np.arange(1, 66), **self.kwargs)

    def test_mismatched_rows(self):
        """Per member columns must all have a row per member."""
        with pytest.raises(ParameterError, match="shape"):
            LazyArray.from_columns(["/a", "/b"], [[1], [2], [3]], **self.kwargs)
        with pytest.raises(ParameterError, match="key"):
            LazyArray.from_columns(["/a", "/b"], [1], key=["x"], **self.kwargs)
        with pytest.raises(ParameterError, match="cast_via"):
            LazyArray.from_columns(["/a", "/b"], [1], cast_via=["f8"], **self.kwargs)
        with pytest.raises(ParameterError, match="origin_id"):
            LazyArray.from_columns(
                "/a", [1], key=["x", "y"], origin_id=["o"] * 3, **self.kwargs
            )

    @pytest.mark.parametrize(
        "source_dtype", [None, ["f4", None], ["f4", np.nan], ["f4", pd.NA]]
    )
    def test_missing_dtype(self, source_dtype):
        """Every member states the dtype it is stored at."""
        with pytest.raises(ParameterError, match="source_dtype"):
            LazyArray.from_columns(
                ["/a", "/b"],
                (2,),
                format="DASDAE",
                version="1",
                source_dtype=source_dtype,
            )

    def test_ndim_disagrees(self):
        """Starts and extents have one number per axis of the shape."""
        with pytest.raises(ParameterError, match="start"):
            LazyArray.from_columns("/a", (2, 2), start=(0,), **self.kwargs)

    @pytest.mark.parametrize(
        "window",
        [
            {"start": (1, 0)},
            {"start": (-1, 0), "extent": (5, 2)},
            {"start": (2, 0), "extent": (3, 2)},
            {"shape": (-1, 2), "extent": (0, 2)},
        ],
    )
    def test_window_outside(self, window):
        """A window must be of positive length and inside its stored array."""
        window = {"shape": (2, 2), **window}
        with pytest.raises(ParameterError, match="outside"):
            LazyArray.from_columns("/a", **window, **self.kwargs)

    @pytest.mark.parametrize("empty", ["path", "format"])
    @pytest.mark.parametrize("missing", ["", None, np.nan])
    def test_unloadable(self, empty, missing):
        """A member needs a path and a format to be read."""
        kwargs = {**self.kwargs, "path": ["/a", "/b"], empty: ["x", missing]}
        with pytest.raises(ParameterError, match="path and a format"):
            LazyArray.from_columns(shape=(2,), **kwargs)


def mixed_sources():
    """Files under and outside a base, keys, origins and constants; mixed adds casts."""
    whole = ArraySource(path="/root/a.h5", format="DASDAE", version="1")
    return [
        whole.describe((3, 2), "f4"),
        replace(whole, key="k1").describe((5, 2), "i2")[1:3],
        ArraySource(path="/elsewhere/é.h5", format="H5Simple", version="2").describe(
            (2, 2), "u1"
        ),
        replace(whole, path="/root/b.h5", origin_id="a" * 32).describe((4, 2), "f4")[
            0:2
        ],
        ArraySource.full((2, 2), 1, dtype="i8"),  # int64 on every platform
        ArraySource(filled=True, value=True).describe((1, 2), "f8"),
        ArraySource(filled=True, value=-0.0).describe((1, 2), "f8"),
        ArraySource(filled=True, value=float("nan")).describe((1, 2), "f8"),
    ]


def mixed():
    """Return an array of every kind of member, some cast on the way."""
    casts = [None, "f8", None, "f4", None, None, "f4", None]
    return LazyArray.from_sources(mixed_sources(), base_uri="/root/", cast_via=casts)


def many_members(count=300):
    """Return an array of many members over a few files, some paths repeated."""
    return LazyArray.from_columns(
        [f"/root/f{x % 97:03d}.h5" for x in range(count)],
        (2, 3),
        format="DASDAE",
        version="1",
        source_dtype=["f4", "i2"] * (count // 2),
        key=[["", "k"][x % 3 == 0] for x in range(count)],
        base_uri="/root/",
    )


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


class TestSourceTable:
    """Members read one sources table, which views share until a join cuts it."""

    def test_views_share_it(self):
        """Slicing, joining and rechunking keep the one sources table."""
        array = many_members()
        sources = array.table.members.sources
        half = array.shape[0] // 2
        views = [
            array[10:500],
            concat([array[:half], array[half:]]),
            *array.rechunk([0, 7, 333, 600]),
            LazyTable.from_arrays([array[:half], array[half:]])[1],
        ]
        assert all(x.table.members.sources is sources for x in views)

    def test_built_apart_is_built_together(self):
        """Arrays built apart and joined are the array built from all of them."""
        sources = [*mixed_sources(), *mixed_sources()[::-1]]
        together = LazyArray.from_sources(sources)
        apart = concat([LazyArray.from_sources(x) for x in (sources[:5], sources[5:])])
        assert apart.data_id == together.data_id
        # A nan is not equal to itself, so the sources are compared as written.
        assert repr(apart.sources) == repr(together.sources)
        pd.testing.assert_frame_equal(apart.to_frame(), together.to_frame())

    def test_compacted_when_little_is_read(self):
        """A join of a few members of a big table keeps only the rows read."""
        array = many_members()
        parts = [array[0:4], array[-6:]]
        joined = concat(parts)
        assert len(joined.table.members.sources) < len(array.table.members.sources)
        assert len(joined.table.members.sources) <= 5
        sources = [*parts[0].sources, *parts[1].sources]
        rebuilt = LazyArray.from_sources(sources, base_uri="/root/")
        assert joined.data_id == rebuilt.data_id
        assert joined.sources == rebuilt.sources
        pd.testing.assert_frame_equal(joined.to_frame(), rebuilt.to_frame())

    def test_strings_round_trip(self):
        """Paths and origins of any text are read back exactly."""
        texts = ["/root/日本/é.h5", "/root/b.h5", "/a.h5", "/root/"]
        origins = ["", "ü" * 3, "", "a" * 32]
        sources = [
            ArraySource(path=x, format="DASDAE", origin_id=y).describe((2,), "f4")
            for x, y in zip(texts, origins)
        ]
        sources.insert(2, ArraySource.full((2,), 0.0))
        array = LazyArray.from_sources(sources, base_uri="/root/")
        assert array.sources == tuple(sources)
        assert array.sources[2].path == ""
        back = LazyArray.from_frame(array.to_frame(), array.shape, array.dtype)
        assert back.sources == tuple(sources)
        assert back.data_id == array.data_id
        first = LazyArray.from_sources(sources[:2], base_uri="/root/")
        joined = concat([array[4:], first, array[8:]])
        assert joined.sources == (*sources[2:], *sources[:2], sources[-1])

    def test_undecodable_path(self):
        """A path holding an escaped undecodable byte is read back as it was."""
        source = stored((2,), path="/root/\udcff.h5")
        array = LazyArray.from_sources([source, source], base_uri="/root/")
        assert array.sources == (source, source)
        joined = concat([array, LazyArray.from_source(source), array[2:]])
        assert joined.sources == (source,) * 4

    def test_constant_values_stay_apart(self):
        """Constants of one dtype whose values compare equal are not one source."""
        values = [1, True, 1.0, -0.0, 0.0, float("nan")]
        sources = [
            ArraySource(filled=True, value=x).describe((1, 2), "f8") for x in values
        ]
        array = LazyArray.from_sources([stored((1, 2), dtype="f8"), *sources])
        assert len(array.table.members.sources) == len(values) + 1
        ids = [array[0 : x + 2].data_id for x in range(len(values))]
        assert len(set(ids)) == len(values)

    def test_one_source_two_casts(self):
        """Members of one source row keep their own casts."""
        source = stored((2,), path="/a/c.h5")
        casts = [None, "f4", "f8"]
        together = LazyArray.from_sources([source] * 3, cast_via=casts, dtype="f8")
        assert len(together.table.members.sources) == 1
        apart = [
            LazyArray.from_sources([source], cast_via=[x], dtype="f8") for x in casts
        ]
        assert together.data_id == concat(apart).data_id
        uncast = LazyArray.from_sources([source] * 3, dtype="f8")
        assert together.data_id != uncast.data_id

    def test_whole_member_of_a_later_source(self):
        """A member left whole is named as its own source, whichever row it reads."""
        first = stored((2,), path="/a/f4.h5", dtype="f4")
        second = stored((2,), path="/a/f8.h5", dtype="f8")
        cut = LazyArray.from_sources([first, second], dtype="f8")[2:]
        assert cut.data_id == LazyArray.from_sources([second]).data_id == second.data_id

    def test_whole_constant_of_a_later_source(self):
        """A constant cut from a mixed array is named as the constant alone."""
        filled = ArraySource.full((2,), 5.0)
        cut = LazyArray.from_sources([stored((2,), dtype="f8"), filled])[2:]
        assert cut.data_id == LazyArray.from_sources([filled]).data_id

    def test_empty_frame_columns(self):
        """An array of no members has the text columns of any other."""
        array = LazyArray.from_sources([stored((2,)), stored((2,), path="/c.h5")])
        expected = array.to_frame().dtypes
        for empty in (array[0:0], concat([array[0:0], constant((0,))])):
            assert empty.to_frame().dtypes.equals(expected)

    def test_join_merges_repeated_sources(self):
        """Windows of one file built apart and joined read one source row."""
        whole = stored((1000,), path="/a/one.h5")
        windows = [whole[x : x + 10] for x in range(0, 1000, 10)]
        joined = concat([LazyArray.from_source(x) for x in windows])
        assert len(joined.table.members.sources) == 1
        assert joined.data_id == LazyArray.from_sources(windows).data_id

    def test_copies_are_named_once(self, monkeypatch):
        """Copies of one array joined keep, and name, one row per source."""
        copies = [many_members(30) for _ in range(20)]
        expected = LazyArray.from_sources(
            [x for y in copies for x in y.sources]
        ).data_id
        joined = concat(copies)
        assert len(joined.table.members.sources) == 30
        hashed = []
        original = lazy_module.H
        monkeypatch.setattr(
            lazy_module, "H", lambda x, y: hashed.append(x) or original(x, y)
        )
        assert joined.data_id == expected
        assert hashed.count("location") == 30

    def test_most_read_is_kept_whole(self):
        """A join reading most of a table shares it rather than cutting it."""
        paths = [f"/a/f{x}.h5" for x in range(300)]
        array = LazyArray.from_columns(paths, (2,), **FORMAT)
        joined = concat([array[:-2], array[:0]])
        assert joined.table.members.sources is array.table.members.sources


class TestMissingText:
    """A missing origin id or frame column reads as it did before."""

    kwargs = MappingProxyType({"format": "F", "version": "1", "source_dtype": "f4"})

    @pytest.mark.parametrize("origin", [None, [None, "x"]])
    def test_columns(self, origin):
        """A None origin id is no origin id, as an empty one is."""
        array = LazyArray.from_columns(
            ["/a", "/b"], (2, 1), origin_id=origin, **self.kwargs
        )
        empty = [x or "" for x in origin] if isinstance(origin, list) else ""
        expected = LazyArray.from_columns(
            ["/a", "/b"], (2, 1), origin_id=empty, **self.kwargs
        )
        assert array.data_id == expected.data_id
        pd.testing.assert_frame_equal(array.to_frame(), expected.to_frame())

    def test_pinned(self):
        """The id is the one an array with no origin id always had."""
        array = LazyArray.from_columns(
            ["/a", "/b"], (2, 1), origin_id=None, **self.kwargs
        )
        assert array.data_id == "bdc10f0ac9771acac56db59e8a5807f3"

    def test_source(self):
        """A source whose origin id is None reads as one with none."""
        source = ArraySource(path="/a", format="F", origin_id=None).describe((2,), "f4")
        array = LazyArray.from_sources([source])
        expected = LazyArray.from_sources([replace(source, origin_id="")])
        assert array.data_id == expected.data_id
        pd.testing.assert_frame_equal(array.to_frame(), expected.to_frame())

    def test_frame(self):
        """A frame whose origin ids read back as missing gives the same array."""
        array = LazyArray.from_columns(["/a", "/b"], (2, 1), **self.kwargs)
        frame = array.to_frame().astype({"origin_id": object})
        frame["origin_id"] = None
        back = LazyArray.from_frame(frame, array.shape, array.dtype)
        assert back.data_id == array.data_id
        assert back.sources == array.sources

    @pytest.mark.parametrize("name", ["key", "origin_id", "dtype", "cast", "value"])
    def test_frame_without_column(self, name):
        """A frame missing a member column reads it as empty."""
        array = LazyArray.from_columns(["/a", "/b"], (2, 1), **self.kwargs)
        frame = array.to_frame()
        back = LazyArray.from_frame(frame.drop(columns=name), array.shape, array.dtype)
        assert back.data_id == array.data_id
