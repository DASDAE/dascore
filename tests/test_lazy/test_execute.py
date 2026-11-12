"""Tests for running a lazy array: the boxes, what each reads, and the result."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.config import config_context
from dascore.exceptions import LazyOperationError
from dascore.lazy.array import LazyArray
from dascore.lazy.chain import Chain, Node
from dascore.lazy.contract import Window
from dascore.lazy.execute import chunk_boxes, read_schedule
from dascore.proc.basic import Abs
from dascore.proc.coords import Select

from .conftest import constant, lazy_patch, stored_patch

# Bytes of one float64 row of the stored patch: 300 distances by 2000 times.
ROW = 2000 * 8


@pytest.fixture(scope="module")
def lazy(tmp_path_factory):
    """A stored float64 patch of random values, its data unread."""
    shape = dc.get_example_patch().shape
    data = np.random.default_rng(0).standard_normal(shape)
    return lazy_patch(stored_patch(tmp_path_factory.mktemp("execute"), data))


@pytest.fixture(scope="module")
def eager(lazy):
    """The same patch loaded, so both start from one id."""
    return lazy.load()


def _covers(boxes, shape):
    """Whether boxes tile a shape: each sample in exactly one."""
    count = np.zeros(shape, np.int64)
    for box in boxes:
        count[box] += 1
    return bool((count == 1).all())


class TestSelect:
    """A selection of lazy data stays lazy and reads only what it keeps."""

    def test_matches_eager(self, lazy, eager, recipe_loads):
        """Coords, attrs and id are the eager ones, and nothing is read."""
        kwargs = {"time": (100, 900), "distance": (..., 50), "samples": True}
        out, expected = lazy.select(**kwargs), eager.select(**kwargs)
        assert not recipe_loads
        assert isinstance(out.data.chain.last.contract, Window)
        assert out.coords == expected.coords
        assert out.attrs == expected.attrs
        assert out.attrs.data_id == expected.attrs.data_id
        assert out.load() == expected

    def test_reads_only_the_window(self, lazy, recipe_loads):
        """What is read has the selection's shape, not the array's."""
        out = lazy.select(time=(100, 900), distance=(10, 50), samples=True)
        out.load()
        assert [x.shape for x in recipe_loads] == [(40, 800)]

    def test_by_value(self, lazy, eager):
        """Coordinate values select as they do eagerly, ends included."""
        time = eager.get_array("time")
        kwargs = {"time": (time[30], time[60])}
        assert lazy.select(**kwargs).load() == eager.select(**kwargs)

    def test_everything_is_the_patch(self, lazy):
        """A selection which keeps everything is a no-op, as eagerly."""
        assert lazy.select(time=(None, None)) is lazy

    def test_selections_compose(self, lazy, eager, recipe_loads):
        """A window of a window reads the samples both keep."""
        out = lazy.select(time=(100, 900), samples=True)
        out = out.select(time=(50, 60), samples=True)
        expected = eager.select(time=(150, 160), samples=True)
        assert np.array_equal(out.load().data, expected.data)
        assert [x.shape for x in recipe_loads] == [(300, 10)]

    def test_values_refused(self, lazy, eager, recipe_loads):
        """Picking out separate values is no window, so it is refused."""
        distance = eager.get_array("distance")
        with pytest.raises(LazyOperationError, match="select"):
            lazy.select(distance=distance[1::2])
        assert not recipe_loads

    def test_empty(self, lazy, eager):
        """A selection which keeps nothing loads as an empty array."""
        out = lazy.select(time=(5, 5), samples=True)
        assert out.shape == (300, 0)
        assert out.load().data.shape == (300, 0)


class TestReadSchedule:
    """What is read is known before anything is read."""

    def test_filter_after_select_stays_in_the_window(self, lazy, recipe_loads):
        """A halo never reaches into what an earlier selection dropped."""
        out = lazy.select(time=(100, 900), samples=True)
        out = out.median_filter(time=5, samples=True).data
        whole, inner = (slice(0, 300), slice(0, 800)), (slice(0, 300), slice(50, 60))
        found = read_schedule(out, [whole, inner])
        # The window's ends are the array's ends to the filter, as eagerly.
        assert found == [
            (slice(0, 300), slice(100, 900)),
            (slice(0, 300), slice(148, 162)),
        ]
        assert not recipe_loads

    def test_select_after_filter_reads_only_its_reach(self, lazy, eager, recipe_loads):
        """A selection made after a filter still narrows what is read."""
        out = lazy.median_filter(time=5, samples=True)
        out = out.select(time=(100, 900), samples=True)
        expected = eager.median_filter(time=5, samples=True)
        expected = expected.select(time=(100, 900), samples=True)
        assert np.array_equal(out.load().data, expected.data)
        assert [x.shape for x in recipe_loads] == [(300, 804)]

    def test_whole_axis_reads_it_all(self, lazy):
        """A selection after a recursive filter cannot narrow its axis."""
        out = lazy.pass_filter(time=(10, 100)).select(
            time=(100, 900), distance=(5, 10), samples=True
        )
        box = (slice(0, 5), slice(0, 800))
        assert read_schedule(out.data, [box]) == [(slice(5, 10), slice(0, 2000))]


class TestChunkBoxes:
    """How an array is cut into the boxes it is computed in."""

    def test_one_box_when_it_fits(self, lazy):
        """An array within the block size is not cut."""
        whole = (slice(0, 300), slice(0, 2000))
        assert chunk_boxes(lazy.abs().data, 300 * ROW) == [whole]

    def test_default_is_the_setting(self, lazy):
        """With no block size given, the setting decides."""
        with config_context(lazy_block_size=100 * ROW):
            assert len(chunk_boxes(lazy.abs().data)) == 4

    @pytest.mark.parametrize("rows", (300, 100, 7, 1))
    def test_boxes_tile_and_fit(self, lazy, rows):
        """Every sample is in one box, and no box is over the block size."""
        boxes = chunk_boxes(lazy.abs().data, rows * ROW)
        assert _covers(boxes, (300, 2000))
        assert all(
            np.prod([x.stop - x.start for x in b]) * 8 <= rows * ROW for b in boxes
        )

    def test_leading_axis_is_cut_first(self, lazy):
        """The pieces are slabs of the first axis while that is enough."""
        boxes = chunk_boxes(lazy.abs().data, 10 * ROW)
        assert {x[1] for x in boxes} == {slice(0, 2000)}

    def test_then_the_next(self, lazy):
        """Below one row, the second axis is cut as well."""
        boxes = chunk_boxes(lazy.abs().data, ROW // 4)
        assert {x[0].stop - x[0].start for x in boxes} == {1}
        assert len({x[1] for x in boxes}) == 4

    def test_whole_axis_is_never_cut(self, lazy):
        """Cutting an axis which is needed whole would save nothing."""
        out = lazy.pass_filter(distance=(None, 0.1)).data
        boxes = chunk_boxes(out, 10 * ROW)
        assert {x[0] for x in boxes} == {slice(0, 300)}
        assert len(boxes) > 1 and _covers(boxes, (300, 2000))

    def test_aimed_for_not_promised(self, lazy):
        """With every axis needed whole there is one box, whatever its size."""
        out = lazy.pass_filter(distance=(None, 0.1)).pass_filter(time=(10, 100))
        assert len(chunk_boxes(out.data, ROW)) == 1

    def test_halo_axis_stays_long(self, lazy):
        """An axis is not cut into pieces short beside what is read around them."""
        out = lazy.gaussian_filter(distance=5, samples=True).data
        (reach, _) = out.chain.last.contract.widths[0]
        boxes = chunk_boxes(out, ROW)
        # All but the last piece have one length.
        longest = max(x[0].stop - x[0].start for x in boxes)
        assert reach <= longest < 300
        assert _covers(boxes, (300, 2000))

    def test_no_contract_is_one_box(self):
        """A step which recorded no contract says nothing about its parts."""
        node = Node(Abs(), {}).settled((40, 30), np.float64, "id")
        array = LazyArray(constant((40, 30), -1.0), Chain((node,)))
        assert len(chunk_boxes(array, 8)) == 1

    def test_nothing_to_compute_is_one_box(self, lazy):
        """Selections alone are read in one piece: cutting only reopens files."""
        out = lazy.select(time=(100, 900), samples=True)
        assert len(chunk_boxes(lazy.data, ROW)) == 1
        assert len(chunk_boxes(out.data, ROW)) == 1
        assert len(chunk_boxes(out.abs().data, ROW)) > 1

    def test_counts_each_steps_dtype(self, tmp_path_factory):
        """A step which widens the dtype is what the block size is held to."""
        data = np.ones(dc.get_example_patch().shape, np.int8)
        small = lazy_patch(stored_patch(tmp_path_factory.mktemp("int8"), data))
        # More than the int8 read and less than the float16 result.
        assert len(chunk_boxes(small.angle().data, 900_000)) > 1

    def test_empty_array(self):
        """An array with no samples is one empty box."""
        array = LazyArray(constant((0, 3)))
        assert chunk_boxes(array, 8) == [(slice(0, 0), slice(0, 3))]


class TestLoad:
    """Loading in chunks gives what loading whole gives."""

    @staticmethod
    def _run(patch):
        """A chain with a window, a halo, a whole axis and a pointwise step."""
        out = patch.select(time=(50, 1500), samples=True)
        out = out.gaussian_filter(time=3, distance=2, samples=True)
        out = out.pass_filter(time=(10, 100))
        return out.abs().select(distance=(20, 250), samples=True)

    def test_keeps_the_dtype(self, tmp_path_factory):
        """Put together from boxes, the array has the dtype the patch stated."""
        data = np.random.default_rng(1).standard_normal((300, 2000))
        directory = tmp_path_factory.mktemp("float32")
        small = lazy_patch(stored_patch(directory, data.astype("float32")))
        with config_context(lazy_block_size=10 * ROW):
            loaded = small.abs().load()
        assert loaded.data.dtype == np.float32
        assert np.array_equal(loaded.data, np.abs(small.load().data))

    @pytest.mark.parametrize("rows", (0, 60, 3))
    def test_any_block_size(self, lazy, eager, recipe_loads, rows):
        """The result and its id do not depend on how the work was cut."""
        out, expected = self._run(lazy), self._run(eager)
        with config_context(lazy_block_size=rows * ROW):
            loaded = out.load()
        assert (len(recipe_loads) > 1) == (rows > 0)
        assert loaded.attrs == expected.attrs
        np.testing.assert_allclose(loaded.data, expected.data, rtol=1e-12, atol=1e-12)


class TestFilterArguments:
    """Arguments which change what a filter reads change its contract."""

    @staticmethod
    def _same(lazy_out, eager_out):
        """Loaded in many small boxes, the result is the eager one."""
        with config_context(lazy_block_size=ROW // 8):
            assert len(chunk_boxes(lazy_out.data)) > 1
            loaded = lazy_out.load()
        assert loaded.data.dtype == eager_out.data.dtype
        np.testing.assert_allclose(loaded.data, eager_out.data, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("mode", ("wrap", "grid-wrap"))
    @pytest.mark.parametrize("name", ("gaussian_filter", "median_filter"))
    def test_wrapping_edges(self, lazy, eager, name, mode):
        """An edge filled from the far end of the array has the axis whole."""
        kwargs = {"distance": 3, "samples": True, "mode": mode}
        self._same(getattr(lazy, name)(**kwargs), getattr(eager, name)(**kwargs))

    def test_truncate(self, lazy, eager):
        """A wider gaussian kernel reads further."""
        kwargs = {"distance": 3, "samples": True, "truncate": 8.0}
        self._same(lazy.gaussian_filter(**kwargs), eager.gaussian_filter(**kwargs))

    @pytest.mark.parametrize("mode", (("wrap", "reflect"), ["wrap", "reflect"]))
    def test_mode_for_each_axis_refused(self, lazy, recipe_loads, mode):
        """A mode given axis by axis is not worked out, so it is refused."""
        with pytest.raises(LazyOperationError, match="gaussian_filter"):
            lazy.gaussian_filter(time=2, distance=2, samples=True, mode=mode)
        assert not recipe_loads


class TestQueries:
    """Only a query which is one range for each dimension is a window."""

    def test_stepped_slice_is_no_window(self):
        """A slice with a step would not be a box."""
        plan = {"indexer": (slice(0, 10, 2), slice(None))}
        assert Select().chunk_contract(np.dtype("f8"), plan) is None

    def test_unselect_refused(self, lazy, recipe_loads):
        """What unselect keeps is spelled as positions, so it is refused."""
        with pytest.raises(LazyOperationError, match="unselect"):
            lazy.unselect(time=(0, 100), samples=True)
        assert not recipe_loads


class TestDtypes:
    """A filter states a contract only for the dtypes its kernel keeps."""

    CALLS = (
        lambda x: x.pass_filter(time=(10, 100)),
        lambda x: x.median_filter(time=5, samples=True),
        lambda x: x.savgol_filter(polyorder=2, time=7, samples=True),
        lambda x: x.gaussian_filter(time=1.5, samples=True),
    )

    @pytest.fixture(scope="class")
    def stored(self, tmp_path_factory):
        """A stored patch of each of three dtypes."""
        directory = tmp_path_factory.mktemp("filter_dtypes")
        data = np.arange(600_000).reshape(dc.get_example_patch().shape) % 97
        return {
            x: stored_patch(directory, data.astype(x), x)
            for x in ("int16", "float32", "float64")
        }

    @pytest.mark.parametrize("call", (CALLS[0], CALLS[2]))
    def test_changed_dtype_refused(self, stored, call, recipe_loads):
        """Filters which turn integers into floats refuse them, reading nothing."""
        with pytest.raises(LazyOperationError):
            call(lazy_patch(stored["int16"]))
        assert not recipe_loads
