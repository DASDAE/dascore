"""Tests for running a lazy array: the boxes, what each reads, and the result."""

from __future__ import annotations

import math
import tracemalloc
import warnings
from collections import defaultdict
from itertools import product
from typing import Any, ClassVar, NamedTuple

import numpy as np
import pytest

import dascore as dc
from dascore.backends import ArrayAPIBackend
from dascore.config import config_context
from dascore.core.processor import register_kernel
from dascore.core.source import ArraySource
from dascore.exceptions import LazyOperationError, ParameterError, PatchError
from dascore.lazy.array import LazyArray
from dascore.lazy.chain import Chain, Node
from dascore.lazy.contract import ChunkContract, Halo, Pointwise, WholeAxis, Window
from dascore.lazy.execute import chunk_boxes, compute, read_schedule
from dascore.lazy.recipe import Recipe, concat
from dascore.proc.basic import Abs
from dascore.proc.coords import Select
from dascore.utils.array_api import array_namespace, backend_name, to_numpy

from .conftest import constant, lazy_patch, stored_patch, unreadable

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


@pytest.fixture()
def gone(lazy, tmp_path):
    """The lazy patch over a copy of its file, deleted: any read of it raises."""
    return lazy_patch(unreadable(lazy, tmp_path))


def _covers(boxes, shape):
    """Whether boxes tile a shape: each sample in exactly one."""
    count = np.zeros(shape, np.int64)
    for box in boxes:
        count[box] += 1
    return bool((count == 1).all())


class TestSelect:
    """A selection of lazy data stays lazy and reads only what it keeps."""

    def test_matches_eager(self, lazy, gone, eager):
        """Coords, attrs and id are the eager ones, and nothing is read."""
        kwargs = {"time": (100, 900), "distance": (..., 50), "samples": True}
        out, expected = gone.select(**kwargs), eager.select(**kwargs)
        assert isinstance(out.data.chain.last.contract, Window)
        assert out.coords == expected.coords
        assert out.attrs == expected.attrs
        assert out.attrs.data_id == expected.attrs.data_id
        assert lazy.select(**kwargs).load() == expected

    def test_reads_only_the_window(self, lazy, recipe_loads):
        """
        What is read has the selection's shape, not the array's.

        A performance guard (rule C) that keeps its spy: reading more of the file
        and cutting it gives the same values, so the shape read is its only
        witness.
        """
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
        """
        A window of a window reads the samples both keep.

        A performance guard (rule C) that keeps its spy, for the reason
        `test_reads_only_the_window` gives.
        """
        out = lazy.select(time=(100, 900), samples=True)
        out = out.select(time=(50, 60), samples=True)
        expected = eager.select(time=(150, 160), samples=True)
        assert np.array_equal(out.load().data, expected.data)
        assert [x.shape for x in recipe_loads] == [(300, 10)]

    def test_values_refused(self, gone, eager):
        """Picking out separate values is no window, so it is refused."""
        distance = eager.get_array("distance")
        with pytest.raises(LazyOperationError, match="select"):
            gone.select(distance=distance[1::2])

    def test_empty(self, lazy, eager):
        """A selection which keeps nothing loads as an empty array."""
        out = lazy.select(time=(5, 5), samples=True)
        assert out.shape == (300, 0)
        assert out.load().data.shape == (300, 0)


class TestReadSchedule:
    """What is read is known before anything is read."""

    def test_filter_after_select_stays_in_the_window(self, gone):
        """A halo never reaches into what an earlier selection dropped."""
        out = gone.select(time=(100, 900), samples=True)
        out = out.median_filter(time=5, samples=True).data
        whole, inner = (slice(0, 300), slice(0, 800)), (slice(0, 300), slice(50, 60))
        found = read_schedule(out, [whole, inner])
        # The window's ends are the array's ends to the filter, as eagerly.
        assert found == [
            (slice(0, 300), slice(100, 900)),
            (slice(0, 300), slice(148, 162)),
        ]

    def test_select_after_filter_reads_only_its_reach(self, lazy, eager, recipe_loads):
        """
        A selection made after a filter still narrows what is read.

        A performance guard (rule C) that keeps its spy, for the reason
        `TestSelect.test_reads_only_the_window` gives.
        """
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

    @pytest.mark.parametrize("rows", (100, 7, 1))
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
        """
        The result and its id do not depend on how the work was cut.

        Keeps its spy: the reads counted show the work was cut at all, and
        a load in one piece gives the same result, so they are the only
        witness.
        """
        out, expected = self._run(lazy), self._run(eager)
        with config_context(lazy_block_size=rows * ROW):
            loaded = out.load()
        assert (len(recipe_loads) > 1) == (rows > 0)
        assert loaded.attrs == expected.attrs
        np.testing.assert_allclose(loaded.data, expected.data, rtol=1e-12, atol=1e-12)


# The windowed cases slowest in boxes an eighth of a row long. WebAssembly
# runs them in larger boxes only; the other cases still run in eighths there.
SLOW_EIGHTHS = ("halo_beside", "halo_inside", "pointwise", "analysis", "fill")


def _block_sizes(names):
    """Each case at each block size, the slowest eighths marked wasm_skip."""
    slow = [(1 / 8, x) for x in SLOW_EIGHTHS]
    return [
        pytest.param(*x, marks=pytest.mark.wasm_skip) if x in slow else x
        for x in product((0, 60, 3, 1 / 8), names)
    ]


class TestWindowed:
    """An operation in a window is computed from the windows a box overlaps."""

    CASES: ClassVar = {
        # An operation with no contract, which eagerly needs its axis whole.
        "detrend": lambda x: x.window(time=64, samples=True).detrend("time"),
        "whole_axis": lambda x: x.window(distance=32, samples=True).pass_filter(
            time=(10, 100)
        ),
        "halo_beside": lambda x: x.window(time=64, samples=True).median_filter(
            distance=3, samples=True
        ),
        "halo_inside": lambda x: x.window(time=64, samples=True).median_filter(
            time=3, samples=True
        ),
        "pointwise": lambda x: x.window(time=50, step=37, samples=True).abs(),
        "two_dims": lambda x: x.window(time=64, distance=16, samples=True).detrend(
            "time"
        ),
        "analysis": lambda x: x.window(
            time=64, overlap=48, samples=True, analysis="hann"
        ).abs(),
        "fill": lambda x: x.window(time=64, samples=True, fill=3.0).abs(),
        # The first samples of each window are empty, and discarded.
        "empty_ends": lambda x: x.window(time=200, samples=True).fbe(
            time=(10, 50), window=0.1
        ),
        # Every fifth sample of each window is kept, so the result is shorter.
        "every_nth": lambda x: x.window(time=200, overlap=50, samples=True).fbe(
            time=(10, 50), window=0.1, step=0.02
        ),
        "every_nth_two_dims": lambda x: x.window(
            time=80, distance=80, overlap=40, samples=True
        ).decimate(time=4, filter_type=None),
        # A length, and a window, which are no multiple of the three skipped.
        "every_nth_ragged": lambda x: (
            x.select(time=(0, 1999), samples=True)
            .window(time=91, overlap=31, samples=True)
            .decimate(time=3, filter_type=None)
        ),
        "in_a_chain": lambda x: (
            x.select(time=(10, 1700), samples=True)
            .window(time=64, samples=True)
            .detrend("time")
            .abs()
        ),
    }

    @pytest.mark.parametrize(("rows", "name"), _block_sizes(CASES))
    def test_any_block_size(self, lazy, eager, name, rows):
        """However the work is cut, the result is the eager one exactly."""
        out, expected = self.CASES[name](lazy), self.CASES[name](eager)
        assert backend_name(out.data) == "lazy"
        with config_context(lazy_block_size=int(rows * ROW)):
            loaded = out.load()
        assert loaded.attrs == expected.attrs
        # NaN where the eager one is: an energy's first samples are empty.
        assert np.array_equal(loaded.data, expected.data, equal_nan=True)
        assert not np.isnan(loaded.data).all()

    def test_windowed_axis_is_cut(self, lazy):
        """A dimension detrend needs whole is cut once it is windowed."""
        array = lazy.window(time=64, samples=True).detrend("time").data
        boxes = chunk_boxes(array, block_size=ROW)
        assert len({box[1] for box in boxes}) > 1
        # Nothing says how far a detrend reaches along distance.
        assert {box[0] for box in boxes} == {slice(0, lazy.shape[0])}

    def test_reads_only_its_windows(self, lazy):
        """A box reads the windows which overlap it, and no more."""
        array = lazy.window(time=64, samples=True).abs().data
        box = (slice(10, 20), slice(1000, 1010))
        # Windows of 64 every 32, the first starting 32 before the array.
        assert read_schedule(array, [box]) == [(slice(10, 20), slice(960, 1056))]

    @pytest.mark.parametrize(
        "window",
        ({}, {"fill": 2.5}, {"analysis": "hann", "overlap": 48}),
        ids=("plain", "fill", "analysis"),
    )
    @pytest.mark.parametrize("name", ("abs", "real"))
    def test_states_the_dtype_of_integers(self, tmp_path_factory, window, name):
        """The dtype stated before anything is read is the one computed."""
        data = np.random.default_rng(2).integers(-9, 9, (300, 2000), np.int16)
        ints = lazy_patch(stored_patch(tmp_path_factory.mktemp("ints"), data))
        # Only integers which reach the blend as integers are warned of: a
        # fill or an analysis window has made floats of them already.
        expected_warning = () if window else (UserWarning,)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = getattr(ints.window(time=64, samples=True, **window), name)()
            eager = ints.load().window(time=64, samples=True, **window)
            expected = getattr(eager, name)()
            # Said once when declared, and not again for each box computed.
            with config_context(lazy_block_size=10 * ROW):
                loaded = out.load()
        blends = [x for x in caught if "Blending windows of" in str(x.message)]
        assert len(blends) == 2 * len(expected_warning)
        assert out.dtype == loaded.dtype == expected.dtype
        assert np.array_equal(loaded.data, expected.data)

    @pytest.mark.parametrize("name, args", (("abs", ()), ("set_units", ("m",))))
    def test_bad_blend_refused_before_reading(self, gone, name, args):
        """A taper whose ramps would cross is refused when it is called."""
        window = gone.window(time=64, overlap=48, samples=True)
        # Whether or not the operation computes anything to blend.
        with pytest.raises(ParameterError, match="ramps would cross"):
            getattr(window, name)(*args)

    def test_declaring_is_quiet(self, gone):
        """The window of zeros run to find the dtype raises no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = gone.window(time=64, samples=True).standardize("time")
        assert out.dtype == np.float64

    def test_metadata_only_adds_no_step(self, lazy):
        """An operation with no kernel leaves the lazy array as it was."""
        out = lazy.window(time=64, samples=True).set_units("m/s")
        assert out.data is lazy.data
        assert out.attrs.data_units == dc.get_quantity("m/s")

    def test_declaring_reads_and_holds_little(self):
        """The dtype is found from a short window, however long the others are."""
        shape = (8, 20_000_000)
        coords = {"distance": np.arange(8.0), "time": np.arange(shape[1])}
        recipe = Recipe.from_source(ArraySource.full(shape, 1, dtype=np.int16))
        long = dc.Patch(data=recipe, coords=coords, dims=("distance", "time"))
        tracemalloc.start()
        try:
            out = long.window(distance=4, samples=True).detrend("distance")
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        # One window whole would be 4 by 20 million samples.
        assert out.dtype == np.float64 and peak < 20e6

    def test_dtype_of_what_cannot_run_short(self, tmp_path_factory):
        """An operation which needs more of a dimension than is tried gets it."""
        data = np.random.default_rng(3).integers(-9, 9, (300, 2000), np.int16)
        ints = lazy_patch(stored_patch(tmp_path_factory.mktemp("short"), data))
        # A fit to 201 samples of time, where the dtype is first tried on 64.
        call = {"polyorder": 2, "time": 201, "samples": True}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            out = ints.window(distance=32, samples=True).savgol_filter(**call)
            eager = ints.load().window(distance=32, samples=True)
            expected = eager.savgol_filter(**call)
        assert out.dtype == expected.dtype

    def test_reduction_still_refused(self, gone):
        """A reduction in a window states no contract yet."""
        with pytest.raises(LazyOperationError):
            gone.window(time=64, samples=True).mean()


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

    @pytest.mark.wasm_skip
    @pytest.mark.parametrize("mode", ("wrap", "grid-wrap"))
    @pytest.mark.parametrize("name", ("gaussian_filter", "median_filter"))
    def test_wrapping_edges(self, lazy, eager, name, mode):
        """An edge filled from the far end of the array has the axis whole."""
        kwargs = {"distance": 3, "samples": True, "mode": mode}
        self._same(getattr(lazy, name)(**kwargs), getattr(eager, name)(**kwargs))

    @pytest.mark.wasm_skip
    @pytest.mark.parametrize("mode", ("wrap", "interp", "mirror", "nearest"))
    def test_savgol_modes(self, lazy, eager, mode):
        """Each savgol edge mode gives the eager result, however it is cut."""
        kwargs = {"distance": 7, "polyorder": 2, "samples": True, "mode": mode}
        self._same(lazy.savgol_filter(**kwargs), eager.savgol_filter(**kwargs))

    def test_savgol_with_a_nan(self, tmp_path_factory):
        """A NaN is never at a cut end of what the default mode fits."""
        data = np.random.default_rng(2).standard_normal((300, 2000))
        data[150, 1000] = np.nan
        holed = lazy_patch(stored_patch(tmp_path_factory.mktemp("nan"), data))
        kwargs = {"distance": 7, "time": 7, "polyorder": 2, "samples": True}
        expected = holed.load().savgol_filter(**kwargs)
        with config_context(lazy_block_size=ROW // 8):
            loaded = holed.savgol_filter(**kwargs).abs().load()
        np.testing.assert_allclose(
            loaded.data, np.abs(expected.data), rtol=1e-12, atol=1e-12
        )

    @pytest.mark.wasm_skip
    def test_truncate(self, lazy, eager):
        """A wider gaussian kernel reads further."""
        kwargs = {"distance": 3, "samples": True, "truncate": 8.0}
        self._same(lazy.gaussian_filter(**kwargs), eager.gaussian_filter(**kwargs))

    @pytest.mark.parametrize("mode", (("wrap", "reflect"), ["wrap", "reflect"]))
    def test_mode_for_each_axis_refused(self, gone, mode):
        """A mode given axis by axis is not worked out, so it is refused."""
        with pytest.raises(LazyOperationError, match="gaussian_filter"):
            gone.gaussian_filter(time=2, distance=2, samples=True, mode=mode)


class Demean(dc.PatchProcessor):
    """Subtract the mean of everything, which no part of the data can do."""

    name = None

    def kernel(self, data):
        """Return the data less their mean."""
        return data - data.mean()


class Doubled(dc.PatchProcessor):
    """A kernel which wrongly repeats every row."""

    name = None

    def kernel(self, data):
        """Return each row twice."""
        return np.repeat(data, 2, axis=0)


class TestSteps:
    """What a step is given, and what is made of what it returns."""

    @staticmethod
    def _array(*nodes, shape=(4, 3)):
        """A lazy array of the rows 0 to 3 with the given steps."""
        rows = [constant((1, shape[1]), float(x)) for x in range(shape[0])]
        return LazyArray(concat(rows), Chain(nodes))

    def test_no_contract_needs_everything(self):
        """A later selection does not narrow a step which stated no contract."""
        whole = Node(Demean(), {}).settled((4, 3), np.float64, "a")
        index = (slice(0, 2), slice(None))
        window = Node(Select(), {}, Window(index)).settled((2, 3), np.float64, "b")
        array = self._array(whole, window)
        box = (slice(0, 2), slice(0, 3))
        assert read_schedule(array, [box]) == [(slice(0, 4), slice(0, 3))]
        assert np.array_equal(compute(array, box)[:, 0], [-1.5, -0.5])

    @pytest.mark.parametrize("contract", (Pointwise(), None))
    def test_oversized_result_refused(self, contract):
        """A kernel which returns too much is refused, not trimmed."""
        node = Node(Doubled(), {}, contract).settled((4, 3), np.float64, "a")
        box = (slice(0, 4), slice(0, 3))
        with pytest.raises(PatchError, match="Doubled computed an array of shape"):
            compute(self._array(node), box)

    def test_empty_box_runs_nothing(self, lazy, eager):
        """An empty selection after a filter is empty, as eagerly."""
        kwargs = {"polyorder": 2, "time": 7, "samples": True}
        out = lazy.savgol_filter(**kwargs).select(distance=(2, 2), samples=True)
        expected = eager.savgol_filter(**kwargs).select(distance=(2, 2), samples=True)
        assert out.load().data.shape == expected.data.shape == (0, 2000)

    def test_load_raises_the_kernels_error(self, lazy):
        """An error from a step is the one `Patch.load` raises."""
        out = lazy.select(time=(0, 5), samples=True).pass_filter(time=(10, 100))
        with pytest.raises(ValueError, match="padlen"):
            out.load()


class TestQueries:
    """Only a query which is one range for each dimension is a window."""

    def test_stepped_slice_is_no_window(self):
        """A slice with a step would not be a box."""
        plan = {"indexer": (slice(0, 10, 2), slice(None))}
        assert Select().chunk_contract(np.dtype("f8"), plan) is None

    def test_unselect_refused(self, gone):
        """What unselect keeps is spelled as positions, so it is refused."""
        with pytest.raises(LazyOperationError, match="unselect"):
            gone.unselect(time=(0, 100), samples=True)


class _Compiling(ArrayAPIBackend):
    """
    Arrays of array_api_strict, whose chains it compiles once for each key.

    What it compiles is a function of the data and the steps' arrays which
    runs the steps it was compiled from, as a compiler's would: run for
    another box or patch, it is handed only what `Step.arrays` holds.
    """

    name = "compiling_strict"
    array_type = "array_api_strict._array_object.Array"

    # What it compiled, by key, and the steps of each run it was handed.
    compiled: ClassVar[dict] = {}
    sections: ClassVar[list] = []

    def run_chain(self, steps, data):
        """Run the steps as compiled for their keys, compiling if need be."""
        type(self).sections.append(steps)
        keys = tuple(x.key for x in steps)
        if None in keys:
            return super().run_chain(steps, data)
        key = (tuple(data.shape), str(data.dtype), keys)
        if key not in self.compiled:
            self.compiled[key] = self._compile(steps)
        arrays = [[self._held(y) for y in x.arrays] for x in steps]
        return self.compiled[key](data, arrays)

    def _held(self, value):
        """Return an array the chain is handed as one of the backend's."""
        return self.from_array(value) if hasattr(value, "shape") else value

    def _compile(self, steps):
        """Return the steps as one function of the data and their arrays."""

        def run(data, arrays):
            for step, given in zip(steps, arrays, strict=True):
                data = self.from_array(step.run(data, given))
            return data

        return run


def _spacing(meta) -> float:
    """Return the distance between channels."""
    return float(meta.get_coord("distance").step)


def _each_sample(processor, dtype, plan):
    """Return that each sample is made from its own."""
    return Pointwise()


class _Gains(NamedTuple):
    """The gains of a plan, held as a named tuple."""

    gain: Any


class _Gained(dc.PatchProcessor):
    """Multiply by the channel spacing, which the plan holds nested, as an array."""

    name = None
    position_free = True
    chunk_contract = _each_sample

    # A float the compiled chain is handed, as a tuning knob would be.
    plan_arrays: ClassVar[tuple[str, ...]] = ("offset",)

    keep: Any = None

    def get_metadata(self, meta):
        """Return the gains, the offset, and a fixed entry."""
        gain = np.asarray([_spacing(meta)])
        nested = {"gains": _Gains(gain)}
        plan = {"unit": np.ones(1), "offset": 0.0, "nested": nested, "n": 1}
        return meta, plan

    def kernel(self, data, *, unit, offset, nested, n):
        """Return the data times the gains, plus the offset."""
        xp = array_namespace(data)
        gain = xp.asarray(nested["gains"].gain)
        return data * xp.asarray(unit) * gain * n + offset


class _Weighted(dc.PatchProcessor):
    """Multiply by weights, as many as the channel spacing."""

    name = None
    position_free = True
    chunk_contract = _each_sample

    def get_metadata(self, meta):
        """Return a weight of one for each unit of spacing."""
        return meta, {"weights": np.ones(int(_spacing(meta)))}

    def kernel(self, data, *, weights):
        """Return the data times the weights."""
        return data * array_namespace(data).asarray(weights)


class _Paired(dc.PatchProcessor):
    """Multiply by weights a container holds, handed over whole as arrays."""

    name = None
    position_free = True
    chunk_contract = _each_sample
    plan_arrays: ClassVar[tuple[str, ...]] = ("pair",)

    def get_metadata(self, meta):
        """Return a weight of one for each unit of spacing, in a container."""
        return meta, {"pair": {"weights": np.ones(int(_spacing(meta)))}}

    def kernel(self, data, *, pair):
        """Return the data times the weights."""
        return data * array_namespace(data).asarray(pair["weights"])


class _Shifted(dc.PatchProcessor):
    """Add the channel spacing, which the plan holds fixed."""

    name = None
    position_free = True
    chunk_contract = _each_sample

    def get_metadata(self, meta):
        """Return the number to add."""
        return meta, {"by": int(_spacing(meta))}

    def kernel(self, data, *, by):
        """Return the data plus the number."""
        return data + by


class _Signed(dc.PatchProcessor):
    """Replace the data by the sign of a zero the plan holds fixed."""

    name = None
    position_free = True
    chunk_contract = _each_sample

    def get_metadata(self, meta):
        """Return a zero, negative for channels closer than 1.5."""
        return meta, {"zero": math.copysign(0.0, _spacing(meta) - 1.5)}

    def kernel(self, data, *, zero):
        """Return the zero's sign everywhere."""
        return data * 0 + math.copysign(1.0, zero)


class _Flagged(dc.PatchProcessor):
    """Double the data for a flag of True, and triple them for a 1."""

    name = None
    position_free = True
    chunk_contract = _each_sample

    def get_metadata(self, meta):
        """Return True for channels one apart, and 1 otherwise."""
        return meta, {"flag": True if _spacing(meta) == 1 else 1}

    def kernel(self, data, *, flag):
        """Return the data times two or three."""
        return data * (2 if flag is True else 3)


class _Defaulted(dc.PatchProcessor):
    """Add what a dict gives for a key it lacks, which its items do not say."""

    name = None
    position_free = True
    chunk_contract = _each_sample

    def get_metadata(self, meta):
        """Return a table whose missing entries are the channel spacing."""
        spacing = _spacing(meta)
        return meta, {"table": defaultdict(lambda: spacing)}

    def kernel(self, data, *, table):
        """Return the data plus the table's missing entry."""
        return data + table["missing"]


class _Stated(dc.PatchProcessor):
    """Hand the data back, with one of two values in the plan by spacing."""

    name = None
    position_free = True

    pair: Any = None
    contract: Any = None

    def get_metadata(self, meta):
        """Return the first value for channels one apart, else the second."""
        return meta, {"value": self.pair[0] if _spacing(meta) == 1 else self.pair[1]}

    def chunk_contract(self, dtype, plan):
        """Return the contract given, or that each sample is its own."""
        return self.contract or Pointwise()

    def kernel(self, data, *, value):
        """Return the data as they are."""
        return data + 0


class _Biased(Pointwise):
    """A pointwise contract holding a value outside its fields."""

    def __init__(self, bias):
        object.__setattr__(self, "bias", bias)


class _Slotted(Pointwise):
    """A pointwise contract holding a value in a slot."""

    __slots__ = ("bias",)

    def __init__(self, bias):
        object.__setattr__(self, "bias", bias)


class _Named(NamedTuple):
    """A named tuple of one value."""

    value: float


class _Tagged(_Named):
    """A named tuple which holds a tag besides its items."""


def _tagged(value, tag):
    """Return a tagged named tuple."""
    out = _Tagged(value)
    out.tag = tag
    return out


class _NumpyOnly(dc.PatchProcessor):
    """An operation with a NumPy kernel alone, which converts on any other."""

    name = None
    position_free = True
    chunk_contract = _each_sample

    def numpy_kernel(self, data):
        """Return the data negated."""
        return -data


class _Spread(dc.PatchProcessor):
    """Hand the data back, read with a sample either side along the first axis."""

    name = None
    position_free = True

    def chunk_contract(self, dtype, plan):
        """Return the sample read either side."""
        return Halo({0: (1, 1)})

    def kernel(self, data):
        """Return the data as they are."""
        return data + 0


class _Whole(dc.PatchProcessor):
    """Hand the data back, needing the first axis whole."""

    name = None
    position_free = True

    def chunk_contract(self, dtype, plan):
        """Return that the first axis is needed whole."""
        return WholeAxis((0,))

    def kernel(self, data):
        """Return the data as they are."""
        return data + 0


class _Placed(ChunkContract):
    """Each sample is its own plus where its box starts in the array."""

    def apply(self, run, data, box, needed):
        """Return the result moved up by the start of the box."""
        return super().apply(run, data, box, needed) + box[0].start


class _PlacedWindow(Window):
    """A window which adds where its box starts in the array."""

    def apply(self, run, data, box, needed):
        """Return the input moved up by the start of the box."""
        return data + box[0].start


class _Moved(dc.PatchProcessor):
    """An operation whose contract writes its own apply, and no key."""

    name = None
    position_free = True

    contract: Any = None

    def chunk_contract(self, dtype, plan):
        """Return the contract which adds where the box is."""
        return self.contract

    def kernel(self, data):
        """Return the data as they are."""
        return data + 0


class _Summed(dc.PatchProcessor):
    """Sum each row, a step with no contract which drops an axis."""

    name = None

    def kernel(self, data):
        """Return the sum of each row."""
        return array_namespace(data).sum(data, axis=1)


def _ramp(spacing: float = 1.0):
    """Return a lazy patch of the row numbers, 40 rows of 6 channels."""
    rows = [constant((1, 6), float(x)) for x in range(40)]
    coords = {"time": np.arange(40), "distance": np.arange(6) * spacing}
    patch = dc.Patch(data=concat(rows), coords=coords, dims=("time", "distance"))
    return patch.set_backend("lazy[compiling_strict]")


def _values(array):
    """Return an array, or a patch's, as NumPy's."""
    return np.asarray(to_numpy(getattr(array, "data", array)))


def _rows(start=0, stop=40):
    """Return the row numbers of a ramp, in its 6 channels."""
    return np.arange(float(start), float(stop))[:, None] * np.ones(6)


@pytest.fixture()
def abs_kernels():
    """Forget, after a test, the kernels it registered for Abs."""
    before = dict(Abs.__dict__.get("_kernels", {}))
    yield
    Abs._kernels = before


class TestRunChain:
    """A backend is handed each run of steps it has kernels for, to compile."""

    @pytest.fixture(autouse=True)
    def _forget(self):
        """Start each test with nothing compiled, and nothing run."""
        pytest.importorskip("array_api_strict")
        _Compiling.compiled.clear()
        _Compiling.sections.clear()
        yield
        _Compiling.compiled.clear()
        _Compiling.sections.clear()

    @staticmethod
    def _names():
        """Return the operations of each run of steps the backend was handed."""
        return [[type(x.processor).__name__ for x in y] for y in _Compiling.sections]

    def test_two_patches_of_one_shape_compile_once(self):
        """The plan's arrays, nested ones too, are handed to what was compiled."""
        window = {"time": (5, 35), "samples": True}
        outs = [_Gained()(_ramp(x)).select(**window).abs().load() for x in (1, 2)]
        assert len(_Compiling.compiled) == 1
        assert np.array_equal(_values(outs[0]), _rows(5, 35))
        assert np.array_equal(_values(outs[1]), _rows(5, 35) * 2)

    @pytest.mark.parametrize("operation", (_Weighted(), _Paired()))
    def test_arrays_of_another_shape_compile_again(self, operation):
        """An array of the plan, or of a container it hands over, compiles by shape."""
        outs = [operation(_ramp(x)).load() for x in (1.0, 6.0)]
        assert len(_Compiling.compiled) == 2
        assert all(np.array_equal(_values(x), _rows()) for x in outs)

    @pytest.mark.parametrize(
        "operation, expected",
        (
            (_Shifted(), (_rows(1, 41), _rows(2, 42))),
            (_Signed(), (-1.0, 1.0)),
            (_Flagged(), (_rows() * 2, _rows() * 3)),
        ),
    )
    def test_a_fixed_entry_compiles_again(self, operation, expected):
        """What the plan holds fixed was compiled, 0.0 and -0.0, True and 1 apart."""
        outs = [_values(operation(_ramp(x)).load()) for x in (1.0, 2.0)]
        assert len(_Compiling.compiled) == 2
        assert all(np.array_equal(x, y + 0 * x) for x, y in zip(outs, expected))

    @pytest.mark.parametrize(
        "pair",
        (
            (complex(-1, 0.0), complex(-1, -0.0)),
            (np.float32(0.0), np.float32(-0.0)),
            pytest.param(
                (np.longdouble(1), np.longdouble(1) + np.longdouble(2) ** -63),
                marks=pytest.mark.skipif(
                    np.finfo(np.longdouble).nmant <= np.finfo(float).nmant,
                    reason="no wider float here",
                ),
            ),
        ),
    )
    def test_values_a_kernel_tells_apart_key_apart(self, pair):
        """Signed zeros of complex numbers, and wide floats, are keyed whole."""
        for spacing in (1.0, 2.0):
            _Stated(pair=pair)(_ramp(spacing)).load()
        first, second = (x[0].key for x in _Compiling.sections)
        assert None not in (first, second) and first != second

    @pytest.mark.parametrize(
        "operation",
        (
            _Stated(pair=(_tagged(1.0, "a"), _tagged(1.0, "b"))),
            _Stated(pair=(1.0, 1.0), contract=_Biased(2.0)),
            _Stated(pair=(1.0, 1.0), contract=_Slotted(2.0)),
        ),
    )
    def test_what_holds_more_than_it_says_has_no_key(self, operation):
        """A named tuple or a contract holding more than its items has no key."""
        operation(_ramp()).load()
        assert _Compiling.sections[0][0].key is None

    def test_a_container_of_its_own_kind_has_no_key(self):
        """A dict which holds more than its items is not keyed by them."""
        outs = [_values(_Defaulted()(_ramp(x)).load()) for x in (1.0, 2.0)]
        assert all(x[0].key is None for x in _Compiling.sections)
        assert np.array_equal(outs[1] - outs[0], np.ones((40, 6)))

    def test_boxes_of_one_shape_compile_once(self):
        """Equal boxes along a patch share a compile; another shape does not."""
        array = _Gained()(_ramp()).data
        boxes = [(slice(a, b), slice(0, 6)) for a, b in ((0, 10), (10, 20), (20, 35))]
        outs = [_values(compute(array, x)) for x in boxes]
        assert len(_Compiling.compiled) == 2
        assert np.array_equal(outs[1], _rows(10, 20))

    @pytest.mark.parametrize(
        "operation, boxes",
        (
            # Both read 11 rows: the first one after it, the last one before.
            (_Spread(), ((0, 10), (30, 40))),
            # Both read all 40 rows, from which they cut 10 and 15.
            (_Whole(), ((0, 10), (0, 15))),
        ),
    )
    def test_boxes_cut_otherwise_compile_apart(self, operation, boxes):
        """Boxes a contract cuts differently share no compile."""
        array = operation(_ramp()).data
        for start, stop in boxes:
            box = (slice(start, stop), slice(0, 6))
            assert np.array_equal(_values(compute(array, box)), _rows(start, stop))
        assert len(_Compiling.compiled) == 2

    @pytest.mark.parametrize(
        "contract", (_Placed(), _PlacedWindow((slice(None), slice(None))))
    )
    def test_a_contract_with_its_own_apply_is_keyed_by_its_boxes(self, contract):
        """An apply which reads where the box is shares no compile between boxes."""
        array = _Moved(contract=contract)(_ramp()).data
        for start in (0, 10):
            box = (slice(start, start + 10), slice(0, 6))
            out = _values(compute(array, box))
            assert np.array_equal(out, _rows(start, start + 10) + start)
        assert len(_Compiling.compiled) == 2

    def test_tiles_compile_for_each_box(self):
        """A tiled step's kernel is told where its tiles are, so no box shares."""
        windowed = _ramp().window(time=8, overlap=4, samples=True).taper(time=0.25)
        expected = windowed.set_backend("lazy").load().data
        # Two windows apart: the same tiles, counted from each box.
        for start in (10, 18):
            box = (slice(start, start + 10), slice(0, 6))
            out = _values(compute(windowed.data, box))
            assert np.allclose(out, expected[box], equal_nan=True)
        assert len(_Compiling.compiled) == 2

    def test_a_step_dropping_an_axis_has_a_key(self):
        """A step with no contract which drops an axis is keyed as a whole."""
        node = Node(_Summed(), {}).settled((40,), np.float64, "a")
        array = LazyArray(_ramp().data.recipe, Chain((node,)), backend=_Compiling())
        assert np.array_equal(
            _values(compute(array, (slice(0, 40),))), _rows()[:, 0] * 6
        )
        assert _Compiling.sections[0][0].key is not None

    def test_a_kernel_registered_later_is_run(self, abs_kernels):
        """What was compiled before a kernel was registered is not reused."""
        windowed = _ramp().window(time=8, overlap=4, samples=True).abs()
        before = _values(windowed.load())

        @register_kernel(Abs, "compiling_strict")
        def _abs(processor, data):
            """Return the absolute values, and 10 more."""
            return array_namespace(data).abs(data) + 10

        after = _values(windowed.load())
        assert np.allclose(after, before + 10, equal_nan=True)
        # Compiled again: a compiler would have kept the kernel it traced.
        assert len(_Compiling.compiled) == 2

    def test_split_at_a_step_which_converts(self):
        """A step with no kernel for the library is run between sections."""
        with config_context(numpy_fallback="ignore"):
            out = _Shifted()(_NumpyOnly()(_Shifted()(_ramp()))).load()
        assert self._names() == [["_Shifted"], ["_Shifted"]]
        assert np.array_equal(_values(out), -_rows())

    def test_split_at_a_window_whose_operation_converts(self):
        """A window converts for the operation it runs, so it splits too."""
        with config_context(numpy_fallback="ignore"):
            window = _ramp().window(time=8, overlap=4, samples=True)
            _Shifted()(window.detrend("time")).abs().load()
        assert self._names() == [["_Shifted", "Abs"]]

    def test_arrays_wherever_the_plan_holds_them(self):
        """Arrays, nested ones and those the processor names are handed over."""
        _Gained()(_ramp(3.0)).load()
        (step,) = _Compiling.sections[0]
        # In the order the plan holds them.
        unit, offset, gain = step.arrays
        assert np.array_equal(gain, [3.0]) and offset == 0.0 and unit[0] == 1
        assert step.static["n"] == 1 and "nested" in step.static
        backend = _Compiling()
        data = backend.from_array(np.ones((40, 6)))
        arrays = (unit, 1.0, backend.from_array(np.asarray([5.0])))
        assert np.array_equal(_values(step.run(data, arrays)), np.full((40, 6), 6.0))

    # A processor holding what has no id says so, and its key is None.
    @pytest.mark.filterwarnings("ignore:No id could be derived")
    @pytest.mark.parametrize(
        "keep, same",
        (
            (np.arange(3), True),
            (np.arange(3) + 1, False),
            ({1, 2}, False),
            (object(), None),
            (np.array([object()]), None),
        ),
    )
    def test_a_processors_array_is_told_apart(self, keep, same):
        """An array a processor holds keys by its values; what has no id, none."""
        for processor in (_Gained(keep=np.arange(3)), _Gained(keep=keep)):
            processor(_ramp()).load()
        first, second = (x[0].key for x in _Compiling.sections)
        assert first is not None
        if same is None:
            assert second is None
        else:
            assert second is not None and (second == first) is same
