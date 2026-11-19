"""Tests for impulse windows: a linear operation cut by its measured reach."""

from __future__ import annotations

import sys
from typing import ClassVar

import numpy as np
import pytest
from pydantic import Field

import dascore as dc
from dascore.config import config_context
from dascore.exceptions import ParameterError, PatchDataError
from dascore.lazy.contract import Halo
from dascore.lazy.execute import chunk_boxes
from dascore.proc import impulse as impulse_module
from dascore.proc.impulse import Impulsed, PatchImpulseWindow
from dascore.utils.array_api import to_numpy
from dascore.warnings import NumpyFallbackWarning


@pytest.fixture(scope="module")
def patch():
    """One channel, long in time, so lazy boxes cut along time."""
    return dc.get_example_patch("random_das", shape=(1, 40000))


@pytest.fixture(scope="module")
def wide():
    """Many channels, so a filter along distance settles inside the axis."""
    return dc.get_example_patch("random_das", shape=(400, 4000))


def _on(patch, backend, tmp_path_factory, name):
    """The patch, lazy or on dask in chunks along both axes."""
    if backend == "lazy":
        return _stored(patch, tmp_path_factory, name)
    dask_array = pytest.importorskip("dask.array")
    return patch.new(data=dask_array.from_array(patch.data, chunks=(100, 1000)))


def _computed(patch) -> np.ndarray:
    """A lazy patch loaded in boxes, or a dask one computed."""
    if patch.backend == "lazy":
        return _loaded_in_boxes(patch).data
    return patch.data.compute()


@pytest.fixture(scope="module")
def two_rows():
    """Two channels, to filter along distance as well."""
    return dc.get_example_patch("random_das", shape=(40, 4000))


def _stored(patch, tmp_path_factory, name):
    """The patch written to a file and read back lazily."""
    path = tmp_path_factory.mktemp(name) / f"{name}.h5"
    dc.write(patch, path, "dasdae")
    return dc.read(path)[0].set_backend("lazy")


@pytest.fixture(scope="module")
def lazy(patch, tmp_path_factory):
    """The long patch, lazy."""
    return _stored(patch, tmp_path_factory, "impulse")


def _reach(operation, patch, **tolerances):
    """The reach measured along each windowed axis, as (axis, before, after)."""
    _, plan = Impulsed(operation, **tolerances).get_metadata(patch.drop_data())
    return plan["halo"]


def _loaded_in_boxes(lazy_patch, boxes=16):
    """A lazy patch loaded with a block size which cuts it into about `boxes`."""
    size = lazy_patch.data.dtype.itemsize * np.prod(lazy_patch.shape) // boxes
    with config_context(lazy_block_size=int(size)):
        assert len(chunk_boxes(lazy_patch.data)) > 1
        return lazy_patch.load()


def _relative_error(found, expected) -> float:
    """The largest difference, as a fraction of the largest value."""
    return np.abs(found - expected).max() / np.abs(expected).max()


class TestInMemory:
    """Data in memory are computed whole, exactly as the operation alone."""

    def test_exact(self, patch):
        """The values are those of the operation alone, to the bit."""
        out = patch.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        expected = patch.pass_filter(time=(1, 20))
        assert np.array_equal(out.data, expected.data)
        assert out.coords == expected.coords and out.dtype == expected.dtype

    def test_history_and_id(self, patch):
        """The window is recorded, and enters the id."""
        out = patch.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        again = patch.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        plain = patch.pass_filter(time=(1, 20))
        history = out.attrs.history[-1]
        assert history.startswith("impulse_window(time='1e-06').pass_filter(")
        assert out.attrs.data_id == again.attrs.data_id != plain.attrs.data_id
        other = patch.impulse_window(time=1e-3).pass_filter(time=(1, 20))
        assert other.attrs.data_id != out.attrs.data_id

    def test_another_library(self, patch):
        """On another library the operation's kernel for it runs, whole."""
        strict = pytest.importorskip("array_api_strict")
        there = patch.set_backend("array_api_strict")
        with pytest.warns(NumpyFallbackWarning, match="pass_filter has no"):
            out = there.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        assert type(out.data) is type(strict.asarray(np.zeros(1)))
        expected = patch.pass_filter(time=(1, 20))
        np.testing.assert_allclose(to_numpy(out.data), expected.data)


class TestReach:
    """The reach is what the operation's response to an impulse shows."""

    def test_zero_phase_is_symmetric(self, patch):
        """A filter run forwards and back reaches as far each way."""
        ((axis, before, after),) = _reach(
            dc.proc.PassFilter(time=(1, 20)), patch, time=1e-6
        )
        assert axis == patch.get_axis("time") and before == after > 0

    def test_causal_reaches_back(self, patch):
        """A causal filter's output reads only the input before it."""
        operation = dc.proc.PassFilter(time=(1, 20), zerophase=False)
        ((axis, before, after),) = _reach(operation, patch, time=1e-6)
        # What follows a sample is only the least a chunk needs for the
        # filter to run at all, which is nothing beside what precedes it.
        least = impulse_module._fits(
            operation,
            operation.get_metadata(patch.drop_data())[1],
            patch.drop_data(),
            axis,
            (axis,),
        )
        assert after == least and before > 50 * after

    def test_smaller_tolerance_reaches_further(self, patch):
        """The less may lie beyond, the further the reach."""
        operation = dc.proc.PassFilter(time=(1, 20))
        reaches = [_reach(operation, patch, time=x)[0][1] for x in (1e-3, 1e-6, 1e-9)]
        assert reaches == sorted(reaches) and len(set(reaches)) == 3

    def test_tail_reach(self):
        """The reach is the fewest samples beyond which the rest is small."""
        response = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 0.25])
        # Beyond 2 samples: 0.5 + 0.25 + 0.25 = 1, an eighth of 8.
        assert impulse_module._tail_reach(response, 1 / 8 + 1e-9) == 2
        assert impulse_module._tail_reach(response, 1 / 8 - 1e-9) == 3
        assert impulse_module._tail_reach(np.zeros(4), 0.1) == 0

    def test_oldest_measurement_is_let_go(self, patch, monkeypatch):
        """Only so many measurements are kept; the oldest goes first."""
        monkeypatch.setattr(impulse_module, "_MEASURED", {})
        monkeypatch.setattr(impulse_module, "_MEASURED_LIMIT", 2)
        for low in (1, 2, 3):
            _reach(dc.proc.PassFilter(time=(low, 20)), patch, time=1e-5)
        assert len(impulse_module._MEASURED) == 2
        kept = " ".join(str(x[3]) for x in impulse_module._MEASURED)
        # What the third and second calls measured; not the first.
        first = dc.proc.PassFilter(time=(1, 20)).get_metadata(patch.drop_data())[1]
        third = dc.proc.PassFilter(time=(3, 20)).get_metadata(patch.drop_data())[1]
        assert str(impulse_module._plan_key(third)) in kept
        assert str(impulse_module._plan_key(first)) not in kept

    def test_measured_once_for_a_plan(self, patch, monkeypatch):
        """The measurement is kept for the plan, and not made again."""
        calls = []
        original = impulse_module._impulsed

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(impulse_module, "_impulsed", counted)
        operation = dc.proc.PassFilter(time=(2, 30), corners=3)
        _reach(operation, patch, time=1e-5)
        first = len(calls)
        _reach(operation, patch, time=1e-5)
        assert first and len(calls) == first

    def test_kept_by_plan(self, patch):
        """The same call on a finer sampling measures again, further."""
        operation = dc.proc.PassFilter(time=(1, 20))
        coarse = _reach(operation, patch, time=1e-6)[0][1]
        step = patch.get_coord("time").step / 4
        finer = patch.update_coords(time_step=step)
        assert _reach(operation, finer, time=1e-6)[0][1] > 3 * coarse
        assert _reach(operation, patch, time=1e-6)[0][1] == coarse

    def test_along_two_dimensions(self, two_rows):
        """A filter along two dimensions has a reach along each."""
        operation = dc.proc.NotchFilter(time=60, distance=0.2, q=30)
        halo = _reach(operation, two_rows, time=1e-6, distance=1e-6)
        assert {x[0] for x in halo} == {0, 1}
        assert all(before > 0 and after > 0 for _, before, after in halo)


class TestLazy:
    """A lazy patch is computed a chunk at a time, within the tolerance."""

    @pytest.mark.parametrize("zerophase", (True, False))
    @pytest.mark.parametrize("tolerance", (1e-3, 1e-6, 1e-9))
    def test_within_tolerance(self, patch, lazy, zerophase, tolerance):
        """Cut along time, the result is the exact one to about the tolerance."""
        expected = patch.pass_filter(time=(1, 20), zerophase=zerophase)
        out = lazy.impulse_window(time=tolerance).pass_filter(
            time=(1, 20), zerophase=zerophase
        )
        assert out.backend == "lazy"
        loaded = _loaded_in_boxes(out)
        assert loaded.attrs == out.attrs and loaded.dtype == expected.dtype
        assert _relative_error(loaded.data, expected.data) < 4 * tolerance

    def test_notch(self, patch, lazy):
        """A notch filter, which cannot run lazily alone, runs in pieces."""
        expected = patch.notch_filter(time=60, q=30)
        out = lazy.impulse_window(time=1e-6).notch_filter(time=60, q=30)
        assert _relative_error(_loaded_in_boxes(out).data, expected.data) < 4e-6

    def test_same_id_as_in_memory(self, patch, lazy):
        """Lazy or in memory, the call is the same and so is its id."""
        out = lazy.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        eager = patch.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        assert out.attrs.data_id == eager.attrs.data_id

    def test_contract(self, patch):
        """A chunk is read with the reach around it along each axis."""
        windowed = Impulsed(dc.proc.PassFilter(time=(1, 20)), time=1e-6)
        _, plan = windowed.get_metadata(patch.drop_data())
        contract = windowed.chunk_contract(patch.dtype, plan)
        ((axis, before, after),) = plan["halo"]
        assert contract == Halo({axis: (before, after)})

    def test_integers_state_the_dtype_made(self, tmp_path_factory):
        """The dtype stated ahead is the one the filter makes of integers."""
        ints = dc.get_example_patch("random_das", shape=(1, 8000))
        ints = ints.new(data=(ints.data * 100).astype(np.int16))
        lazy = _stored(ints, tmp_path_factory, "ints")
        out = lazy.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        expected = ints.pass_filter(time=(1, 20))
        assert out.dtype == expected.dtype == np.dtype(np.float64)
        loaded = _loaded_in_boxes(out, boxes=4)
        assert _relative_error(loaded.data, expected.data) < 4e-6


class TestXarray:
    """On a DataArray an impulse window gives back a DataArray."""

    def test_comes_back_a_data_array(self, patch):
        """The window is forwarded through the accessor, as a window is."""
        xr = pytest.importorskip("xarray")
        pytest.importorskip("dascore.xarray")
        array = patch.io.to_xarray()
        out = array.dc.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        assert isinstance(out, xr.DataArray)
        np.testing.assert_array_equal(out.values, patch.pass_filter(time=(1, 20)).data)


class TestDask:
    """A dask array is computed a chunk at a time, read with the reach."""

    @pytest.mark.parametrize("zerophase", (True, False))
    def test_causal_and_zero_phase(self, patch, zerophase):
        """A causal filter reads only before each chunk, on dask too."""
        dask_array = pytest.importorskip("dask.array")
        there = patch.new(data=dask_array.from_array(patch.data, chunks=(1, 5000)))
        out = there.impulse_window(time=1e-6).pass_filter(
            time=(1, 20), zerophase=zerophase
        )
        expected = patch.pass_filter(time=(1, 20), zerophase=zerophase)
        assert _relative_error(out.data.compute(), expected.data) < 4e-6

    @pytest.mark.parametrize("chunks", ((1, 500), (1, 7000), (1, 40000)))
    def test_within_tolerance(self, patch, chunks):
        """Whatever the chunks, nothing is computed at the call, and it is close."""
        dask_array = pytest.importorskip("dask.array")
        callbacks = pytest.importorskip("dask.callbacks")
        there = patch.new(data=dask_array.from_array(patch.data, chunks=chunks))
        computed = []
        with callbacks.Callback(posttask=lambda *x: computed.append(1)):
            out = there.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        assert out.backend == "dask" and not computed
        # In the chunks the data came in, though they are shorter than the reach.
        assert out.data.chunks == there.data.chunks
        expected = patch.pass_filter(time=(1, 20))
        assert _relative_error(out.data.compute(), expected.data) < 4e-6


class TestReviewedLimits:
    """What a probe and a chunk need where a reach is long or a filter large."""

    @pytest.mark.parametrize("backend", ("lazy", "dask"))
    def test_one_axis_whole_beside_one_cut(
        self, wide, tmp_path_factory, monkeypatch, backend
    ):
        """An axis needed whole keeps the reach of the others."""
        original = impulse_module._measured

        def unsettled_in_distance(operation, plan, meta, axis, *args):
            if axis == 0:
                return None
            return original(operation, plan, meta, axis, *args)

        monkeypatch.setattr(impulse_module, "_measured", unsettled_in_distance)
        monkeypatch.setattr(impulse_module, "_MEASURED", {})
        there = _on(wide, backend, tmp_path_factory, "whole_beside")
        window = there.impulse_window(on_unsettled="whole", time=1e-6, distance=1e-6)
        out = window.notch_filter(time=60, distance=0.2, q=3)
        expected = wide.notch_filter(time=60, distance=0.2, q=3)
        assert _relative_error(_computed(out), expected.data) < 4e-6

    @pytest.mark.parametrize("backend", ("lazy", "dask"))
    def test_two_axes_cut(self, wide, tmp_path_factory, backend):
        """Cut along both axes a filter reaches along, each read with its reach."""
        there = _on(wide, backend, tmp_path_factory, "two_cut")
        window = there.impulse_window(time=1e-6, distance=1e-6)
        out = window.notch_filter(time=60, distance=0.2, q=3)
        halo = _reach(
            dc.proc.NotchFilter(time=60, distance=0.2, q=3),
            wide,
            time=1e-6,
            distance=1e-6,
        )
        # Both reaches shorter than their axes, so both axes are cut.
        assert all(max(b, a) < wide.shape[x] for x, b, a in halo)
        expected = wide.notch_filter(time=60, distance=0.2, q=3)
        assert _relative_error(_computed(out), expected.data) < 4e-6

    def test_dask_reach_longer_than_the_axis(self):
        """A reach past the axis reads it whole, rather than failing."""
        dask_array = pytest.importorskip("dask.array")
        short = dc.get_example_patch("random_das", shape=(2, 500))
        there = short.new(data=dask_array.from_array(short.data, chunks=(1, 100)))
        out = there.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        assert out.data.chunks == there.data.chunks
        expected = short.pass_filter(time=(1, 20))
        np.testing.assert_allclose(out.data.compute(), expected.data, atol=1e-12)

    def test_dask_states_the_dtype_made(self):
        """Integers on dask are said to become what the filter makes of them."""
        dask_array = pytest.importorskip("dask.array")
        ints = dc.get_example_patch("random_das", shape=(1, 8000))
        ints = ints.new(data=(ints.data * 100).astype(np.int16))
        there = ints.new(data=dask_array.from_array(ints.data, chunks=(1, 2000)))
        out = there.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        assert out.dtype == out.data.compute().dtype == np.dtype(np.float64)

    def test_high_order_filter(self, patch, lazy):
        """A filter which pads its ends further than a short probe still runs."""
        expected = patch.pass_filter(time=(1, 20), corners=22)
        window = lazy.impulse_window(time=1e-6)
        out = window.pass_filter(time=(1, 20), corners=22)
        in_memory = patch.impulse_window(time=1e-6).pass_filter(
            time=(1, 20), corners=22
        )
        assert np.array_equal(in_memory.data, expected.data)
        assert _relative_error(out.load().data, expected.data) < 4e-6

    def test_many_dimensions_within_the_budget(self, monkeypatch):
        """Probes of a patch of many dimensions stay within the budget."""
        sizes = []
        original = impulse_module._impulsed

        def recorded(operation, plan, shape, *args, **kwargs):
            sizes.append(int(np.prod(shape)))
            return original(operation, plan, shape, *args, **kwargs)

        monkeypatch.setattr(impulse_module, "_impulsed", recorded)
        data = np.zeros((40, 40, 40, 400))
        coords = {
            "a": np.arange(40.0),
            "b": np.arange(40.0),
            "c": np.arange(40.0),
            "time": dc.to_datetime64(np.arange(400) / 250),
        }
        many = dc.Patch(data=data, coords=coords, dims=("a", "b", "c", "time"))
        many.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        # One axis at a time, the rest one sample long: never the four.
        assert max(sizes) <= 8 * 400

    def test_shortest_length_longer_than_the_axis(self):
        """A filter which cannot run on the whole axis is not looked for forever."""
        short = dc.get_example_patch("random_das", shape=(1, 60))
        operation = dc.proc.PassFilter(time=(1, 20), corners=22)
        meta = short.drop_data()
        plan = operation.get_metadata(meta)[1]
        with pytest.raises(ValueError, match="padlen"):
            impulse_module._fits(operation, plan, meta, 1, (1,))

    def test_probe_past_the_budget_is_refused(self, patch, monkeypatch):
        """A probe which would pass the budget is refused, not made."""
        monkeypatch.setattr(impulse_module, "_MEASURED", {})
        monkeypatch.setattr(impulse_module, "_PROBE_BUDGET", 1)
        with pytest.raises(ParameterError, match="needs a probe of more than"):
            patch.impulse_window(time=1e-6).pass_filter(time=(1, 20))

    def test_decay_past_the_probe(self):
        """A response decaying at the probe's end is extrapolated past it."""
        beyond = impulse_module._beyond
        decaying = 0.5 ** np.arange(64.0)
        assert beyond(decaying, decaying.sum()) < 1e-12
        flat = np.ones(64)
        assert beyond(flat, flat.sum()) == np.inf
        assert beyond(np.ones(4), 4.0) == np.inf
        # Nothing at all near the end, then something: not decaying.
        rising = np.r_[np.ones(48), np.zeros(8), np.ones(8)]
        assert beyond(rising, rising.sum()) == np.inf

    @pytest.mark.skipif(
        sys.platform == "emscripten", reason="emscripten does not support threads"
    )
    def test_kept_across_threads(self, patch, monkeypatch):
        """Measurements shared by threads are kept and let go safely."""
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        monkeypatch.setattr(impulse_module, "_MEASURED", {})
        monkeypatch.setattr(impulse_module, "_MEASURED_LIMIT", 4)

        def reach(low):
            operation = dc.proc.PassFilter(time=(1 + low % 7, 20))
            return _reach(operation, patch, time=1e-5)

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(reach, range(64)))
        assert len(impulse_module._MEASURED) <= 4

    def test_filter_too_long_for_the_patch(self):
        """A filter which cannot run on the patch at all says why, in the probe."""
        short = dc.get_example_patch("random_das", shape=(1, 60))
        # Its own padding of the ends is longer than the whole axis.
        with pytest.raises(ValueError, match="padlen"):
            short.impulse_window(time=1e-6).pass_filter(time=(1, 20), corners=22)

    def test_response_at_even_lags_only(self, patch, tmp_path_factory):
        """An impulse spreading past its neighbours is found to reach."""
        # At 250 Hz, a notch at 62.5 Hz answers at even lags alone.
        step = dc.to_timedelta64(1 / 250)
        sampled = patch.update_coords(time_step=step)
        operation = dc.proc.NotchFilter(time=62.5, q=30)
        ((_, before, after),) = _reach(operation, sampled, time=1e-6)
        assert before > 10 and after > 10
        lazy = _stored(sampled, tmp_path_factory, "even_lags")
        out = lazy.impulse_window(time=1e-6).notch_filter(time=62.5, q=30)
        expected = sampled.notch_filter(time=62.5, q=30)
        assert _relative_error(_loaded_in_boxes(out).data, expected.data) < 4e-6

    def test_response_too_slow_to_settle(self, patch):
        """A response which does not decay inside the probe reads its axis whole."""
        step = dc.to_timedelta64(1 / 250)
        sampled = patch.update_coords(time_step=step)
        operation = dc.proc.NotchFilter(time=60, q=1e9)
        ((_, before, after),) = _reach(operation, sampled, time=1e-6)
        # Decaying too slowly to tell inside four axis lengths: all of it.
        assert before == after == sampled.shape[1]

    def test_shortest_chunk_the_filter_runs_on(self, patch):
        """Chunks shorter than the filter's padding are read with enough around them."""
        dask_array = pytest.importorskip("dask.array")
        short = patch.select(time=(0, 4000), samples=True)
        there = short.new(data=dask_array.from_array(short.data, chunks=(1, 1)))
        out = there.impulse_window(time=0.1).pass_filter(time=(None, 100))
        expected = short.pass_filter(time=(None, 100))
        assert np.isfinite(out.data.compute()).all()
        assert _relative_error(out.data.compute(), expected.data) < 1

    def test_probe_memory_is_bounded(self, monkeypatch):
        """However long the response, a probe holds no more than the budget."""
        sizes = []
        original = impulse_module._impulsed

        def recorded(operation, plan, shape, *args, **kwargs):
            sizes.append(int(np.prod(shape)))
            return original(operation, plan, shape, *args, **kwargs)

        monkeypatch.setattr(impulse_module, "_impulsed", recorded)
        monkeypatch.setattr(impulse_module, "_MEASURED", {})
        wide = dc.get_example_patch("random_das", shape=(300, 4000))
        # A sharp notch along both axes, whose response is long.
        window = wide.impulse_window(on_unsettled="whole", time=1e-6, distance=1e-6)
        window.notch_filter(time=60, distance=0.2, q=3000)
        assert sizes and max(sizes) <= impulse_module._PROBE_BUDGET

    def test_reach_past_the_axis_is_the_axis(self, two_rows):
        """The probe stops growing once the reach is past the axis."""
        operation = dc.proc.NotchFilter(distance=0.2, q=300)
        halo = _reach(operation, two_rows, distance=1e-6)
        ((_, before, after),) = [x for x in halo if x[0] == 0]
        assert before <= two_rows.shape[0] and after <= two_rows.shape[0]


class TestUnsettled:
    """A response which does not fall under its tolerance in the longest probe."""

    def test_raises_by_default(self, patch, monkeypatch):
        """Refused, naming the tolerance and what to do."""
        monkeypatch.setattr(impulse_module, "_PROBE_BUDGET", 256)
        monkeypatch.setattr(impulse_module, "_FIRST_PROBE", 128)
        with pytest.raises(ParameterError, match="does not fall under"):
            patch.impulse_window(time=1e-9).pass_filter(time=(0.5, 2))

    def test_whole_axis(self, patch, monkeypatch):
        """Or the dimension is needed whole, as the operation alone needs it."""
        monkeypatch.setattr(impulse_module, "_PROBE_BUDGET", 256)
        monkeypatch.setattr(impulse_module, "_FIRST_PROBE", 128)
        operation = dc.proc.PassFilter(time=(0.5, 2))
        windowed = Impulsed(operation, on_unsettled="whole", time=1e-9)
        _, plan = windowed.get_metadata(patch.drop_data())
        # A reach as long as the axis, which a halo cuts at its ends.
        whole = patch.shape[1]
        assert windowed.chunk_contract(patch.dtype, plan) == Halo({1: (whole, whole)})
        out = patch.impulse_window(on_unsettled="whole", time=1e-9).pass_filter(
            time=(0.5, 2)
        )
        assert np.array_equal(out.data, patch.pass_filter(time=(0.5, 2)).data)

    @pytest.mark.parametrize("on_unsettled", ("raise", "whole"))
    def test_budget_passed_while_padding(self, patch, monkeypatch, on_unsettled):
        """A probe lengthened past the budget for padding is unsettled, too."""
        original = impulse_module._response

        def padded_past(operation, plan, shape, axis, *args):
            if shape[axis] >= impulse_module._FIRST_PROBE:
                raise impulse_module._OverBudgetError("lengthened past the budget")
            return original(operation, plan, shape, axis, *args)

        monkeypatch.setattr(impulse_module, "_response", padded_past)
        monkeypatch.setattr(impulse_module, "_MEASURED", {})
        windowed = Impulsed(
            dc.proc.PassFilter(time=(1, 20)), on_unsettled=on_unsettled, time=1e-6
        )
        if on_unsettled == "raise":
            with pytest.raises(ParameterError, match="does not fall under"):
                windowed.get_metadata(patch.drop_data())
            return
        _, plan = windowed.get_metadata(patch.drop_data())
        whole = patch.shape[1]
        assert windowed.chunk_contract(patch.dtype, plan) == Halo({1: (whole, whole)})

    def test_probe_never_past_four_axes(self, monkeypatch):
        """A short axis is never probed longer than four times itself."""
        lengths = []
        original = impulse_module._response

        def recorded(operation, plan, shape, axis, *args):
            lengths.append(shape[axis])
            return original(operation, plan, shape, axis, *args)

        monkeypatch.setattr(impulse_module, "_response", recorded)
        monkeypatch.setattr(impulse_module, "_MEASURED", {})
        short = dc.get_example_patch("random_das", shape=(1, 100))
        short.impulse_window(time=1e-9).pass_filter(time=(1, 20))
        assert max(lengths) <= 400


class TestNonFinite:
    """A NaN carried along a filtered axis is never cut short."""

    @pytest.fixture(scope="class")
    def holed(self, patch):
        """The long trace with one NaN early on."""
        data = patch.data.copy()
        data[0, 100] = np.nan
        return patch.new(data=data)

    def test_in_memory_as_the_filter(self, holed):
        """In memory the NaN goes where the filter alone takes it."""
        out = holed.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        expected = holed.pass_filter(time=(1, 20))
        np.testing.assert_array_equal(out.data, expected.data)

    def test_lazy_cut_refused(self, holed, tmp_path_factory):
        """A lazy patch cut along the filtered axis refuses it."""
        lazy = _stored(holed, tmp_path_factory, "holed")
        out = lazy.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        with pytest.raises(PatchDataError, match="NaN or infinity"):
            _loaded_in_boxes(out)

    def test_dask_cut_refused(self, holed):
        """A dask array cut along the filtered axis refuses it."""
        dask_array = pytest.importorskip("dask.array")
        there = holed.new(data=dask_array.from_array(holed.data, chunks=(1, 5000)))
        out = there.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        with pytest.raises(PatchDataError, match="NaN or infinity"):
            out.data.compute()

    def test_dask_whole_as_the_filter(self, holed):
        """In one chunk along the filtered axis the NaN goes as the filter takes it."""
        dask_array = pytest.importorskip("dask.array")
        there = holed.new(data=dask_array.from_array(holed.data, chunks=-1))
        out = there.impulse_window(time=1e-6).pass_filter(time=(1, 20))
        expected = holed.pass_filter(time=(1, 20))
        np.testing.assert_array_equal(out.data.compute(), expected.data)


class TestRefused:
    """What an impulse window cannot run is refused when it is called."""

    def test_settles_from_its_data(self, patch):
        """An operation which settles its result from the data is refused."""

        class Settled(dc.proc.PassFilter):
            """A filter which settles its result afterwards."""

            name = None

            def reconcile(self, data, out, meta):
                return out

        with pytest.raises(ParameterError, match="settles its result"):
            Impulsed(Settled(time=(1, 20)), time=1e-6)(patch)

    def test_kernel_replaced_by_a_subclass(self, patch):
        """A subclass with a kernel of its own must say again it is linear."""

        class Squared(dc.proc.PassFilter):
            """A filter whose result is squared."""

            name = None

            def numpy_kernel(self, data, **kwargs):
                return super().numpy_kernel(data, **kwargs) ** 2

        class Kept(dc.proc.PassFilter):
            """A filter which only renames itself."""

            name = None

        with pytest.raises(ParameterError, match="says is linear"):
            Impulsed(Squared(time=(1, 20)), time=1e-6)(patch)
        out = Impulsed(Kept(time=(1, 20)), time=1e-6)(patch)
        assert np.array_equal(out.data, patch.pass_filter(time=(1, 20)).data)

    def test_computes_no_data(self, patch):
        """An operation without a kernel has nothing to measure."""

        class Renamed(dc.PatchProcessor):
            """A linear operation on metadata alone."""

            name = None
            linear = True
            position_free = True

        with pytest.raises(ParameterError, match="computes no data"):
            Impulsed(Renamed(), time=1e-6).get_metadata(patch.drop_data())

    def test_inner_checks_run(self, patch):
        """What the operation requires of a patch is asked of it."""

        class Calibrated(dc.proc.PassFilter):
            """A filter which needs an attribute."""

            name = None
            linear = True
            required_attrs: ClassVar[dict] = {"calibrated_by": "someone"}

        with pytest.raises(Exception, match="calibrated_by"):
            Impulsed(Calibrated(time=(1, 20)), time=1e-6)(patch)

    def test_dimension_named_like_a_parameter(self, patch):
        """A dimension named like one of its parameters is refused clearly."""
        renamed = patch.rename_coords(distance="operation")
        with pytest.raises(ParameterError, match="names one of its own"):
            renamed.impulse_window(operation=1e-6, time=1e-6)

    def test_kept_by_every_field(self, patch, monkeypatch):
        """A field the repr does not show still tells two operations apart."""

        class Gained(dc.proc.PassFilter):
            """A filter with a gain its repr hides."""

            name = None
            linear = True
            gain: float = Field(1.0, repr=False)

        monkeypatch.setattr(impulse_module, "_MEASURED", {})
        one, two = Gained(time=(1, 20), gain=1.0), Gained(time=(1, 20), gain=2.0)
        assert repr(one) == repr(two)
        # One with no faithful id is matched by nothing, not even itself.
        unkeyed = Gained(time=(1, 20), gain=1.0)
        object.__setattr__(unkeyed, "gain", object())
        key = impulse_module._operation_key
        assert key(unkeyed) != key(unkeyed)
        assert impulse_module._operation_key(one) != impulse_module._operation_key(two)

    def test_operation_not_linear(self, patch):
        """An operation which does not say it is linear says nothing by its response."""
        with pytest.raises(ParameterError, match="linear and acts the same"):
            patch.impulse_window(time=1e-6).detrend("time")

    def test_dimension_without_tolerance(self, patch):
        """Every dimension the operation reaches along needs a tolerance."""
        with pytest.raises(ParameterError, match=r"gives no\s+tolerance"):
            patch.impulse_window(distance=1e-6).pass_filter(time=(1, 20))

    def test_dimension_not_reached(self, two_rows):
        """A tolerance along a dimension it does not reach needs nothing."""
        operation = dc.proc.PassFilter(time=(1, 20))
        halo = _reach(operation, two_rows, time=1e-6, distance=1e-6)
        assert two_rows.shape[0] > 1 and (0, 0, 0) in halo

    def test_unknown_dimension(self, patch):
        """A dimension the patch lacks is refused."""
        with pytest.raises(ParameterError, match="lacks"):
            patch.impulse_window(depth=1e-6).pass_filter(time=(1, 20))

    @pytest.mark.parametrize("value", (0, 1, 1.5, -1e-3, True, "small"))
    def test_tolerance_is_a_fraction(self, patch, value):
        """A tolerance is a fraction of the whole response."""
        with pytest.raises(ParameterError, match="between 0 and 1"):
            patch.impulse_window(time=value).pass_filter(time=(1, 20))

    def test_no_tolerance(self, patch):
        """At least one dimension and its tolerance."""
        with pytest.raises(ParameterError, match="needs a dimension"):
            patch.impulse_window().pass_filter(time=(1, 20))

    def test_unsettled_option(self, patch):
        """What to do when it does not settle is one of two."""
        with pytest.raises(ParameterError, match="on_unsettled"):
            Impulsed(dc.proc.PassFilter(time=(1, 20)), on_unsettled="x", time=1e-6)(
                patch
            )

    def test_not_a_processor(self, patch):
        """A function is not an operation."""
        with pytest.raises(ParameterError, match="runs a PatchProcessor"):
            Impulsed(np.abs, time=1e-6)(patch)

    def test_shape_changes(self, patch):
        """An operation which changes the shape is refused."""

        class Halves(dc.PatchProcessor):
            """A linear operation which keeps every second sample."""

            name = None
            linear = True
            position_free = True

            def get_metadata(self, meta):
                coords, _ = meta.coords.isel(time=slice(None, None, 2))
                return meta.new(coords=coords), {}

            def kernel(self, data):
                return data[..., ::2]

        with pytest.raises(ParameterError, match="changes the patch's shape"):
            Impulsed(Halves(), time=1e-6)(patch)

    def test_takes_another_patch(self, patch):
        """An operation given another patch is refused."""

        class Adds(dc.PatchProcessor):
            """A linear operation which adds another patch."""

            name = None
            linear = True
            position_free = True
            other: object = None

            def kernel(self, data):
                return data

        with pytest.raises(ParameterError, match="takes another patch"):
            Impulsed(Adds(other=patch), time=1e-6)(patch)

    @pytest.mark.parametrize("name", ("not_a_method", "_private", "rolling"))
    def test_not_an_operation(self, patch, name):
        """Only an operation written as a processor is found on the window."""
        window = patch.impulse_window(time=1e-6)
        assert isinstance(window, PatchImpulseWindow)
        with pytest.raises(AttributeError):
            getattr(window, name)
