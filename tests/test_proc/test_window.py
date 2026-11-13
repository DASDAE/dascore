"""Tests for windows of a patch: the declaration, reductions, and apply."""

from __future__ import annotations

import copy
import dataclasses
import warnings
from typing import ClassVar

import numpy as np
import pytest
from scipy.signal import detrend

import dascore as dc
from dascore.core.processor import PatchProcessor, captured_operation
from dascore.core.source import ArraySource
from dascore.exceptions import (
    LazyOperationError,
    ParameterError,
    PatchAttributeError,
    PatchCoordinateError,
    PatchError,
)
from dascore.lazy.recipe import Recipe
from dascore.proc import window as window_module
from dascore.proc.rolling import _rolling_pandas
from dascore.proc.window import PatchWindow, Windowed, WindowReduce
from dascore.transform.hilbert import Envelope
from dascore.units import percent
from dascore.warnings import NumpyFallbackWarning

REDUCERS = ("mean", "median", "min", "max", "std", "sum")


@pytest.fixture(scope="module")
def ramp():
    """Six samples counting up along time, labelled by their position."""
    return dc.Patch(data=np.arange(6.0), coords={"time": np.arange(6)}, dims=("time",))


@pytest.fixture(scope="module")
def patch():
    """A two dimensional patch of random values, so no reduction is trivial."""
    example = dc.get_example_patch()
    data = np.random.default_rng(0).standard_normal(example.shape)
    return example.new(data=data)


def _rolled(patch, name="mean", **kwargs):
    """What the deprecated rolling gives, with its warning swallowed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return getattr(patch.rolling(**kwargs), name)()


class TestDeclaration:
    """`Patch.window` states a window and computes nothing."""

    def test_needs_a_dimension(self, patch):
        """A window of nothing is refused."""
        with pytest.raises(ParameterError, match="needs a dimension"):
            patch.window()

    def test_overlap_and_step_refused(self, patch):
        """The two say the same thing, so only one may be given."""
        with pytest.raises(ParameterError, match="mutually exclusive"):
            patch.window(time=0.5, overlap=0.1, step=0.1)

    @pytest.mark.parametrize("fill", ("zero", [0, 1], np.zeros(2)))
    def test_fill_must_be_one_number(self, patch, fill):
        """A fill which is not a single value is refused when declared."""
        with pytest.raises(ParameterError, match="single number"):
            patch.window(time=0.5, fill=fill)

    @pytest.mark.parametrize("option", ({"ends": "shift"}, {"align": "middle"}))
    def test_unknown_option_refused(self, patch, option):
        """A misspelt option is refused when the window is declared."""
        with pytest.raises(ParameterError, match="must be one of"):
            patch.window(time=0.5, **option)

    def test_the_operation_checks_too(self, patch):
        """The operation behind a reduction refuses the same misspellings."""
        with pytest.raises(ParameterError, match="must be one of"):
            WindowReduce("mean", time=3, samples=True, align="middle")(patch)

    def test_unknown_dimension_refused(self, patch):
        """A dimension the patch lacks is refused by the operation."""
        with pytest.raises(ParameterError, match="bob"):
            patch.window(bob=3, samples=True).mean()


class TestReduce:
    """A reduction gives one value for each window, labelled by a sample."""

    def test_keeps_the_shape(self, ramp):
        """Every sample gets a window; the incomplete ones at the ends are NaN."""
        out = ramp.window(time=3, samples=True).mean()
        np.testing.assert_array_equal(out.data, [np.nan, 1, 2, 3, 4, np.nan])
        np.testing.assert_array_equal(out.get_array("time"), np.arange(6))

    def test_drop_leaves_out_the_incomplete(self, ramp):
        """With ends='drop' only the complete windows remain, and their labels."""
        out = ramp.window(time=3, samples=True, ends="drop").mean()
        np.testing.assert_array_equal(out.data, [1, 2, 3, 4])
        np.testing.assert_array_equal(out.get_array("time"), [1, 2, 3, 4])

    def test_step(self, ramp):
        """A step keeps every step-th window, labelled by every step-th sample."""
        out = ramp.window(time=3, step=2, samples=True).mean()
        np.testing.assert_array_equal(out.data, [np.nan, 2, 4])
        np.testing.assert_array_equal(out.get_array("time"), [0, 2, 4])

    def test_step_and_drop(self, ramp):
        """The complete windows of a stepped window keep their own labels."""
        out = ramp.window(time=3, step=2, samples=True, ends="drop").mean()
        np.testing.assert_array_equal(out.data, [2, 4])
        np.testing.assert_array_equal(out.get_array("time"), [2, 4])

    @pytest.mark.parametrize(
        "align, expected",
        (
            ("center", [np.nan, np.nan, 1.5, 2.5, 3.5, np.nan]),
            ("right", [np.nan, np.nan, np.nan, 1.5, 2.5, 3.5]),
            ("left", [1.5, 2.5, 3.5, np.nan, np.nan, np.nan]),
        ),
    )
    def test_align(self, ramp, align, expected):
        """The label is the later middle sample, the last, or the first."""
        out = ramp.window(time=4, samples=True, align=align).mean()
        np.testing.assert_array_equal(out.data, expected)

    def test_window_as_long_as_the_data(self, ramp):
        """One window fits, and it is the middle sample's."""
        out = ramp.window(time=6, samples=True).mean()
        expected = np.full(6, np.nan)
        expected[3] = 2.5
        np.testing.assert_array_equal(out.data, expected)

    def test_longer_than_the_data_refused(self, ramp):
        """A window which could never be complete is refused."""
        with pytest.raises(ParameterError, match="larger than coordinate"):
            ramp.window(time=7, samples=True).mean()

    def test_no_complete_window(self, ramp):
        """With no window complete, every value is NaN, or nothing is left."""
        window = {"time": 6, "step": 4, "samples": True}
        np.testing.assert_array_equal(ramp.window(**window).mean().data, [np.nan] * 2)
        assert ramp.window(**window, ends="drop").mean().shape == (0,)

    @pytest.mark.parametrize("size", (3, 150))
    def test_one_dimension_uses_numpys_std(self, size):
        """A patch of one dimension is reduced as any other: ddof of zero."""
        data = np.arange(400.0) ** 2
        coords = {"time": np.arange(400)}
        squares = dc.Patch(data=data, coords=coords, dims=("time",))
        out = squares.window(time=size, samples=True, ends="drop").std()
        view = np.lib.stride_tricks.sliding_window_view(data, size)
        np.testing.assert_allclose(out.data, view.std(axis=-1), rtol=1e-9)

    def test_small_window_of_one_dimension(self):
        """The first complete windows of a short patch, by hand."""
        data = np.arange(6.0) ** 2
        squares = dc.Patch(data=data, coords={"time": np.arange(6)}, dims=("time",))
        out = squares.window(time=3, samples=True, align="right").std()
        np.testing.assert_allclose(
            out.data[2:], [np.std(data[i : i + 3]) for i in range(4)]
        )

    def test_other_layouts(self):
        """Three dimensions and a descending coordinate reduce the same way."""
        data = np.random.default_rng(3).standard_normal((3, 4, 20))
        coords = {"a": np.arange(3), "b": np.arange(4), "time": np.arange(20)[::-1]}
        cube = dc.Patch(data=data, coords=coords, dims=("a", "b", "time"))
        out = cube.window(time=5, step=2, samples=True, ends="drop").max()
        np.testing.assert_array_equal(
            out.get_array("time"), [17, 15, 13, 11, 9, 7, 5, 3]
        )
        np.testing.assert_array_equal(out.data[..., 0], data[..., 0:5].max(axis=-1))
        np.testing.assert_array_equal(out.data[..., -1], data[..., 14:19].max(axis=-1))

    def test_complex_stays_complex(self):
        """Complex data keep both parts."""
        data = np.arange(6) * 1j
        turned = dc.Patch(data=data, coords={"time": np.arange(6)}, dims=("time",))
        out = turned.window(time=3, samples=True, ends="drop").mean()
        np.testing.assert_array_equal(out.data, np.array([1j, 2j, 3j, 4j]))

    def test_unknown_name_refused(self, patch):
        """A reduction named by a string must be one of the six."""
        with pytest.raises(ParameterError, match="A reduction is a function"):
            patch.window(time=3, samples=True).reduce("var")

    def test_fast_engine_is_taken(self, patch, monkeypatch):
        """A long window named by a string goes to pandas; arguments keep it out."""
        calls = []

        def counted(*args, **kwargs):
            """Note the call and make it."""
            calls.append(args[1])
            return _rolling_pandas(*args, **kwargs)

        monkeypatch.setattr(window_module, "_rolling_pandas", counted)
        window = patch.window(time=400, samples=True)
        window.std()
        window.reduce(np.std)
        window.reduce("std", ddof=1)
        patch.window(time=3, samples=True).std()
        assert calls == ["std"]

    def test_arguments_to_a_named_reduction(self, patch):
        """A named reduction given arguments is NumPy's with them, any length."""
        long = patch.window(time=400, samples=True, ends="drop")
        out = long.reduce("std", ddof=1)
        expected = np.std(patch.data[:, :400], ddof=1, axis=-1)
        np.testing.assert_allclose(out.data[:, 0], expected)
        assert out.attrs.history[-1].endswith(".reduce(ddof='1',function='std')")

    @pytest.mark.parametrize("ends, fill", (("drop", None), ("pad", 2.5)))
    @pytest.mark.parametrize("align", ("center", "right", "left"))
    def test_fast_engine_with_drop_and_fill(self, patch, ends, fill, align):
        """The fast engine agrees when windows are dropped or filled out."""
        window = patch.window(time=400, samples=True, align=align, ends=ends, fill=fill)
        np.testing.assert_allclose(
            window.mean().data, window.reduce(np.mean).data, rtol=1e-9, atol=1e-9
        )

    @pytest.mark.parametrize("name", REDUCERS)
    @pytest.mark.parametrize("align", ("center", "right", "left"))
    @pytest.mark.parametrize("step", (1, 3, 4))
    @pytest.mark.parametrize("size", (400, 401))
    def test_fast_engine_agrees(self, patch, name, align, step, size):
        """A long window, which another engine answers, gives NumPy's values."""
        # An odd window centres where pandas does, so its answer is used whole.
        window = patch.window(time=size, step=step, samples=True, align=align)
        named = getattr(window, name)()
        by_function = window.reduce(getattr(np, name))
        np.testing.assert_allclose(
            named.data, by_function.data, equal_nan=True, rtol=1e-9, atol=1e-9
        )
        assert np.isnan(named.data).sum() == np.isnan(by_function.data).sum() > 0

    def test_each_dimension(self, patch):
        """Either dimension can be windowed, and the other is untouched."""
        out = patch.window(distance=5, samples=True).max()
        assert out.shape == patch.shape
        expected = patch.data[0:5].max(axis=0)
        np.testing.assert_array_equal(out.data[2], expected)

    def test_arguments_reach_the_function(self, patch):
        """What is given after the function is passed on to it, by name too."""
        window = patch.window(time=50, samples=True, ends="drop")
        expected = np.percentile(patch.data[:, :50], 80, axis=-1)
        np.testing.assert_allclose(
            window.reduce(np.percentile, 80).data[:, 0], expected
        )
        np.testing.assert_allclose(
            window.reduce(np.percentile, q=80).data[:, 0], expected
        )

    def test_units_and_percent(self, patch):
        """A step in coordinate units and an overlap in percent say the same."""
        step = patch.get_coord("time").step
        by_step = patch.window(time=20 * step, step=5 * step).mean()
        by_overlap = patch.window(time=20, overlap=75 * percent, samples=True).mean()
        assert by_step.shape == by_overlap.shape == (300, 400)
        np.testing.assert_array_equal(by_step.data, by_overlap.data)

    def test_integers_become_floats(self, patch):
        """The result is float64, so an incomplete window can be NaN."""
        ints = patch.new(data=np.ones(patch.shape, np.int16))
        out = ints.window(time=3, samples=True).sum()
        assert out.dtype == out.data.dtype == np.float64
        assert np.isnan(out.data[:, 0]).all() and (out.data[:, 1:-1] == 3).all()
        # With nothing to pad, the values and the stated dtype are still floats.
        whole = ints.window(time=3, samples=True, ends="drop").reduce(np.sum)
        assert whole.dtype == whole.data.dtype == np.float64

    def test_one_dimension_only(self, patch):
        """A reduction over two windowed dimensions is refused."""
        with pytest.raises(ParameterError, match="exactly one dimension"):
            patch.window(time=3, distance=3, samples=True).mean()

    @pytest.mark.parametrize("option", ({"taper": "hann"}, {"analysis": "hann"}))
    def test_blending_options_refused(self, patch, option):
        """Nothing is blended, so a taper or analysis window is refused."""
        with pytest.raises(ParameterError, match="blends nothing"):
            patch.window(time=3, samples=True, **option).mean()

    def test_fill_completes_the_ends(self, ramp):
        """With a fill, the end windows are computed over the filled-out data."""
        out = ramp.window(time=3, samples=True, fill=0).mean()
        np.testing.assert_allclose(out.data, [1 / 3, 1, 2, 3, 4, 3])
        stepped = ramp.window(time=3, step=2, samples=True, fill=np.nan)
        np.testing.assert_allclose(stepped.reduce(np.nanmean).data, [0.5, 2, 4])

    def test_fill_on_integers(self):
        """A fill the data's dtype cannot hold is not cut down to fit it."""
        coords = {"time": np.arange(6)}
        ints = dc.Patch(data=np.arange(6), coords=coords, dims=("time",))
        out = ints.window(time=3, samples=True, fill=0.5).sum()
        np.testing.assert_allclose(out.data, [1.5, 3, 6, 9, 12, 9.5])
        holed = ints.window(time=3, samples=True, fill=np.nan).reduce(np.nanmean)
        np.testing.assert_allclose(holed.data, [0.5, 1, 2, 3, 4, 4.5])

    def test_fill_out_of_the_dtypes_range(self):
        """A fill outside what the data's integers hold is still itself."""
        coords = {"time": np.arange(6)}
        small = dc.Patch(data=np.arange(6, dtype="u1"), coords=coords, dims=("time",))
        out = small.window(time=3, samples=True, fill=-1).min()
        np.testing.assert_array_equal(out.data, [-1, 0, 1, 2, 3, -1])
        short = dc.Patch(data=np.arange(6, dtype="i2"), coords=coords, dims=("time",))
        out = short.window(time=3, samples=True, fill=100_000).max()
        np.testing.assert_array_equal(out.data, [100_000, 2, 3, 4, 5, 100_000])

    def test_fill_with_drop_refused(self, ramp):
        """A fill completes the windows which drop would leave out."""
        with pytest.raises(ParameterError, match="one or the other"):
            ramp.window(time=3, samples=True, ends="drop", fill=9.0).mean()

    def test_fill_agrees_with_no_fill_inside(self, patch):
        """A fill changes only the windows which reach past the data."""
        plain = patch.window(time=9, samples=True, align="right").std()
        filled = patch.window(time=9, samples=True, align="right", fill=1.0).std()
        np.testing.assert_allclose(filled.data[:, 8:], plain.data[:, 8:])
        assert not np.isnan(filled.data).any()

    def test_lazy_refused(self):
        """A lazy patch is refused, and nothing is read."""
        recipe = Recipe.from_source(ArraySource.full((4, 30), 1.0))
        coords = {"distance": np.arange(4), "time": np.arange(30)}
        lazy = dc.Patch(data=recipe, coords=coords, dims=("distance", "time"))
        with pytest.raises(LazyOperationError):
            lazy.window(time=3, samples=True).mean()


class TestRollingParity:
    """A window labelled like `rolling` gives what `rolling` gave."""

    @pytest.mark.parametrize("name", REDUCERS)
    @pytest.mark.parametrize("size, step", ((3, 1), (4, 1), (10, 3), (7, 7), (250, 2)))
    def test_right_is_rolling(self, patch, name, size, step):
        """align='right' is rolling's default label, step or no step."""
        window = patch.window(time=size, step=step, samples=True, align="right")
        out = getattr(window, name)()
        old = _rolled(patch, name, time=size, step=step, samples=True)
        assert out.coords == old.coords
        np.testing.assert_allclose(out.data, old.data, equal_nan=True)

    def test_stepped_center_is_not_rollings(self):
        """With a step, a window is on its label, where rolling's was off it."""
        coords = {"time": np.arange(12)}
        ramp = dc.Patch(data=np.arange(12.0), coords=coords, dims=("time",))
        out = ramp.window(time=3, step=2, samples=True).mean()
        np.testing.assert_array_equal(out.data, [np.nan, 2, 4, 6, 8, 10])
        old = _rolled(ramp, time=3, step=2, samples=True, center=True, engine="numpy")
        assert not np.array_equal(out.data, old.data, equal_nan=True)

    @pytest.mark.parametrize("size", (3, 4, 9, 250))
    def test_center_is_rolling_centered(self, patch, size):
        """With no step, the default label is rolling's center=True."""
        out = patch.window(time=size, samples=True).mean()
        old = _rolled(patch, time=size, samples=True, center=True)
        assert out.coords == old.coords
        np.testing.assert_allclose(out.data, old.data, equal_nan=True)

    def test_rolling_warns(self, patch):
        """`rolling` still works, and says what replaces it."""
        with pytest.warns(DeprecationWarning, match=r"Patch\.window"):
            patch.rolling(time=3, samples=True).mean()

    def test_operations_built_on_it_do_not_warn(self, patch):
        """The transforms which roll inside themselves warn of nothing."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            patch.stalta(time=(0.01, 0.1))
            patch.fbe(time=(10, 50), window=0.1)


class TestIdentity:
    """A windowed operation records its window in history and in its id."""

    def test_reduction_history(self, patch):
        """The call is written as it is made: the window, then the reduction."""
        out = patch.window(time=3, samples=True, ends="drop").mean()
        assert out.attrs.history[-1] == (
            "window(ends='drop',samples='True',time='3').mean()"
        )

    def test_reduce_history(self, patch):
        """A function is named, with what it was given."""
        out = patch.window(time=0.5, step=0.25).reduce(np.percentile, 80)
        assert out.attrs.history[-1] == (
            "window(step='0.25',time='0.5').reduce(args='(80)',function='percentile')"
        )

    def test_every_option_is_written(self, patch):
        """Each option a call set is in what history says of it."""
        window = {"time": 3, "samples": True, "align": "right", "fill": 1.0}
        out = patch.window(**window, overlap=1).max()
        assert out.attrs.history[-1] == (
            "window(align='right',fill='1.0',overlap='1',samples='True',time='3').max()"
        )
        tiles = patch.window(time=64, step=32, samples=True, fill=1.0)
        stacked = tiles.apply(np.positive, mode="stack", engine="numpy")
        assert stacked.attrs.history[-1] == (
            "window(fill='1.0',samples='True',step='32',time='64')"
            ".apply(engine='numpy',function='positive',mode='stack')"
        )

    def test_apply_history(self, patch):
        """An applied function is named after its window."""
        out = patch.window(time=0.5, taper="triang").apply(np.positive)
        assert out.attrs.history[-1] == (
            "window(taper='triang',time='0.5').apply(function='positive')"
        )

    @pytest.mark.parametrize(
        "change",
        ({"time": 4}, {"step": 2}, {"ends": "drop"}, {"align": "right"}, {"fill": 0}),
    )
    def test_window_is_part_of_the_id(self, patch, change):
        """The same call has one id; any change to the window makes another."""
        base = {"time": 3, "samples": True}
        first = patch.window(**base).mean()
        again = patch.window(**base).mean()
        other = patch.window(**(base | change)).mean()
        assert first.attrs.data_id == again.attrs.data_id != patch.attrs.data_id
        assert other.attrs.data_id != first.attrs.data_id


class TestApply:
    """`apply` hands the windows to a function and blends or stacks them."""

    def test_step_is_the_overlap_it_leaves(self, patch):
        """A step says what an overlap of the rest of the window says."""

        def demean(tiles):
            """Take each window's own mean from it, so the tiling shows."""
            return tiles - tiles.mean(axis=-1, keepdims=True)

        by_step = patch.window(time=64, step=48, samples=True).apply(demean)
        by_overlap = patch.window(time=64, overlap=16, samples=True).apply(demean)
        other = patch.window(time=64, overlap=32, samples=True).apply(demean)
        np.testing.assert_array_equal(by_step.data, by_overlap.data)
        assert not np.allclose(by_step.data, other.data)

    def test_gaps_refused(self, patch):
        """Windows which leave samples uncovered cannot be blended."""
        with pytest.raises(ParameterError, match="gaps"):
            patch.window(time=64, step=100, samples=True).apply(lambda x: x)

    @pytest.mark.parametrize("mode", ("overlap_add", "stack"))
    @pytest.mark.parametrize("compiled", (False, True))
    def test_fill_is_what_a_window_holds_past_the_data(self, patch, mode, compiled):
        """Either engine and either mode hand the function the fill."""
        numba = pytest.importorskip("numba") if compiled else None
        channel = patch.select(distance=0, samples=True).squeeze()

        def lowest(tiles):
            """Every sample replaced by the lowest its window (or stack) holds."""
            return np.full(tiles.shape, tiles.min())

        function = numba.njit(lowest) if compiled else lowest
        window = {"time": 64, "samples": True}
        plain = channel.window(**window).apply(function, mode=mode)
        filled = channel.window(**window, fill=-99.0).apply(function, mode=mode)
        # Only the fill is that low, and without one nothing is.
        assert filled.data.min() == -99.0 and plain.data.min() > -99.0

    def test_fill_widens_integers(self, patch):
        """An integer patch given a fractional fill holds it whole."""
        ints = patch.new(data=np.ones(patch.shape, np.int16))
        seen = []

        def peek(tiles):
            """Keep the lowest value handed over."""
            seen.append(tiles.min())
            return tiles

        ints.window(time=64, samples=True, fill=0.5).apply(peek)
        assert set(seen) == {0.5}

    def test_complex_fill_reaches_a_compiled_function(self, patch):
        """The compiled engine keeps both parts of a complex fill."""
        numba = pytest.importorskip("numba")
        window = patch.window(time=8, samples=True, fill=1 + 2j)
        # A window's sum reaches every sample of it, so the fill reaches the data.
        compiled = window.apply(numba.njit(lambda tile: tile + tile.sum()))
        plain = window.apply(lambda tiles: tiles + tiles.sum(axis=-1, keepdims=True))
        assert np.iscomplexobj(compiled.data)
        np.testing.assert_allclose(compiled.data, plain.data, atol=1e-6)

    def test_fill_is_part_of_the_id(self, patch):
        """Two fills are two results."""
        window = {"time": 64, "samples": True}
        first = patch.window(**window).apply(np.negative)
        other = patch.window(**window, fill=1.0).apply(np.negative)
        assert first.attrs.data_id != other.attrs.data_id

    @pytest.mark.parametrize("option", ({"ends": "drop"}, {"align": "right"}))
    def test_reduction_options_refused(self, patch, option):
        """Blended windows have no label to align and are always padded."""
        with pytest.raises(ParameterError, match="apply to reductions"):
            patch.window(time=64, samples=True, **option).apply(lambda x: x)


class _Shrinks(dc.PatchProcessor):
    """An operation whose kernel returns less than its metadata states."""

    name = None

    def kernel(self, data):
        """Return all but the last sample of the last axis."""
        return data[..., :-1]


class _NameOnly(dc.PatchProcessor):
    """An operation which writes only its name into history."""

    name = None
    history = "method_name"

    def kernel(self, data):
        """Return the data doubled."""
        return data * 2


def _demeaned(tiles):
    """What a constant detrend along time makes of a stack of windows."""
    return detrend(tiles, axis=-1, type="constant")


class TestOperations:
    """A patch operation called on a window runs in each window."""

    @pytest.mark.parametrize(
        "window",
        (
            {"blend": "hann"},
            {"taper": "triang"},
            {"step": 75},
            {"overlap": 10},
            {"analysis": "hann", "overlap": 75},
            {"fill": 2.5},
            {"distance": 40},
        ),
        ids=lambda x: next(iter(x)),
    )
    def test_window_options_reach_the_blend(self, patch, window):
        """Each option gives what the same function gives through apply."""
        window = {k: v for k, v in window.items() if k != "blend"}
        declared = patch.window(time=100, samples=True, **window)
        out = declared.detrend("time", "constant")
        expected = declared.apply(_demeaned)
        np.testing.assert_allclose(out.data, expected.data, rtol=1e-10, atol=1e-10)
        # A mean along time is the same whichever distances share its window.
        if window and "distance" not in window:
            plain = patch.window(time=100, samples=True).detrend("time", "constant")
            assert not np.allclose(out.data, plain.data)

    def test_fill_widens_integers(self, patch):
        """A fill integers cannot hold is not cut down to one they can."""
        ints = patch.new(data=np.ones(patch.shape, np.int16))
        window = ints.window(time=100, samples=True, fill=2.5)
        out = window.detrend("time", "constant")
        expected = window.apply(_demeaned)
        np.testing.assert_allclose(out.data, expected.data, atol=1e-10)
        # The first window is half fill, so its ones are not its mean.
        assert not np.isclose(out.data[0, 0], 0) and np.isclose(out.data[0, 1000], 0)

    def test_taper_is_the_operation(self, patch):
        """`taper` on a window is the patch operation, not the window's option."""
        out = patch.window(time=100, samples=True).taper(time=0.1)
        expected = Windowed(dc.proc.Taper(time=0.1), time=100, samples=True)(patch)
        assert out.equals(expected)
        assert not np.allclose(out.data, patch.data)

    def test_no_option_hides_an_operation(self, patch):
        """No field of the window has the name of an operation it could run."""
        fields = {x.name for x in dataclasses.fields(PatchWindow)}
        operations = {
            x for x in dir(dc.Patch) if hasattr(getattr(dc.Patch, x), "__processor__")
        }
        assert len(operations) > 50 and not fields & operations

    def test_data_type_is_the_operations(self, patch):
        """What the operation says its result is, the windowed result is."""
        window = patch.window(time=100, samples=True)
        assert window.envelope("time").attrs.data_type == "envelope"
        assert window.hilbert("time").attrs.data_type == ""

    def test_what_is_returned_is_the_windowed_call(self, patch):
        """The method found on a window is named for the operation alone."""
        window = patch.window(time=100, samples=True)
        assert window.detrend.__name__ == "detrend"
        # The bypass of the operation is not the bypass of this call.
        assert not hasattr(window.detrend, "raw_function")
        assert copy.copy(window).detrend("time").equals(window.detrend("time"))

    def test_is_the_function_applied(self, patch):
        """An operation in a window is its kernel applied to each window."""
        window = patch.window(time=0.5, distance=40)
        out = window.abs()
        expected = window.apply(np.abs)
        assert out.shape == patch.shape and out.coords == patch.coords
        np.testing.assert_allclose(out.data, expected.data, rtol=1e-12, atol=1e-12)

    def test_each_window_is_treated_alone(self, patch):
        """Windows which abut each have their own trend taken out."""
        length = patch.shape[1] // 100 * 100
        whole = patch.select(time=(0, length), samples=True)
        out = whole.window(time=100, overlap=0, samples=True).detrend("time")
        pieces = [
            whole.select(time=(x, x + 100), samples=True).detrend("time").data
            for x in range(0, length, 100)
        ]
        np.testing.assert_allclose(out.data, np.concatenate(pieces, axis=1), atol=1e-12)
        assert not np.allclose(out.data, whole.detrend("time").data)

    def test_arguments_reach_the_operation(self, patch):
        """Positional and keyword arguments are the method's own."""
        window = patch.window(time=100, samples=True)
        by_position = window.detrend("time", "constant")
        by_name = window.detrend(dim="time", type="constant")
        assert by_position.equals(by_name)
        assert not np.allclose(by_position.data, window.detrend("time").data)

    def test_dimensions_in_any_order(self, patch):
        """The windows are the same whichever dimension is named first."""
        one = patch.window(time=100, distance=30, samples=True).detrend("time")
        other = patch.window(distance=30, time=100, samples=True).detrend("time")
        np.testing.assert_array_equal(one.data, other.data)

    def test_analysis_window(self, patch):
        """Under an analysis window an untouched window still comes back."""
        window = patch.window(time=64, overlap=48, samples=True, analysis="hann")
        np.testing.assert_allclose(window.real().data, patch.data, atol=1e-10)

    def test_fill_is_what_a_window_holds_past_the_data(self, patch):
        """The operation sees the fill where its window leaves either end."""
        window = {"time": 100, "samples": True}
        zeros = patch.window(**window).detrend("time", "constant")
        filled = patch.window(**window, fill=5.0).detrend("time", "constant")
        assert not np.allclose(zeros.data[:, :50], filled.data[:, :50])
        assert not np.allclose(zeros.data[:, -50:], filled.data[:, -50:])
        np.testing.assert_allclose(zeros.data[:, 100:-150], filled.data[:, 100:-150])

    def test_metadata_only_operation(self, patch):
        """An operation with no kernel sets its attrs and blends nothing."""
        ints = patch.new(data=np.ones(patch.shape, np.int16))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = ints.window(time=100, samples=True).set_units("m/s")
        assert out.attrs.data_units == dc.get_quantity("m/s")
        assert out.dtype == np.int16 and np.array_equal(out.data, ints.data)

    def test_history_of_an_operation_recording_its_name(self, patch):
        """An operation which records only its name is written as a call."""
        out = Windowed(_NameOnly(), time=100, samples=True)(patch)
        assert out.attrs.history[-1].endswith("time='100')._NameOnly()")

    def test_history(self, patch):
        """The window is written before the operation, as it is called."""
        out = patch.window(time=100, overlap=20, samples=True).detrend("time")
        expected = (
            "window(overlap='20',samples='True',time='100')"
            ".detrend(dim='time',type='linear')"
        )
        assert out.attrs.history[-1] == expected

    def test_window_and_operation_are_the_id(self, patch):
        """The same call has one id; another window or argument has another."""

        def run(size=100, kind="linear"):
            """The id of a detrend in a window."""
            window = patch.window(time=size, samples=True)
            return window.detrend("time", kind).attrs.data_id

        assert run() == run()
        assert len({run(), run(size=50), run(kind="constant")}) == 3
        assert run() != patch.detrend("time").attrs.data_id

    @pytest.mark.parametrize(
        "option",
        ({"taper": "triang"}, {"analysis": "hann"}, {"fill": 1.0}, {"step": 60}),
        ids=lambda x: next(iter(x)),
    )
    def test_every_option_is_the_id(self, patch, option):
        """Each option of the window changes the id and is written down."""
        plain = patch.window(time=100, samples=True).detrend("time")
        out = patch.window(time=100, samples=True, **option).detrend("time")
        assert out.attrs.data_id != plain.attrs.data_id
        assert next(iter(option)) in out.attrs.history[-1]

    def test_integers_warn_and_become_floats(self, patch):
        """Blended integers are floats, which the call says where it is made."""
        ints = patch.new(data=(patch.data * 100).astype(np.int16))
        match = "Blending windows of int16 data gives float32"
        with pytest.warns(UserWarning, match=match) as caught:
            out = ints.window(time=100, samples=True).abs()
        assert out.dtype == np.float32
        assert [x.filename for x in caught] == [__file__]

    def test_booleans_warn_too(self, patch):
        """Blended booleans are floats as blended integers are."""
        flags = patch.new(data=patch.data > 0)
        with pytest.warns(UserWarning, match="Blending windows of bool"):
            out = flags.window(time=100, samples=True).real()
        assert out.dtype.kind == "f"

    def test_floats_made_of_integers_do_not_warn(self, patch):
        """An operation which makes floats of integers blends floats."""
        ints = patch.new(data=(patch.data * 100).astype(np.int16))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = ints.window(time=100, samples=True).detrend("time")
        assert out.dtype == np.float64

    def test_processor_is_the_call(self, patch):
        """The processor built directly is the method on the window."""
        op = Windowed(dc.proc.Detrend(dim="time"), time=100, samples=True)
        out = patch.window(time=100, samples=True).detrend("time")
        assert op(patch).equals(out)
        assert op(patch).attrs.data_id == out.attrs.data_id


class _Typed(dc.PatchProcessor):
    """An operation which plans for a dtype, and is handed that dtype."""

    name = None
    position_free = True

    def get_metadata(self, meta):
        """Return whether the data are complex."""
        return meta, {"complex": np.dtype(meta.dtype).kind == "c"}

    def kernel(self, data, *, complex):
        """Return the data, which are what the plan was told."""
        assert (data.dtype.kind == "c") is complex
        return data


class TestWhatTheOperationIsHanded:
    """The operation plans for the window it will be given."""

    @pytest.mark.parametrize("window", ({"fill": 1 + 2j}, {"analysis": "hann"}))
    def test_plans_for_the_dtype_it_is_handed(self, patch, window):
        """A fill or an analysis window changes the dtype before the plan."""
        ints = patch.new(data=np.ones(patch.shape, np.int16))
        out = Windowed(_Typed(), time=100, overlap=20, samples=True, **window)(ints)
        assert out.dtype.kind == ("c" if "fill" in window else "f")


class TestOperationsRefused:
    """What cannot run in a window is refused when it is called."""

    @pytest.mark.parametrize("option", ({"ends": "drop"}, {"align": "right"}))
    def test_reduction_options(self, patch, option):
        """Ends and align say nothing about windows which are blended."""
        with pytest.raises(ParameterError, match="apply to reductions"):
            patch.window(time=100, samples=True, **option).detrend("time")

    def test_shape_changes(self, patch):
        """An operation which changes a window's shape is not blended."""
        with pytest.raises(ParameterError, match="keep their shape"):
            patch.window(time=100, samples=True).decimate(time=2)

    @pytest.mark.parametrize(
        "change",
        (
            lambda x: x.convert_units(distance="ft"),
            # Equality of coordinates passes over their units.
            lambda x: x.set_units(distance="ft"),
            lambda x: x.rename_coords(time="t"),
            lambda x: x.drop_private_coords(),
        ),
        ids=("values", "units", "dims", "private"),
    )
    def test_coordinate_changes(self, patch, change):
        """Nor one which changes the coordinates, which the result would lose."""
        hidden = patch.update_coords(_hidden=("time", np.arange(patch.shape[1])))
        with pytest.raises(ParameterError, match="dimensions or coordinates"):
            change(hidden.window(time=100, samples=True))

    def test_array_shaped_like_the_patch(self, patch):
        """An array which lines up with the whole patch cannot be cut yet."""
        mask = np.ones(patch.shape, bool)
        with pytest.raises(ParameterError, match="array shaped like the patch"):
            patch.window(time=100, samples=True).where(mask)

    def test_error_of_the_operation_names_the_window(self, patch):
        """What the operation raises for one window says that is where."""
        window = patch.window(time=100, samples=True)
        with pytest.raises(ValueError, match="out of bounds") as caught:
            window.taper_range(time=(0.1, 0.2, 0.3, 0.4), relative=False)
        assert "one window of Patch.window" in caught.value.__notes__[0]

    @pytest.mark.parametrize(
        "needs, error",
        (
            ({"required_attrs": {"data_type": "not_this"}}, PatchAttributeError),
            ({"required_dims": ("not_a_dim",)}, PatchCoordinateError),
            ({"required_coords": ("not_a_coord",)}, PatchCoordinateError),
        ),
    )
    def test_what_the_operation_requires(self, patch, needs, error):
        """A window is checked for what the operation needs of a patch."""
        needy = type("_Needy", (_NameOnly,), {"name": None, **needs})
        with pytest.raises(error):
            Windowed(needy(), time=100, samples=True)(patch)

    def test_taper_with_analysis(self, patch):
        """An analysis window is blended under its dual, never a taper too."""
        window = patch.window(time=64, samples=True, taper="triang", analysis="hann")
        with pytest.raises(ParameterError, match="a taper cannot be given"):
            window.abs()

    def test_reconciling_operation(self, patch):
        """An operation which settles its result from the data is refused."""
        with pytest.raises(ParameterError, match="from the data it computed"):
            patch.window(time=100, samples=True).dropna("time")

    def test_operation_taking_a_patch(self, patch):
        """So is one given another patch."""
        with pytest.raises(ParameterError, match="takes another patch"):
            patch.window(time=100, samples=True).where(patch > 0)

    def test_window_longer_than_the_dimension(self, patch):
        """There must be one whole window to decide from."""
        with pytest.raises(ParameterError, match="larger than coordinate"):
            patch.window(time=patch.shape[1] + 1, samples=True).detrend("time")

    def test_not_a_processor(self, patch):
        """The processor takes an operation, not a function."""
        with pytest.raises(ParameterError, match="runs a PatchProcessor"):
            Windowed(np.abs, time=100, samples=True)(patch)

    def test_kernel_which_changes_the_shape(self, patch):
        """A kernel which does not keep the shape it stated is caught."""
        with pytest.raises(PatchError, match="computed a window of shape"):
            Windowed(_Shrinks(), time=100, samples=True)(patch)

    @pytest.mark.parametrize("name", ("add", "rolling", "not_a_method", "_private"))
    def test_not_an_operation(self, patch, name):
        """Only an operation written as a processor is found on a window."""
        window = patch.window(time=100, samples=True)
        assert not hasattr(window, name)
        with pytest.raises(AttributeError):
            getattr(window, name)


def _subclasses(cls):
    """Yield every subclass of a class, at any depth."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


# Every one of DASCore's own operations which says its plan ignores position.
POSITION_FREE = sorted(
    {
        x.__name__
        for x in _subclasses(PatchProcessor)
        if x.__dict__.get("position_free") and x.__module__.startswith("dascore.")
    }
)

# How each is called. An operation declared later needs an entry.
FREE_CALLS = {
    "Abs": lambda x: x.abs(),
    "Real": lambda x: x.real(),
    "Imag": lambda x: x.imag(),
    "Angle": lambda x: x.angle(),
    "Demean": lambda x: x.demean("time"),
    "Demedian": lambda x: x.demedian("time"),
    "Standardize": lambda x: x.standardize("time"),
    "Normalize": lambda x: x.normalize("time", window=5, samples=True),
    "Detrend": lambda x: x.detrend("time"),
    "PassFilter": lambda x: x.pass_filter(time=(None, 20)),
    "SobelFilter": lambda x: x.sobel_filter("time"),
    # The class the median and Savitzky-Golay filters take their plan from.
    "_WindowFilter": lambda x: x.savgol_filter(polyorder=2, time=5, samples=True),
    "NotchFilter": lambda x: x.notch_filter(q=30, time=20),
    "GaussianFilter": lambda x: x.gaussian_filter(time=0.01),
    "HampelFilter": lambda x: x.hampel_filter(threshold=3, time=5, samples=True),
    "WienerFilter": lambda x: x.wiener_filter(time=5, samples=True),
    "Hilbert": lambda x: x.envelope("time"),
    "Taper": lambda x: x.taper(time=0.1),
    "SetUnits": lambda x: x.set_units("m/s"),
    "SimplifyUnits": lambda x: x.simplify_units(),
    "ConvertUnits": lambda x: x.convert_units("mm/s"),
    "Differentiate": lambda x: x.differentiate("time"),
    "Integrate": lambda x: x.integrate("time", definite=False),
    "Whiten": lambda x: x.whiten(smooth_size=5, time=None),
    "Roll": lambda x: x.roll(time=5, samples=True),
    "SlopeFilter": lambda x: x.slope_filter(filt=(2e3, 2.2e3, 8e3, 2e4)),
    "Stalta": lambda x: x.stalta(time=(0.05, 0.3)),
    "Kurtosis": lambda x: x.kurtosis(time=0.1),
    "VelocityToStrainRate": lambda x: x.velocity_to_strain_rate(),
    "Where": lambda x: x.where(np.True_),
    # These three say so only for values measured from a coordinate's start.
    "LineMute": lambda x: x.line_mute(time=(0.1, 0.2)),
    "TaperRange": lambda x: x.taper_range(time=(0.01, 0.02, 0.1, 0.2), relative=True),
    "PowCoord": lambda x: x.pow_coord(distance=1),
}
# Declared although a length found against a float coordinate can round a
# sample either way between windows: one plan for every window is the point.
ROUNDED = {"Taper", "TaperRange"}


class _Sized(dc.PatchProcessor):
    """An operation which plans, and does not say its plan ignores position."""

    name = None

    def get_metadata(self, meta):
        """Return the length of the last axis."""
        return meta, {"length": meta.shape[-1]}

    def kernel(self, data, *, length):
        """Return the data over the length."""
        return data / length


class _Captures:
    """Stands in for a patch, and gives back the operation a method builds."""

    def __init__(self, patch):
        self._patch = patch

    def __getattr__(self, name):
        method = getattr(type(self._patch), name)
        return lambda *args, **kwargs: captured_operation(
            method, self._patch, *args, **kwargs
        )


def _same_plan(one, other) -> bool:
    """Whether two plans hold the same values, exactly, arrays included."""
    if isinstance(one, dict) and isinstance(other, dict):
        same = all(_same_plan(one[x], other[x]) for x in one if x in other)
        return one.keys() == other.keys() and same
    if isinstance(one, list | tuple) and isinstance(other, list | tuple):
        return len(one) == len(other) and all(map(_same_plan, one, other))
    if isinstance(one, np.ndarray) or isinstance(other, np.ndarray):
        nullable = np.asarray(one).dtype.kind in "fcmM"
        return np.array_equal(one, other, equal_nan=nullable)
    return type(one) is type(other) and (one == other or one != one)


class TestPositionFree:
    """Only an operation which says its plan ignores position runs in a window."""

    def test_the_declared_are_called(self):
        """Every operation which says so has a call here to hold it to it."""
        assert POSITION_FREE == sorted(FREE_CALLS)

    @pytest.mark.parametrize("name", FREE_CALLS)
    @pytest.mark.parametrize("time", ("datetime", "float"))
    def test_declaration_holds(self, patch, name, time):
        """Asked for its plan in many windows, a declared operation gives one."""
        if time == "float" and name in ROUNDED:
            pytest.skip("a length rounded against floats may differ by a sample")
        if time == "float":
            # Steps which no float holds exactly, so rounding has its say.
            floats = np.arange(patch.shape[1]) * 0.004 + 1e6 / 3
            patch = patch.update_coords(time=floats)
        operation = FREE_CALLS[name](_Captures(patch))
        assert operation._is_position_free()
        meta = patch.drop_data()
        plans = []
        for start in (0, 37, 250, 251, 999, 1400, 1899):
            coords, _ = meta.coords.isel(time=slice(start, start + 100))
            plans.append(operation.get_metadata(meta.new(coords=coords))[1])
        assert all(_same_plan(plans[0], x) for x in plans)

    @pytest.mark.parametrize("name", FREE_CALLS)
    def test_declared_runs_in_a_window(self, patch, name):
        """Each declared operation is run in windows and gives a patch back."""
        patch = patch.update_attrs(data_type="velocity")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = FREE_CALLS[name](patch.window(time=100, samples=True))
        assert out.shape == patch.shape

    def test_who_is_free(self):
        """One which plans nothing is; one which overrides the plan says again."""

        class Planless(dc.PatchProcessor):
            """An operation with no plan."""

            name = None

        class Overridden(dc.proc.Detrend):
            """A detrend whose plan is its own, with nothing said of it."""

            name = None

            def get_metadata(self, meta):
                """Return the parent's plan."""
                return super().get_metadata(meta)

        assert Planless()._is_position_free()
        assert dc.proc.Detrend(dim="time")._is_position_free()
        assert Envelope(dim="time")._is_position_free()
        assert not Overridden(dim="time")._is_position_free()
        assert not _Sized()._is_position_free()

    def test_undeclared_is_refused(self, patch):
        """An operation which does not say so is not run, however few windows."""
        with pytest.raises(ParameterError, match="setting position_free"):
            Windowed(_Sized(), time=100, samples=True)(patch)

    def test_arguments_can_decide(self, patch):
        """Values from a coordinate's start are every window's; absolute are not."""
        window = patch.window(time=100, samples=True)
        assert window.line_mute(time=(0.1, 0.2)).shape == patch.shape
        with pytest.raises(ParameterError, match="setting position_free"):
            window.line_mute(time=(0.1, 0.2), relative=False)

    def test_many_windows(self):
        """A declared operation is one plan, whatever the number of windows."""
        coords = {"distance": np.arange(2.0), "time": np.arange(30_000.0)}
        patch = dc.Patch(np.ones((2, 30_000)), coords, ("distance", "time"))
        out = patch.window(time=10, samples=True).detrend("time")
        assert out.shape == patch.shape

    def test_taper_is_one_taper_in_every_window(self):
        """Every window gets the first one's ramps, whatever floats would round to."""
        coords = {
            "distance": np.arange(2.0),
            "time": 12345.678 + np.arange(30_000) / 1e4,
        }
        patch = dc.Patch(np.ones((2, 30_000)), coords, ("distance", "time"))
        window = patch.window(time=161, overlap=0, samples=True)
        out = window.taper(time=0.1).data[0, : 161 * 186].reshape(186, 161)
        # Ones in, so each row is its window's taper; every row is the first.
        np.testing.assert_array_equal(out, np.broadcast_to(out[0], out.shape))
        assert out[0, 0] == 0 and out[0, 80] == 1


def _on(patch, library: str):
    """The patch with its data on another array library, if it is installed."""
    if library == "dask":
        dask_array = pytest.importorskip("dask.array")
        return patch.new(data=dask_array.from_array(patch.data, chunks=(100, 500)))
    strict = pytest.importorskip("array_api_strict")
    return patch.new(data=strict.asarray(patch.data))


def _values(patch) -> np.ndarray:
    """The patch's data as a NumPy array, computed if they are a dask graph."""
    data = patch.data
    return np.asarray(data.compute() if hasattr(data, "compute") else data)


@pytest.mark.parametrize("library", ("array_api_strict", "dask"))
class TestOtherBackends:
    """Windows are cut, run and blended on the array library the data are on."""

    CASES: ClassVar = {
        "one_dim": lambda x: x.window(time=100, samples=True).abs(),
        "two_dims": lambda x: x.window(time=100, distance=30, samples=True).abs(),
        "step": lambda x: x.window(time=50, step=37, samples=True).abs(),
        "triang": lambda x: x.window(time=100, samples=True, taper="triang").abs(),
        "analysis": lambda x: x.window(
            time=64, overlap=48, samples=True, analysis="hann"
        ).abs(),
        "fill": lambda x: x.window(time=100, samples=True, fill=2.5).abs(),
    }

    @pytest.mark.parametrize("name", CASES)
    def test_stays_on_the_library(self, patch, library, name):
        """Nothing is converted, and the values are NumPy's to rounding."""
        expected = self.CASES[name](patch)
        with warnings.catch_warnings():
            # A conversion through NumPy would warn, and so fail here.
            warnings.simplefilter("error", NumpyFallbackWarning)
            out = self.CASES[name](_on(patch, library))
        assert out.backend == library
        np.testing.assert_allclose(_values(out), expected.data, rtol=0, atol=1e-13)
        assert out.attrs.data_id != patch.attrs.data_id

    def test_dask_is_not_computed(self, patch, library):
        """On dask the result is still a graph: no window was computed."""
        if library != "dask":
            pytest.skip("only dask defers")
        out = _on(patch, library).window(time=100, samples=True).abs()
        assert hasattr(out.data, "compute") and not isinstance(out.data, np.ndarray)

    def test_integers(self, patch, library):
        """Integers are blended to floats without the library mixing kinds."""
        ints = patch.new(data=(patch.data * 100).astype(np.int16))
        with pytest.warns(UserWarning, match="Blending windows of int16"):
            expected = ints.window(time=100, samples=True).abs()
        with pytest.warns(UserWarning, match="Blending windows of int16"):
            out = _on(ints, library).window(time=100, samples=True).abs()
        assert out.backend == library and _values(out).dtype == expected.dtype
        np.testing.assert_allclose(_values(out), expected.data, rtol=0, atol=1e-5)

    def test_operation_without_the_library_converts_a_window(self, patch, library):
        """Only the operation with no implementation falls back, and says so."""
        expected = patch.window(time=100, samples=True).detrend("time")
        with pytest.warns(NumpyFallbackWarning, match="detrend has no"):
            out = _on(patch, library).window(time=100, samples=True).detrend("time")
        assert out.backend == library
        np.testing.assert_allclose(_values(out), expected.data, rtol=0, atol=1e-12)

    def test_kernel_which_changes_the_shape(self, patch, library):
        """A kernel which does not keep the shape it stated is caught here too."""
        with pytest.raises(PatchError, match="computed a window of shape"):
            Windowed(_Shrinks(), time=100, samples=True)(_on(patch, library))

    def test_metadata_only_operation(self, patch, library):
        """An operation which computes nothing leaves the array where it is."""
        there = _on(patch, library)
        out = there.window(time=100, samples=True).set_units("m/s")
        assert out.data is there.data

    def test_a_box_of_windows(self, patch, library):
        """Given only some windows, as a lazy chunk gives, the result is the same."""
        operation = Windowed(dc.proc.Abs(), time=64, samples=True)
        _, plan = operation.get_metadata(patch.drop_data())
        whole = operation.numpy_kernel(patch.data, **plan)
        # Windows 10 to 13 of 64 samples every 32, the first starting at -32.
        box = _on(patch, library).data[:, 288:416]
        out = operation.kernel(box, **plan, tiles=((10, 13),))
        # Samples under every one of their windows: all but the outer halves.
        inner = np.asarray(out.compute() if hasattr(out, "compute") else out)
        np.testing.assert_allclose(inner[:, 32:96], whole[:, 320:384], atol=1e-13)
