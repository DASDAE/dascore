"""Tests for windows of a patch: the declaration, reductions, and apply."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import dascore as dc
from dascore.core.source import ArraySource
from dascore.exceptions import LazyOperationError, ParameterError
from dascore.lazy.recipe import Recipe
from dascore.proc.tile_apply import TileApply
from dascore.proc.window import PatchWindow
from dascore.units import percent

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

    def test_returns_a_window(self, patch):
        """The window holds the patch and what was asked for."""
        window = patch.window(time=0.5, overlap=0.1)
        assert isinstance(window, PatchWindow)
        assert window.patch is patch
        assert dict(window.dims) == {"time": 0.5}

    def test_needs_a_dimension(self, patch):
        """A window of nothing is refused."""
        with pytest.raises(ParameterError, match="needs a dimension"):
            patch.window()

    def test_overlap_and_step_refused(self, patch):
        """The two say the same thing, so only one may be given."""
        with pytest.raises(ParameterError, match="mutually exclusive"):
            patch.window(time=0.5, overlap=0.1, step=0.1)

    @pytest.mark.parametrize("option", ({"ends": "shift"}, {"align": "middle"}))
    def test_unknown_option_refused(self, patch, option):
        """A misspelt option is refused when the window is declared."""
        with pytest.raises(ParameterError, match="must be one of"):
            patch.window(time=0.5, **option)

    def test_unknown_dimension_refused(self, patch):
        """A dimension the patch lacks is refused by the operation."""
        with pytest.raises(Exception, match="bob"):
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
        with pytest.raises(ParameterError):
            ramp.window(time=7, samples=True).mean()

    @pytest.mark.parametrize("name", REDUCERS)
    def test_named_is_the_numpy_function(self, patch, name):
        """Each named reduction is NumPy's function of each window."""
        window = patch.window(time=7, step=3, samples=True)
        named = getattr(window, name)()
        by_function = window.reduce(getattr(np, name))
        np.testing.assert_allclose(named.data, by_function.data, equal_nan=True)
        view = np.lib.stride_tricks.sliding_window_view(patch.data, 7, axis=1)
        # Output 1 is labelled by sample 3, whose window starts at sample 0.
        expected = getattr(np, name)(view[:, 0], axis=-1)
        np.testing.assert_allclose(named.data[:, 1], expected)

    @pytest.mark.parametrize("name", REDUCERS)
    @pytest.mark.parametrize("align", ("center", "right", "left"))
    def test_fast_engine_agrees(self, patch, name, align):
        """A long window, which another engine answers, gives NumPy's values."""
        window = patch.window(time=400, samples=True, align=align)
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
        """What is given after the function is passed on to it."""
        out = patch.window(time=50, samples=True, ends="drop").reduce(np.percentile, 80)
        expected = np.percentile(patch.data[:, :50], 80, axis=-1)
        np.testing.assert_allclose(out.data[:, 0], expected)

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

    def test_one_dimension_only(self, patch):
        """A reduction over two windowed dimensions is refused."""
        with pytest.raises(ParameterError):
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

    @pytest.mark.parametrize("size", (3, 4, 9, 250))
    def test_center_is_rolling_centered(self, patch, size):
        """The default label is rolling's center=True, even windows included."""
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

    def test_apply_history(self, patch):
        """An applied function is named after its window."""
        out = patch.window(time=0.5, taper="triang").apply(np.positive)
        assert out.attrs.history[-1] == (
            "window(taper='triang',time='0.5').apply(function='positive')"
        )

    @pytest.mark.parametrize(
        "change",
        ({"time": 4}, {"step": 2}, {"ends": "drop"}, {"align": "right"}),
    )
    def test_window_is_part_of_the_id(self, patch, change):
        """The same call has one id; any change to the window makes another."""
        base = {"time": 3, "samples": True}
        first = patch.window(**base).mean()
        again = patch.window(**base).mean()
        other = patch.window(**(base | change)).mean()
        assert first.attrs.data_id == again.attrs.data_id != patch.attrs.data_id
        assert other.attrs.data_id != first.attrs.data_id

    def test_reduction_is_part_of_the_id(self, patch):
        """Two reductions of one window are two results."""
        window = patch.window(time=3, samples=True)
        assert window.mean().attrs.data_id != window.std().attrs.data_id


class TestApply:
    """`apply` hands the windows to a function and blends or stacks them."""

    def test_is_the_operation(self, patch):
        """The window's apply is the tile operation with the window's options."""
        out = patch.window(time=64, overlap=16, samples=True).apply(np.negative)
        direct = TileApply(np.negative, time=64, overlap=16, samples=True)(patch)
        np.testing.assert_array_equal(out.data, direct.data)
        assert out.attrs.data_id == direct.attrs.data_id

    def test_unchanged_windows_blend_back(self, patch):
        """Windows nothing was done to sum to the patch."""
        out = patch.window(time=64, distance=16, samples=True).apply(lambda x: x)
        np.testing.assert_allclose(out.data, patch.data, atol=1e-12)

    def test_step_is_the_overlap_it_leaves(self, patch):
        """A step says what an overlap of the rest of the window says."""
        by_step = patch.window(time=64, step=48, samples=True).apply(np.negative)
        by_overlap = patch.window(time=64, overlap=16, samples=True).apply(np.negative)
        np.testing.assert_array_equal(by_step.data, by_overlap.data)

    def test_gaps_refused(self, patch):
        """Windows which leave samples uncovered cannot be blended."""
        with pytest.raises(ParameterError, match="gaps"):
            patch.window(time=64, step=100, samples=True).apply(lambda x: x)

    def test_stack(self, patch):
        """Stacked windows come back as a patch which reassembles."""
        tiles = patch.window(time=64, samples=True).apply(lambda x: x, mode="stack")
        assert tiles.dims == (*patch.dims, "time_offset")
        np.testing.assert_allclose(tiles.reassemble().data, patch.data, atol=1e-12)

    def test_fill_is_what_a_window_holds_past_the_data(self, patch):
        """The function sees the fill, and unchanged windows still blend back."""
        seen = []

        def peek(tiles):
            """Keep the first window, which starts before the data."""
            seen.append(tiles[0].copy())
            return tiles

        window = {"time": 64, "samples": True}
        patch.select(distance=0, samples=True).window(**window).apply(peek)
        filled = patch.select(distance=0, samples=True).window(**window, fill=7.0)
        out = filled.apply(peek)
        assert (seen[0][:32] == 0).all() and (seen[1][:32] == 7.0).all()
        expected = patch.select(distance=0, samples=True).data
        np.testing.assert_allclose(out.data, expected, atol=1e-12)

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
