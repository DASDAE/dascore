"""Tests for windows of a patch: the declaration, reductions, and apply."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import dascore as dc
from dascore.core.source import ArraySource
from dascore.exceptions import LazyOperationError, ParameterError
from dascore.lazy.recipe import Recipe
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

    def test_one_dimension_uses_numpys_std(self):
        """A patch of one dimension is reduced as any other: ddof of zero."""
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

    @pytest.mark.parametrize("name", REDUCERS)
    @pytest.mark.parametrize("align", ("center", "right", "left"))
    @pytest.mark.parametrize("step", (1, 3, 4))
    def test_fast_engine_agrees(self, patch, name, align, step):
        """A long window, which another engine answers, gives NumPy's values."""
        window = patch.window(time=400, step=step, samples=True, align=align)
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

    def test_fill_with_drop_changes_nothing(self, ramp):
        """Nothing is incomplete once the incomplete are dropped."""
        out = ramp.window(time=3, samples=True, ends="drop", fill=9.0).mean()
        np.testing.assert_array_equal(out.data, [1, 2, 3, 4])

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
        by_step = patch.window(time=64, step=48, samples=True).apply(np.negative)
        by_overlap = patch.window(time=64, overlap=16, samples=True).apply(np.negative)
        np.testing.assert_array_equal(by_step.data, by_overlap.data)

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
