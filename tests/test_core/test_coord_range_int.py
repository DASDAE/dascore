"""Tests for exact integer-grid range coordinates (CoordRangeInt)."""

from __future__ import annotations

import pickle
from fractions import Fraction

import numpy as np
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core.coords import (
    CoordPartial,
    CoordRange,
    CoordRangeInt,
    CoordSegmented,
    CoordSummary,
    concat_coords,
    get_coord,
)
from dascore.exceptions import CoordError

T0 = np.datetime64("2020-01-01T00:00:00.000000000")
ONE_S = np.timedelta64(1, "s")


def _floor_labels(start_tick, num, den, offset, count):
    """The labels the plan defines: start + floor((offset + i*num) / den)."""
    i = np.arange(count, dtype=np.int64)
    return start_tick + (offset + i * num) // den


@pytest.fixture(scope="session")
def hz_1024():
    """An hour at 1024 Hz: the drifting case."""
    return get_coord(start=T0, step=(1, 1024), shape=(3_686_400,))


@pytest.fixture(scope="session")
def hz_3000():
    """Ten seconds at 3000 Hz: a step that is not a binary fraction."""
    return get_coord(start=T0, step=Fraction(1, 3000), shape=(30_000,))


@pytest.fixture(scope="session")
def int_frac():
    """An integer grid with a three-halves step."""
    return get_coord(start=-7, step=(3, 2), shape=(50,), units="m")


class TestDispatch:
    """get_coord picks the exact class for time and integer ranges."""

    def test_time_scalar_step(self):
        """A timedelta step on a datetime start makes a whole-tick exact grid."""
        coord = get_coord(start=T0, step=ONE_S, shape=(10,))
        assert isinstance(coord, CoordRangeInt)
        assert coord.step_denominator == 1
        assert coord.step == ONE_S

    def test_timedelta(self):
        """A timedelta coordinate is an exact grid in its own unit."""
        coord = get_coord(start=np.timedelta64(0, "ns"), step=ONE_S, shape=(3,))
        assert isinstance(coord, CoordRangeInt)
        assert coord.dtype == np.dtype("timedelta64[ns]")
        assert coord.step_exact == 1

    def test_int(self):
        """Integer ranges are exact grids."""
        coord = get_coord(start=0, stop=10, step=2)
        assert isinstance(coord, CoordRangeInt)
        assert np.array_equal(coord.values, [0, 2, 4, 6, 8])

    def test_float_stays_range(self):
        """Float ranges keep the plain class."""
        coord = get_coord(start=0.0, stop=10, step=2.5)
        assert type(coord) is CoordRange

    def test_int_with_float_step_stays_range(self):
        """An integer start with a float step is a float range, as before."""
        coord = get_coord(start=0, stop=10, step=2.5)
        assert type(coord) is CoordRange

    def test_int_uneven_span_stays_float(self):
        """Span/count that does not divide makes floats, as before."""
        coord = get_coord(start=0, stop=10, shape=3)
        assert type(coord) is CoordRange
        assert np.allclose(coord.values, [0, 10 / 3, 20 / 3])
        even = get_coord(start=0, stop=12, shape=3)
        assert isinstance(even, CoordRangeInt)
        assert np.array_equal(even.values, [0, 4, 8])

    def test_fraction_on_float_raises(self):
        """A fraction step has no meaning on a float coordinate."""
        with pytest.raises(CoordError, match="fraction step"):
            get_coord(start=0.0, step=(1, 3), shape=(3,))

    def test_sub_nanosecond_stays_range(self):
        """Finer than nanosecond time keeps the legacy class."""
        start = np.datetime64("2020-01-01T00:00:00.000000000001")
        step = np.timedelta64(1000, "ps")
        coord = get_coord(start=start, stop=start + step * 10, step=step)
        assert type(coord) is CoordRange

    def test_coarse_units_kept(self):
        """A microsecond coordinate stays microseconds and works past 2262."""
        start = np.datetime64("2500-01-01T00:00:00", "us")
        coord = get_coord(start=start, step=np.timedelta64(2, "us"), shape=(5,))
        assert isinstance(coord, CoordRangeInt)
        assert coord.dtype == np.dtype("datetime64[us]")
        assert coord.values[-1] == start + np.timedelta64(8, "us")
        assert coord.step_exact == Fraction(2, 10**6)

    def test_coarse_start_fine_step_resolves_finer(self):
        """A scalar step resolves the unit as start + step does."""
        coord = get_coord(start=np.datetime64("2020-01-01"), step=ONE_S, shape=(3,))
        assert coord.dtype == np.dtype("datetime64[s]")

    def test_fraction_step_with_coarse_start_uses_nanoseconds(self):
        """A fraction step is in seconds, so nanoseconds hold it."""
        day = np.datetime64("2020-01-01")
        coord = get_coord(start=day, step=(1, 1024), shape=(4,))
        assert coord.dtype == np.dtype("datetime64[ns]")
        assert coord.step_exact == Fraction(1, 1024)

    def test_legacy_construction_untouched(self):
        """CoordRange built directly still accepts time and has no grid."""
        coord = CoordRange(start=T0, step=ONE_S, shape=(10,))
        assert type(coord) is CoordRange
        assert coord.step_exact == 1

    def test_data_path(self):
        """Evenly spaced integer and time arrays become exact grids."""
        coord = get_coord(data=np.arange(10) * 3)
        assert isinstance(coord, CoordRangeInt)
        time = get_coord(data=T0 + np.arange(5) * ONE_S)
        assert isinstance(time, CoordRangeInt)

    def test_single_with_step(self):
        """A one-sample array with a step is an exact grid."""
        coord = get_coord(data=[5], step=2)
        assert isinstance(coord, CoordRangeInt)
        assert len(coord) == 1 and coord.stop == 7

    def test_partial_fallbacks(self):
        """Under-specified inputs are partial coordinates, as before."""
        assert isinstance(get_coord(start=10, shape=10), CoordPartial)
        assert isinstance(get_coord(shape=10, step=1), CoordPartial)


class TestConstruction:
    """The validator resolves start, stop, step, and shape combinations."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(start=0, stop=10, step=2),
            dict(start=0, step=2, shape=(5,)),
            dict(stop=10, step=2, shape=(5,)),
            dict(start=0, stop=10, shape=(5,)),
        ],
    )
    def test_three_of_four(self, kwargs):
        """Any three of start, stop, step, and shape define the range."""
        coord = get_coord(**kwargs)
        assert np.array_equal(coord.values, [0, 2, 4, 6, 8])
        assert coord.stop == 10

    def test_stop_and_shape_fraction(self):
        """Start is the floor of the ideal origin count steps before stop."""
        coord = get_coord(stop=10, step=(3, 2), shape=(4,))
        # ideal origin = 10 - 4 * 1.5 = 4
        assert coord.start == 4 and coord.origin_offset == 0
        assert np.array_equal(coord.values, [4, 5, 7, 8])
        coord = get_coord(stop=11, step=(3, 2), shape=(4,))
        # ideal origin 5; labels 5, 6.5->6, 8, 9.5->9
        assert np.array_equal(coord.values, [5, 6, 8, 9])

    def test_fraction_and_tuple_agree(self):
        """A Fraction and a tuple state the same step."""
        a = get_coord(start=T0, step=(1, 1024), shape=(100,))
        b = get_coord(start=T0, step=Fraction(1, 1024), shape=(100,))
        assert a == b

    def test_step_rounds_ties_to_even(self, hz_1024):
        """976562.5 ns rounds to 976562 ns, matching the old np.round."""
        assert hz_1024.step == np.timedelta64(976562, "ns")
        odd = get_coord(start=0, step=(5, 2), shape=(3,))  # 2.5 -> 2
        assert odd.step == 2
        assert np.array_equal(odd.values, [0, 2, 5])

    def test_below_one_tick_raises(self):
        """A step below one tick would repeat labels."""
        with pytest.raises(ValidationError, match="smaller than one tick"):
            get_coord(start=0, step=(1, 3), shape=(3,))
        with pytest.raises(ValidationError, match="smaller than one tick"):
            get_coord(start=0, stop=10, step=(1, 3))

    def test_bad_sign_raises(self):
        """The step sign must match the span."""
        with pytest.raises(ValidationError, match="Sign of step"):
            get_coord(start=0, stop=10, step=-1)

    def test_zero_count_is_partial(self):
        """A zero-length shape is a partial coordinate."""
        coord = get_coord(start=0, step=1, shape=(0,))
        assert isinstance(coord, CoordPartial)

    @pytest.mark.parametrize("ms", [10_020, 10_060, 9_990])
    def test_stop_rounding_parity(self, ms):
        """A stop stated a hair off a sample counts as CoordRange always has."""
        legacy = CoordRange(start=0.0, stop=ms / 1000, step=1.0)
        exact = get_coord(start=T0, stop=T0 + np.timedelta64(ms, "ms"), step=ONE_S)
        assert len(exact) == len(legacy)

    def test_start_equals_stop(self):
        """Equal start and stop give one sample, as before."""
        coord = get_coord(start=5, stop=5, step=1)
        assert len(coord) == 1

    def test_zero_step(self):
        """A zero step is a single repeated label."""
        coord = get_coord(start=5, stop=5, step=0)
        assert np.array_equal(coord.values, [5])
        assert np.array_equal(coord.select((4, 6))[0].values, [5])
        assert isinstance(coord.select((6, 7))[0], CoordPartial)

    def test_overflow_raises(self):
        """A grid past the int64 range is refused at construction."""
        with pytest.raises(ValidationError, match="int64"):
            get_coord(start=2**62, step=2**40, shape=(2**30,))

    def test_grid_normalized_jointly(self):
        """(6, 2, 1) must not reduce to (3, 1, 0): that moves the origin."""
        coord = get_coord(start=0, step=(3, 2), shape=(20,))[1::2]
        assert (coord.step_numerator, coord.step_denominator) == (6, 2)
        assert coord.origin_offset == 1
        assert np.array_equal(coord.values, [1, 4, 7, 10, 13, 16, 19, 22, 25, 28])
        strided = get_coord(start=0, step=(3, 2), shape=(20,))[::2]
        assert (strided.step_numerator, strided.step_denominator) == (3, 1)
        odd_origin = get_coord(start=0, step=(3, 2), shape=(20,))[1:]
        assert (odd_origin.step_numerator, odd_origin.step_denominator) == (3, 2)
        assert odd_origin.origin_offset == 1
        assert (odd_origin[::2].step_numerator, odd_origin[::2].step_denominator) == (
            6,
            2,
        )
        assert odd_origin[::2].origin_offset == 1


class TestLabels:
    """Labels come from the ideal grid, quantized once."""

    def test_matches_floor_definition(self, hz_1024):
        """Labels are start + floor((offset + i * num) / den)."""
        labels = hz_1024.values.astype(np.int64)
        expected = _floor_labels(int(T0.astype(np.int64)), 1953125, 2, 0, len(hz_1024))
        assert np.array_equal(labels, expected)

    def test_no_drift_over_an_hour(self, hz_1024):
        """A rounded 976562 ns step drifts 1.8 ms an hour; the grid does not."""
        assert hz_1024.max() == T0 + np.timedelta64(3600 * 10**9 - 976563, "ns")
        assert hz_1024.stop == T0 + np.timedelta64(1, "h")

    def test_3000_hz(self, hz_3000):
        """A third of a microsecond is held exactly."""
        assert hz_3000.step_exact == Fraction(1, 3000)
        assert hz_3000.stop == T0 + np.timedelta64(10, "s")
        assert hz_3000[-1] == T0 + np.timedelta64(10 * 10**9 - 333_334, "ns")

    def test_negative_origin(self, int_frac):
        """Floor quantization holds below zero."""
        expected = _floor_labels(-7, 3, 2, 0, 50)
        assert np.array_equal(int_frac.values, expected)
        assert int_frac.min() == -7

    def test_getitem_int(self, hz_3000):
        """Integer indexing evaluates one label."""
        assert hz_3000[5] == hz_3000.values[5]
        assert hz_3000[-1] == hz_3000.values[-1]
        with pytest.raises(IndexError):
            _ = hz_3000[len(hz_3000)]

    def test_min_max_descending(self):
        """Min and max are the end labels whichever way the grid runs."""
        coord = get_coord(start=10, stop=0, step=-3)
        assert (coord.min(), coord.max()) == (1, 10)
        assert coord.reverse_sorted and not coord.sorted

    def test_index_values_negative(self, hz_1024):
        """Negative indices count from the end."""
        vals = hz_1024._get_index_values(np.array([-1, 0]))
        assert vals[0] == hz_1024.max() and vals[1] == hz_1024.min()


class TestSlicing:
    """Slices reproduce the materialized original exactly."""

    @pytest.mark.parametrize(
        "sl",
        [
            slice(None),
            slice(1, None),
            slice(3, 1000, 7),
            slice(None, None, -1),
            slice(-5, None),
            slice(999, 3, -5),
            slice(2, 2000, 3),
        ],
    )
    def test_slice_equals_values(self, hz_1024, hz_3000, int_frac, sl):
        """Slicing the coordinate equals slicing its values."""
        for coord in (hz_1024, hz_3000, int_frac):
            sub = coord[sl]
            expected = coord.values[sl]
            if len(expected) == 0:
                assert isinstance(sub, CoordPartial)
                continue
            assert isinstance(sub, CoordRangeInt)
            assert np.array_equal(sub.values, expected)
            assert sub.dtype == coord.dtype

    def test_nested(self, hz_1024):
        """Nested slices and reversals reproduce the values exactly."""
        a = hz_1024[100:3000][7:2000:3]
        assert np.array_equal(a.values, hz_1024.values[100:3000][7:2000:3])
        b = hz_1024[::-1][::2]
        assert np.array_equal(b.values, hz_1024.values[::-1][::2])
        assert b.step_exact == Fraction(-1, 512)

    def test_ideal_positions_survive(self, hz_1024):
        """The origin offset of a slice is compared, not only its labels."""
        sub = hz_1024[1:]
        assert sub.origin_offset == 1 and sub.step_denominator == 2
        assert sub._ideal_origin == hz_1024._ideal_origin + Fraction(1953125, 2)

    def test_random_nested_slices_against_oracle(self):
        """Random grids and nested slices agree with fraction arithmetic."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            num = int(rng.integers(-40, 40)) or 7
            den = int(rng.integers(1, 12))
            if abs(num) < den:
                num = den * int(np.sign(num) or 1)
            offset = int(rng.integers(0, den))
            count = int(rng.integers(1, 60))
            start = int(rng.integers(-1000, 1000))
            coord = CoordRangeInt(
                start=start,
                shape=(count,),
                step_numerator=num,
                step_denominator=den,
                origin_offset=offset,
            )
            values = _floor_labels(start, num, den, offset, count)
            assert np.array_equal(coord.values, values)
            for _ in range(3):
                a, b = sorted(rng.integers(-count - 2, count + 2, size=2))
                s = int(rng.integers(-4, 5)) or 1
                sl = slice(int(a), int(b), s)
                expected = values[sl]
                sub = coord[sl]
                if len(expected) == 0:
                    assert len(sub) == 0
                    break
                assert np.array_equal(sub.values, expected)
                coord, values = sub, expected

    def test_slice_is_metadata_only(self):
        """A billion-sample grid slices without evaluating its labels."""
        coord = get_coord(start=T0, step=(1, 1024), shape=(10**9,))
        sub = coord[10**8 : 9 * 10**8 : 1000]
        assert len(sub) == 800_000
        assert sub.min() == coord[10**8]
        rev = coord[::-1]
        assert rev.max() == coord.max()
        assert coord.max() == T0 + np.timedelta64((10**9 - 1) * 1953125 // 2, "ns")

    def test_empty_slice_is_partial(self, hz_1024):
        """An empty slice is a typed partial coordinate."""
        out = hz_1024[5:5]
        assert isinstance(out, CoordPartial)
        assert out.dtype == hz_1024.dtype

    def test_array_getitem(self, int_frac):
        """Indexing with an array selects those labels."""
        out = int_frac[np.array([1, 3, 9])]
        assert np.array_equal(out.values, int_frac.values[[1, 3, 9]])

    def test_sort(self, hz_1024):
        """Sorting reverses the grid and back."""
        rev, sl = hz_1024.sort(reverse=True)
        assert sl == slice(None, None, -1)
        assert np.array_equal(rev.values, hz_1024.values[::-1])
        back, _ = rev.sort()
        assert back == hz_1024

    def test_change_length(self, hz_1024):
        """Changing the length keeps the grid and offset."""
        longer = hz_1024.change_length(len(hz_1024) + 10)
        assert np.array_equal(longer.values[: len(hz_1024)], hz_1024.values)
        assert hz_1024[1:].change_length(5).origin_offset == 1


class TestSelect:
    """Value selection is exact on the grid."""

    def test_time_window(self, hz_1024):
        """A window between two on-grid times selects exactly those samples."""
        one, two = T0 + ONE_S, T0 + 2 * ONE_S
        sub, sl = hz_1024.select((one, two))
        assert sl == slice(1024, 2049)
        assert sub.min() == one and sub.max() == two

    def test_between_samples(self, hz_1024):
        """Bounds between samples exclude the samples they fall past."""
        lo = hz_1024[10] + np.timedelta64(1, "ns")
        hi = hz_1024[20] - np.timedelta64(1, "ns")
        sub, sl = hz_1024.select((lo, hi))
        assert sl == slice(11, 20)
        assert np.array_equal(sub.values, hz_1024.values[11:20])

    def test_float_bounds_on_int_grid(self, int_frac):
        """Float bounds on an integer grid round toward the selection."""
        sub, _ = int_frac.select((0.5, 10.5))
        values = int_frac.values
        assert np.array_equal(sub.values, values[(values >= 0.5) & (values <= 10.5)])

    def test_infinite_bounds(self, int_frac):
        """Infinite bounds open one side."""
        sub, _ = int_frac.select((3.0, np.inf))
        assert sub.min() >= 3 and sub.max() == int_frac.max()
        sub, _ = int_frac.select((-np.inf, 3))
        assert sub.min() == int_frac.min() and sub.max() <= 3

    def test_descending(self):
        """Selection on a descending grid."""
        coord = get_coord(start=10, stop=0, step=-3)  # 10 7 4 1
        sub, sl = coord.select((2, 8))
        assert np.array_equal(sub.values, [7, 4])
        assert sl == slice(1, 3)

    def test_descending_fraction(self):
        """Selection on a descending fractional grid."""
        coord = get_coord(start=T0, step=(1, 1024), shape=(2048,))[::-1]
        lo, hi = coord[1500], coord[100]
        sub, sl = coord.select((lo, hi))
        assert sl == slice(100, 1501)
        assert sub.max() == hi and sub.min() == lo

    def test_degenerate(self, int_frac):
        """A window past the coordinate is degenerate."""
        sub, sl = int_frac.select((1000, 2000))
        assert sl == slice(0, 0)
        assert isinstance(sub, CoordPartial)

    def test_open_ended(self, hz_1024):
        """None and Ellipsis open a side."""
        _, sl = hz_1024.select((None, hz_1024[10]))
        assert sl == slice(None, 11)
        _, sl = hz_1024.select((hz_1024[10], ...))
        assert sl == slice(10, None)

    def test_relative(self, hz_1024):
        """Relative bounds are durations from the ends."""
        _, sl = hz_1024.select((1, 2), relative=True)
        assert sl == slice(1024, 2049)

    def test_samples(self, hz_1024):
        """Sample bounds are positions."""
        sub, _ = hz_1024.select((10, 20), samples=True)
        assert np.array_equal(sub.values, hz_1024.values[10:20])

    def test_array_bounds(self, int_frac):
        """The array path of _get_index agrees with the scalar path."""
        probes = np.array([-100, -7, -6, 0, 1, 2, 60, 100])
        for forward in (True, False):
            arr = int_frac._get_index(probes, forward=forward)
            for probe, got in zip(probes, arr):
                scalar = int_frac._index_of(int(probe), forward)
                assert got == scalar

    def test_get_next_index(self, hz_1024):
        """The next index past a value between samples."""
        idx = hz_1024.get_next_index(hz_1024[100] + np.timedelta64(1, "ns"))
        assert idx == 101


class TestUpdateLimits:
    """update_limits translates or re-grids, and keeps the exact grid."""

    def test_min_translates(self, hz_1024):
        """A new min translates the grid."""
        out = hz_1024.update_limits(min=T0 + ONE_S)
        assert out.min() == T0 + ONE_S
        assert out.step_exact == hz_1024.step_exact
        assert len(out) == len(hz_1024)

    def test_max_translates(self, hz_1024):
        """A new max translates the grid."""
        out = hz_1024.update_limits(max=T0)
        assert out.max() == T0
        assert np.array_equal(
            out.values - out.values[0], hz_1024.values - hz_1024.values[0]
        )

    def test_descending_min_max(self):
        """Limits on a descending grid are inclusive and keep the direction."""
        coord = get_coord(start=10, stop=0, step=-1)
        assert coord.update_limits(max=20).values.tolist() == list(range(20, 10, -1))
        assert coord.update_limits(min=100).values.tolist() == list(range(109, 99, -1))

    def test_step_scalar(self, hz_1024):
        """A scalar step re-grids from the same start."""
        out = hz_1024.update_limits(step=np.timedelta64(2, "ms"))
        assert out.step_exact == Fraction(1, 500)
        assert out.min() == hz_1024.min() and len(out) == len(hz_1024)

    def test_step_fraction(self, hz_1024):
        """A fraction step re-grids from the same start."""
        out = hz_1024.update_limits(step=(1, 512))
        assert out.step_exact == Fraction(1, 512)

    def test_step_and_min(self, hz_1024):
        """A step and a min apply in turn."""
        out = hz_1024.update_limits(step=(1, 512), min=T0 + ONE_S)
        assert out.step_exact == Fraction(1, 512) and out.min() == T0 + ONE_S

    def test_both_limits_exact(self):
        """Both limits keep the count; the spacing becomes exact."""
        coord = get_coord(start=T0, step=ONE_S, shape=(10,))
        out = coord.update_limits(min=T0, max=T0 + np.timedelta64(65, "s"))
        assert len(out) == 10
        assert out.stop == T0 + np.timedelta64(65, "s")
        assert out.step_exact == Fraction(65, 10)

    def test_both_limits_below_tick_makes_floats(self):
        """Both limits closer than a tick per sample make floats, as before."""
        coord = get_coord(start=0, stop=10, step=1)
        out = coord.update_limits(min=2, max=8)
        assert type(out) is CoordRange and len(out) == 10

    def test_too_many(self, hz_1024):
        """All three parameters is an error."""
        with pytest.raises(ValueError, match="At most two"):
            hz_1024.update_limits(min=T0, max=T0, step=ONE_S)

    def test_update_via_coordmanager(self, hz_1024):
        """The patch-level update keeps the exact grid."""
        patch = dc.get_example_patch().update_coords(time=hz_1024.change_length(2000))
        out = patch.update_coords(time_min=T0 + ONE_S)
        assert out.get_coord("time").step_exact == Fraction(1, 1024)
        assert out.get_coord("time").min() == T0 + ONE_S


class TestNewAndRoundTrips:
    """Dumps, summaries, pickles, and `new` keep the grid."""

    def test_model_dump_round_trip(self, hz_1024):
        """A dumped coordinate rebuilds itself through get_coord."""
        sub = hz_1024[1:]
        assert get_coord(**sub.model_dump()) == sub

    def test_new_translation_keeps_offset(self, hz_1024):
        """New with a shifted start and stop keeps the grid."""
        sub = hz_1024[1:]
        out = sub.new(start=sub.start + ONE_S, stop=sub.stop + ONE_S)
        assert out.origin_offset == 1 and out.step_exact == sub.step_exact
        assert len(out) == len(sub)

    def test_new_stop_changes_count(self, hz_1024):
        """New with a stop re-derives the count."""
        out = hz_1024.new(stop=hz_1024[100])
        assert len(out) == 100

    def test_new_step_replaces_grid(self, hz_1024):
        """New with a step drops the old grid."""
        out = hz_1024[1:].new(step=ONE_S)
        assert out.step_denominator == 1 and out.origin_offset == 0
        assert out.step == ONE_S

    def test_new_units(self, int_frac):
        """New with units keeps the grid."""
        out = int_frac.new(units="ft")
        assert out.units == dc.get_quantity("ft") and out.step_exact == Fraction(3, 2)

    def test_summary(self, hz_1024):
        """The summary carries the grid and rebuilds the coordinate."""
        summary = hz_1024[1:].to_summary(dims=("time",))
        assert summary.is_exact_grid
        assert summary.step_numerator == 1953125
        assert summary.step_denominator == 2
        assert summary.origin_offset == 1
        assert summary.to_coord() == hz_1024[1:]

    def test_summary_descending(self, hz_1024):
        """A descending grid survives the summary."""
        rev = hz_1024[::-1]
        assert rev.to_summary().to_coord() == rev

    def test_summary_without_grid_rebuilds_exact_class(self):
        """A legacy summary rebuilds onto the exact class."""
        summary = CoordSummary(min=T0, max=T0 + 9 * ONE_S, step=ONE_S, len=10)
        coord = summary.to_coord()
        assert isinstance(coord, CoordRangeInt) and len(coord) == 10

    def test_pickle(self, hz_1024):
        """Pickling keeps the grid."""
        out = pickle.loads(pickle.dumps(hz_1024[3:]))
        assert out == hz_1024[3:]

    def test_patch_round_trip_through_coords(self, hz_1024):
        """A patch holding a fractional grid selects on it exactly."""
        patch = dc.get_example_patch().update_coords(time=hz_1024[:2000])
        assert patch.get_coord("time").step_exact == Fraction(1, 1024)
        sub = patch.select(time=(T0 + ONE_S, None))
        assert sub.get_coord("time").min() == T0 + ONE_S
        assert np.array_equal(
            sub.get_coord("time").values, patch.get_coord("time").values[1024:]
        )


class TestIdentity:
    """Equality and fingerprints compare the grid, never the class."""

    def test_legacy_fingerprint_parity(self):
        """A whole-tick grid fingerprints as the legacy class does."""
        legacy = CoordRange(start=T0, step=ONE_S, shape=(10,))
        exact = get_coord(start=T0, step=ONE_S, shape=(10,))
        assert legacy.fingerprint() == exact.fingerprint()
        legacy = CoordRange(start=0, stop=10, step=2, units="m")
        exact = get_coord(start=0, stop=10, step=2, units="m")
        assert legacy.fingerprint() == exact.fingerprint()

    def test_fraction_grid_changes_fingerprint(self, hz_1024):
        """A fractional grid is a different identity from its rounding."""
        rounded = get_coord(start=T0, step=hz_1024.step, shape=(len(hz_1024),))
        assert rounded.fingerprint() != hz_1024.fingerprint()
        assert hz_1024[1:].fingerprint() != hz_1024[:-1].fingerprint()

    def test_fingerprint_stable(self, hz_1024):
        """The fingerprint survives a no-op slice and a summary round trip."""
        assert hz_1024.fingerprint() == hz_1024[:].fingerprint()
        assert hz_1024.fingerprint() == hz_1024.to_summary().to_coord().fingerprint()

    def test_equality_is_field_equality(self):
        """== stays pydantic field equality."""
        legacy = CoordRange(start=0, stop=10, step=2)
        exact = get_coord(start=0, stop=10, step=2)
        assert exact == get_coord(data=np.arange(0, 10, 2))
        assert legacy != exact  # different classes, as pydantic sees it


class TestUnits:
    """Unit conversion of integer grids."""

    def test_time_units_fixed(self, hz_1024):
        """Time coordinates do not convert units."""
        assert hz_1024.convert_units("s") is hz_1024
        assert hz_1024.units == dc.get_quantity("s")

    def test_whole_tick_converts_to_floats(self):
        """A whole-tick integer grid converts to floats, as before."""
        coord = get_coord(start=0, stop=10, step=2, units="m")
        out = coord.convert_units("km")
        assert type(out) is CoordRange
        assert np.allclose(out.values, coord.values / 1000)

    def test_fraction_grid_refuses(self, int_frac):
        """A fractional integer grid refuses conversion but accepts set_units."""
        with pytest.raises(CoordError, match="fractional step"):
            int_frac.convert_units("km")
        assert int_frac.set_units("ft").units == dc.get_quantity("ft")

    def test_fraction_grid_fingerprints(self, int_frac):
        """A fractional grid fingerprints without converting units."""
        assert int_frac.fingerprint() == int_frac.set_units("m").fingerprint()


class TestRepr:
    """The class split does not leak into text."""

    def test_header_is_coord_range(self, hz_1024):
        """The repr header reads CoordRange and states the exact step."""
        text = str(hz_1024)
        assert text.startswith("CoordRange(")
        assert "CoordRangeInt" not in text
        assert "1/1024 s" in text and "1024 Hz" in text

    def test_whole_tick_repr_unchanged(self):
        """A whole-tick grid prints as the legacy class does."""
        legacy = CoordRange(start=T0, step=ONE_S, shape=(10,))
        exact = get_coord(start=T0, step=ONE_S, shape=(10,))
        assert str(legacy) == str(exact)

    def test_int_fraction_repr(self, int_frac):
        """An integer fraction step is stated with its units."""
        assert "3/2 m" in str(int_frac)

    def test_coord_manager_repr(self, hz_1024):
        """The coordinate manager states the range kind, not the class."""
        patch = dc.get_example_patch().update_coords(time=hz_1024[:2000])
        assert "CoordRangeInt" not in str(patch.coords)


class TestStepExact:
    """step_exact on every coordinate kind."""

    def test_legacy_range(self):
        """Timedelta and integer steps are exact; float steps are not."""
        assert CoordRange(start=T0, step=ONE_S, shape=(3,)).step_exact == 1
        assert CoordRange(start=0, stop=10, step=2).step_exact == 2
        assert CoordRange(start=0.0, stop=1.0, step=0.1).step_exact is None

    def test_partial_and_array(self):
        """Coordinates without a step have no exact step."""
        assert get_coord(shape=(3,)).step_exact is None
        assert get_coord(data=[1.0, 2.5, 3.0]).step_exact is None

    def test_time_units_are_seconds(self, hz_1024):
        """step_exact of a time coordinate is in seconds."""
        assert hz_1024.step_exact == Fraction(1, 1024)
        start = np.datetime64("2020-01-01", "ms")
        ms = get_coord(start=start, step=np.timedelta64(4, "ms"), shape=(3,))
        assert ms.step_exact == Fraction(4, 1000)


class TestSegments:
    """Exact grids inside segmented coordinates."""

    def test_adjacent_fraction_runs_fuse(self, hz_1024):
        """Adjacent runs of one grid fuse back into it."""
        a, b = hz_1024[:1000], hz_1024[1000:2000]
        out = concat_coords(a, b)
        assert isinstance(out, CoordRangeInt)
        assert out == hz_1024[:2000]

    def test_off_grid_neighbours_do_not_fuse(self, hz_1024):
        """A run a tick off the grid stays a separate segment."""
        a = hz_1024[:1000]
        b = hz_1024[1000:2000].update_limits(min=a.stop + np.timedelta64(1, "ns"))
        out = concat_coords(a, b)
        assert isinstance(out, CoordSegmented)

    def test_same_labels_different_origin_do_not_fuse(self):
        """Labels can align while the ideal origins do not."""
        a = get_coord(start=0, step=(3, 2), shape=(4,))  # 0 1 3 4, stop 6
        b = get_coord(start=6, step=(3, 2), shape=(4,))[:]
        fused = concat_coords(a, b)
        assert isinstance(fused, CoordRangeInt)
        shifted = CoordRangeInt(
            start=6, shape=(4,), step_numerator=3, step_denominator=2, origin_offset=1
        )
        out = concat_coords(a, shifted)
        assert isinstance(out, CoordSegmented)

    def test_segment_dump_round_trip(self, hz_1024):
        """Segments rebuild onto the exact class."""
        seg = concat_coords(hz_1024[:10], hz_1024[20:30])
        assert isinstance(seg, CoordSegmented)
        rebuilt = get_coord(**seg.model_dump())
        assert rebuilt == seg
        assert all(isinstance(s, CoordRangeInt) for s in rebuilt.segments)


class TestPersistenceGuards:
    """Formats that cannot hold the grid say so instead of rounding it."""

    def test_dasdae_v1_refuses_fraction(self, hz_1024, tmp_path):
        """DASDAE format 1 refuses a fractional step rather than rounding it."""
        patch = dc.get_example_patch().update_coords(time=hz_1024[:2000])
        with pytest.raises(NotImplementedError, match="fractional step"):
            patch.io.write(tmp_path / "frac.h5", "dasdae")

    def test_dasdae_v1_writes_whole_ticks(self, tmp_path):
        """Whole-tick grids round-trip through DASDAE."""
        patch = dc.get_example_patch()
        assert isinstance(patch.get_coord("time"), CoordRangeInt)
        path = tmp_path / "whole.h5"
        patch.io.write(path, "dasdae")
        assert dc.spool(path)[0].get_coord("time") == patch.get_coord("time")

    def test_xarray_lazy_index_skipped(self, hz_1024):
        """A fractional grid travels to xarray as values, not a rounded index."""
        pytest.importorskip("xarray")
        from dascore.xarray.patch import patch_to_xarray  # noqa: PLC0415

        patch = dc.get_example_patch().update_coords(time=hz_1024[:2000])
        array = patch_to_xarray(patch, lazy_coords={"time"})
        assert type(array.xindexes["time"]).__name__ != "TemporalRangeIndex"
        assert np.array_equal(array["time"].values, patch.get_coord("time").values)


class TestValidationErrors:
    """Every refusal of the exact class states why."""

    def test_summary_needs_length(self, hz_1024):
        """An exact summary without a length cannot rebuild."""
        summary = hz_1024.to_summary().model_copy(update=dict(len=None))
        with pytest.raises(CoordError, match="length"):
            summary.to_coord()

    def test_value_off_grid(self):
        """A value finer than the tick does not sit on the grid."""
        start = np.datetime64("2020-01-01T00:00:00.000001", "us")
        coord = get_coord(start=start, step=np.timedelta64(1, "us"), shape=(4,))
        with pytest.raises(CoordError, match="does not sit"):
            coord._new_grid(start + np.timedelta64(1, "ns"), coord.step, 4)

    def test_non_time_value_on_time_grid(self):
        """A value that is not a time at all is refused."""
        with pytest.raises(ValidationError, match="not a"):
            CoordRangeInt(start=T0, stop="tomorrow", step=(1, 2))

    def test_non_integer_float(self):
        """A non-integer float cannot be a tick of an integer grid."""
        with pytest.raises(ValidationError, match="non-integer"):
            CoordRangeInt(start=0, stop=10, step=2.5)

    def test_month_unit(self):
        """A month is not a fixed number of seconds; a fraction step uses ns."""
        start = np.datetime64("2020-01", "M")
        from dascore.core.coords import _range_class  # noqa: PLC0415

        assert _range_class(start, None, np.timedelta64(1, "M"), (3,)) is CoordRange
        exact = get_coord(start=start, step=(1, 2), shape=(3,))
        assert exact.dtype == np.dtype("datetime64[ns]")

    def test_direct_construction_errors(self):
        """The class itself refuses what get_coord would turn into a partial."""
        with pytest.raises(ValidationError, match="requires start or stop"):
            CoordRangeInt(step=1, shape=(3,))
        with pytest.raises(ValidationError, match="1D"):
            CoordRangeInt(start=0, step=1, shape=(2, 3))
        with pytest.raises(ValidationError, match="at least one sample"):
            CoordRangeInt(start=0, step=1, shape=(0,))
        with pytest.raises(ValidationError, match="Three of"):
            CoordRangeInt(start=0, shape=(3,))
        with pytest.raises(ValidationError, match=r"Three of|are needed"):
            CoordRangeInt(start=0, step=1)
        with pytest.raises(ValidationError, match="positive"):
            CoordRangeInt(start=0, shape=(3,), step_numerator=1, step_denominator=-2)
        with pytest.raises(ValidationError, match="origin_offset"):
            CoordRangeInt(
                start=0,
                shape=(3,),
                step_numerator=3,
                step_denominator=2,
                origin_offset=2,
            )
        with pytest.raises(ValidationError, match="fixed tick"):
            month = np.datetime64("2020-01", "M")
            CoordRangeInt(start=month, step=np.timedelta64(1, "M"), shape=(3,))

    def test_update_limits_step_below_tick(self, int_frac):
        """A new step below one tick is refused."""
        with pytest.raises(CoordError, match="smaller than one tick"):
            int_frac.update_limits(step=(1, 3))

    def test_new_grid_helper(self, int_frac):
        """The parent's grid constructor builds a whole-tick grid."""
        out = int_frac._new_grid(4, 2, 5)
        assert np.array_equal(out.values, [4, 6, 8, 10, 12])

    def test_descending_array_index(self):
        """The vectorized index of a descending whole-tick grid."""
        coord = get_coord(start=10, stop=0, step=-3)  # 10 7 4 1
        probes = np.array([11, 10, 8, 7, 1, 0])
        for forward in (True, False):
            got = coord._get_index(probes, forward=forward)
            expected = [coord._index_of(int(p), forward) for p in probes]
            assert got.tolist() == expected

    def test_integral_float_is_a_tick(self):
        """A float that is a whole number sits on an integer grid."""
        coord = CoordRangeInt(start=0, stop=10.0, step=2)
        assert np.array_equal(coord.values, [0, 2, 4, 6, 8])

    def test_time_units_do_not_convert(self, hz_1024):
        """Time coordinates hand themselves back from unit conversion."""
        assert hz_1024._convert_units("ms") is hz_1024

    def test_float_range_failure_is_still_partial(self):
        """A float range that cannot validate stays a partial, as before."""
        out = get_coord(start=0.0, stop=10.0, step=-1.0, shape=(10,))
        assert isinstance(out, CoordPartial)


class TestUnitHelpers:
    """Time unit parsing behind the exact grid."""

    def test_generic_unit_has_no_tick(self):
        """A generic time dtype states no unit and so no tick."""
        from dascore.core.coords import _seconds_per_tick, _time_unit  # noqa: PLC0415

        assert _time_unit(np.dtype("datetime64")) == ("generic", 1)
        assert _seconds_per_tick(np.dtype("datetime64")) is None
        assert _seconds_per_tick(np.dtype("datetime64[10us]")) == Fraction(1, 10**5)

    def test_mixed_kinds_stay_legacy(self):
        """A time start with a non-time stop is left to the legacy validator."""
        from dascore.core.coords import _range_class  # noqa: PLC0415

        assert _range_class(T0, "tomorrow", ONE_S, None) is CoordRange


class TestLegacyCoordRange:
    """CoordRange built directly keeps its time and integer behaviour."""

    @pytest.fixture(scope="class")
    def legacy_time(self):
        """A time CoordRange built without get_coord."""
        return CoordRange(start=T0, stop=T0 + 10 * ONE_S, step=ONE_S)

    @pytest.fixture(scope="class")
    def legacy_int(self):
        """An integer CoordRange built without get_coord."""
        return CoordRange(start=0, stop=10, step=1)

    def test_one_element_arrays_unbox(self):
        """One-element arrays are accepted as scalars."""
        coord = CoordRange(start=np.array([0]), stop=np.array([10]), step=np.array([2]))
        assert len(coord) == 5

    def test_needs_three_values(self):
        """Fewer than three of start, stop, step, shape is an error."""
        with pytest.raises(ValidationError, match="Three of"):
            CoordRange(start=0, step=1)

    def test_getitem(self, legacy_time, legacy_int):
        """Integer and slice indexing on legacy ranges."""
        assert legacy_time[3] == T0 + 3 * ONE_S
        assert np.array_equal(legacy_time[2:5].values, legacy_time.values[2:5])
        assert np.array_equal(legacy_int[::3].values, [0, 3, 6, 9])
        with pytest.raises(IndexError):
            _ = legacy_time[10]

    def test_index_values(self, legacy_time, legacy_int):
        """Requested samples evaluate on the same grid as values."""
        assert np.array_equal(
            legacy_time._get_index_values([0, 9]), legacy_time.values[[0, 9]]
        )
        assert np.array_equal(legacy_int._get_index_values([0, 9]), [0, 9])

    def test_time_units_fixed(self, legacy_time):
        """Time coordinates do not convert units."""
        assert legacy_time._convert_units("ms") is legacy_time

    def test_zero_d_array_bound(self, legacy_int):
        """A 0-d array bound is unboxed."""
        assert legacy_int._get_index(np.array(4)) == 4

    def test_update_limits(self, legacy_time, legacy_int):
        """Translation, both limits, and the three-argument error."""
        assert legacy_int.update_limits(max=20).values.tolist() == list(range(11, 21))
        both = legacy_int.update_limits(min=2, max=8)
        assert len(both) == 10 and both.stop == 8
        assert legacy_time.update_limits(max=T0).max() == T0
        with pytest.raises(ValueError, match="At most two"):
            legacy_time.update_limits(min=T0, max=T0, step=ONE_S)

    def test_out_of_bounds_index(self):
        """Positions past the ends of a float range use the rounded step."""
        coord = CoordRange(start=0.0, stop=10.0, step=1.0)
        assert coord.get_next_index(12.0, allow_out_of_bounds=True) == 12

    def test_tile_offsets(self, legacy_time):
        """Tile offsets on a legacy time range step from the first sample."""
        from dascore.proc.tile_apply import _offset_values  # noqa: PLC0415

        out = _offset_values(legacy_time, np.array([0, 2]))
        assert np.array_equal(out, legacy_time.values[[0, 2]])

    def test_select_and_sort(self, legacy_time):
        """Selection and sorting work as they always have."""
        sub, sl = legacy_time.select((T0 + ONE_S, T0 + 3 * ONE_S))
        assert sl == slice(1, 4) and len(sub) == 3
        rev, _ = legacy_time.sort(reverse=True)
        assert rev.values[0] == legacy_time.max()


class TestReviewFindings:
    """Consumers that rebuilt from the rounded step now stay on the grid."""

    def test_summary_keeps_coarse_tick_unit(self):
        """A microsecond grid summarised in nanoseconds rebuilds its labels."""
        start = np.datetime64("2020-01-01", "us")
        coord = get_coord(start=start, step=np.timedelta64(2, "us"), shape=(5,))
        rebuilt = coord.to_summary().to_coord()
        assert np.array_equal(rebuilt.values, coord.values)
        assert rebuilt.step_exact == coord.step_exact

    def test_pad_extends_the_grid(self, hz_1024):
        """Padding a fractional grid keeps every existing label."""
        patch = dc.get_example_patch().update_coords(time=hz_1024[:2000])
        out = patch.pad(time=(3, 1), expand_coords=True)
        time = out.get_coord("time")
        assert time.step_exact == Fraction(1, 1024)
        assert time.min() == T0 - 3 * ONE_S
        assert np.array_equal(time.values[3072:5072], hz_1024[:2000].values)

    def test_halfway_count_parity(self):
        """A span of 1.05 steps rounds as a float did: to two samples."""
        assert len(get_coord(start=0, stop=21, step=20)) == 2
        assert len(get_coord(start=0.0, stop=21.0, step=20.0)) == 2

    def test_step_update_redispatches(self):
        """A step the grid cannot hold hands over to the class that can."""
        floats = get_coord(start=0, stop=10, step=1).update_limits(step=0.5)
        assert type(floats) is CoordRange and floats.step == 0.5
        moved = get_coord(start=0, stop=10, step=1).update_limits(step=0.5, min=1)
        assert type(moved) is CoordRange and moved.min() == 1 and moved.step == 0.5
        start = np.datetime64("2020-01-01", "ms")
        ms = get_coord(start=start, step=np.timedelta64(1, "ms"), shape=(4,))
        finer = ms.update_limits(step=np.timedelta64(500, "us"))
        assert finer.step_exact == Fraction(1, 2000) and len(finer) == 4
        assert finer.min() == ms.min()

    def test_sample_count_uses_exact_step(self, hz_1024):
        """A second at 1024 Hz is 1024 samples, not 1025."""
        assert hz_1024.get_sample_count(1) == 1024
        assert hz_1024.get_sample_count(dc.to_timedelta64(0.5)) == 512
        assert hz_1024.get_sample_count(np.timedelta64(1, "ns")) == 1
        assert get_coord(start=0, stop=10, step=1).get_sample_count(0.1) == 1

    def test_out_of_bounds_index_uses_grid(self):
        """Positions past the ends come from the grid, not the rounded step."""
        coord = get_coord(start=0, step=(3, 2), shape=(5,))  # 0 1 3 4 6
        assert coord.get_next_index(9, allow_out_of_bounds=True) == 6
        assert coord.get_next_index(-3, allow_out_of_bounds=True) == -2

    def test_tile_offsets_on_grid(self, hz_1024):
        """Tile coordinates evaluate offsets on the exact grid."""
        from dascore.proc.tile_apply import _offset_values  # noqa: PLC0415

        out = _offset_values(hz_1024, np.array([0, 1024, 1025]))
        assert out[1] == T0 + ONE_S
        assert np.array_equal(out, hz_1024.values[[0, 1024, 1025]])
