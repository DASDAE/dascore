"""Tests for the run-table coordinate."""

from __future__ import annotations

import datetime
import json
import math
import sys
from fractions import Fraction
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core._run_kernels import (
    TickKernel,
    _record_dtype,
    _rows,
    float_rows,
    float_terms,
    get_kernel,
)
from dascore.core.coords import (
    _NS_PER_S,
    CoordPartial,
    CoordSummary,
    NumericND,
    _canonical,
    _out_of_ns,
    _to_tick,
    concat_coords,
    concat_tables,
    get_coord,
)
from dascore.exceptions import CoordError, ParameterError

T0 = np.datetime64("2020-01-01", "ns")
MS = np.timedelta64(1, "ms")
NS = np.timedelta64(1, "ns")


def grid_1024(count=2048, start=T0):
    """A 1024 Hz coordinate, whose period is 976562.5 ns."""
    return NumericND.from_run(start=start, step=Fraction(1, 1024), shape=(count,))


def exact_labels(start, count, stride=1, phase=Fraction(0)):
    """Labels of a 1024 Hz grid, as the floor of each ideal position."""
    ideal = [phase + Fraction(i * stride * 1953125, 2) for i in range(count)]
    return start + np.asarray([int(x // 1) for x in ideal]).astype("timedelta64[ns]")


def table(coord):
    """The run table as plain numbers, for comparing two coordinates."""
    return coord.runs.tolist()


@pytest.fixture(scope="module")
def holed():
    """Three runs of ten at 4 ms, each a hole of ten samples apart."""
    step = np.timedelta64(4, "ms")
    runs = [
        NumericND.from_run(start=T0 + step * 20 * i, step=step, shape=(10,))
        for i in range(3)
    ]
    return concat_tables(*runs)


@pytest.fixture(scope="module")
def mixed():
    """A grid run, a stored run of jittered labels, then a grid run."""
    step = np.timedelta64(4, "ms")
    first = NumericND.from_run(start=T0, step=step, shape=(10,))
    wobble = np.asarray([0, 1, -3, 2, 0]) * NS
    jitter = T0 + step * np.asarray([20, 23, 27, 28, 34]) + wobble
    stored = NumericND.from_array(jitter)
    last = NumericND.from_run(start=T0 + step * 50, step=step, shape=(8,))
    return concat_tables(first, stored, last)


class TestSingleRun:
    """One run is an ordinary range."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(start=0.0, step=0.1, shape=(50,)),
            dict(start=3, step=2, shape=(10,)),
            dict(start=T0, step=np.timedelta64(4, "ms"), shape=(100,)),
            dict(start=np.timedelta64(0, "s"), step=np.timedelta64(1, "s"), shape=(5,)),
        ],
    )
    def test_values_match_range(self, kwargs):
        """The labels are those of the range coordinate built the same way."""
        expected = get_coord(**kwargs).values
        out = NumericND.from_run(**kwargs)
        assert out.runs_count == 1
        assert out.evenly_sampled
        if out.dtype.kind in "mM":
            # A time is counted in nanosecond ticks whatever unit it was given in.
            assert out.dtype == np.dtype(f"{out.dtype.kind}8[ns]")
            np.testing.assert_array_equal(out.values, expected.astype(out.dtype))
        else:
            assert out.dtype == expected.dtype
            np.testing.assert_array_equal(out.values, expected)

    def test_exact_grid_does_not_drift(self):
        """A rate with no whole-tick period is counted from the origin."""
        coord = grid_1024()
        np.testing.assert_array_equal(coord.values, exact_labels(T0, 2048))
        assert coord.step == np.timedelta64(976562, "ns")
        assert coord.step_exact == Fraction(1, 1024)
        assert table(coord)[0][2:5] == (1953125, 2, 0)

    def test_limits(self):
        """Min and max come from the ends without computing the labels."""
        coord = grid_1024(10)
        assert coord.min() == T0
        assert coord.max() == exact_labels(T0, 10)[-1]
        assert coord.sorted and not coord.reverse_sorted
        assert not coord.holes

    def test_integer_start_refuses_a_fractional_float_step(self):
        """An integer coordinate cannot hold labels between integers."""
        with pytest.raises(CoordError, match="non-integer"):
            NumericND.from_run(start=0, step=0.5, shape=(3,))

    def test_label_at_index(self):
        """An integer index gives one label, counted from either end."""
        coord = grid_1024(10)
        assert coord[3] == coord.values[3]
        assert coord[-1] == coord.values[-1]
        with pytest.raises(IndexError):
            coord[10]

    def test_values_takes_the_single_run_path(self, monkeypatch):
        """One run is evaluated without repeating any column."""

        def _boom(*args, **kwargs):
            raise AssertionError("the many-run path ran for a single run")

        coord = grid_1024(1000)
        monkeypatch.setattr(np, "repeat", _boom)
        assert len(coord.values) == 1000


class TestCanonical:
    """Equal labels give one table, whatever made them."""

    def test_reduced_grid_equals_whole_tick_grid(self):
        """A stride-two 1024 Hz slice is the 2048 Hz-period grid it labels."""
        odd = grid_1024(5000)[1::2]
        fresh = NumericND.from_run(odd.start, np.timedelta64(1953125, "ns"), 2500)
        np.testing.assert_array_equal(odd.values, fresh.values)
        assert table(odd) == table(fresh)
        assert odd.data_id == fresh.data_id
        assert odd == fresh

    def test_unreduced_input_keeps_its_labels(self):
        """A step stated in unreduced terms is reduced with its phase."""
        coord = NumericND.from_run(0, (6, 4), 5, origin_offset=2)
        assert table(coord)[0][2:5] == (3, 2, 1)
        np.testing.assert_array_equal(coord.values, [0, 2, 3, 5, 6])

    @pytest.mark.parametrize("step", [(3906250, 2), (6, 4), (12, 8), (10, 4), (9, 6)])
    def test_reduction_preserves_every_label(self, step):
        """Reducing a grid moves no label, at any phase it can hold."""
        for offset in range(step[1]):
            coord = NumericND.from_run(0, step, 50, origin_offset=offset)
            expected = [(offset + i * step[0]) // step[1] for i in range(50)]
            np.testing.assert_array_equal(coord.values, expected)

    def test_phase_past_its_denominator(self):
        """A phase of more than one tick is carried into the start."""
        coord = NumericND.from_run(0, (3, 2), 4, origin_offset=5)
        np.testing.assert_array_equal(coord.values, [2, 4, 5, 7])


class TestOneRowArithmetic:
    """One run is canonicalised in python; it must answer as the table does."""

    # (dtype, start, length, num, den, offset): grids which fit and grids
    # which leave their dtype at either end.
    CASES: ClassVar = [
        ("datetime64[ns]", 0, 10, 1, 1, 0),
        ("datetime64[ns]", 0, 100, 3, 2, 1),
        ("datetime64[ns]", 0, 2**62, 8, 1, 0),
        ("datetime64[ns]", 2**62, 2**62, 4, 1, 0),
        ("datetime64[ns]", -(2**62), 10, -(2**60), 1, 0),
        ("timedelta64[ns]", 5, 3, 1, 1024, 3),
        ("int64", 0, 10, 0, 1, 0),
        ("int8", 0, 100, 1, 1, 0),
        ("int8", 100, 100, 1, 1, 0),
        ("int8", -100, 100, -1, 1, 0),
        ("uint8", 200, 100, 1, 1, 0),
        # A uint64 label past int64's ceiling: every tick is counted in
        # int64, so it is out of range whatever uint64 could hold.
        ("uint64", 2**63 - 10, 20, 1, 1, 0),
        ("uint64", 2**62, 20, 1, 1, 0),
        ("float64", 0.0, 10, 0, 0, 0),  # a stored run states no grid
    ]

    def _raised(self, func) -> bool:
        """Whether a check refused the run it was handed."""
        try:
            func()
        except CoordError:
            return True
        return False

    @pytest.mark.parametrize("case", CASES)
    def test_one_row_check_matches_the_vectorised_one(self, case):
        """A few rows are checked one at a time; a long table answers the same."""
        name, *fields = case
        dtype = np.dtype(name)
        kernel = get_kernel(dtype)
        rows = np.asarray([(*fields, b"")], _record_dtype(dtype))
        scalar = self._raised(lambda: kernel.check_range(rows, dtype))
        vector = self._raised(lambda: kernel.check_range(np.repeat(rows, 20), dtype))
        assert scalar == vector

    @pytest.mark.parametrize("length", [0, -1])
    def test_a_run_of_no_samples_is_dropped(self, length):
        """A run holding nothing leaves an empty coordinate, not a bad shape."""
        coord = NumericND.from_run(0, 1, (length,))
        assert coord.shape == (0,) and len(coord) == 0

    def test_a_derived_table_is_already_canonical(self):
        """A slice, reversal or shift of a canonical table is canonical."""
        coord = grid_1024(500)
        derived = [
            coord[3::7],
            coord[::-1],
            coord[100:200],
            coord._translated(MS),
            coord[::2][::3],
            get_coord(start=0.0, step=0.1, shape=(50,))[2::5],
        ]
        for out in derived:
            assert np.array_equal(_canonical(out.runs, out.dtype), out.runs)


class TestSlicing:
    """Slices stay on the grid, phase included."""

    def test_stride_keeps_phase(self):
        """Every third sample of a 1024 Hz grid, from the seventh, stays on it."""
        coord = grid_1024()
        out = coord[7:1000:3]
        assert out.runs_count == 1
        np.testing.assert_array_equal(out.values, coord.values[7:1000:3])

    def test_stride_across_holes_gives_each_run_its_phase(self):
        """Runs cut at different phases keep their own."""
        a, b = grid_1024(100), grid_1024(100, T0 + np.timedelta64(1, "s"))
        coord = concat_tables(a, b)
        out = coord[1:200:3]
        assert out.runs_count == 2
        np.testing.assert_array_equal(out.values, coord.values[1:200:3])

    @pytest.mark.parametrize(
        "item",
        [
            slice(7, None, 3),
            slice(1, 400, 2),
            slice(None, None, -1),
            slice(3, None, -2),
            slice(None, None, 1024),
        ],
    )
    def test_slices_match_their_labels(self, item):
        """A slice of the table holds the labels that slice of the values does."""
        coord = grid_1024(5000)
        np.testing.assert_array_equal(coord[item].values, coord.values[item])

    def test_reversed(self):
        """A negative stride reverses the runs and their steps."""
        a, b = grid_1024(10), grid_1024(10, T0 + np.timedelta64(1, "s"))
        coord = concat_tables(a, b)
        out = coord[::-1]
        assert out.reverse_sorted
        np.testing.assert_array_equal(out.values, coord.values[::-1])
        assert table(out[::-1]) == table(coord)
        assert out.min() == coord.min() and out.max() == coord.max()

    def test_index_array(self):
        """An index array gives the labels it picks, in that order."""
        coord = grid_1024(10)
        picked = coord[np.array([1, 5, 2])]
        np.testing.assert_array_equal(picked.values, coord.values[[1, 5, 2]])

    def test_empty_slice(self):
        """A slice of nothing is a coordinate of no samples."""
        empty = grid_1024(10)[5:2]
        assert len(empty) == 0
        assert empty.dtype == np.dtype("datetime64[ns]")

    def test_padding_extends_the_grid(self):
        """A single run may be sliced before its start, which padding needs."""
        coord = grid_1024(100)
        padded = coord._sliced(-10, 1, 120)
        assert len(padded) == 120
        np.testing.assert_array_equal(padded.values[10:110], coord.values)
        # The label before the first is the floor of its ideal position,
        # half a tick further out than the rounded step.
        assert padded.values[9] == T0 - np.timedelta64(976563, "ns")
        assert padded[10:110] == coord


class TestRuns:
    """Holes and changes of rate are runs in one table."""

    def test_hole_is_a_second_run(self):
        """Runs with a hole between them stay two runs."""
        step = np.timedelta64(4, "ms")
        a = NumericND.from_run(start=T0, step=step, shape=(100,))
        b = NumericND.from_run(start=T0 + step * 110, step=step, shape=(50,))
        coord = concat_tables(a, b)
        assert coord.runs_count == 2
        assert not coord.evenly_sampled
        assert coord.holes
        assert coord.step == step  # one rate, still, across the hole
        np.testing.assert_array_equal(
            coord.values, np.concatenate([a.values, b.values])
        )

    def test_contiguous_runs_fuse(self):
        """Equal samples give an equal coordinate however they were assembled."""
        whole = grid_1024(300)
        parts = [whole[:100], whole[100:250], whole[250:]]
        rebuilt = concat_tables(*parts)
        assert rebuilt.runs_count == 1
        assert rebuilt == whole
        assert rebuilt.data_id == whole.data_id
        # A slice with a phase fuses only onto the grid it came from.
        strided = whole[1::2]
        assert concat_tables(strided[:50], strided[50:]) == strided

    def test_change_of_rate_without_a_hole(self):
        """A run at a new rate directly after another is two runs, no hole."""
        a = NumericND.from_run(start=0, step=2, shape=(5,))
        b = NumericND.from_run(start=10, step=3, shape=(4,))
        coord = concat_tables(a, b)
        assert coord.runs_count == 2
        assert coord.step is None
        assert not coord.holes
        np.testing.assert_array_equal(coord.values, [0, 2, 4, 6, 8, 10, 13, 16, 19])

    def test_mixed_dtypes_refused(self):
        """Runs must share a dtype."""
        with pytest.raises(CoordError, match="dtype"):
            concat_tables(
                NumericND.from_run(start=0, step=1, shape=(3,)),
                NumericND.from_run(start=0.0, step=1.0, shape=(3,)),
            )

    def test_mixed_units_refused(self):
        """Runs must share units."""
        with pytest.raises(CoordError, match="units"):
            concat_tables(
                NumericND.from_run(start=0, step=1, shape=(3,), units="m"),
                NumericND.from_run(start=3, step=1, shape=(3,), units="ft"),
            )


class TestMixedRuns:
    """Stored labels and grids live in one table."""

    def test_values(self, mixed):
        """The table holds the labels of all three runs."""
        assert mixed.runs_count == 3
        assert len(mixed) == 23
        assert mixed.sorted and mixed.holes
        assert mixed.step is None  # a stored run states no rate
        assert mixed.runs["den"].tolist() == [1, 0, 1]  # the middle run is stored
        assert len(mixed._flat_labels) == 5
        step = np.timedelta64(4, "ms")
        wobble = np.asarray([0, 1, -3, 2, 0]) * NS
        expected = np.concatenate(
            [
                T0 + step * np.arange(10),
                T0 + step * np.asarray([20, 23, 27, 28, 34]) + wobble,
                T0 + step * (50 + np.arange(8)),
            ]
        )
        np.testing.assert_array_equal(mixed.values, expected)

    def test_select_across_runs(self, mixed):
        """A window over the stored run keeps the samples inside it."""
        values = mixed.values
        out, indexer = mixed.select((values[8], values[12]))
        assert indexer == slice(8, 13)
        np.testing.assert_array_equal(out.values, values[8:13])

    def test_select_between_stored_labels(self, mixed):
        """A bound between two stored labels selects from the next one in."""
        values = mixed.values
        out, _ = mixed.select((values[11] + NS, values[13] - NS))
        np.testing.assert_array_equal(out.values, values[12:13])

    def test_strided_slice(self, mixed):
        """A stride cuts every run, stored labels included."""
        out = mixed[1::4]
        np.testing.assert_array_equal(out.values, mixed.values[1::4])

    def test_reversed(self, mixed):
        """Reversing turns the stored labels round with the grids."""
        out = mixed[::-1]
        np.testing.assert_array_equal(out.values, mixed.values[::-1])
        assert out.reverse_sorted
        assert table(out[::-1]) == table(mixed)

    def test_data_id_is_stable(self, mixed):
        """Splitting and rejoining gives back the same table and data_id."""
        # Grid runs fuse back together; a stored run keeps its own bounds.
        rebuilt = concat_tables(mixed[:7], mixed[7:15], mixed[15:])
        assert table(rebuilt) == table(mixed)
        assert rebuilt.data_id == mixed.data_id

    def test_labels_at(self, mixed):
        """Labels can be asked for at scattered indices."""
        picked = np.asarray([0, 11, 22, 13])
        np.testing.assert_array_equal(mixed._labels(picked), mixed.values[picked])


class TestFromArray:
    """Labels state their own runs, or are kept as they are."""

    def test_even_array_is_one_run(self):
        """An exactly evenly sampled array is a grid run."""
        coord = NumericND.from_array(np.arange(10) * 3)
        assert coord.runs_count == 1
        assert coord.evenly_sampled
        assert coord.step == 3
        np.testing.assert_array_equal(coord.values, np.arange(10) * 3)

    def test_hole_gives_two_runs(self):
        """An array with a hole in it becomes two grid runs."""
        values = np.asarray([0.0, 1, 2, 3, 10, 11, 12, 13])
        coord = NumericND.from_array(values)
        assert coord.runs_count == 2
        assert coord.holes
        np.testing.assert_array_equal(coord.values, values)

    def test_stray_labels_are_stored(self):
        """Labels which state no grid are kept, beside the runs which do."""
        values = np.asarray([0, 1, 2, 3, 20, 33, 34, 35, 36])
        coord = NumericND.from_array(values)
        assert coord.runs_count == 3
        assert coord._flat_labels.tolist() == [20]
        np.testing.assert_array_equal(coord.values, values)

    def test_jitter_is_one_stored_run(self):
        """A dense jittered array is kept whole rather than split per sample."""
        rng = np.random.default_rng(42)
        count = 1_000_000
        values = T0 + np.arange(count) * MS + rng.integers(-500, 500, count) * NS
        coord = NumericND.from_array(values)
        assert coord.runs_count == 1
        assert coord.sources is not None
        assert coord.step is None
        np.testing.assert_array_equal(coord.values, values)

    def test_two_dimensional(self):
        """An array of any shape is one stored run carrying its shape."""
        coord = NumericND.from_array(np.zeros((3, 4)))
        assert coord.shape == (3, 4)
        assert coord.ndim == 2
        assert coord.runs_count == 1
        assert coord.runs["length"][0] == 12
        np.testing.assert_array_equal(coord[1], np.zeros(4))
        assert coord[:2].shape == (2, 4)

    def test_float_grid_below_one_is_detected(self):
        """A float spacing below one is a step like any other."""
        values = np.concatenate([np.arange(10) * 0.5, 100 + np.arange(10) * 0.5])
        coord = NumericND.from_array(values, detect=True)
        assert coord.runs_count == 2
        assert np.all(float_terms(coord.runs)[0] == 0.5)
        assert coord.step == 0.5
        np.testing.assert_array_equal(coord.values, values)

    def test_a_huge_spacing_is_still_a_grid(self):
        """Any finite double is a step; none is too large to state."""
        values = np.asarray([0.0, 1e20, 2e20, 1e21, 1.1e21, 1.2e21])
        coord = NumericND.from_array(values, detect=True)
        np.testing.assert_array_equal(coord.values, values)
        np.testing.assert_array_equal(get_coord(data=values).values, values)
        big = get_coord(start=0.0, step=1e30, shape=(5,))
        np.testing.assert_array_equal(big.values, np.arange(5) * 1e30)

    def test_get_coord_reads_a_float_grid(self):
        """The shared reader constructor sees the same grid."""
        coord = get_coord(data=np.arange(20) * 0.25)
        assert coord.evenly_sampled and coord.step == 0.25
        # a float spacing is the double it is, which has no exact form
        assert coord.step_exact is None

    def test_rank_zero_stays_rank_zero(self):
        """A scalar label is a coordinate of no axis, not one of one sample."""
        coord = NumericND.from_array(np.array(3.0))
        assert coord.shape == () and coord.ndim == 0
        assert coord.values.shape == ()
        assert get_coord(data=np.array(3.0)).shape == ()

    def test_patch_keeps_a_rank_zero_coord(self):
        """A non-dimensional scalar coordinate attaches to a patch."""
        patch = dc.get_example_patch()
        out = patch.update_coords(source=((), np.array(3.0)))
        assert out.get_coord("source").shape == ()

    def test_unsorted_labels(self):
        """Arbitrary labels are stored, sorted on request, and hashed."""
        coord = NumericND.from_array(np.asarray([3.0, 1.0, 2.0]))
        assert not coord.sorted and not coord.reverse_sorted
        assert coord.min() == 1.0 and coord.max() == 3.0
        ordered, order = coord.sort()
        np.testing.assert_array_equal(ordered.values, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(order, [1, 2, 0])
        assert coord.data_id != ordered.data_id

    def test_monotonic_select(self):
        """A sorted stored run selects by searching its labels."""
        coord = NumericND.from_array(np.asarray([1.0, 2.5, 4.0, 8.0]))
        out, indexer = coord.select((2.0, 5.0))
        np.testing.assert_array_equal(out.values, [2.5, 4.0])
        assert indexer == slice(1, 3)


class TestOverlappingRuns:
    """Runs which overlap are not sorted, and are selected by their labels."""

    @pytest.fixture()
    def overlapping(self):
        """Ten samples from zero, then ten more from five."""
        first = NumericND.from_run(start=0, step=1, shape=(10,))
        return concat_tables(first, NumericND.from_run(start=5, step=1, shape=(10,)))

    def test_not_sorted(self, overlapping):
        """An overlap is not a sorted coordinate, however each run reads."""
        assert not overlapping.sorted
        assert not overlapping.reverse_sorted

    def test_select_matches_the_labels(self, overlapping):
        """Selection falls back to matching every label, in both runs."""
        np.testing.assert_array_equal(overlapping.values, [*range(10), *range(5, 15)])
        out, indexer = overlapping.select((3, 7))
        assert np.flatnonzero(indexer).tolist() == [3, 4, 5, 6, 7, 10, 11, 12]
        np.testing.assert_array_equal(out.values, [3, 4, 5, 6, 7, 5, 6, 7])

    def test_lookup_refuses(self, overlapping):
        """Looking one value up needs an order the coordinate does not have."""
        with pytest.raises(CoordError, match="sorted"):
            overlapping._get_index(4)


class TestDegenerate:
    """Flat, short, and empty coordinates behave."""

    def test_zero_step(self):
        """A step of zero labels every sample the same."""
        coord = NumericND.from_run(start=5, step=0, shape=(4,))
        np.testing.assert_array_equal(coord.values, [5, 5, 5, 5])
        assert coord.sorted and not coord.reverse_sorted
        assert coord.step == 0
        out, indexer = coord.select((5, 5))
        assert len(out) == 4 and indexer == slice(None, 4)
        out, _ = coord.select((6, 7))
        assert len(out) == 0

    def test_single_sample(self):
        """A run of one sample selects, reverses, and states its ends."""
        coord = grid_1024(1)
        assert len(coord) == 1
        assert coord.min() == coord.max() == T0
        np.testing.assert_array_equal(coord[::-1].values, coord.values)
        out, indexer = coord.select((T0, T0))
        assert len(out) == 1 and indexer == slice(None, 1)

    def test_empty(self):
        """An empty coordinate is a run of no samples, not an empty table."""
        empty = grid_1024(10)[5:2]
        assert len(empty) == 0
        assert empty.runs_count == 1
        assert empty.values.dtype == np.dtype("datetime64[ns]")
        assert empty.step is None
        assert len(empty[:]) == 0
        # An empty coordinate states a null end, as every other empty
        # coordinate in DASCore does.
        assert pd.isnull(empty.min()) and pd.isnull(empty.max())

    def test_empty_concatenates_away(self):
        """An empty run leaves nothing behind in a table it joins."""
        whole = grid_1024(300)
        parts = [whole[:100], whole[100:100], whole[100:], whole[300:]]
        assert concat_tables(*parts) == whole

    def test_empty_select(self):
        """Selecting on an empty coordinate selects nothing."""
        out, indexer = grid_1024(10)[5:2].select((T0, T0 + MS))
        assert len(out) == 0
        assert indexer == slice(0, 0)


class TestOverflow:
    """Grids which leave int64 are refused rather than silently wrapped."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(start=0, step=10**12, shape=(10**7,), dtype="int64"),
            dict(start=T0, step=np.timedelta64(1, "h"), shape=(3_000_000,)),
            dict(start=2**62, step=2**61, shape=(4,), dtype="int64"),
        ],
    )
    def test_refused(self, kwargs):
        """A grid whose ticks do not fit cannot be built."""
        with pytest.raises(CoordError, match="exceeds"):
            NumericND.from_run(**kwargs)

    def test_extending_is_checked(self):
        """A grid extended past what int64 holds is refused too."""
        coord = NumericND.from_run(start=0, step=10**11, shape=(10,), dtype="int64")
        with pytest.raises(CoordError, match="exceeds"):
            coord.change_length(10**9)


class TestFloats:
    """A float run is ``start + step * (k0 + k * stride)`` in plain doubles."""

    # Axes as numpy users build them; each must come back bit for bit.
    AXES: ClassVar = {
        "arange * step": np.arange(5000) * 0.1,
        "start + arange * step": 3.7 + np.arange(5000) * 0.1,
        "gauge length": np.arange(5000) * 1.02,
        "linspace": np.linspace(0, 1000, 5000),
        "large origin": 1e6 + np.arange(5000) * 0.25,
        "negative step": 10.0 - np.arange(5000) * 0.1,
    }

    def test_step_is_the_double_it_was_given(self):
        """A float spacing is stored as itself, with a stride of one."""
        coord = NumericND.from_run(start=0.0, step=0.1, shape=(11,))
        step, stride, k0 = float_terms(coord.runs)
        assert (step[0], stride[0], k0[0]) == (0.1, 1, 0)
        assert coord.step == 0.1
        assert coord.step_exact is None
        np.testing.assert_array_equal(coord.values, np.arange(11) * 0.1)

    @pytest.mark.parametrize("step", [math.pi, 1e-15, 5e-324, 1e300])
    def test_any_finite_step_is_a_grid(self, step):
        """No spacing is too fine or too coarse to be a step."""
        coord = NumericND.from_run(start=0.0, step=step, shape=(5,))
        assert coord.evenly_sampled and coord.step == step
        np.testing.assert_array_equal(coord.values, np.arange(5) * step)

    @pytest.mark.parametrize("step", [np.nan, np.inf, -np.inf])
    def test_a_step_which_is_no_number_is_refused(self, step):
        """A NaN or infinite step raises the coordinate's own error."""
        with pytest.raises(CoordError, match="finite"):
            NumericND.from_run(start=0.0, step=step, shape=(5,))

    @pytest.mark.parametrize("name", AXES)
    def test_array_comes_back_bit_for_bit(self, name):
        """An axis read from labels is one grid and gives back every double."""
        values = self.AXES[name]
        coord = get_coord(data=values)
        assert coord.evenly_sampled
        np.testing.assert_array_equal(coord.values, values)

    @pytest.mark.parametrize(
        "values",
        [
            np.arange(5000) / 250,
            -np.arange(5000) / 1000.0,
            (119_999_999 + np.arange(5000) * 100_000) / 1e9,
        ],
    )
    def test_non_multiplication_formulas_stay_stored(self, values):
        """Automatic inference does not search rates or decimal time scales."""
        coord = NumericND.from_array(values)
        assert not coord.evenly_sampled
        np.testing.assert_array_equal(coord.values, values)

    @pytest.mark.parametrize("name", AXES)
    def test_no_slice_moves_a_label(self, name):
        """Slicing and reversal change integers only, so labels cannot move."""
        values = self.AXES[name]
        coord = get_coord(data=values)
        for item in (slice(7, None, 3), slice(None, None, -1), slice(4000, 10, -7)):
            np.testing.assert_array_equal(coord[item].values, values[item])
        chained = coord[100:][250:][::2][::-1]
        np.testing.assert_array_equal(chained.values, values[350::2][::-1])

    def test_routes_to_one_slice_give_one_row(self):
        """``c[a:][b:]`` is ``c[a + b:]``: equal, with equal run hashes."""
        coord = get_coord(data=self.AXES["start + arange * step"])
        one, two = coord[350:], coord[100:][250:]
        assert one == two
        np.testing.assert_array_equal(one.runs, two.runs)
        assert coord[::-1][::-1] == coord

    def test_a_slice_arriving_as_labels_is_found_on_its_parent_grid(self):
        """A slice needing formula search stays stored and bit exact."""
        values = np.arange(10_000) * 0.1
        coord = NumericND.from_array(values[5000:])
        assert not coord.evenly_sampled
        np.testing.assert_array_equal(coord.values, values[5000:])

    @pytest.mark.parametrize("rate", [250.0, -250.0])
    def test_explicit_divided_grid_slices_without_moving_labels(self, rate):
        """Persisted divided-grid rows retain their slice guarantees."""
        rows = float_rows("float64", [0.0], [5000], [rate], [-1], [0])
        coord = get_coord(runs=rows, dtype="float64")
        values = np.arange(5000) / rate
        np.testing.assert_array_equal(coord.values, values)
        for item in (slice(7, None, 3), slice(None, None, -1)):
            np.testing.assert_array_equal(coord[item].values, values[item])

    def test_labels_no_grid_reproduces_are_stored(self):
        """Near-even labels are kept as they are, never fitted."""
        values = np.cumsum(np.full(1000, 0.1))
        coord = NumericND.from_array(values)
        assert not coord.evenly_sampled and coord.sources is not None
        np.testing.assert_array_equal(coord.values, values)

    def test_float32_axis_round_trips(self):
        """A narrower float is verified in its own dtype."""
        values = np.arange(1000, dtype=np.float32) * np.float32(0.5)
        coord = get_coord(data=values)
        assert coord.dtype == np.float32 and coord.evenly_sampled
        np.testing.assert_array_equal(coord.values, values)

    def test_holes_share_one_grid(self):
        """Runs either side of a hole sit on one grid, a grid index apart."""
        values = np.arange(3000) * 0.1
        kept = np.concatenate([values[:1000], values[2000:]])
        coord = NumericND.from_array(kept)
        assert coord.runs_count == 2 and coord.holes
        assert coord.runs["start"][0] == coord.runs["start"][1]
        assert coord.step == 0.1
        np.testing.assert_array_equal(coord.values, kept)

    def test_slice_keeps_its_first_label(self):
        """A slice is exactly the labels its parent gave those samples."""
        coord = NumericND.from_run(start=1e6, step=0.1, shape=(1000,))
        for first in (0, 11, 137):
            sliced = coord[first : first + 9]
            np.testing.assert_array_equal(
                sliced.values, coord.values[first : first + 9]
            )

    def test_split_runs_fuse_back(self):
        """Pieces of one grid rejoin into the run they were cut from."""
        coord = NumericND.from_run(start=-0.3, step=0.1, shape=(6,))
        rejoined = concat_tables(coord[:3], coord[3:])
        assert rejoined.runs_count == 1 and rejoined == coord

    def test_grids_built_apart_fuse_only_when_exact(self):
        """Two ranges meeting end to end fuse where one grid holds both."""
        first = get_coord(start=0.0, stop=10.0, step=1.0)
        second = get_coord(start=10.0, stop=20.0, step=1.0)
        joined = concat_tables(first, second)
        assert joined.runs_count == 1
        np.testing.assert_array_equal(joined.values, np.arange(20.0))
        # 0.1 * k and 1.0 + 0.1 * k are different doubles, so these do not
        awkward = concat_tables(
            get_coord(start=0.0, step=0.1, shape=(10,)),
            get_coord(start=1.0, step=0.1, shape=(10,)),
        )
        expected = np.concatenate([np.arange(10) * 0.1, 1.0 + np.arange(10) * 0.1])
        np.testing.assert_array_equal(awkward.values, expected)
        assert not awkward.holes and awkward.step == 0.1

    def test_select(self):
        """A bound within a rounding of a label is that label."""
        coord = NumericND.from_run(start=0.0, step=0.1, shape=(20,))
        out, _ = coord.select((0.3, 0.7))
        np.testing.assert_array_equal(out.values, coord.values[3:8])

    def test_select_near_a_label(self):
        """A bound a hair inside a label picks the next label in."""
        coord = NumericND.from_run(start=1e6, step=0.1, shape=(1000,))
        values = coord.values
        out, _ = coord.select((values[10], values[20]))
        np.testing.assert_array_equal(out.values, values[10:21])
        out, _ = coord.select((values[10] + 1e-6, values[20] - 1e-6))
        np.testing.assert_array_equal(out.values, values[11:20])


class TestLookup:
    """The table answers value windows as the existing coordinates do."""

    @pytest.fixture()
    def segmented(self):
        """Three ten-sample runs at 4 ms with holes, as DASCore states them."""
        step = np.timedelta64(4, "ms")
        segments = [
            get_coord(start=T0 + step * 30 * i, step=step, shape=(10,))
            for i in range(3)
        ]
        return concat_coords(*segments)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_random_windows(self, segmented, reverse):
        """Random windows keep exactly the NumPy labels within their bounds."""
        coord = segmented[::-1] if reverse else segmented
        values = np.asarray(coord.values)
        rng = np.random.default_rng(0)
        bounds = [*values, *(values + MS), *(values - MS), None]
        for _ in range(100):
            low, high = (bounds[i] for i in rng.integers(0, len(bounds), 2))
            if low is not None and high is not None and low > high:
                low, high = high, low
            out, indexer = coord.select((low, high))
            keep = np.ones(len(values), dtype=bool)
            if low is not None:
                keep &= values >= low
            if high is not None:
                keep &= values <= high
            expected = np.flatnonzero(keep)
            np.testing.assert_array_equal(np.arange(len(values))[indexer], expected)
            np.testing.assert_array_equal(out.values, values[expected])

    def test_window_across_a_hole(self, holed):
        """A window spanning a hole keeps the samples on both sides."""
        values = holed.values
        out, indexer = holed.select((values[5], values[15]))
        np.testing.assert_array_equal(out.values, values[5:16])
        assert indexer == slice(5, 16)
        assert out.runs_count == 2

    def test_window_inside_a_hole(self, holed):
        """A window which falls entirely in a hole selects nothing."""
        values = holed.values
        out, indexer = holed.select((values[9] + MS, values[10] - MS))
        assert len(out) == 0
        assert indexer == slice(0, 0)

    def test_open_ends(self, holed):
        """An open bound reaches the coordinate's end."""
        values = holed.values
        out, _ = holed.select((values[12], None))
        np.testing.assert_array_equal(out.values, values[12:])
        out, _ = holed.select((None, values[12]))
        np.testing.assert_array_equal(out.values, values[:13])

    def test_bounds_between_samples(self):
        """A bound between two labels selects from the next label inward."""
        coord = grid_1024(100)
        values = coord.values
        out, _ = coord.select((values[10] + NS, values[20] - NS))
        np.testing.assert_array_equal(out.values, values[11:20])

    @pytest.mark.parametrize("reverse", [False, True])
    def test_fractional_grid_windows(self, reverse):
        """Fractional-grid windows keep the NumPy labels within their bounds."""
        coord = grid_1024(500)
        coord = coord[::-1] if reverse else coord
        values = coord.values
        windows = [
            (min(values[100], values[200]), max(values[100], values[200])),
            (min(values[100], values[200]) + NS, max(values[100], values[200]) - NS),
            (None, values[10]),
            (values[10], None),
        ]
        for low, high in windows:
            out, indexer = coord.select((low, high))
            keep = np.ones(len(values), dtype=bool)
            if low is not None:
                keep &= values >= low
            if high is not None:
                keep &= values <= high
            expected = np.flatnonzero(keep)
            np.testing.assert_array_equal(np.arange(len(values))[indexer], expected)
            np.testing.assert_array_equal(out.values, values[expected])

    def test_reverse_sorted(self, holed):
        """Selection on a reversed coordinate mirrors the forward one."""
        reversed_coord = holed[::-1]
        values = holed.values
        out, _ = reversed_coord.select((values[5], values[15]))
        np.testing.assert_array_equal(out.values, values[5:16][::-1])

    def test_relative_and_samples(self, holed):
        """The base class's relative and sample selection still apply."""
        out, _ = holed.select((2, 5), samples=True)
        np.testing.assert_array_equal(out.values, holed.values[2:5])
        out, _ = holed.select((0, np.timedelta64(8, "ms")), relative=True)
        np.testing.assert_array_equal(out.values, holed.values[:3])


class TestFromExisting:
    """The existing coordinates convert without losing what they state."""

    def test_units_survive(self):
        """Units go along with the labels."""
        coord = get_coord(start=0, step=1, shape=(4,), units="m")
        assert coord.units == dc.get_quantity("m")
        assert coord[1:].units == coord.units

    def test_from_rows_round_trip(self):
        """A table states itself: its rows rebuild the coordinate."""
        coord = grid_1024(100)[3::7]
        rebuilt = NumericND.from_rows(
            coord.runs, sources=coord.sources, dtype=coord.dtype, units=coord.units
        )
        assert rebuilt == coord

    def test_get_coord_rebuilds_from_runs(self, mixed):
        """`get_coord(runs=...)`, the constructor an index row hands back."""
        out = get_coord(
            runs=mixed.runs,
            sources=mixed.sources,
            dtype=mixed.dtype,
            units=mixed.units,
        )
        assert out == mixed
        assert out.data_id == mixed.data_id

    def test_model_dump_round_trip(self, mixed, holed):
        """A dumped coordinate states enough to be rebuilt."""
        for coord in (mixed, holed, grid_1024(10)):
            assert get_coord(**coord.model_dump()) == coord
            assert coord.model_validate(coord.model_dump()) == coord

    def test_class_takes_only_its_own_fields(self):
        """A start, a step or an array is read by `get_coord`, not the class."""
        forms = [
            dict(start=0, stop=10, step=1),
            dict(values=np.arange(5)),
            dict(data=np.arange(5)),
            dict(segments=(get_coord(start=0, stop=5, step=1),)),
        ]
        for form in forms:
            with pytest.raises(ValidationError, match="built from its own run table"):
                NumericND(**form)

    def test_summary_dump_round_trip(self, mixed, holed):
        """A dumped summary keeps the runs it states."""
        for coord in (mixed, holed):
            summary = coord.to_summary(dims=("time",))
            back = type(summary).model_validate(summary.model_dump())
            assert back == summary
            assert hash(back) == hash(summary)


class TestHashing:
    """A coordinate is its table, and a stored run is the id it carries."""

    def test_order_matters(self):
        """The same runs in another order are another coordinate."""
        step = np.timedelta64(4, "ms")
        early = NumericND.from_run(start=T0, step=step, shape=(5,))
        late = NumericND.from_run(start=T0 + step * 50, step=step, shape=(5,))
        assert concat_tables(early, late) != concat_tables(late, early)

    def test_stored_labels_are_named_by_their_id(self, mixed):
        """Two stored runs of different labels carry different ids."""
        moved = concat_tables(mixed[:10], mixed[10:15]._translated(NS), mixed[15:])
        assert moved.runs["source_id"].tolist() != mixed.runs["source_id"].tolist()
        assert moved != mixed and moved.data_id != mixed.data_id

    def test_an_array_is_hashed_once(self, monkeypatch, mixed):
        """No view of a stored coordinate digests its labels a second time."""

        def _refuse(array):
            raise AssertionError("the labels were hashed again")

        monkeypatch.setattr(dc.core.coords, "_array_id", _refuse)
        for out in (mixed[2:20], mixed[::-1], mixed[::3], mixed.fuse()):
            assert out.data_id
        assert concat_tables(mixed[:10], mixed[10:]).data_id == mixed.data_id

    def test_an_unused_source_is_dropped(self, mixed):
        """A table which no longer reads a source stops carrying it."""
        assert set(mixed.sources) == {mixed.runs["source_id"][1]}
        assert mixed[:4].sources is None

    def test_equal_after_split_and_concat(self, mixed):
        """Cutting a coordinate up and rejoining it gives the same identity."""
        parts = [mixed[:4], mixed[4:9], mixed[9:20], mixed[20:]]
        assert concat_tables(*parts).data_id == mixed.data_id

    def test_dtype_separates_tables(self):
        """An integer table and a float table of the same numbers differ."""
        ints = NumericND.from_run(start=0, step=1, shape=(5,))
        floats = NumericND.from_run(start=0.0, step=1.0, shape=(5,))
        assert ints.data_id != floats.data_id
        assert ints != floats

    def test_not_hashable(self):
        """Coordinates state a data_id rather than a python hash."""
        with pytest.raises(TypeError, match="not hashable"):
            hash(grid_1024(10))


class TestUpdates:
    """Limits, lengths, and steps of a table."""

    def test_translate_a_table(self, mixed):
        """Every label of every run moves by the same amount."""
        moved = mixed._translated(np.timedelta64(7, "ms"))
        np.testing.assert_array_equal(
            moved.values, mixed.values + np.timedelta64(7, "ms")
        )
        assert moved.runs_count == mixed.runs_count

    def test_update_limits_min(self, holed):
        """A new minimum shifts the whole table."""
        out = holed.update_limits(min=T0 + np.timedelta64(1, "s"))
        assert out.min() == T0 + np.timedelta64(1, "s")
        assert out.max() - out.min() == holed.max() - holed.min()

    def test_update_limits_max(self, holed):
        """A new maximum shifts the whole table."""
        out = holed.update_limits(max=T0)
        assert out.max() == T0

    def test_update_limits_step(self):
        """A single run may state a new cadence."""
        coord = NumericND.from_run(start=0, step=1, shape=(5,))
        out = coord.update_limits(step=3)
        np.testing.assert_array_equal(out.values, [0, 3, 6, 9, 12])

    def test_float_step_below_one(self):
        """A float coordinate has no tick, so any fraction of a unit steps it."""
        coord = NumericND.from_run(start=0.0, step=0.5, shape=(5,))
        out = coord.update_limits(step=0.25)
        np.testing.assert_allclose(
            out.values, [0.0, 0.25, 0.5, 0.75, 1.0], rtol=1e-12, atol=0
        )
        assert coord.new(step=0.25).step == 0.25

    def test_patch_takes_a_float_step_below_one(self):
        """The same through the API users reach it by."""
        patch = dc.get_example_patch()
        distance = np.arange(patch.shape[0]) * 0.5
        floats = patch.update_coords(distance=distance)
        assert (
            floats.update_coords(distance_step=0.25).get_coord("distance").step == 0.25
        )

    def test_sub_tick_step_still_refused(self):
        """An integer coordinate cannot step by half a tick and keep its labels."""
        coord = NumericND.from_run(start=0, step=1, shape=(5,))
        with pytest.raises(CoordError, match="smaller than one tick"):
            coord.update_limits(step=Fraction(1, 2))

    def test_grid_fields_refused_beside_labels(self):
        """Labels state their own grid; the exact fields cannot restate it."""
        with pytest.raises(CoordError, match="grid of a range"):
            get_coord(data=np.arange(5), step_numerator=1, step_denominator=2)
        segments = [get_coord(start=0, stop=5, step=1)]
        with pytest.raises(CoordError, match="grid of a range"):
            get_coord(segments=segments, origin_offset=1)

    def test_legacy_grid_fields_need_their_bounds(self):
        """The old three fields state a run only beside a min and a max."""
        summary = dict(dtype="int64", min=0, max=np.nan, step=1, step_numerator=1)
        with pytest.raises(CoordError, match="both of its bounds"):
            dc.core.coords.CoordSummary(**summary).to_coord()
        bad = dict(dtype="int64", min=0, max=10, step=1, step_denominator=0)
        with pytest.raises(CoordError, match="positive denominator"):
            dc.core.coords.CoordSummary(**bad).to_coord()

    def test_empty_keeps_its_rank(self):
        """An N-D coordinate emptied whole is still N-D."""
        coord = NumericND.from_array(np.zeros((3, 4)))
        assert coord.empty().shape == (0, 0)
        assert coord.empty(0).shape == (0, 4)

    def test_next_index_at_the_first_label(self):
        """The first label's own index is its index, on any coordinate."""
        array = get_coord(data=np.asarray([0.0, 1.0, 4.0, 9.0, 16.0]))
        assert array.get_next_index(0.0) == 0
        assert array.get_next_index(-1.0, allow_out_of_bounds=True) == 0
        assert get_coord(data=np.asarray([5.0])).get_next_index(5.0) == 0
        # the grid it is meant to agree with
        assert get_coord(start=0.0, stop=5.0, step=1.0).get_next_index(0.0) == 0

    def test_new_step_rederives_the_count(self):
        """A new cadence keeps both ends and states how many samples fit."""
        coord = get_coord(start=0.0, stop=300.0, step=1.0)
        out = coord.new(step=0.5)
        assert len(out) == 600
        assert out.min() == 0.0 and out.max() == 299.5

    def test_update_limits_step_keeps_the_count(self):
        """`update_limits` moves the end instead, as it always has."""
        coord = get_coord(start=0.0, stop=300.0, step=1.0)
        out = coord.update_limits(step=0.5)
        assert len(out) == len(coord)
        assert out.min() == 0.0 and out.max() == 149.5

    def test_update_step_refused_on_a_table(self, holed):
        """A table of several runs has no one step to change."""
        with pytest.raises(ParameterError, match="no single step"):
            holed.update_limits(step=np.timedelta64(1, "ms"))

    def test_change_length(self):
        """A grid run may be lengthened or shortened on its own grid."""
        coord = grid_1024(100)
        assert len(coord.change_length(120)) == 120
        np.testing.assert_array_equal(
            coord.change_length(120).values[:100], coord.values
        )
        assert coord.change_length(100) is coord

    def test_new_units(self):
        """New units are set without touching the labels."""
        coord = NumericND.from_run(start=0.0, step=1.0, shape=(4,))
        out = coord.new(units="m")
        assert out.units == dc.get_quantity("m")
        np.testing.assert_array_equal(out.values, coord.values)

    def test_new_values(self):
        """New labels state their own table."""
        coord = NumericND.from_run(start=0.0, step=1.0, shape=(4,))
        out = coord.new(values=np.asarray([0.0, 2.0, 4.0, 6.0]))
        assert out.step == 2.0

    def test_summary_states_the_grid(self):
        """A single grid run summarises with the grid it sits on."""
        coord = grid_1024(10)
        summary = coord.to_summary()
        assert summary.min == coord.min() and summary.max == coord.max()
        assert summary.len == 10
        terms = (summary.step_numerator, summary.step_denominator)
        assert terms == (1953125, 2)
        assert summary.is_exact_grid

    def test_exact_grid_means_a_tick_grid(self):
        """A float has no tick, so its grid is not an exact one."""
        assert get_coord(start=0, stop=10, step=1).to_summary().is_exact_grid
        assert not get_coord(start=0.0, stop=10.0, step=1.0).to_summary().is_exact_grid
        assert (
            not get_coord(data=np.asarray([1.0, 2.5, 7.0])).to_summary().is_exact_grid
        )

    def test_legacy_grid_fields_still_state_a_run(self):
        """A summary written before the run table still rebuilds its grid."""
        summary = dc.core.coords.CoordSummary(
            dtype="datetime64",
            min=T0,
            max=T0 + np.timedelta64(1, "s"),
            step=np.timedelta64(976562, "ns"),
            len=1025,
            step_numerator=1953125,
            step_denominator=2,
            origin_offset=0,
        )
        assert summary.is_exact_grid
        assert summary.to_coord() == grid_1024(1025)

    def test_legacy_grid_counts_its_floored_labels(self):
        """A 3/2 grid puts two labels in [0, 1], not one."""
        summary = dc.core.coords.CoordSummary(
            dtype="int64",
            min=0,
            max=1,
            step=2,
            step_numerator=3,
            step_denominator=2,
        )
        np.testing.assert_array_equal(summary.to_coord().values, [0, 1])

    def test_summary_states_each_run(self, holed):
        """A table of several runs summarises each of them."""
        summary = holed.to_summary()
        assert summary.runs is not None
        assert len(summary.runs) == holed.runs_count

    def test_summaries_compare_by_their_runs(self, holed):
        """Two summaries of one coordinate are equal, run table and all."""
        assert holed.to_summary() == holed.to_summary()
        assert holed.to_summary() != holed[:-1].to_summary()

    def test_a_summary_carrying_runs_hashes(self, holed):
        """Its runs are arrays, but their own hashes stand in for them."""
        summary = holed.to_summary()
        assert hash(summary) == hash(holed.to_summary())
        assert len({summary, holed.to_summary()}) == 1
        assert hash(summary) != hash(holed[:-1].to_summary())

    def test_patch_summary_hashes(self):
        """A whole patch summary hashes, which is how dc.scan results travel."""
        patch = dc.get_example_patch()
        assert hash(patch.summary) == hash(patch.summary)
        assert len({patch.summary, patch.summary}) == 1


class TestRepr:
    """The table prints without crashing, whatever it holds."""

    @pytest.mark.parametrize("count", [0, 1, 10])
    def test_single_run(self, count):
        """A run of any length prints its facts."""
        assert "NumericND" in str(grid_1024(10)[:count].__rich__())

    def test_table(self, mixed):
        """A table states how many runs it holds."""
        assert "runs" in str(mixed.__rich__())


class TestNarrowDtypes:
    """A coordinate keeps the numeric dtype its labels were given in."""

    @pytest.mark.parametrize(
        "dtype",
        ["int8", "int16", "int32", "int64", "uint16", "uint32", "float32", "float64"],
    )
    def test_dtype_survives(self, dtype):
        """The table records ticks, but the labels keep their own width."""
        values = np.arange(0, 12, 2).astype(dtype)
        coord = NumericND.from_array(values)
        assert coord.dtype == np.dtype(dtype)
        assert coord.values.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(coord.values, values)
        np.testing.assert_array_equal(coord[1::2].values, values[1::2])

    def test_unsigned_label_past_the_signed_range_is_refused(self):
        """An unsigned label above int64 max has no tick to sit on."""
        for values in ([0, 2**63 + 5], [0, 1, 2**63 + 5]):
            with pytest.raises(CoordError, match="int64 range"):
                NumericND.from_array(np.asarray(values, dtype=np.uint64))

    def test_narrow_range_is_guarded(self):
        """A grid which would wrap a narrow integer is refused."""
        with pytest.raises(CoordError, match="exceeds"):
            NumericND.from_run(np.int8(100), np.int8(2), (50,))

    def test_from_coord_keeps_a_narrow_range(self):
        """Converting a narrow range does not widen it."""
        reference = get_coord(start=np.int8(0), stop=np.int8(100), step=np.int8(2))
        coord = reference
        assert coord.dtype == reference.dtype
        np.testing.assert_array_equal(coord.values, reference.values)


class TestGaps:
    """Seams, holes, and the discontinuity report read from the columns."""

    def test_discontinuities_all(self, holed):
        """Every seam between runs is a boundary."""
        df = holed.get_discontinuities("all")
        assert len(df) == holed.runs_count - 1
        assert (df["index"] == [10, 20]).all()
        assert df["delta"].tolist() == [pd.Timedelta("44ms")] * 2

    def test_discontinuities_gaps(self, holed):
        """Each seam of ten missing samples is a gap."""
        assert len(holed.get_discontinuities("gaps")) == 2

    def test_no_discontinuities_in_one_run(self):
        """A single grid run has no boundary to report."""
        assert len(grid_1024(10).get_discontinuities("all")) == 0

    @pytest.mark.parametrize(
        "values",
        [[3.0, 1.0, 2.0, 9.0, 4.0], [1.0, 1.0, 2.0, 2.0, 3.0]],
    )
    def test_labels_in_no_order_report_nothing(self, values):
        """Unordered or repeated labels state no spacing to depart from."""
        coord = get_coord(data=np.asarray(values))
        assert len(coord.get_discontinuities("all")) == 0

    def test_a_sorted_stored_run_still_reports(self):
        """Ordered labels do have an expected spacing, and say where it breaks."""
        coord = get_coord(data=np.asarray([0.0, 1.0, 2.0, 5.0, 6.0, 7.0]))
        assert len(coord.get_discontinuities("all")) == 1

    def test_missing_counts_the_grid_positions(self):
        """A declared grid says how many positions have no sample."""
        coord = get_coord(data=[1, 3, 4, 10, 11, 12], step=1)
        table = coord
        assert table.missing().count == 6
        assert not table.missing().complete
        np.testing.assert_array_equal(table.missing().positions(), [2, 5, 6, 7, 8, 9])

    def test_missing_is_empty_for_a_full_grid(self):
        """A run with no holes misses nothing."""
        assert NumericND.from_run(0, 1, (11,)).missing().complete

    def test_a_null_label_is_no_distance(self):
        """A NaN at a boundary states no spacing, so it opens no hole."""
        grid = NumericND.from_run(start=10.0, step=1.0, shape=(3,))
        for values in ([0.0, 1.0, np.nan], [0.0, np.nan, 2.0]):
            stored = NumericND.from_array(np.asarray(values), detect=False)
            assert not concat_tables(stored, grid).holes
        stated = NumericND.from_array(np.asarray([0.0, 1.0, 2.1]), detect=False)
        assert concat_tables(stated, grid).holes

    def test_a_late_tick_run_after_a_stored_one_is_a_hole(self):
        """Ticks past 2**53 still difference exactly, so the hole is seen."""
        stored = NumericND.from_array(T0 + np.asarray([0, 10, 21]) * NS, detect=False)
        far = NumericND.from_run(T0 + 100 * NS, np.timedelta64(10, "ns"), (4,))
        assert concat_tables(stored, far).holes
        near = NumericND.from_run(T0 + 31 * NS, np.timedelta64(10, "ns"), (4,))
        assert not concat_tables(stored, near).holes

    def test_holes_across_a_change_of_rate(self):
        """A change of rate at the next sample is not a hole."""
        a = NumericND.from_run(start=0, step=2, shape=(5,))
        b = NumericND.from_run(start=10, step=3, shape=(4,))
        assert not concat_tables(a, b).holes


class TestSimplifyAndSnap:
    """A table re-fits its runs, or forces one grid over them."""

    def test_simplify_exact_is_a_no_op(self, holed):
        """With no tolerance the holes stay holes."""
        assert holed.fuse().runs_count == holed.runs_count

    def test_simplify_absorbs_a_small_gap(self):
        """A tolerance wide enough collapses the runs into one grid."""
        values = np.asarray([0.0, 1, 2, 3, 10, 11, 12, 13])
        coord = NumericND.from_array(values)
        assert coord.runs_count == 2
        out = coord.fuse(tolerance=4.0)
        assert out.runs_count == 1 and out.evenly_sampled
        assert np.max(np.abs(out.values - values)) <= 4.0

    def test_simplify_keeps_every_sample(self, holed):
        """Simplifying never changes how many samples there are."""
        assert len(holed.fuse(tolerance=np.timedelta64(1, "s"))) == len(holed)

    def test_snap_forces_one_grid(self, holed):
        """Snapping forces one grid of the same length from the same start."""
        out = holed.snap()
        assert out.evenly_sampled
        assert len(out) == len(holed)
        assert out.min() == holed.min()
        assert out.step == np.timedelta64(6758621, "ns")
        assert out.max() - holed.max() == np.timedelta64(9, "ns")

    def test_snap_of_a_grid_is_itself(self):
        """A coordinate already on one grid snaps to itself."""
        coord = grid_1024(10)
        assert coord.snap() is coord


class TestUnitsAndRanges:
    """Units, spans, and sample counts."""

    def test_convert_units(self):
        """Converting scales every label and keeps the sample count."""
        coord = NumericND.from_run(start=0.0, step=1.0, shape=(11,), units="m")
        out = coord.convert_units("cm")
        assert out.units == dc.get_quantity("cm")
        np.testing.assert_array_equal(out.values, coord.values * 100)

    def test_stored_run_converts_its_labels(self):
        """A stored run's own labels are scaled, and its declared step with them."""
        coord = NumericND.from_array(np.asarray([1.0, 2.5, 7.0]), units="m")
        assert coord.sources is not None
        out = coord.convert_units("cm")
        np.testing.assert_allclose(
            out.values, [100.0, 250.0, 700.0], rtol=1e-12, atol=0
        )
        assert out.units == dc.get_quantity("cm")

    def test_every_run_is_converted(self):
        """A run beside a stored one is scaled too, not merely relabelled."""
        coord = concat_tables(
            NumericND.from_run(start=0.0, step=1.0, shape=(5,), units="m"),
            NumericND.from_array(np.asarray([10.0, 11.5, 14.0]), units="m"),
        )
        out = coord.convert_units("cm")
        np.testing.assert_allclose(out.values, coord.values * 100, rtol=1e-12, atol=0)
        assert out.runs_count == coord.runs_count

    def test_a_long_numerator_converts_in_float(self):
        """A float spacing's fraction can be wide; the conversion is not ticks."""
        step = np.pi * 1e6
        coord = concat_tables(
            NumericND.from_run(0.0, step, (2000,), units="m"),
            NumericND.from_run(2e10, step, (2000,), units="m"),
        )
        out = coord.convert_units("ft")
        assert out.sorted
        np.testing.assert_allclose(
            out.values, coord.values / 0.3048, rtol=1e-12, atol=0
        )

    def test_convert_units_of_a_time_is_a_no_op(self):
        """A time coordinate is always in seconds."""
        coord = grid_1024(10)
        assert coord.convert_units("s") is coord

    def test_fractional_integer_grid_refuses_conversion(self):
        """An integer grid with a fractional step is not an integer grid scaled."""
        coord = NumericND.from_run(0, (3, 2), 10, units="m")
        with pytest.raises(CoordError, match="fractional step"):
            coord.convert_units("cm")

    def test_coord_range(self):
        """The span reaches the exclusive end of the last sample."""
        coord = NumericND.from_run(start=0.0, step=2.0, shape=(5,))
        assert coord.coord_range() == 10.0
        assert coord.coord_range(extend=False) == 8.0

    def test_coord_range_needs_one_grid(self, holed):
        """An extended span is only defined for an evenly sampled coordinate."""
        with pytest.raises(CoordError, match="evenly sampled"):
            holed.coord_range()

    def test_get_sample_count(self):
        """A duration counts the samples it spans on an exact grid."""
        coord = grid_1024(2048)
        assert coord.get_sample_count(np.timedelta64(1, "s")) == 1024

    def test_get_next_index_array(self):
        """An array of values gives an array of indices."""
        coord = NumericND.from_run(start=0, step=1, shape=(10,))
        out = coord.get_next_index(np.asarray([0, 3, 9]))
        np.testing.assert_array_equal(out, [0, 3, 9])


class TestManagerContract:
    """The pieces a coordinate manager and the patch API lean on."""

    def test_index_with_a_slice_keeps_the_runs(self, holed):
        """Indexing by a slice keeps the table rather than its labels."""
        out = holed.index(slice(0, 12))
        assert isinstance(out, NumericND)
        np.testing.assert_array_equal(out.values, holed.values[:12])

    def test_align_to(self):
        """Two tables align on the labels they share."""
        a = NumericND.from_run(start=0, step=1, shape=(10,))
        b = NumericND.from_run(start=5, step=1, shape=(10,))
        c1, c2, i1, i2 = a.align_to(b)
        np.testing.assert_array_equal(c1.values, c2.values)
        np.testing.assert_array_equal(a.values[i1], np.arange(5, 10))
        np.testing.assert_array_equal(b.values[i2], np.arange(5, 10))

    def test_approx_equal(self):
        """Labels a hair apart are approximately, but not exactly, equal."""
        a = NumericND.from_run(start=0.0, step=1.0, shape=(10,))
        b = NumericND.from_array(a.values + 1e-12)
        assert a.approx_equal(b)
        assert a != b

    def test_empty_of_an_axis(self):
        """An N-D stored run can be emptied along one axis."""
        coord = NumericND.from_array(np.zeros((3, 4)))
        assert coord.empty(axes=0).shape == (0, 4)

    def test_sort_of_an_unsorted_table(self):
        """Sorting gives back the order which produced it."""
        coord = NumericND.from_array(np.asarray([3.0, 1.0, 2.0]))
        out, order = coord.sort()
        np.testing.assert_array_equal(out.values, coord.values[order])
        assert out.sorted

    def test_update_limits_min_and_max(self):
        """Both ends together re-derive the step over the same count."""
        coord = NumericND.from_run(start=0, step=1, shape=(10,))
        out = coord.update_limits(min=0, max=20)
        assert len(out) == 10 and out.step == 2

    def test_reduce_coord(self):
        """Reducing a table to one sample keeps its dtype."""
        out = grid_1024(10).reduce_coord("mean")
        assert len(out) == 1


class TestIndependentContracts:
    """Independent checks for selection and exact summaries."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(start=0, stop=100, step=2),
            dict(start=T0, stop=T0 + np.timedelta64(1, "s"), step=MS),
            dict(start=0.0, stop=10.0, step=0.5),
        ],
    )
    def test_select_keeps_the_samples_inside_the_window(self, kwargs):
        """A window keeps exactly the labels which lie within it."""
        coord = get_coord(**kwargs)
        values = coord.values
        for low, high in [
            (values[3], values[10]),
            (None, values[5]),
            (values[5], None),
            (values[0], values[0]),
        ]:
            _, index = coord.select((low, high))
            keep = np.ones(len(values), dtype=bool)
            if low is not None:
                keep &= values >= low
            if high is not None:
                keep &= values <= high
            assert list(range(len(values))[index]) == list(np.flatnonzero(keep))

    def test_summary_states_the_exact_grid(self):
        """A fractional grid's summary carries the run that describes it."""
        coord = get_coord(start=T0, step=Fraction(1, 1024), shape=(500,))
        summary = coord.to_summary()
        assert summary.min == T0 and summary.len == 500
        assert summary.max == coord.max()
        # 1/1024 s is 1953125/2 ns, which the grid fields state as it is.
        terms = (
            summary.step_numerator,
            summary.step_denominator,
            summary.origin_offset,
        )
        assert terms == (1953125, 2, 0)
        assert summary.to_coord() == coord


class TestExactArrayConstruction:
    """Labels are evidence to preserve, not a request to fit a grid."""

    @pytest.mark.parametrize("factory", [NumericND.from_array])
    @pytest.mark.parametrize(
        "values",
        [
            np.arange(5000) * 0.1,
            np.arange(1, 100, 1 / 3),
            np.arange(-1, -100, -1 / 3),
            np.linspace(41.5, 41.6, 10),
            np.arange(550) * 1.0213001907746815,
            np.array([0.0, 1.0, 2.0005, 3.0015, 4.002]),
            np.array([-0.0, 1.0, 2.0]),
        ],
    )
    def test_float_bits_survive(self, factory, values):
        """Exact detection and canonicalization preserve all floating bits."""
        coord = factory(data=values)
        assert coord.values.tobytes() == values.tobytes()

    @pytest.mark.parametrize("rate", [49, 1024])
    @pytest.mark.parametrize("first,stride", [(0, 1), (7, 3)])
    def test_rational_time_detection(self, rate, first, stride):
        """Fractional-rate timestamps stay stored unless the grid is declared."""
        source = get_coord(start=T0, step=Fraction(1, rate), shape=(10000,))
        labels = source.values[first::stride]
        coord = NumericND.from_array(labels)
        assert not coord.evenly_sampled
        assert coord.runs_count == 1 and not coord.holes
        np.testing.assert_array_equal(coord.values, labels)

    @pytest.mark.parametrize("rate", [49, 12345, 44100])
    @pytest.mark.parametrize("count", [200, 999])
    def test_short_fractional_time_stays_one_stored_run(self, rate, count):
        """Alternating tick roundoff does not create phantom gaps."""
        source = NumericND.from_run(T0, Fraction(1, rate), count)
        coord = NumericND.from_array(source.values)
        assert coord.runs_count == 1 and not coord.holes
        np.testing.assert_array_equal(coord.values, source.values)

    @pytest.mark.parametrize("item", [slice(None, None, -1), slice(3, None, 7)])
    def test_short_fractional_time_views_stay_one_stored_run(self, item):
        """Descending and strided fractional labels do not fragment."""
        values = NumericND.from_run(T0, Fraction(1, 44100), 500).values[item]
        coord = NumericND.from_array(values)
        assert coord.runs_count == 1 and not coord.holes
        np.testing.assert_array_equal(coord.values, values)

    def test_short_time_array_keeps_a_real_gap(self):
        """Meaningful uniform runs on either side of a gap remain segmented."""
        values = T0 + np.asarray([0, 1, 2, 4, 5, 6]) * NS
        coord = get_coord(data=values)
        assert coord.runs_count == 2 and coord.holes
        np.testing.assert_array_equal(coord.values, values)

    def test_short_many_real_gaps_stay_segmented(self):
        """Many short runs remain distinct and split a spool at their gaps."""
        step = np.timedelta64(4, "ms")
        segments = [
            get_coord(start=T0 + i * 100 * MS, step=step, shape=(3,)) for i in range(10)
        ]
        values = concat_coords(*segments).values
        coord = get_coord(data=values)
        assert coord.runs_count == 10 and coord.holes
        np.testing.assert_array_equal(coord.values, values)

        patch = dc.Patch(data=np.arange(30), coords={"time": coord}, dims=("time",))
        chunks = dc.spool(patch).chunk(time=None)
        assert len(chunks) == 10
        assert [len(x.get_coord("time")) for x in chunks] == [3] * 10

    def test_short_contiguous_rate_change_stays_segmented(self):
        """A short coordinate keeps meaningful runs even without a gap."""
        values = T0 + np.asarray([0, 2, 4, 6, 7, 8, 9]) * MS
        coord = get_coord(data=values)
        assert coord.runs_count == 2
        assert coord.get_discontinuities("gaps").empty
        np.testing.assert_array_equal(coord.values, values)

    def test_timestamp_jitter_is_not_fitted_by_the_builder(self):
        """The builder keeps near-even stamps; `get_coord` reads them as a grid."""
        labels = T0 + np.array([0, 1000000, 2000100, 3000150]) * NS
        kept = NumericND.from_array(labels)
        assert not kept.evenly_sampled
        np.testing.assert_array_equal(kept.values, labels)
        # An array handed to `get_coord` -- as a patch hands one -- is read
        # as the grid its spacings cluster around, as it always has been.
        patch = dc.Patch(data=np.zeros(4), coords={"time": labels}, dims=("time",))
        assert patch.get_coord("time").evenly_sampled

    def test_two_samples_state_an_exact_grid(self):
        """Two distinct values still determine an exact spacing."""
        coord = get_coord(data=[2.0, 5.0])
        assert coord.evenly_sampled
        assert coord.step == 3.0
        np.testing.assert_array_equal(coord.values, [2.0, 5.0])

    @pytest.mark.parametrize("indexer", [slice(10, 15), [20, 10, -1]])
    def test_index_does_not_expand_source(self, monkeypatch, indexer):
        """Selecting a few labels never allocates a billion-label axis."""
        coord = get_coord(start=0, step=2, shape=(10**9,))

        def fail(_self):
            raise AssertionError("source labels were materialized")

        with monkeypatch.context() as local:
            local.setattr(NumericND, "values", property(fail))
            selected = coord.index(indexer)
        expected = (
            [20, 22, 24, 26, 28] if isinstance(indexer, slice) else [40, 20, 1999999998]
        )
        np.testing.assert_array_equal(selected.values, expected)

    @pytest.mark.parametrize(
        "indices", [[10], [-11], np.array([2**64 - 1], dtype="uint64")]
    )
    def test_fancy_index_bounds(self, indices):
        """Lazy indexing rejects out-of-bounds values instead of extending."""
        coord = get_coord(start=0, step=1, shape=(10,))
        with pytest.raises(IndexError):
            coord[indices]

    @pytest.mark.parametrize(
        "values,step", [(np.arange(5000) * 0.1, 0.1), (np.array([-0.0]), -1.0)]
    )
    def test_declared_array_keeps_its_bits(self, values, step):
        """A step declaration does not permit relabeling floating samples."""
        coord = get_coord(data=values, step=step)
        assert coord.values.tobytes() == values.tobytes()
        assert coord.step == step

    @pytest.mark.parametrize("dtype", ["int8", "uint8", "int64", "datetime64[ns]"])
    @pytest.mark.parametrize("count", [2, 3])
    @pytest.mark.parametrize("segmented", [False, True])
    def test_array_at_dtype_limit(self, dtype, count, segmented):
        """Valid labels stay usable when their exclusive grid stop would overflow."""
        integer = "int64" if dtype == "datetime64[ns]" else dtype
        last = int(np.iinfo(integer).max)
        values = np.asarray(list(range(last - count + 1, last + 1)), dtype=dtype)
        if segmented:
            values = np.concatenate([values, np.asarray([0, 1], dtype=dtype)])
        coord = get_coord(data=values)
        np.testing.assert_array_equal(coord.values, values)
        assert coord.dtype == values.dtype

    @pytest.mark.parametrize("factory", [get_coord, NumericND.from_array])
    @pytest.mark.parametrize(
        "values",
        [
            np.array(["1500-01-01", "2020-01-01"], dtype="datetime64[D]"),
            np.array(["2020-01-01", "3000-01-01"], dtype="datetime64[D]"),
            np.array([-500000, 0, 500000], dtype="timedelta64[D]"),
        ],
    )
    def test_temporal_overflow_is_refused(self, factory, values):
        """Coarse dates outside the nanosecond range must never wrap."""
        with pytest.raises(CoordError, match="outside the nanosecond range"):
            factory(data=values)


@pytest.mark.skipif(
    sys.maxsize <= 2**31,
    reason="a run of more samples than this machine indexes cannot be asked its length",
)
class TestRunLimitedByItsLabels:
    """A tick run holds any labels its dtype can; int64 asks only ``length * den``."""

    DAY = 86_400

    @pytest.mark.parametrize(
        ("rate", "days"), [(9999, 20), (12345, 60), (99999, 2), (49, 3000)]
    )
    def test_long_fractional_rate_runs_build(self, rate, days):
        """Rates the product ``length * num`` could not reach are ordinary grids."""
        count = rate * self.DAY * days
        coord = NumericND.from_run(T0, Fraction(1, rate), count)
        assert len(coord) == count
        # the last label against python integers, which cannot wrap
        last = int(T0.astype("int64")) + ((count - 1) * 10**9) // rate
        assert int(coord.max().astype("int64")) == last
        assert int(coord[count - 1].astype("int64")) == last

    def test_long_runs_slice_reverse_and_fuse_exactly(self):
        """Every route through the arithmetic agrees past the old limit."""
        rate, count = 9999, 9999 * self.DAY * 20
        coord = NumericND.from_run(T0, Fraction(1, rate), count)
        cut = count // 2 + 7
        assert concat_tables(coord[:cut], coord[cut:]) == coord
        assert coord[::-1][::-1] == coord
        probe = np.asarray([0, 1, cut - 1, cut, count - 2, count - 1])
        ticks = [int(T0.astype("int64")) + (int(k) * 10**9) // rate for k in probe]
        np.testing.assert_array_equal(
            coord[probe].values.astype("int64"), np.asarray(ticks)
        )
        late = [int(T0.astype("int64")) + ((cut + k) * 10**9) // rate for k in range(4)]
        np.testing.assert_array_equal(coord[cut:][:4].values.astype("int64"), late)

    def test_runs_far_apart_on_a_long_denominator_do_not_meet(self):
        """A difference of starts which would wrap int64 is no continuation."""
        rows = [(0, 4, 1, 2**32, 0), (2**32, 4, 1, 2**32, 4)]
        coord = NumericND.from_rows(rows, dtype="int64")
        assert coord.runs_count == 2
        np.testing.assert_array_equal(coord.values, [0] * 4 + [2**32] * 4)
        ends = NumericND.from_rows(
            [(-(2**62), 2, 1, 1, 0), (2**62 + 2, 2, 1, 1, 0)], dtype="int64"
        )
        assert ends.runs_count == 2 and ends.sorted

    def test_a_run_past_its_dtype_names_what_it_reaches(self):
        """The refusal states the label, not a range the labels never left."""
        with pytest.raises(CoordError, match="exceeds the datetime64"):
            NumericND.from_run(T0, np.timedelta64(1, "h"), 3_000_000)

    def test_a_label_far_outside_the_coordinate_is_refused(self):
        """Extending a grid past int64 raises rather than wrapping."""
        coord = NumericND.from_run(0, 10**12, 10, dtype="int64")
        with pytest.raises(CoordError, match="leaves int64"):
            coord._labels([10**10])


class TestTableIntegrity:
    """What the class refuses, and what makes two tables one."""

    def test_raw_constructor_refuses_a_non_canonical_table(self):
        """An unreduced step handed straight to the class is not a coordinate."""
        good = NumericND.from_run(0, Fraction(1, 2), 4, dtype="int64")
        for row in [(0, 4, 2, 4, 0), (0, 4, 1, 2, 5), (0, -1, 1, 2, 0)]:
            runs = np.asarray([(*row, b"")], good.runs.dtype)
            # pydantic reports a validator's error as its own ValueError
            with pytest.raises(ValueError, match="canonical"):
                NumericND(runs=runs, dtype=good.dtype, shape=(4,))
        again = NumericND(runs=good.runs, dtype=good.dtype, shape=good.shape)
        assert again == good

    def test_byte_order_is_no_part_of_a_run(self):
        """A table read from a big-endian file hashes as its native twin does."""
        coord = grid_1024(300)
        swapped = coord.runs.astype(coord.runs.dtype.newbyteorder(">"))
        other = NumericND.from_rows(swapped, dtype=coord.dtype)
        assert other == coord and other.data_id == coord.data_id

    def test_one_label_is_one_coordinate(self):
        """A run of one sample shows no spacing or phase of its own.

        Where it was cut from, and how its grid was spelled, are no part of
        what it is. The grid itself still is: `step` answers for it, and a
        lone label on a finer grid is another coordinate.
        """
        parent = grid_1024(10)
        label = parent.values[3]
        floats = get_coord(data=np.arange(10) * 0.1)
        families = (
            [
                parent[3:4],
                NumericND.from_run(label, Fraction(1, 1024), 1),
                NumericND.from_array(np.asarray([label]), step=parent.step),
            ],
            [
                NumericND.from_run(0.5, 0.1, 1),
                floats[5:6],
            ],
        )
        for spellings in families:
            assert len({x.data_id for x in spellings}) == 1
            assert len({x.step for x in spellings}) == 1
            assert all(x == spellings[0] for x in spellings)
        # the same label on another grid, or on none at all
        assert parent[3:4].data_id != NumericND.from_run(label, MS, 1).data_id
        assert parent[3:4].data_id != NumericND.from_array(np.asarray([label])).data_id

    @pytest.mark.parametrize("start", [np.datetime64("NaT"), np.nan])
    def test_a_run_needs_a_first_label(self, start):
        """A null start raises the coordinate's own error."""
        step = MS if isinstance(start, np.datetime64) else 1.0
        with pytest.raises(CoordError, match="first label"):
            NumericND.from_run(start, step, 3)

    def test_extreme_integers_keep_their_order(self):
        """Labels a difference of which leaves int64 still read as sorted."""
        coord = get_coord(data=np.asarray([-(2**62), 2**62]))
        assert coord.sorted and not coord.reverse_sorted
        assert get_coord(data=np.asarray([2**62, -(2**62)])).reverse_sorted


class TestEdges:
    """The corners of the table: what it refuses, and what it falls back to."""

    def test_documents_state_runs_as_summaries(self):
        """A summary read from a document turns its runs back into coordinates."""
        run = dict(dtype="int64", min=0, max=4, step=1, len=5, step_numerator=1)
        summary = CoordSummary(
            dtype="int64",
            min=0,
            max=9,
            len=10,
            runs=[run, {**run, "min": 5, "max": 9}],
        )
        assert len(summary.runs) == 2
        np.testing.assert_array_equal(summary.to_coord().values, np.arange(10))

    @pytest.mark.parametrize(
        ("dtype", "units", "expected"),
        [
            ("int64", None, [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]),
            ("float64", "m", [0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ],
    )
    def test_a_summary_states_a_summary_per_run(self, dtype, units, expected):
        """A coordinate of several runs summarises each of them."""
        low, high = (0, 12) if dtype == "int64" else (0.0, 12.0)
        step = 1 if dtype == "int64" else 1.0

        def segment(start, stop):
            return {
                "dtype": dtype,
                "min": start,
                "max": stop,
                "step": step,
                "units": units,
                "len": 5,
                "object_type": "CoordSummary",
            }

        document = {
            "dtype": dtype,
            "min": low,
            "max": high,
            "step": None,
            "units": units,
            "len": 10,
            "runs": [segment(low, low + 4), segment(high - 4, high)],
            "object_type": "CoordSummary",
        }
        summary = CoordSummary.model_validate(json.loads(json.dumps(document)))
        assert len(summary.runs) == 2
        np.testing.assert_array_equal(summary.to_coord().values, expected)

    def test_a_legacy_float_grid_is_read_as_its_step(self):
        """The fraction an older float summary stated is the step it divides to."""
        # the scalar step disagrees, so it is the fraction which is honoured
        summary = CoordSummary(
            dtype="float64",
            min=0.0,
            max=0.4,
            step=0.25,
            step_numerator=1,
            step_denominator=10,
        )
        np.testing.assert_array_equal(summary.to_coord().values, np.arange(5) * 0.1)

    def test_a_declared_step_is_exact_beside_stored_labels(self):
        """Labels held as they are still state the grid they were declared on."""
        ticks = np.delete(np.arange(3000, dtype="int64"), np.arange(500, 2500)) * 10**6
        coord = get_coord(data=T0 + ticks.astype("timedelta64[ns]"), step=MS)
        assert coord.step_exact == Fraction(1, 1000)
        assert coord.missing().count == 2000

    @pytest.mark.parametrize(
        "step", [(1, 0), (1, -2), (1, 2, 3), Fraction(2**70, 3)], ids=str
    )
    def test_a_step_no_run_can_hold_is_refused(self, step):
        """A malformed or oversized fraction raises the coordinate's own error."""
        with pytest.raises(CoordError):
            NumericND.from_run(T0, step, 3)

    def test_stored_labels_with_a_declared_step(self):
        """The declared grid is what a hole after stored labels is held against."""
        labels = T0 + np.asarray([0, 1, 2, 5]) * MS
        stored = NumericND.from_array(labels, step=MS, detect=False)
        assert stored.sources is not None and stored.step_exact == Fraction(1, 1000)
        rows = np.concatenate(
            [stored.runs, NumericND.from_run(T0 + 10 * MS, MS, 3).runs]
        )
        joined = NumericND.from_rows(rows, labels=labels, dtype=labels.dtype, step=MS)
        assert joined.step == MS and joined.holes
        # a step the labels' dtype cannot hold declares nothing
        ints = NumericND.from_rows(
            [(0, 3, 0, 0, 0)], labels=np.asarray([0, 1, 3]), dtype="int64", step=0.5
        )
        assert ints.step is None

    def test_nothing_but_nulls(self):
        """Labels which are all missing are held, in nanoseconds like any time."""
        nulls = np.asarray(["NaT", "NaT"], dtype="datetime64[s]")
        coord = NumericND.from_array(nulls)
        assert (
            coord.dtype == np.dtype("datetime64[ns]") and pd.isnull(coord.values).all()
        )
        assert CoordPartial(shape=(3,), step=1.0).missing().count == 0

    def test_a_label_past_int64_is_refused(self):
        """An integer no tick can be is named, not wrapped."""
        with pytest.raises(CoordError, match="outside the int64 range"):
            _to_tick(2**64)

    def test_a_float_step_on_integer_labels_declares_nothing(self):
        """A step the labels' dtype cannot hold is refused."""
        with pytest.raises(CoordError, match="non-integer"):
            get_coord(data=np.asarray([0, 1, 3]), step=0.5)

    def test_labels_no_grid_floors_to_are_stored(self):
        """Spacings a tick apart need not be floors of any one line."""
        uneven = (np.asarray([0, 1, 2, 4, 6, 7, 8]) + 10).astype("datetime64[ns]")
        vast = np.asarray([0, 2**61, 2**62 + 1]).astype("datetime64[ns]")
        for values in (uneven, vast):
            coord = NumericND.from_array(values)
            assert not coord.evenly_sampled
            np.testing.assert_array_equal(coord.values, values)

    def test_python_time_inputs(self):
        """A python timedelta states a duration as numpy's does."""
        coord = get_coord(
            start=datetime.timedelta(0), step=datetime.timedelta(seconds=1), shape=3
        )
        assert coord.dtype == np.dtype("timedelta64[ns]") and len(coord) == 3

    def test_from_run_takes_a_shape_of_any_rank(self):
        """A shape is a sample count, however many axes spell it."""
        assert NumericND.from_run(0, 1, (2, 3)).shape == (6,)

    def test_stored_labels_answer_the_range_questions(self):
        """Labels of their own have a stop, and refuse what only a grid can do."""
        coord = get_coord(data=[1.0, 2.0, 4.0])
        assert coord.stop == 4.0
        with pytest.raises(NotImplementedError, match="change its length"):
            coord.change_length(5)
        np.testing.assert_array_equal(coord.new(start=0).values, coord.values)
        with pytest.raises(CoordError, match="declared step"):
            coord.missing()

    def test_indexing_corners(self):
        """A mask must fit, and an index of one axis gives one label."""
        coord = get_coord(start=0, stop=10, step=1)
        with pytest.raises(IndexError, match="Boolean index"):
            coord[np.asarray([True, False])]
        assert coord[(3,)] == 3
        np.testing.assert_array_equal(coord.new(min=5).values[:2], [5, 6])

    def test_a_stride_past_int64_is_refused(self):
        """Whichever path slices a run, its step has to stay a step."""
        one = NumericND.from_run(0, 2**61, 3, dtype="int64")
        with pytest.raises(CoordError, match="past int64"):
            one._sliced(0, 8, 2)
        late = NumericND.from_run(2**63 - 2**62, 2**59, 2, dtype="int64")
        two = concat_tables(one, late)
        with pytest.raises(CoordError, match="past int64"):
            two[::16]

    def test_runs_share_a_dtype_and_stand_alone(self):
        """Tables of different widths do not join, and runs state everything."""
        wide = get_coord(start=0.0, stop=3.0, step=1.0)
        narrow = NumericND.from_array(np.asarray([5, 6, 7], dtype="f4"))
        with pytest.raises(CoordError, match="share a dtype"):
            concat_tables(wide, narrow)
        with pytest.raises(CoordError, match="cannot be combined"):
            get_coord(runs=wide.runs, data=[1, 2])
        ints = get_coord(start=0, stop=3, step=1)
        late = get_coord(start=5, stop=8, step=1)
        assert concat_coords(ints.model_dump(), late).runs_count == 2

    def test_unordered_runs_are_left_as_they_are(self):
        """Runs in no order have no seams to close."""
        early, late = (
            get_coord(start=0, stop=5, step=1),
            get_coord(start=3, stop=8, step=1),
        )
        mixed = concat_tables(late, early)
        assert mixed.fuse(10) is mixed

    def test_a_hole_after_stored_labels(self):
        """Stored labels state the spacing the next run is held against."""
        jitter = NumericND.from_array(np.asarray([0.0, 1.0, 2.1]), detect=False)
        one = NumericND.from_array(np.asarray([7.0]), detect=False)
        grid = NumericND.from_run(20.0, 1.0, 4)
        assert concat_tables(jitter, grid).holes
        # one label states no spacing, so nothing says the next run is late
        assert not concat_tables(one, grid).holes
        declared = NumericND.from_array(
            np.asarray([0.0, 1.0, 2.0, 5.0]), step=1.0, detect=False
        )
        assert declared.get_next_index(9.0, allow_out_of_bounds=True) == 9


class TestKernelCorners:
    """What each kernel refuses, asked of it directly."""

    def test_rows_from_scalars(self):
        """A table of one run may be spelled without a single list."""
        rows = _rows(np.dtype("int64"), 0, 3, 1, 1, 0)
        assert rows.shape == (1,) and rows["length"][0] == 3

    def test_a_run_too_long_for_its_denominator(self):
        """``length * den`` is the one thing int64 asks of a tick run."""
        step = Fraction(2**30 + 1, 2**30)
        with pytest.raises(CoordError, match="too long for int64"):
            NumericND.from_run(0, step, 2**40, dtype="int64")
        short = NumericND.from_run(0, Fraction(10**6 + 1, 10**6), 10, dtype="int64")
        with pytest.raises(CoordError, match="too long for int64"):
            short._labels([10**13])
        assert get_kernel(short.dtype).labels(short.runs, 0, []).shape == (0,)

    def test_float_runs_stay_countable_and_finite(self):
        """A grid index past 2**53, or a label past the floats, is refused."""
        with pytest.raises(CoordError, match="range a float64 counts"):
            NumericND.from_rows(float_rows("f8", [0.0], [4], 1.0, 1, 2**53), dtype="f8")
        with pytest.raises(CoordError, match="not finite"):
            NumericND.from_run(1e308, 1e308, 10)
        two = concat_tables(
            NumericND.from_run(0.0, 1.0, 4), NumericND.from_run(10.0, 1.0, 4)
        )
        with pytest.raises(CoordError, match="range a float64 counts"):
            two[:: 2**54]

    def test_flat_float_runs_share_a_step(self):
        """Runs of no spacing meet on a grid only where they all start alike."""
        flat = NumericND.from_run(1.0, 0.0, 3)
        assert concat_tables(flat, flat).step == 0.0
        assert concat_tables(flat, NumericND.from_run(2.0, 0.0, 3)).step is None


class TestOtherCoordsAnswerTheSameQuestions:
    """What the run table asks of itself, a string or a shape answers too."""

    def test_partial(self):
        """A coordinate without values has nothing to fuse, empty, or measure."""
        coord = CoordPartial(shape=(3, 4), units="m")
        assert not coord.has_values and get_coord(data=[1, 2]).has_values
        assert coord.set_units("m") is coord and coord.fuse(1) is coord
        assert coord.empty(axes=0).shape == (0, 4)
        with pytest.raises(CoordError, match="evenly sampled"):
            coord.coord_range()
        assert pd.isnull(coord.coord_range(extend=False))

    def test_string(self):
        """Text has no runs, so no seams and no holes."""
        coord = get_coord(data=np.asarray(["a", "b"]))
        assert len(coord.get_discontinuities()) == 0


class TestReviewFindings:
    """Corners a correctness review found, each pinned by what it broke."""

    def test_a_label_far_from_its_origin_is_found(self):
        """Inverting a label far from the origin loses more than a rounding."""
        coord = get_coord(start=1e9, step=0.1, shape=20)
        label = coord[1]
        assert coord.get_next_index(label) == 1
        np.testing.assert_array_equal(coord.select((label, label))[0].values, [label])

    def test_a_narrow_float_boundary_label_is_found(self):
        """A float32 label rounding below its run's wider head is still its run's."""
        base = get_coord(data=(np.arange(10) * 0.1).astype("float32"))
        coord = concat_coords(base[:3], base[7:])
        label = coord[3]
        assert coord.get_next_index(label) == 3
        np.testing.assert_array_equal(coord.select((label, label))[0].values, [label])

    def test_float_grid_index_arithmetic_cannot_wrap(self):
        """A grid index past what int64 and float64 both count is refused."""
        rows = float_rows("f8", [0.0], [4], 1.0, 2**62)
        with pytest.raises(CoordError, match="float64 counts"):
            NumericND.from_rows(rows, dtype="f8")

    def test_a_phase_carry_cannot_wrap(self):
        """A first label carried past int64 by its phase is refused."""
        with pytest.raises(CoordError, match="carries its first label"):
            NumericND.from_rows([(2**63 - 2, 2, 0, 1, 2), (0, 2, 1, 1, 0)], dtype="i8")
        with pytest.raises(CoordError, match="carries its first label"):
            NumericND.from_rows([(2**63 - 2, 2, 0, 1, 2)], dtype="i8")

    def test_equal_labels_are_equal_however_stated(self):
        """A short run cannot show its phase; its labels still decide equality."""
        declared = NumericND.from_run(T0, Fraction(1, 3), 3)
        rebuilt = get_coord(data=declared.values)
        assert declared == rebuilt
        assert declared != NumericND.from_run(T0 + NS, Fraction(1, 3), 3)
        # the same ends and different interiors
        inner = get_coord(data=T0 + np.asarray([0, 2, 3]) * NS)
        assert inner != get_coord(data=T0 + np.asarray([0, 1, 3]) * NS)

    def test_a_negative_zero_origin_is_kept(self):
        """The sign of a zero is part of the label an array gave."""
        values = np.asarray([-0.0, -1.0, -2.0, -10.0, -11.0, -12.0])
        coord = get_coord(data=values)
        np.testing.assert_array_equal(np.signbit(coord.values), np.signbit(values))

    def test_an_estimate_past_the_label_is_walked_back(self):
        """The division can land a rounding past a label the value sits below."""
        rows = float_rows("f8", [0.0], [1000], 0.1, 1, -297)
        coord = NumericND.from_rows(rows, dtype="f8")
        value = 10.999999999999998  # a double below the label 0.1 * (407 - 297)
        assert coord.values[407] == 11.0
        assert coord._index_sorted(value, forward=True) == 407
        # a value a rounding below a label is that label
        assert coord._index_sorted(np.nextafter(10.9, 0), forward=False) == 406

    def test_a_value_beyond_what_a_float_counts(self):
        """An index past 2**53 is no label, only the side it lies on."""
        coord = get_coord(start=0.0, step=0.1, shape=10)
        assert coord._index_sorted(1e300, forward=True) >= len(coord)

    def test_stored_n_d_labels_differ_by_their_labels(self):
        """Arrays of any rank are equal exactly when their labels are."""
        one = get_coord(data=np.arange(6.0).reshape(2, 3))
        assert one == get_coord(data=np.arange(6.0).reshape(2, 3))
        assert one != get_coord(data=np.arange(1.0, 7.0).reshape(2, 3))

    def test_a_lossy_time_cast_names_the_label(self):
        """A cast which drops part of a label reports that label."""
        value = np.asarray(["2020-01-01T00:00:00.5"], dtype="datetime64[ns]")
        assert _out_of_ns(value, np.dtype("datetime64[s]")) == value[0]


class TestSecondReviewFindings:
    """Corners a second review found, each pinned by what it broke."""

    def test_selecting_one_narrow_float_label_takes_one_sample(self):
        """The rounding a float32 label allows must not reach its neighbour."""
        coord = get_coord(data=(1e6 + np.arange(10) * 0.07).astype("float32"))
        label = coord.values[2]
        out = coord.select((label, label))[0]
        np.testing.assert_array_equal(out.values, [label])

    def test_a_missing_time_is_no_minimum(self):
        """NaT held as a tick would sort first and be handed back as the min."""
        values = np.asarray(["NaT", "2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
        coord = get_coord(data=values)
        assert not coord.sorted and not coord.reverse_sorted
        assert coord.min() == values[1]
        assert coord.max() == values[2]

    def test_runs_far_apart_on_a_fine_grid_stay_one_grid(self):
        """A tick difference times its denominator leaves int64; the grid holds."""
        # two ten-sample runs of one 9999 Hz grid, twenty days apart. Built
        # as rows rather than sliced from the run between them, which is
        # 1.7e10 samples long and has no index on a 32 bit build.
        away = 9999 * 86400 * 20
        num, den = _NS_PER_S, 9999
        start = int(T0.astype("i8"))
        far, phase = divmod(away * num, den)
        coord = NumericND.from_rows(
            [(start, 10, num, den, 0), (start + far, 10, num, den, phase)],
            dtype="datetime64[ns]",
        )
        assert coord.runs_count == 2
        assert coord.step == NumericND.from_run(T0, Fraction(1, 9999), 2).step
        # and the hole is counted on the runs' own grid, not on the whole
        # ticks its step rounds to, which disagree after a few hours
        (missing,) = coord.missing().runs
        assert missing[2] == away - 10


class TestBotReviewFindings:
    """The narrow and wide floats, and the ends of int64, a bot review found."""

    def test_equal_narrow_float_labels_hash_alike(self):
        """A float32 run is its labels, not the float64 arithmetic behind them."""
        one = NumericND.from_run(0.1, 0.001, 6, dtype="float32")
        # a start a double apart, which float32 rounds to the same label
        two = NumericND.from_run(np.nextafter(0.1, 1.0), 0.001, 6, dtype="float32")
        assert one.runs.tolist() != two.runs.tolist()
        np.testing.assert_array_equal(one.values, two.values)
        assert one == two
        assert one.data_id == two.data_id

    def test_promotion_keeps_the_labels_it_was_given(self):
        """A float32 run under float64 arithmetic would lose its own rounding."""
        narrow = NumericND.from_run(0.1, 0.001, 6, dtype="float32")
        wide = NumericND.from_run(1.0, 0.001, 4, dtype="float64")
        joined = concat_coords(narrow, wide)
        assert joined.dtype == np.dtype("float64")
        np.testing.assert_array_equal(joined.values[:6], narrow.values.astype("f8"))

    @pytest.mark.skipif(
        np.dtype(np.longdouble).itemsize <= 8, reason="longdouble is a double here"
    )
    def test_an_extended_float_keeps_its_own_labels(self):
        """A grid counted in float64 cannot state a label wider than one."""
        start = np.nextafter(np.longdouble(1), np.longdouble(2))
        coord = get_coord(start=start, step=np.longdouble("0.1"), shape=(5,))
        assert coord.dtype == np.dtype(np.longdouble)
        assert coord.values[0] == start
        # and snapping it does not send the labels through a double either
        assert coord.snap() is coord

    def test_a_shift_past_the_end_of_int64_is_refused(self):
        """A wrapped label would validate as an ordinary one."""
        coord = get_coord(data=np.asarray([0, 1, 5], dtype="int64"))
        with pytest.raises(CoordError, match="outside int64"):
            coord.update_limits(min=2**63 - 3)

    def test_a_bound_the_dtype_cannot_hold_is_refused(self):
        """The difference to it overflows before any shift is worked out."""
        coord = get_coord(data=np.asarray([0, 1, 5], dtype="int64"))
        for bound in (2**63 + 10, -(2**63) - 10):
            with pytest.raises(CoordError, match="not a label"):
                coord.update_limits(min=bound)

    def test_a_narrow_float_run_reaching_infinity_is_refused(self):
        """Finite float64 ends can still be infinite once cast to float32."""
        with pytest.raises(CoordError, match="finite"):
            NumericND.from_run(np.float32(3e38), 3e37, 10, dtype="float32")


class TestThirdReviewFindings:
    """What the second round of fixes left, each pinned by what it broke."""

    def test_a_narrow_float_label_of_its_own_run_is_found(self):
        """Two labels of one run can round together; the run still holds both."""
        coord = NumericND.from_run(1e6, 0.04, 2, dtype="float32")
        # float32 counts by 0.0625 here, so the row's 1e6 + 0.04 and its
        # 1e6 + 0.08 are one label, and only the first is the run's
        np.testing.assert_array_equal(coord.values, [1000000.0, 1000000.0625])
        label = coord.values[1]
        assert coord.get_next_index(label) == 1
        np.testing.assert_array_equal(coord.select((label, label))[0].values, [label])


def object_coord(values):
    """A coordinate over labels numpy has no arithmetic for."""
    out = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        out[index] = value
    return get_coord(data=out)


class TestFourthReviewFindings:
    """What the fourth review found, each pinned by what it broke."""

    def test_a_hole_is_spelled_on_the_run_s_own_grid(self):
        """A fractional step places a hole's labels; the whole ticks round it."""
        grid = NumericND.from_run(0, (3, 2), 40, dtype="int64")
        coord = concat_coords(grid[:5], grid[15:])
        missing = coord.missing()
        np.testing.assert_array_equal(missing.positions(), grid[5:15].values)
        assert list(missing.iter_runs()) == [(grid.values[5], grid.values[14])]
        assert missing.count == 10
        assert f"[{grid.values[5]} … {grid.values[14]}]" in str(missing)

    def test_a_long_hole_on_a_fine_grid_ends_where_it_should(self):
        """At 1024 Hz the rounded step drifts a microsecond over one outage."""
        grid = grid_1024(4096 * 3)
        coord = concat_coords(grid[:4096], grid[8192:])
        missing = coord.missing()
        assert list(missing.iter_runs()) == [(grid.values[4096], grid.values[8191])]

    def test_a_lone_label_keeps_the_step_it_declares(self):
        """One label shows no spacing, so its declared grid is part of its id."""
        fine = get_coord(data=np.array([1.0]), step=0.5)
        coarse = get_coord(data=np.array([1.0]), step=2.0)
        assert fine.step == 0.5 and coarse.step == 2.0
        assert fine.data_id != coarse.data_id
        one = get_coord(data=np.array([T0]), step=MS)
        two = get_coord(data=np.array([T0]), step=2 * MS)
        assert one.data_id != two.data_id

    def test_one_label_with_no_arithmetic_is_named_by_its_source(self):
        """An object label lives only in its source, which its id must name."""
        first, second = object_coord([(1,)]), object_coord([(2,)])
        assert first.data_id != second.data_id
        assert first != second
        assert first == object_coord([(1,)])

    def test_object_labels_survive_an_end_probe(self):
        """Comparing two object coordinates reads their labels, not a float."""
        values = [(1,), (2,), (3,)]
        assert object_coord(values) != object_coord([(1,), (2,), (4,)])
        assert object_coord(values) == object_coord(values)

    def test_object_labels_concatenate(self):
        """Two runs of labels with no arithmetic still hand back their own."""
        first, second = object_coord([(1,), (2,)]), object_coord([(3,), (4,)])
        coord = concat_tables(first, second)
        assert coord.runs_count == 2
        assert list(coord.values) == [(1,), (2,), (3,), (4,)]
        assert list(coord._labels([0, 3])) == [(1,), (4,)]

    def test_index_of_reads_a_falling_grid(self):
        """A run heading down maps a tick the same way one heading up does."""
        assert TickKernel.index_of((-33, 2, -7, 1, 0), -83, False) == 7
        rng = np.random.default_rng(0)
        window = range(-40, 40)
        for _ in range(300):
            num = int(rng.integers(-9, 9))
            num = num if num else 1
            den = int(rng.integers(1, 6))
            start, offset = int(rng.integers(-50, 50)), int(rng.integers(0, den))
            row = (start, 12, num, den, offset)
            labels = {k: start + (offset + k * num) // den for k in window}
            reached = (lambda a, b: a >= b) if num > 0 else (lambda a, b: a <= b)
            for anchor in range(min(labels.values()) + 1, max(labels.values())):
                ahead = [k for k in window if reached(labels[k], anchor)]
                behind = [k for k in window if not reached(labels[k], anchor)]
                behind += [k for k in window if labels[k] == anchor]
                assert TickKernel.index_of(row, anchor, True) == min(ahead)
                assert TickKernel.index_of(row, anchor, False) == max(behind)
