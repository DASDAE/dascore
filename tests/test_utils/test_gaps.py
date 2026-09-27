"""Tests for coordinate gap utilities."""

from __future__ import annotations

import numpy as np
import pytest

from dascore.core.coords import concat_coords, get_coord
from dascore.exceptions import ParameterError, UnitError
from dascore.units import get_quantity
from dascore.utils.gaps import (
    GapTolerance,
    contiguous_windows,
    gap_boundaries,
    get_gap_edges,
    is_monotonic_and_finite,
)


class TestGetGapEdges:
    """Tests for constructing coordinate cell edges."""

    def test_numeric_without_gaps(self):
        """Numeric coordinates produce midpoint cell edges."""
        edges, gaps = get_gap_edges([0, 1, 2])
        np.testing.assert_allclose(edges, [-0.5, 0.5, 1.5, 2.5])
        assert not np.any(gaps)

    def test_numeric_with_gap(self):
        """Large numeric intervals are expanded into a gap."""
        edges, gaps = get_gap_edges([0, 1, 5, 6], GapTolerance.samples(1.5))
        np.testing.assert_allclose(edges, [-0.5, 0.5, 1.5, 4.5, 5.5, 6.5])
        np.testing.assert_array_equal(gaps, [False, True, False])

    def test_timedelta(self):
        """Timedeltas are converted to seconds before constructing edges."""
        values = np.array([0, 1, 2], dtype="timedelta64[s]")
        edges, gaps = get_gap_edges(values)
        np.testing.assert_allclose(edges, [-0.5, 0.5, 1.5, 2.5])
        assert not np.any(gaps)

    def test_datetime(self):
        """Datetime edges retain datetime dtype and support gaps."""
        values = np.array(
            ["2020-01-01", "2020-01-02", "2020-01-06"],
            dtype="datetime64[D]",
        )
        edges, gaps = get_gap_edges(values, GapTolerance.samples(1.5))
        expected = np.array(
            [
                "2019-12-31T12:00:00",
                "2020-01-01T12:00:00",
                "2020-01-03T06:00:00",
                "2020-01-04T18:00:00",
                "2020-01-07T06:00:00",
            ],
            dtype="datetime64[ns]",
        )
        assert np.issubdtype(edges.dtype, np.datetime64)
        np.testing.assert_array_equal(edges, expected)
        np.testing.assert_array_equal(gaps, [False, True])

    def test_declared_step_sets_the_cells(self):
        """A coordinate's declared step, not the median spacing, sizes gaps."""
        coord = get_coord(data=[0, 3, 6, 7], step=1)
        edges, gaps = get_gap_edges(coord, GapTolerance.samples(1.5))
        # every three-wide spacing is a gap against the step of one, and
        # each side of it gets a cell one step wide
        np.testing.assert_array_equal(gaps, [True, True, False])
        np.testing.assert_allclose(edges, [-0.5, 0.5, 2.5, 3.5, 5.5, 6.5, 7.5])
        # the median spacing (3) would have called none of them a gap
        _, by_median = get_gap_edges(coord.values, GapTolerance.samples(1.5))
        assert not by_median.any()

    def test_declared_datetime_step(self):
        """A datetime coordinate's declared step sizes its gap cells."""
        t0 = np.datetime64("2020-01-01T00:00:00")
        second = np.timedelta64(1, "s")
        coord = get_coord(data=t0 + np.array([0, 1, 5, 6]) * second, step=second)
        edges, gaps = get_gap_edges(coord, GapTolerance.samples(1.5))
        np.testing.assert_array_equal(gaps, [False, True, False])
        half = np.timedelta64(500, "ms")
        expected = t0 + np.array([-1, 1, 3, 9, 11, 13]) * half
        np.testing.assert_array_equal(edges, expected.astype("datetime64[ns]"))

    def test_absolute_seconds_on_datetime(self):
        """An absolute tolerance in seconds measures datetime spacings."""
        t0 = np.datetime64("2020-01-01T00:00:00")
        values = t0 + np.array([0, 1, 2, 4]) * np.timedelta64(1, "s")
        _, gaps = get_gap_edges(values, GapTolerance.absolute(0.001))
        np.testing.assert_array_equal(gaps, [False, False, True])

    def test_singleton_datetime(self):
        """Singleton datetimes warn and receive a one-day default cell width."""
        with pytest.warns(UserWarning, match="Singleton coordinate"):
            edges, gaps = get_gap_edges(np.array(["2020-01-01"], dtype="datetime64[D]"))
        expected = np.array(
            ["2019-12-31T12:00:00", "2020-01-01T12:00:00"], dtype="datetime64[ns]"
        )
        np.testing.assert_array_equal(edges, expected)
        assert not len(gaps)

    def test_singleton_numeric(self):
        """Singleton numeric coordinates warn and receive a unit cell width."""
        with pytest.warns(UserWarning, match="Singleton coordinate"):
            edges, gaps = get_gap_edges([10])
        np.testing.assert_allclose(edges, [9.5, 10.5])
        assert not len(gaps)


class TestIsMonotonicAndFinite:
    """Tests for validating mesh coordinate centers."""

    @pytest.mark.parametrize("values", [[0, 1, 2], [2, 1, 0]])
    def test_valid(self, values):
        """Ascending and descending finite values are valid."""
        assert is_monotonic_and_finite(values)

    @pytest.mark.parametrize("values", [[0, 2, 1], [0, np.nan, 2]])
    def test_invalid(self, values):
        """Nonmonotonic and nonfinite values are invalid."""
        assert not is_monotonic_and_finite(values)


class TestGapTolerance:
    """One tolerance object, two spellings, one predicate."""

    def test_str(self):
        """Each spelling reads as what it counts."""
        assert str(GapTolerance.samples(2)) == "2 samples"

    def test_from_user_forms(self):
        """Numbers count steps; quantities and timedeltas are absolute."""
        assert GapTolerance.from_user(2) == GapTolerance.samples(2.0)
        assert GapTolerance.from_user(get_quantity("2")) == GapTolerance.samples(2.0)
        absolute = GapTolerance.from_user(get_quantity("3 m"))
        assert absolute.excess == get_quantity("3 m") and absolute.count is None
        delta = GapTolerance.from_user(np.timedelta64(1, "s"))
        assert delta.excess == np.timedelta64(1_000_000_000, "ns")
        tol = GapTolerance.samples(1.5)
        assert GapTolerance.from_user(tol) is tol

    def test_predicate(self):
        """A gap is past the count of steps, or past one step plus the excess."""
        count = GapTolerance.samples(1.5)
        assert count.is_gap(1.6, 1.0) and not count.is_gap(1.4, 1.0)
        excess = GapTolerance.absolute(0.5)
        assert excess.is_gap(1.6, 1.0) and not excess.is_gap(1.4, 1.0)
        # an unknown step is never a gap against a count, nothing against an excess
        assert not count.is_gap(10.0, np.nan)
        assert excess.is_gap(0.6, np.nan) and not excess.is_gap(0.4, np.nan)
        # vectorised over spacings, magnitudes only
        np.testing.assert_array_equal(
            count.is_gap([-2.0, 1.0], [1.0, 1.0]), [True, False]
        )

    @pytest.mark.parametrize(
        "bad, error",
        [
            (-1, ParameterError),
            (np.nan, ParameterError),
            ([1, 2], ParameterError),
            ("1 s", ParameterError),
            (get_quantity("1 MB"), UnitError),
            (get_quantity("10 %"), UnitError),
            (get_quantity("inf s"), ParameterError),
        ],
    )
    def test_rejected(self, bad, error):
        """What cannot measure a gap is refused with a reason."""
        with pytest.raises(error):
            GapTolerance.from_user(bad, "time")

    def test_infinite_count_allowed(self):
        """No boundary is ever a gap against an infinite count."""
        assert GapTolerance.from_user(np.inf).count == np.inf

    def test_one_form_only(self):
        """A tolerance is a count or an excess, never both or neither."""
        with pytest.raises(ParameterError):
            GapTolerance(count=1.0, excess=1.0)
        with pytest.raises(ParameterError):
            GapTolerance()


class TestGapBoundaries:
    """The reach sweep over runs."""

    def test_reach_shields_nested_runs(self):
        """A run behind the furthest stop seen never opens a gap."""
        starts = np.array([0.0, 20.0, 5.0, 40.0])
        stops = np.array([30.0, 25.0, 8.0, 45.0])
        steps = np.ones(4)
        order, reach, has_gap = gap_boundaries(
            starts, stops, steps, GapTolerance.samples(1.5)
        )
        assert order.tolist() == [0, 2, 1, 3]
        assert reach.tolist() == [30.0, 30.0, 30.0, 30.0]
        assert has_gap.tolist() == [False, False, False, True]

    def test_unstated_steps(self):
        """Rows stating no step (None) are measured without one."""
        steps = np.array([None, None], dtype=object)
        _, _, has_gap = gap_boundaries(
            [0.0, 10.0], [5.0, 12.0], steps, GapTolerance.samples(1.5)
        )
        assert has_gap.tolist() == [False, False]

    def test_first_row_never_a_gap(self):
        """Nothing lies behind the first run."""
        _, _, has_gap = gap_boundaries([10.0], [12.0], [1.0], GapTolerance.samples(1.5))
        assert has_gap.tolist() == [False]


class TestContiguousWindows:
    """Positional windows between a coordinate's run boundaries and holes."""

    def test_one_grid_is_one_window(self):
        """An evenly sampled coordinate is a single window."""
        assert contiguous_windows(get_coord(start=0, step=1, shape=(5,))) == [(0, 5)]

    def test_runs_and_holes(self):
        """Windows break at a run boundary and at each hole inside a run."""
        coord = concat_coords(
            get_coord(data=np.delete(np.arange(9), [2, 5]), step=1),
            get_coord(start=20, step=1, shape=(3,)),
        )
        assert contiguous_windows(coord) == [(0, 2), (2, 4), (4, 7), (7, 10)]

    def test_reverse_holes(self):
        """A descending coordinate breaks at its holes too."""
        coord = get_coord(data=np.array([9, 8, 7, 4, 3, 1, 0]), step=-1)
        assert contiguous_windows(coord) == [(0, 3), (3, 5), (5, 7)]

    def test_grids_of_differing_steps_split(self):
        """Evenly sampled runs split even when they share no step."""
        coord = concat_coords(
            get_coord(start=0.0, step=1.0, shape=(3,)),
            get_coord(start=3.0, step=1.5, shape=(3,)),
        )
        assert coord.runs_count == 2 and coord.step is None
        assert contiguous_windows(coord) == [(0, 3), (3, 6)]

    def test_irregular_runs_are_one_window(self):
        """Runs including irregular labels and no shared step are not split."""
        coord = concat_coords(
            get_coord(start=0.0, step=1.0, shape=(3,)),
            get_coord(data=np.array([3.0, 4.5, 7.0])),
        )
        assert coord.runs_count == 2 and coord.step is None
        assert contiguous_windows(coord) == [(0, 6)]

    @pytest.mark.parametrize(
        "coord",
        [
            get_coord(data=np.array(["a", "b"])),
            get_coord(data=np.sort(np.random.default_rng(1).random(9))),
        ],
    )
    def test_nothing_to_split(self, coord):
        """Strings and irregular labels without a step are one window."""
        assert contiguous_windows(coord) == [(0, len(coord))]
