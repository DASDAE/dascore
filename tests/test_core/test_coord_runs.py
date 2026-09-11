"""Tests for runs against a declared step, missing positions, and one gap verdict."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import (
    CoordMonotonicArray,
    CoordRange,
    CoordSegmented,
    Missing,
    concat_coords,
    get_coord,
)
from dascore.exceptions import CoordError, ParameterError
from dascore.units import get_quantity
from dascore.utils.gaps import GapTolerance, get_gap_edges

PRESENT = [1, 3, 4, 10, 11, 12]
T0 = np.datetime64("2020-01-01T00:00:00")
MS = np.timedelta64(1, "ms")


@pytest.fixture(scope="module")
def design_case():
    """The plan's design case: six labels on a step-1 grid with six missing."""
    return get_coord(data=PRESENT, step=1)


class TestDeclaredStep:
    """A step on array data declares the grid the values sit on."""

    def test_runs_of_the_step(self, design_case):
        """Consecutive positions become ranges, singletons included."""
        assert isinstance(design_case, CoordSegmented)
        assert design_case.segment_count == 3
        assert all(isinstance(x, CoordRange) for x in design_case.segments)
        assert [len(x) for x in design_case.segments] == [1, 2, 3]
        assert design_case.step == 1
        np.testing.assert_array_equal(design_case.values, PRESENT)

    def test_no_step_makes_no_claim(self):
        """Without a step the same values are one monotonic array."""
        coord = get_coord(data=PRESENT)
        assert isinstance(coord, CoordMonotonicArray)
        assert coord.step is None

    def test_off_grid_raises(self):
        """A value off the grid is refused rather than moved."""
        with pytest.raises(CoordError, match="not on a grid"):
            get_coord(data=[1, 3, 4.5], step=1)

    def test_non_monotonic_with_step_raises(self):
        """A grid claim needs an order to hold it against."""
        with pytest.raises(CoordError, match="monotonic"):
            get_coord(data=[3, 1, 2], step=1)

    def test_fractional_step_on_values_raises(self):
        """Labels on a fractional grid do not state it; only a range can."""
        values = get_coord(start=T0, step=(1, 1024), shape=(8,)).values
        with pytest.raises(CoordError, match="fractional step"):
            get_coord(data=values, step=(1, 1024))

    def test_full_grid_is_a_range(self):
        """Values filling their grid come back as the range they are."""
        coord = get_coord(data=np.arange(10) * 0.1, step=0.1)
        assert isinstance(coord, CoordRange)
        assert coord.step == 0.1

    def test_float_grid(self):
        """Floats sit on a grid within a small tolerance of the step."""
        coord = get_coord(data=[0.0, 0.1, 0.2, 0.5, 0.6], step=0.1)
        assert coord.segment_count == 2
        assert coord.missing().count == 2

    @pytest.mark.parametrize("step", [1, -1])
    def test_descending_values(self, step):
        """Descending values keep their order; the step's sign follows them."""
        coord = get_coord(data=PRESENT[::-1], step=step)
        assert coord.reverse_sorted
        assert coord.step == -1
        np.testing.assert_array_equal(coord.values, PRESENT[::-1])
        assert coord.missing().count == 6

    def test_time_grid(self):
        """A time coordinate declares a timedelta step."""
        labels = T0 + np.array([0, 4, 8, 20, 24]) * MS
        coord = get_coord(data=labels, step=4 * MS)
        assert coord.segment_count == 2
        missing = coord.missing()
        assert missing.count == 2
        assert list(missing.iter_runs()) == [(T0 + 12 * MS, T0 + 16 * MS)]

    def test_two_values(self):
        """Two values apart by more than a step are two runs."""
        coord = get_coord(data=[0, 5], step=1)
        assert coord.segment_count == 2
        assert coord.missing().count == 4

    def test_dense_guard_keeps_the_step(self):
        """Past the guard the values stay one array, step and missing intact."""
        sparse = np.arange(300)[np.arange(300) % 3 != 2]
        dense = np.arange(3000)[np.arange(3000) % 3 != 2]
        as_runs = get_coord(data=sparse, step=1)
        as_array = get_coord(data=dense, step=1)
        assert isinstance(as_runs, CoordSegmented)
        assert isinstance(as_array, CoordMonotonicArray)
        assert as_array.step == 1
        # the trailing removed position lies past the last sample
        assert as_runs.missing().count == 99
        assert as_array.missing().count == 999
        # slicing and units keep the declaration
        assert as_array[10:50].step == 1
        assert as_array.set_units("m").convert_units("ft").step != 1

    def test_direct_array_checks_the_grid(self):
        """Constructing an array coordinate with a step checks its values."""
        assert CoordMonotonicArray(values=np.array([0, 2, 5]), step=1).step == 1
        with pytest.raises(ValueError, match="not on a grid"):
            CoordMonotonicArray(values=np.array([0, 2, 5.5]), step=1)
        with pytest.raises(ValueError, match="monotonic"):
            CoordMonotonicArray(values=np.array([[0, 1], [2, 3]]), step=1)

    def test_range_values_update_drops_the_step(self):
        """New values on a range state their own grid, not the range's."""
        rolled = np.roll(get_coord(start=0, stop=6, step=1).values, 2)
        coord = get_coord(start=0, stop=6, step=1).update(values=rolled)
        assert coord.step is None

    def test_summary_stays_an_envelope(self, design_case):
        """A summary with a step rebuilds a range, which runs are not."""
        assert design_case.to_summary().step is None
        array = get_coord(data=np.arange(3000)[np.arange(3000) % 3 != 2], step=1)
        assert array.to_summary().step is None

    def test_mixed_steps_report_none(self):
        """Runs of different steps share no grid."""
        coord = concat_coords(
            get_coord(start=0.0, stop=5.0, step=1.0),
            get_coord(start=8.0, stop=13.0, step=0.5),
        )
        assert coord.step is None
        with pytest.raises(CoordError, match="declared step"):
            coord.missing()


class TestFusion:
    """Array segments fuse only when nothing says they should not."""

    def test_undeclared_arrays_fuse(self):
        """Arrays without a step carry no expectation and fuse as before."""
        left = CoordMonotonicArray(values=np.array([0.0, 1.0, 2.5]))
        right = CoordMonotonicArray(values=np.array([9.0, 9.7, 11.0]))
        coord = concat_coords(left, right)
        assert isinstance(coord, CoordMonotonicArray)

    def test_declared_arrays_fuse_across_one_step(self):
        """Arrays on one grid meeting one step apart are one array."""
        left = CoordMonotonicArray(values=np.array([0, 2, 3]), step=1)
        right = CoordMonotonicArray(values=np.array([4, 7, 9]), step=1)
        coord = concat_coords(left, right)
        assert isinstance(coord, CoordMonotonicArray)
        assert coord.step == 1

    def test_declared_arrays_keep_a_gap(self):
        """A seam wider than a step stays a seam."""
        left = CoordMonotonicArray(values=np.array([0, 2, 3]), step=1)
        right = CoordMonotonicArray(values=np.array([6, 9, 11]), step=1)
        coord = concat_coords(left, right)
        assert isinstance(coord, CoordSegmented)
        assert coord.step == 1
        assert coord.missing().count == 6

    def test_one_declared_one_not_keeps_the_seam(self):
        """An expectation on one side is not shared by the other."""
        left = CoordMonotonicArray(values=np.array([0.0, 2.0, 3.0]), step=1.0)
        right = CoordMonotonicArray(values=np.array([4.0, 7.5, 9.0]))
        assert isinstance(concat_coords(left, right), CoordSegmented)


class TestMissing:
    """The positions a grid has no sample at."""

    def test_design_case(self, design_case):
        """[1, 3, 4, 10, 11, 12] on a step-1 grid misses six positions."""
        missing = design_case.missing()
        assert isinstance(missing, Missing)
        assert missing.count == 6
        assert not missing.complete
        assert list(missing.iter_runs()) == [(2, 2), (5, 9)]
        np.testing.assert_array_equal(missing.positions(), [2, 5, 6, 7, 8, 9])
        assert "[2] | [5 … 9]" in str(missing)

    def test_range_is_complete(self):
        """A range misses nothing."""
        missing = get_coord(start=0, stop=10, step=1).missing()
        assert missing.complete and missing.count == 0
        assert not len(missing.positions())

    def test_positions_limit(self, design_case):
        """Spelling out more positions than the limit raises."""
        with pytest.raises(ParameterError, match="exceed the limit"):
            design_case.missing().positions(limit=5)

    def test_no_step_raises(self):
        """Missing is relative to a grid nothing declared."""
        for coord in (get_coord(data=[1.0, 2.5, 7.0]), get_coord(shape=(5,))):
            with pytest.raises(CoordError, match="declared step"):
                coord.missing()

    def test_array_holes(self):
        """A dense array with a step reports its holes from the spacings."""
        coord = CoordMonotonicArray(values=np.array([0, 1, 5, 6, 9]), step=1)
        missing = coord.missing()
        assert list(missing.iter_runs()) == [(2, 4), (7, 8)]
        np.testing.assert_array_equal(missing.positions(), [2, 3, 4, 7, 8])

    def test_both_sides_of_the_guard_agree(self):
        """Runs and the guarded array answer missing the same way."""
        values = np.arange(3000)[np.arange(3000) % 3 != 2]
        runs = CoordSegmented.from_array(values[values < 300], step=1)
        array = CoordSegmented.from_array(values, step=1)
        assert isinstance(runs, CoordSegmented) and isinstance(
            array, CoordMonotonicArray
        )
        assert (
            list(runs.missing().iter_runs()) == list(array.missing().iter_runs())[:99]
        )


class TestDiscontinuities:
    """One frame builder and one predicate for every representation."""

    def test_array_seams_against_declared_step(self):
        """Every spacing off the declared step is a discontinuity."""
        coord = CoordMonotonicArray(values=np.array([0, 1, 5, 6, 9]), step=1)
        seams = coord.get_discontinuities()
        assert seams["index"].tolist() == [2, 4]
        assert seams["excess"].tolist() == [3, 2]
        assert coord.get_discontinuities("gaps", tolerance=2)["index"].tolist() == [2]

    def test_array_seams_against_median(self):
        """Without a step the median spacing is the expectation."""
        coord = CoordMonotonicArray(values=np.array([0.0, 1.0, 2.0, 5.0, 6.0]))
        gaps = coord.get_discontinuities("gaps", tolerance=0.5)
        assert gaps["index"].tolist() == [3]
        assert coord.get_discontinuities("gaps", tolerance=3)["index"].tolist() == []

    def test_count_tolerance(self, design_case):
        """A sample count measures the spacing in steps."""
        gaps = design_case.get_discontinuities("gaps", GapTolerance.samples(1.5))
        assert gaps["index"].tolist() == [1, 3]
        wide = design_case.get_discontinuities("gaps", GapTolerance.samples(3))
        assert wide["index"].tolist() == [3]

    def test_quantity_tolerance(self):
        """A quantity converts to the coordinate's units as an excess."""
        coord = get_coord(data=[0.0, 1.0, 2.0, 4.5, 5.5], step=None, units="m")
        assert isinstance(coord, CoordMonotonicArray)
        gaps = coord.get_discontinuities("gaps", get_quantity("100 cm"))
        assert gaps["index"].tolist() == [3]
        assert not len(coord.get_discontinuities("gaps", get_quantity("2 m")))

    def test_range_reports_none(self):
        """An evenly sampled coordinate has no seams."""
        assert get_coord(start=0, stop=5, step=1).get_discontinuities().empty

    def test_single_value_array(self):
        """One value has no spacing to judge."""
        assert CoordMonotonicArray(values=np.array([3.0])).get_discontinuities().empty

    def test_negative_tolerance_raises(self, design_case):
        """A negative excess cannot be met."""
        with pytest.raises(ParameterError, match="negative"):
            design_case.get_discontinuities("gaps", tolerance=-1)


class TestOneVerdict:
    """Coordinate, chunk, and waterfall agree on one (spacing, step, tolerance)."""

    @pytest.fixture(scope="class")
    def patches(self):
        """Three patches: the first two 1.4 steps apart, the third 1.6 steps on."""
        base = dc.get_example_patch()
        time = base.get_coord("time")
        step = time.step
        second = base.update_coords(time_min=time.max() + 1.4 * step)
        third = second.update_coords(
            time_min=second.get_coord("time").max() + 1.6 * step
        )
        return base, second, third

    def test_chunk_and_coordinate_agree(self, patches):
        """The default tolerance splits at 1.6 steps and merges 1.4."""
        merged = dc.spool(patches).chunk(time=None)
        assert len(merged) == 2
        values = np.concatenate([p.get_coord("time").values for p in patches])
        coord = CoordSegmented.from_array(values)
        gaps = coord.get_discontinuities("gaps", GapTolerance.samples(1.5))
        assert len(gaps) == 1
        assert gaps["before"].iloc[0] == patches[1].get_coord("time").max()
        _, mask = get_gap_edges(coord, GapTolerance.samples(1.5))
        assert mask.sum() == 1
        assert np.flatnonzero(mask)[0] + 1 == gaps["index"].iloc[0]

    def test_absolute_tolerance_agrees(self, patches):
        """An absolute excess reads the same at both sites."""
        step = patches[0].get_coord("time").step
        excess = 0.5 * step
        merged = dc.spool(patches).chunk(time=None, tolerance=excess)
        assert len(merged) == 2
        values = np.concatenate([p.get_coord("time").values for p in patches])
        gaps = CoordSegmented.from_array(values).get_discontinuities("gaps", excess)
        assert len(gaps) == 1
