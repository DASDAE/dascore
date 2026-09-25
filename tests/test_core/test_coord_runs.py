"""Tests for runs against a declared step, missing positions, and one gap verdict."""

from __future__ import annotations

import copy
import pickle
from fractions import Fraction

import h5py
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core.coords import (
    Grid,
    Labels,
    Missing,
    NumericCoord,
    concat_coords,
    get_coord,
)
from dascore.exceptions import CoordError, ParameterError
from dascore.io.dasdae.utils import _read_coord
from dascore.units import get_quantity
from dascore.utils.gaps import GapTolerance, get_gap_edges

PRESENT = [1, 3, 4, 10, 11, 12]
T0 = np.datetime64("2020-01-01T00:00:00")
MS = np.timedelta64(1, "ms")


def _is_stored(coord):
    """Whether the coordinate holds its labels as one stored run."""
    return coord.runs_count == 1 and isinstance(coord.runs[0], Labels)


@pytest.fixture(scope="module")
def design_case():
    """The plan's design case: six labels on a step-1 grid with six missing."""
    return get_coord(data=PRESENT, step=1)


class TestDeclaredStep:
    """A step on array data declares the grid the values sit on."""

    def test_runs_of_the_step(self, design_case):
        """Consecutive positions become ranges, singletons included."""
        assert design_case.runs_count == 3
        assert all(x.evenly_sampled for x in design_case.segments)
        assert [len(x) for x in design_case.segments] == [1, 2, 3]
        assert design_case.step == 1
        np.testing.assert_array_equal(design_case.values, PRESENT)

    def test_no_step_makes_no_claim(self):
        """Without a step the same values are one monotonic array."""
        coord = get_coord(data=PRESENT)
        assert _is_stored(coord) and coord.sorted
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
        assert coord.evenly_sampled
        assert coord.step == 0.1

    def test_float_grid(self):
        """Floats sit on a grid within a small tolerance of the step."""
        coord = get_coord(data=[0.0, 0.1, 0.2, 0.5, 0.6], step=0.1)
        assert coord.runs_count == 2
        assert coord.missing().count == 2

    @pytest.mark.parametrize("step", [1, -1])
    def test_descending_values(self, step):
        """Descending values keep their order; the step's sign follows them."""
        coord = get_coord(data=PRESENT[::-1], step=step)
        assert coord.reverse_sorted
        assert coord.step == -1
        np.testing.assert_array_equal(coord.values, PRESENT[::-1])
        missing = coord.missing()
        assert list(missing.iter_runs()) == [(9, 5), (2, 2)]
        np.testing.assert_array_equal(missing.positions(), [9, 8, 7, 6, 5, 2])
        dense = NumericCoord.from_labels(np.array([9, 6, 5, 1]), step=1)
        assert list(dense.missing().iter_runs()) == [(8, 7), (4, 2)]

    def test_time_grid(self):
        """A time coordinate declares a timedelta step."""
        labels = T0 + np.array([0, 4, 8, 20, 24]) * MS
        coord = get_coord(data=labels, step=4 * MS)
        assert coord.runs_count == 2
        missing = coord.missing()
        assert missing.count == 2
        assert list(missing.iter_runs()) == [(T0 + 12 * MS, T0 + 16 * MS)]

    def test_two_values(self):
        """Two values apart by more than a step are two runs."""
        coord = get_coord(data=[0, 5], step=1)
        assert coord.runs_count == 2
        assert coord.missing().count == 4

    def test_dense_guard_keeps_the_step(self):
        """Past the guard the values stay one array, step and missing intact."""
        sparse = np.arange(300)[np.arange(300) % 3 != 2]
        dense = np.arange(3000)[np.arange(3000) % 3 != 2]
        as_runs = get_coord(data=sparse, step=1)
        as_array = get_coord(data=dense, step=1)
        assert as_runs.runs_count > 1
        assert _is_stored(as_array)
        assert as_array.step == 1
        # the trailing removed position lies past the last sample
        assert as_runs.missing().count == 99
        assert as_array.missing().count == 999
        # slicing and units keep the declaration; a fancy index that
        # loses the order loses it, since nothing could hold it
        assert as_array[10:50].step == 1
        assert as_array[[5, 0, 3]].step is None
        in_feet = as_array.set_units("m").convert_units("ft")
        assert in_feet.step == pytest.approx(1 / 0.3048)

    def test_direct_array_checks_the_grid(self):
        """Constructing an array coordinate with a step checks its values."""
        assert NumericCoord.from_labels(np.array([0, 2, 5]), step=1).step == 1
        with pytest.raises(ValueError, match="not on a grid"):
            NumericCoord.from_labels(np.array([0, 2, 5.5]), step=1)
        with pytest.raises(ValueError, match="monotonic"):
            NumericCoord.from_labels(np.array([[0, 1], [2, 3]]), step=1)

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

    def test_mixed_steps_cannot_say_what_is_missing(self):
        """Runs of different steps share no grid to be missing from."""
        coord = concat_coords(
            get_coord(start=0.0, stop=5.0, step=1.0),
            get_coord(start=8.0, stop=13.0, step=0.5),
        )
        assert coord.step is None and coord.missing().complete


class TestFusion:
    """Array segments fuse only when nothing says they should not."""

    def test_undeclared_arrays_fuse(self):
        """Arrays without a step carry no expectation and claim no grid."""
        left = NumericCoord.from_labels(np.array([0.0, 1.0, 2.5]))
        right = NumericCoord.from_labels(np.array([9.0, 9.7, 11.0]))
        coord = concat_coords(left, right)
        assert coord.sorted and not coord.evenly_sampled
        assert coord.step is None

    def test_declared_arrays_fuse_across_one_step(self):
        """Arrays on one grid meeting one step apart leave no gap at the seam."""
        left = NumericCoord.from_labels(np.array([0, 2, 3]), step=1)
        right = NumericCoord.from_labels(np.array([4, 7, 9]), step=1)
        coord = concat_coords(left, right)
        assert coord.step == 1
        assert coord.missing().count == 4

    def test_declared_arrays_keep_a_gap(self):
        """A seam wider than a step stays a seam."""
        left = NumericCoord.from_labels(np.array([0, 2, 3]), step=1)
        right = NumericCoord.from_labels(np.array([6, 9, 11]), step=1)
        coord = concat_coords(left, right)
        assert coord.runs_count > 1
        assert coord.step == 1
        assert coord.missing().count == 6

    def test_one_declared_one_not_keeps_the_seam(self):
        """An expectation on one side is not shared by the other."""
        left = NumericCoord.from_labels(np.array([0.0, 2.0, 3.0]), step=1.0)
        right = NumericCoord.from_labels(np.array([4.0, 7.5, 9.0]))
        assert concat_coords(left, right).runs_count > 1


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

    def test_no_step_misses_nothing(self):
        """Missing is relative to a grid nothing declared."""
        for coord in (get_coord(data=[1.0, 2.5, 7.0]), get_coord(shape=(5,))):
            assert coord.missing().complete

    def test_array_holes(self):
        """A dense array with a step reports its holes from the spacings."""
        coord = NumericCoord.from_labels(np.array([0, 1, 5, 6, 9]), step=1)
        missing = coord.missing()
        assert list(missing.iter_runs()) == [(2, 4), (7, 8)]
        np.testing.assert_array_equal(missing.positions(), [2, 3, 4, 7, 8])

    def test_both_sides_of_the_guard_agree(self):
        """Runs and the guarded array answer missing the same way."""
        values = np.arange(3000)[np.arange(3000) % 3 != 2]
        runs = get_coord(data=values[values < 300], step=1)
        array = get_coord(data=values, step=1)
        assert runs.runs_count > 1 and _is_stored(array)
        assert (
            list(runs.missing().iter_runs()) == list(array.missing().iter_runs())[:99]
        )


class TestFractionalGaps:
    """Gap consumers retain the cadence and phase of fractional grids."""

    @pytest.fixture(params=[1024, 3000])
    def full(self, request):
        """A fractional grid whose rounded labels alternate spacings."""
        return get_coord(start=T0, step=Fraction(1, request.param), shape=(30,))

    @pytest.mark.parametrize("reverse", [False, True])
    def test_missing_labels(self, full, reverse):
        """Holes are slices of the original grid, including after reversal."""
        if reverse:
            full = full[::-1]
        coord = concat_coords(full[2:7], full[11:18], full[22:])
        missing = coord.missing()
        assert coord.step_exact == full.step_exact
        assert missing.count == 8
        assert missing == Missing(missing.step, missing.runs, missing.dtype)
        np.testing.assert_array_equal(
            missing.positions(), np.concatenate([full.values[7:11], full.values[18:22]])
        )

    def test_one_verdict(self, full):
        """A one-sample hole meets a two-step spacing limit everywhere."""
        coord = concat_coords(full[:5], full[6:])
        assert coord.get_discontinuities("gaps", tolerance=float(full.step_exact)).empty
        assert coord.get_discontinuities("gaps", pd.Timedelta(milliseconds=1)).empty
        assert coord.get_discontinuities("gaps", GapTolerance.samples(2)).empty
        assert len(coord.get_discontinuities("gaps", GapTolerance.samples(1.5))) == 1
        assert not get_gap_edges(coord, GapTolerance.samples(2))[1].any()
        assert get_gap_edges(coord, GapTolerance.samples(1.5))[1].sum() == 1

    @pytest.mark.parametrize("one_patch", [False, True])
    @pytest.mark.parametrize("selected", [False, True])
    def test_spool(self, full, one_patch, selected):
        """Indexed run seams and boundaries between patches agree."""
        coords = (full[:5], full[6:])
        if one_patch:
            coords = (concat_coords(*coords),)
        patches = [
            dc.Patch(data=np.ones(len(c)), coords={"time": c}, dims=("time",))
            for c in coords
        ]
        spool = dc.spool(patches)
        if selected:
            spool = spool.select(time=(full.values[2], full.values[-3]))
            full = full[2:-2]
        assert spool.get_gaps(tolerance=2).empty
        assert spool.get_gaps(tolerance=np.inf).empty
        absolute = GapTolerance.absolute(float(full.step_exact))
        assert spool.get_gaps(tolerance=absolute).empty
        assert len(spool.get_gaps(tolerance=1.5)) == 1
        assert len(spool.chunk(time=None, tolerance=1.5)) == 2
        out = spool.chunk(time=None, tolerance=2, fill_value=np.nan)
        assert len(out) == 1
        np.testing.assert_array_equal(out[0].get_coord("time").values, full.values)
        assert np.flatnonzero(np.isnan(out[0].data)).tolist() == [3 if selected else 5]
        between = full.values[0] + np.array([10, 20], dtype="timedelta64[ns]")
        assert spool.select(time=tuple(between)).get_gaps().empty

    def test_allowance_is_bounded(self, full):
        """The rounding allowance is a few nanoseconds, not a looser tolerance."""
        tolerance, step = GapTolerance.samples(2), full.step
        assert not tolerance.is_gap(2 * step + np.timedelta64(4, "ns"), step)
        assert tolerance.is_gap(2 * step + np.timedelta64(5, "ns"), step)
        # a whole extra sample is never within the allowance
        step = np.timedelta64(10_000, "ns")
        assert GapTolerance.samples(10_000).is_gap(10_001 * step, step)
        # nor at steps of a few nanoseconds
        step = np.timedelta64(4, "ns")
        assert GapTolerance.samples(1.5).is_gap(2 * step, step)
        assert GapTolerance.samples(1.9).is_gap(2 * step, step)

    def test_new_keeps_labels(self, full):
        """Rebuilding a joined fractional coordinate keeps its labels."""
        coord = concat_coords(full[:5], full[6:])
        assert coord.new(dtype=coord.dtype) == coord
        rebuilt = coord.new(dtype=coord.dtype, units="s")
        np.testing.assert_array_equal(rebuilt.values, coord.values)

    def test_long_outage_labels(self):
        """A weeks-long hole on a fractional grid keeps ordered end labels."""
        start = np.datetime64("2024-01-01", "ns")
        a = get_coord(start=start, step=Fraction(1, 3001), shape=(5,))
        b = get_coord(
            start=start + np.timedelta64(60, "D"), step=a.step_exact, shape=(5,)
        )
        ((first, last, count),) = concat_coords(a, b).missing().runs
        assert first < last < b.min()
        assert count == 60 * 86_400 * 3001 - 5

    def test_declared_finer_step(self):
        """A declared step finer than the runs' grid names the holes."""
        coord = NumericCoord(
            runs=(Grid(0, 2, 1, 1), Grid(4, 2, 1, 1)), dtype="int64", step=1
        )
        assert coord.missing().positions().tolist() == [1, 2, 3]
        assert coord.step_exact == 1

    def test_elapsed_time_edges(self):
        """Elapsed-time seams get the same verdict as the coordinate report."""
        full = get_coord(
            start=np.timedelta64(0, "ns"), step=Fraction(1, 3000), shape=(30,)
        )
        coord = concat_coords(full[:5], full[6:])
        assert coord.get_discontinuities("gaps", GapTolerance.samples(2)).empty
        assert not get_gap_edges(coord, GapTolerance.samples(2))[1].any()

    def test_off_lattice_runs(self):
        """Runs whose origins fall between each other's positions share no grid."""
        full = get_coord(start=100, step=Fraction(10, 3), shape=(5,), dtype="int64")
        other = get_coord(start=121, step=Fraction(10, 3), shape=(5,), dtype="int64")
        assert concat_coords(full, other).step_exact is None

    def test_metadata_only(self, monkeypatch):
        """A long outage is counted without allocating the coordinate labels."""
        full = get_coord(start=T0, step=Fraction(1, 1024), shape=(2_000_000_000,))
        coord = concat_coords(full[:5], full[-5:])

        def refuse_values(self):
            raise AssertionError("materialized the coordinate")

        monkeypatch.setattr(NumericCoord, "values", property(refuse_values))
        assert coord.missing().count == 1_999_999_990

    def test_index_round_trip(self, tmp_path):
        """A reopened file index retains fractional run phases after selection."""
        full = get_coord(start=T0, step=Fraction(1, 3000), shape=(30,))
        coord = concat_coords(full[:8], full[9:])
        patch = dc.Patch(
            data=np.ones(len(coord)), coords={"time": coord}, dims=("time",)
        )
        dc.write(patch, tmp_path / "gapped.h5", "dasdae")
        dc.spool(tmp_path).update()
        spool = dc.spool(tmp_path).select(time=(full.values[2], full.values[-3]))
        assert spool.get_gaps(tolerance=2).empty
        assert len(spool.get_gaps(tolerance=1.5)) == 1
        assert len(spool.chunk(time=None, tolerance=1.5)) == 2

    def test_fuse_keeps_hole(self, full):
        """Fusing with keep_step leaves a whole-sample hole in place."""
        coord = concat_coords(full[:5], full[7:])
        fused = coord.fuse(GapTolerance.samples(3), keep_step=True)
        assert fused.missing().count == 2

    def test_partial_fuse(self, full):
        """Fusing some runs off the lattice keeps the others as they were."""
        coord = concat_coords(full[:5], full[6:10], full[25:])
        fused = coord.fuse(GapTolerance.samples(1))
        assert fused.runs_count == 2
        np.testing.assert_array_equal(fused.values[-5:], full.values[25:])

    @pytest.mark.parametrize("rate", [1024, 3000])
    def test_independent_files(self, rate):
        """Files whose start times were rounded apart still meet a tolerance."""
        patches, position = [], 0
        for dropped in (0, 1, 0, 2, 0):
            start = T0 + np.timedelta64(round(position * 10**9 / rate), "ns")
            coord = get_coord(start=start, step=Fraction(1, rate), shape=(50,))
            patches.append(
                dc.Patch(data=np.ones(50), coords={"time": coord}, dims=("time",))
            )
            position += 50 + dropped
        spool = dc.spool(patches)
        found = [len(spool.get_gaps(tolerance=x)) for x in (1.5, 2, 3)]
        assert found == [2, 1, 0]
        assert [len(spool.chunk(time=None, tolerance=x)) for x in (2, 3)] == [2, 1]

    @pytest.mark.parametrize("step", [Fraction(10, 3), -3])
    def test_integer_grid(self, step):
        """Fractional grids of integer labels also retain their exact holes."""
        full = get_coord(start=100, step=step, shape=(30,), dtype="int64")
        coord = concat_coords(full[:5], full[9:])
        assert coord.step_exact == full.step_exact
        np.testing.assert_array_equal(coord.missing().positions(), full.values[5:9])


class TestDiscontinuities:
    """One frame builder and one predicate for every representation."""

    def test_array_seams_against_declared_step(self):
        """Every spacing off the declared step is a discontinuity."""
        coord = NumericCoord.from_labels(np.array([0, 1, 5, 6, 9]), step=1)
        seams = coord.get_discontinuities()
        assert seams["index"].tolist() == [2, 4]
        assert seams["excess"].tolist() == [3, 2]
        assert coord.get_discontinuities("gaps", tolerance=2)["index"].tolist() == [2]

    def test_array_seams_against_median(self):
        """Without a step the median spacing is the expectation."""
        coord = NumericCoord.from_labels(np.array([0.0, 1.0, 2.0, 5.0, 6.0]))
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
        assert _is_stored(coord)
        gaps = coord.get_discontinuities("gaps", get_quantity("100 cm"))
        assert gaps["index"].tolist() == [3]
        assert not len(coord.get_discontinuities("gaps", get_quantity("2 m")))

    def test_single_value_array(self):
        """One value has no spacing to judge."""
        assert NumericCoord.from_labels(np.array([3.0])).get_discontinuities().empty

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
        coord = get_coord(data=values, snap=False)
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
        coord = get_coord(data=values, snap=False)
        gaps = coord.get_discontinuities("gaps", excess)
        assert len(gaps) == 1


class TestReviewFindings:
    """Cases a review found the first cut got wrong."""

    def test_singleton_runs_keep_their_order(self):
        """Descending values whose runs are all singletons stay descending."""
        coord = get_coord(data=[9, 6, 3], step=1)
        np.testing.assert_array_equal(coord.values, [9, 6, 3])
        assert coord.step == -1 and coord.missing().count == 4

    def test_promotion_respects_the_declared_grid(self):
        """Even spacing on a finer declared grid is not a range of that spacing."""
        coord = get_coord(data=np.arange(0, 2000, 2), step=1)
        assert _is_stored(coord)
        assert concat_coords(coord).missing().count == coord.missing().count == 999

    def test_gapped_runs_are_not_evenly_sampled(self, design_case):
        """A step on runs does not admit them where even sampling is required."""
        patch = dc.get_example_patch().select(distance=(0, 6), samples=True)
        patch = patch.update_coords(distance=design_case)
        with pytest.raises(CoordError, match="not evenly sampled"):
            patch.get_coord("distance", require_evenly_sampled=True)
        with pytest.raises(CoordError, match="not evenly sampled"):
            patch.update_attrs(data_type="velocity").velocity_to_strain_rate_edgeless()

    def test_median_spacing_ignores_orientation(self):
        """Either orientation of the same labels reports the same gap."""
        tolerance = GapTolerance.samples(1.2)
        up = NumericCoord.from_labels(np.array([0.0, 1.0, 4.0]))
        down = NumericCoord.from_labels(np.array([4.0, 1.0, 0.0]))
        assert len(up.get_discontinuities("gaps", tolerance)) == 1
        assert len(down.get_discontinuities("gaps", tolerance)) == 1

    def test_declared_step_converts_as_a_difference(self):
        """An affine unit's offset does not reach the step."""
        coord = NumericCoord.from_labels(
            np.array([0.0, 2.0, 3.0]), step=1, units="degC"
        )
        converted = coord.convert_units("degF")
        assert converted.step == pytest.approx(1.8)
        assert converted.missing().count == 1

    def test_bare_absolute_excess_in_chunk(self):
        """An excess stated as a bare number is in the coordinate's units."""
        first = dc.get_example_patch()
        first = first.update_coords(distance=first.get_coord("distance").values * 1.0)
        dist = first.get_coord("distance")
        second = first.update_coords(distance_min=dist.max() + 2.4 * dist.step)
        spool = dc.spool([first, second])
        assert len(spool.chunk(distance=None, tolerance=GapTolerance.absolute(2))) == 1
        assert len(spool.chunk(distance=None, tolerance=GapTolerance.absolute(1))) == 2

    def test_dasdae_round_trip(self, tmp_path):
        """Version 2 stores a declared step; version 1 keeps the values alone."""
        dense = get_coord(data=np.arange(3000)[np.arange(3000) % 3 != 2], step=1)
        assert _is_stored(dense)
        time = dc.get_example_patch().get_coord("time")[:3]
        coords = {"distance": dense, "time": time}
        data = np.zeros((len(dense), 3))
        patch = dc.Patch(data=data, coords=coords, dims=("distance", "time"))
        back = dc.read(dc.write(patch, tmp_path / "v2.h5", "dasdae"))[0]
        assert back.get_coord("distance") == dense
        old = dc.read(dc.write(patch, tmp_path / "v1.h5", "dasdae", file_version="1"))[
            0
        ]
        np.testing.assert_array_equal(old.get_coord("distance").values, dense.values)
        assert old.get_coord("distance").step is None

    def test_offset_grids_share_no_step(self):
        """Runs of one step on grids half a step apart declare no common grid."""
        coord = concat_coords(
            get_coord(start=0.0, stop=3.0, step=1.0),
            get_coord(start=5.5, stop=8.5, step=1.0),
        )
        assert coord.step is None
        left = NumericCoord.from_labels(np.array([0.0, 2.0, 3.0]), step=1.0)
        right = NumericCoord.from_labels(np.array([4.5, 7.5, 9.5]), step=1.0)
        assert concat_coords(left, right).runs_count > 1

    def test_complete_time_grid_positions_dtype(self):
        """No missing positions still come back in the coordinate's dtype."""
        coord = get_coord(start=T0, step=np.timedelta64(1, "s"), shape=(5,))
        assert coord.missing().positions().dtype == coord.dtype

    def test_data_id_sees_the_declared_step(self):
        """Declaring a grid changes what the coordinate is, its spelling does not."""
        plain = NumericCoord.from_labels(np.array([0, 1, 5]))
        declared = NumericCoord.from_labels(np.array([0, 1, 5]), step=1)
        as_float = NumericCoord.from_labels(np.array([0, 1, 5]), step=1.0)
        assert plain.data_id != declared.data_id
        assert declared.data_id == as_float.data_id

    def test_bare_absolute_excess_on_time(self):
        """A bare excess on a time dimension is seconds, as coordinates read it."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        second = first.update_coords(time_min=time.max() + 2.4 * time.step)
        spool = dc.spool([first, second])
        wide = float(2 * time.step / np.timedelta64(1, "s"))
        assert len(spool.chunk(time=None, tolerance=GapTolerance.absolute(wide))) == 1
        narrow = float(time.step / np.timedelta64(1, "s"))
        assert len(spool.chunk(time=None, tolerance=GapTolerance.absolute(narrow))) == 2

    def test_seam_after_an_undeclared_singleton(self):
        """A seam whose run states no spacing reports no excess and no gap."""
        one = NumericCoord.from_labels(np.array([T0]))
        runs = get_coord(start=T0 + 5 * MS, step=MS, shape=(3,))
        later = get_coord(start=T0 + 20 * MS, step=MS, shape=(3,))
        coord = concat_coords(one, runs, later)
        seams = coord.get_discontinuities()
        assert len(seams) == 2 and pd.isnull(seams["excess"].iloc[0])
        assert coord.get_discontinuities("gaps")["index"].tolist() == [4]

    def test_waterfall_paints_gaps_by_declared_step(self):
        """The mesh opens a band wherever the declared step says a sample is missing."""
        pytest.importorskip("matplotlib")
        base = dc.get_example_patch().select(distance=(0, 4), samples=True)
        distance = get_coord(data=[0, 3, 6, 7], step=1)
        patch = base.update_coords(distance=distance)
        ax = patch.viz.waterfall(gap_color="white", cbar=False)
        array = ax.collections[0].get_array()
        # two three-wide spacings against a step of one: two bands
        assert array.shape[0] == patch.shape[0] + 2
        undeclared = base.update_coords(distance=np.array([0, 3, 6, 7]))
        ax = undeclared.viz.waterfall(gap_color="white", cbar=False)
        # the median spacing (3) sees no gap at all
        assert ax.collections[0].get_array().shape[0] == patch.shape[0]


class TestLegacySnapStep:
    """The DASDAE version 1 snap path and a stored nominal step."""

    def test_single_sample_takes_the_step(self, tmp_path):
        """One value cannot state its spacing, so the stored step is used."""
        with h5py.File(tmp_path / "legacy.h5", "w") as h5:
            h5.create_dataset("_coord_x", data=np.array([5.0]))
            h5.create_dataset("_coord_y", data=np.array([0.0, 1.0, 2.05]))
            single = _read_coord(h5["_coord_x"], "x", {"x_step": 2.0}, snap=True)
            jittered = _read_coord(h5["_coord_y"], "y", {"y_step": 1.0}, snap=True)
        assert single.evenly_sampled and single.step == 2.0
        # longer values keep today's tolerant reading; the nominal step is
        # not a claim they must meet
        assert jittered.step is None or jittered.evenly_sampled
        np.testing.assert_allclose(jittered.values, [0.0, 1.0, 2.05], atol=0.06)


class TestReviewRoundTwo:
    """Cases the PR review found."""

    @pytest.mark.parametrize("bad", [0, 0.0, np.inf])
    def test_zero_or_non_finite_step_rejected(self, bad):
        """A step which spans nothing is no grid; NaN declares none at all."""
        with pytest.raises(CoordError, match="finite non-zero"):
            get_coord(data=[0, 1, 2], step=bad)
        with pytest.raises(ValueError, match="finite non-zero"):
            NumericCoord.from_labels(np.array([0, 1, 2]), step=bad)
        assert get_coord(data=[0, 1, 5], step=np.nan).step is None

    def test_fractional_step_rejected_on_direct_arrays(self):
        """The array constructor holds the same rule as the factory."""
        with pytest.raises(ValueError, match="fractional step"):
            NumericCoord.from_labels(np.array([0, 1, 3]), step=Fraction(1, 2))

    def test_whole_fraction_is_seconds_for_time(self):
        """A whole fraction on time values means seconds, as a range reads it."""
        second = np.timedelta64(1, "s")
        labels = T0 + np.array([0, 1, 2, 5]) * second
        coord = get_coord(data=labels, step=(1, 1))
        assert coord.step == second and coord.missing().count == 2
        off = T0 + np.array([0, 1]) * np.timedelta64(1, "ns")
        with pytest.raises(CoordError, match="not on a grid"):
            get_coord(data=off, step=(1, 1))

    def test_unsigned_values_descending(self):
        """Unsigned spacings do not wrap when the values descend."""
        values = np.array([9, 6, 5, 1], dtype=np.uint8)
        coord = NumericCoord.from_labels(values, step=1)
        assert list(coord.missing().iter_runs()) == [(8, 7), (4, 2)]
        plain = NumericCoord.from_labels(values)
        seams = plain.get_discontinuities()
        assert seams["delta"].tolist() == [-1, -4]

    def test_hole_ends_share_one_anchor(self):
        """Float noise on the far label does not reach the hole's last position."""
        coord = NumericCoord.from_labels(np.array([0.0, 0.30000005]), step=0.1)
        missing = coord.missing()
        (run,) = missing.iter_runs()
        assert run[1] == pytest.approx(0.2) and run[1] == missing.positions()[-1]

    def test_segmented_strided_index_keeps_the_step(self, design_case):
        """A strided selection of runs still states the grid."""
        assert design_case[::2].step == 1
        assert design_case[[3, 0, 1]].step is None

    def test_segments_with_a_contradicting_step_raise(self, design_case):
        """A step passed beside segments must agree with them."""
        with pytest.raises(CoordError, match="contradicts"):
            get_coord(segments=design_case.segments, step=2)
        assert get_coord(segments=design_case.segments, step=1) == design_case

    def test_new_values_with_an_explicit_step(self):
        """New values with a step declare that grid, not an inferred one."""
        coord = get_coord(start=0, stop=5, step=1).new(
            values=np.array([0, 2, 4]), step=1
        )
        assert coord.step == 1 and coord.missing().count == 2

    def test_exporters_refuse_runs(self, design_case):
        """Formats holding one contiguous trace refuse a coordinate with holes."""
        pytest.importorskip("obspy")
        patch = dc.get_example_patch().select(time=(0, 6), samples=True)
        time = get_coord(data=T0 + np.array(PRESENT) * MS, step=MS)
        patch = patch.update_coords(time=time)
        with pytest.raises(CoordError, match="not evenly sampled"):
            patch.io.to_obspy()

    def test_prebuilt_tolerances_are_validated(self):
        """The constructors hold the same rules as from_user."""
        for bad in (GapTolerance.samples, GapTolerance.absolute):
            with pytest.raises(ParameterError):
                bad(-1)
            with pytest.raises(ParameterError):
                bad(np.nan)
        with pytest.raises(ParameterError):
            GapTolerance.absolute(np.inf)
        assert GapTolerance.samples(np.inf).count == np.inf

    def test_dasdae_keeps_declared_steps_on_array_segments(self, tmp_path):
        """An array segment's declared step survives a version 2 round trip."""
        left = NumericCoord.from_labels(np.array([0.0, 2.0, 3.0]), step=1.0)
        right = NumericCoord.from_labels(np.array([6.0, 9.0, 11.0]), step=1.0)
        coord = concat_coords(left, right)
        base = dc.get_example_patch().select(distance=(0, 6), samples=True)
        patch = base.update_coords(distance=coord)
        back = dc.read(dc.write(patch, tmp_path / "seg.h5", "dasdae"))[0]
        assert back.get_coord("distance") == coord
        assert back.get_coord("distance").step == 1.0

    def test_singleton_edges_use_the_declared_step(self, recwarn):
        """A single value with a step gets a cell of that width, without a warning."""
        edges, _ = get_gap_edges(get_coord(data=[10], step=5))
        np.testing.assert_allclose(edges, [7.5, 12.5])
        second = np.timedelta64(1, "s")
        edges, _ = get_gap_edges(get_coord(data=[T0], step=second))
        half = np.timedelta64(500, "ms")
        expected = np.array([T0 - half, T0 + half], dtype="datetime64[ns]")
        np.testing.assert_array_equal(edges, expected)
        assert not [w for w in recwarn if "Singleton" in str(w.message)]


class TestReviewRoundThree:
    """Cases the third review round found."""

    def test_uneven_block_keeps_its_labels(self):
        """Neighbouring spacings that differ do not become one grid."""
        values = np.array([0, 1, 2, 4, 6])
        assert np.array_equal(get_coord(data=values, snap=False).values, values)
        floats = np.array([0.0, 0.1, 0.2])
        assert np.array_equal(get_coord(data=floats, snap=False).values, floats)

    def test_full_slice_keeps_the_declared_step(self):
        """Slicing everything changes nothing, runs and step included."""
        coord = concat_coords(
            NumericCoord.from_labels(np.array([0, 2, 4]), step=1),
            NumericCoord.from_labels(np.array([6, 8, 10]), step=1),
        )
        out = coord[:]
        assert out.step == coord.step == 1
        assert out.missing().count == coord.missing().count == 5

    def test_integer_index_of_nd_labels(self):
        """An integer index of an N-D coordinate returns a coordinate."""
        coord = get_coord(data=np.arange(12).reshape(3, 4), units="m")
        row = coord[0]
        assert isinstance(row, NumericCoord)
        assert row == coord[0, :]
        assert np.array_equal(row.values, np.arange(4))

    def test_new_shape_extends_the_grid(self):
        """A new length keeps the step rather than re-deriving one."""
        coord = get_coord(start=0, step=1, shape=(5,))
        out = coord.new(shape=(10,))
        assert out.step == 1
        assert np.array_equal(out.values, np.arange(10))

    def test_strided_multi_run_fractional_grids(self):
        """A stride across fractional grids gives what numpy indexing gives."""
        coord = concat_coords(
            get_coord(start=0, step=Fraction(3, 2), shape=(5,)),
            get_coord(start=10, step=Fraction(3, 2), shape=(5,)),
        )
        values = coord.values
        for stride in (2, 3, -3):
            assert np.array_equal(coord[::stride].values, values[::stride])

    def test_labels_copy_a_read_only_view(self):
        """A read-only view cannot change the labels under their stored id."""
        base = np.array([1.0, 2.0, 3.0])
        view = base[:]
        view.flags.writeable = False
        coord = NumericCoord.from_labels(view)
        before = coord.data_id
        base[0] = 99.0
        assert coord.values[0] == 1.0 and coord.data_id == before

    def test_unsorted_run_is_not_sorted(self):
        """A coordinate never assumes its runs ascend."""
        coord = NumericCoord(
            runs=(np.array([3.0, 1.0, 2.0]), Grid(10.0, 1.0, 0, 4)),
            dtype="float64",
        )
        assert not coord.sorted and not coord.reverse_sorted
        assert len(coord.select((1.0, 2.0))[0]) == 2

    def test_empty_runs_are_dropped(self):
        """An empty run states nothing, so it is not kept."""
        coord = NumericCoord(
            runs=(np.array([], dtype="float64"), Grid(10.0, 1.0, 0, 4)),
            dtype="float64",
        )
        assert coord.runs_count == 1 and coord.sorted

    def test_descending_single_sample_segments(self, tmp_path):
        """Segments of one sample keep the order they were written in."""
        coord = get_coord(data=np.array([9, 7, 4]), step=1)
        base = dc.get_example_patch().select(distance=(0, 3), samples=True)
        patch = base.update_coords(distance=coord)
        back = dc.read(dc.write(patch, tmp_path / "down.h5", "dasdae"))[0]
        assert np.array_equal(back.get_coord("distance").values, [9, 7, 4])
        assert np.array_equal(back.data, patch.data)

    def test_unit_conversion_keeps_the_seam(self):
        """Converting units scales each run rather than re-reading the labels."""
        coord = concat_coords(
            get_coord(start=0.0, step=1.0, shape=(10,), units="m"),
            get_coord(start=10.0001, step=1.0, shape=(10,), units="m"),
        )
        out = coord.convert_units("cm")
        assert out.runs_count == coord.runs_count
        assert np.allclose(out.values, coord.values * 100, rtol=0, atol=1e-6)
        assert out.values[10] != out.values[9] + 100

    def test_a_trim_keeps_labels_the_step_contradicts(self):
        """A trimmed run on another spacing is not the declared grid."""
        coord = NumericCoord.from_labels(np.array([0.0, 2.0, 4.0, 6.0, 9.0]), step=1.0)
        out = coord[0:3]
        assert out.step == 1.0 and out.missing().count == 2

    def test_constant_labels_keep_their_count(self):
        """Labels which never change are not one sample repeated."""
        coord = get_coord(data=np.zeros(5, dtype="float32"), snap=False)
        assert len(coord[1:3]) == 2

    def test_unit_conversion_keeps_declared_holes(self):
        """A declared step's missing positions survive a change of units."""
        coord = concat_coords(
            NumericCoord.from_labels(np.array([0.0, 2.0, 4.0]), step=1.0, units="m"),
            NumericCoord.from_labels(np.array([10.0, 12.0, 14.0]), step=1.0, units="m"),
        )
        out = coord.convert_units("cm")
        assert out.step == 100.0
        assert out.missing().count == coord.missing().count

    def test_metadata_updates_do_not_materialize(self, monkeypatch):
        """A step or unit change on a huge grid stays arithmetic."""

        def _boom(self):
            raise AssertionError("the labels were materialized")

        monkeypatch.setattr(NumericCoord, "values", property(_boom))
        coord = get_coord(start=0, step=1, shape=(10**9,), units="m")
        assert coord.update_limits(step=2).step == 2
        assert get_quantity(coord.convert_units("cm").units) == get_quantity("cm")


class TestReviewRoundFour:
    """Findings from the fourth adversarial pass over the run model."""

    def test_a_fit_which_needs_floats_gets_them(self):
        """Integer labels re-fit within a tolerance the integers cannot hold."""
        coord = concat_coords(get_coord(data=[0, 1, 2]), get_coord(data=[4, 5, 6]))
        out = coord.fuse(0.5)
        assert out.evenly_sampled and out.dtype == np.dtype("float64")
        assert np.max(np.abs(out.values - coord.values)) <= 0.5

    def test_a_fit_which_needs_finer_time_gets_it(self):
        """A fit spaced in nanoseconds is not read back as milliseconds."""
        labels = T0.astype("datetime64[ms]") + np.array([0, 1, 3], dtype="m8[ms]")
        coord = get_coord(data=labels, snap=False)
        out = coord.fuse(np.timedelta64(1, "ms"))
        assert out.evenly_sampled and out.min() == labels[0]
        assert out.max() - out.min() == np.timedelta64(3, "ms")

    def test_a_shift_which_needs_floats_gets_them(self):
        """Labels moved half a step land off the integers they came from."""
        out = get_coord(data=[0, 2, 5]).update(min=0.5)
        assert out.dtype == np.dtype("float64")
        assert out.to_summary().min == 0.5 and out.to_summary().max == 5.5

    def test_repeated_labels_are_never_a_grid(self):
        """One label repeated is not one sample answering for every position."""
        coord = get_coord(data=[1, 1, 2], snap=False)[:2]
        assert np.array_equal(coord.select((1, 1))[0].values, [1, 1])

    def test_unordered_runs_look_every_label_up(self):
        """Runs which do not chain have no order for a sparse search."""
        coord = NumericCoord(
            runs=(np.array([0, 1, 2, 50, 4, 5]), Grid(60, 1, 1, 4)), dtype="int64"
        )
        patch = dc.Patch(data=np.arange(10.0), coords={"x": coord}, dims=("x",))
        assert patch.sel(x=50).data == 3

    def test_a_declared_step_does_not_move_labels(self):
        """A step says which grid the labels sit on, not that one restates them."""
        values = np.array([0.0, 1.00000001, 2.0, 4.0])
        coord = get_coord(data=values, step=1.0, snap=False)
        assert coord.step == 1.0
        np.testing.assert_array_equal(coord.values, values)

    def test_an_input_may_fill_another_gap(self):
        """Runs, not whole coordinates, are what may not overlap."""
        first = concat_coords(get_coord(data=[0, 1, 2]), get_coord(data=[6, 7, 8]))
        out = concat_coords(first, get_coord(data=[3, 4, 5]))
        np.testing.assert_array_equal(out.values, np.arange(9))

    def test_runs_which_share_no_step_miss_nothing(self):
        """A segment of a stepless coordinate has no grid to be missing from."""
        coord = concat_coords(
            NumericCoord.from_labels(np.array([0.0, 1.0, 3.0]), step=1.0),
            get_coord(start=10.0, step=2.0, shape=(3,)),
        )
        assert coord.step is None
        assert coord.segments[0].missing().complete


class TestReviewRoundFive:
    """Findings from the fifth adversarial pass over the run model."""

    @pytest.fixture
    def stored(self):
        """A coordinate holding three labels as one stored run."""
        return NumericCoord.from_labels(np.array([0.0, 2.0, 5.0]))

    def test_a_tampered_dump_cannot_join_its_original(self, stored):
        """One id may not name two different arrays of labels."""
        dump = stored.model_dump()
        key = stored.runs[0].id
        dump["sources"] = {key: np.array([10.0, 12.0, 15.0])}
        with pytest.raises(CoordError, match="Two different arrays"):
            concat_coords(stored, get_coord(**dump))

    def test_equal_arrays_under_one_id_still_join(self, stored):
        """The same labels reached twice are the same labels."""
        other = NumericCoord.from_labels(np.array([0.0, 2.0, 5.0]) + 10)
        out = concat_coords(stored, other)
        np.testing.assert_array_equal(out.values, [0, 2, 5, 10, 12, 15])

    @pytest.mark.parametrize("protocol", [2, 5])
    def test_pickled_sources_stay_sealed(self, stored, protocol):
        """Unpickling must not hand back a writable array under a kept id."""
        out = pickle.loads(pickle.dumps(stored, protocol=protocol))
        assert all(not x.flags.writeable for x in out.sources.values())

    @pytest.mark.parametrize(
        "copier", [copy.deepcopy, lambda x: x.model_copy(deep=True)]
    )
    def test_copied_sources_stay_sealed(self, stored, copier):
        """Deep copies must not hand back a writable array under a kept id."""
        assert all(not x.flags.writeable for x in copier(stored).sources.values())

    def test_a_window_may_not_overrun_its_source(self, stored):
        """A count larger than the stored array would lie about the shape."""
        key = stored.runs[0].id
        with pytest.raises(ValidationError, match="reads outside"):
            NumericCoord(
                runs=(Labels(key, 20),), sources=stored.sources, dtype="float64"
            )

    def test_a_window_may_not_start_before_its_source(self, stored):
        """A negative offset read backwards would quietly return everything."""
        key = stored.runs[0].id
        with pytest.raises(ValidationError, match="reads outside"):
            NumericCoord(
                runs=(Labels(key, 2, -1, -1),), sources=stored.sources, dtype="float64"
            )

    def test_a_reversed_window_is_still_valid(self, stored):
        """The whole array read backwards stays inside it."""
        np.testing.assert_array_equal(stored[::-1].values, [5, 2, 0])

    def test_a_partial_fit_promotes_its_kept_grids(self):
        """A fit which needs floats does not fail on the runs it left alone."""
        out = get_coord(data=[0, 1, 2, 10, 11, 13], snap=False).fuse(0.6)
        assert out.dtype == np.dtype("float64")
        np.testing.assert_allclose(out.values, [0, 1, 2, 10, 11.5, 13])

    @pytest.mark.parametrize("dtype", ["uint8", "int16", "int64"])
    def test_labels_at_the_dtype_ceiling_stay_stored(self, dtype):
        """A grid ending one past the top does not reject labels which fit."""
        top = np.iinfo(dtype).max
        values = np.array([0, top - 2, top - 1, top], dtype=dtype)
        out = NumericCoord.from_labels(values)[1:]
        np.testing.assert_array_equal(out.values, values[1:])

    def test_only_one_run_may_be_n_dimensional(self):
        """Several N-D runs would report a shape their labels contradict."""
        runs = (np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3) + 6)
        with pytest.raises(ValidationError, match="exactly one run"):
            NumericCoord(runs=runs)


class TestReviewRoundSix:
    """Findings from the sixth adversarial pass over the run model."""

    def test_identical_sources_holding_nan_still_join(self):
        """An unused NaN in a shared source must not look like a collision."""
        values = np.array([0.0, 1.5, 4.0, 7.0, 9.5, 20.0, np.nan])
        coord = NumericCoord.from_labels(values)
        out = concat_coords(coord[:3], copy.deepcopy(coord)[3:6])
        np.testing.assert_array_equal(out.values, values[:6])

    def test_a_copied_cache_stays_sealed(self):
        """A copy of a materialized coordinate hands back sealed labels."""
        coord = concat_coords(
            NumericCoord.from_labels(np.array([0.0, 1.5, 4.0])),
            NumericCoord.from_labels(np.array([7.0, 9.5, 20.0])),
        )
        assert not coord.values.flags.writeable
        assert not copy.deepcopy(coord).values.flags.writeable

    def test_a_reversed_unsigned_grid(self):
        """Reversing a grid whose end is zero does not run past the dtype."""
        out = get_coord(data=np.arange(5, dtype="uint8"), snap=False)[::-1]
        np.testing.assert_array_equal(out.values, [4, 3, 2, 1, 0])
        assert out.coord_range() == 5

    def test_a_strided_slice_at_the_dtype_top(self):
        """A stride whose next sample would wrap keeps the samples it has."""
        values = np.array([250, 251, 252, 253, 254], dtype="uint8")
        out = get_coord(data=values, snap=False)[::3]
        np.testing.assert_array_equal(out.values, [250, 253])

    def test_labels_at_the_dtype_top_are_kept(self):
        """Labels ending on the top of their dtype are a coordinate."""
        values = np.array([252, 253, 254, 255], dtype="uint8")
        out = get_coord(data=values, snap=False)
        np.testing.assert_array_equal(out.values, values)

    def test_labels_past_int64_stay_stored(self):
        """Unsigned labels above the signed range are never grid arithmetic."""
        values = np.array([2**63 + 4, 2**63 + 5, 2**63 + 6], dtype="uint64")
        out = NumericCoord.from_labels(values)[1:]
        np.testing.assert_array_equal(out.values, values[1:])

    @pytest.mark.parametrize("flip", [False, True])
    def test_mixed_time_units_are_refused(self, flip):
        """Promoting ns ticks to another time unit would restate them."""
        step = np.timedelta64(10, "ns")
        nano = get_coord(start=np.datetime64(1000, "ns"), step=step, shape=(3,))
        pico = get_coord(
            start=np.datetime64(2000000, "ps"),
            step=np.timedelta64(10000, "ps"),
            shape=(3,),
        )
        pair = (pico, nano) if flip else (nano, pico)
        with pytest.raises(CoordError, match="compatible dtypes"):
            concat_coords(*pair)
