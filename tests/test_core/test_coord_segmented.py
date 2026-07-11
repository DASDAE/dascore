"""Tests for segmented (piecewise) coordinates."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core.coords import (
    CoordMonotonicArray,
    CoordPartial,
    CoordRange,
    CoordSegmented,
    concat_coords,
    get_coord,
)
from dascore.exceptions import CoordError, ParameterError
from dascore.units import get_quantity


@pytest.fixture(scope="session")
def float_gap_coord() -> CoordSegmented:
    """Two evenly sampled float blocks (0..9, 15..24) separated by a gap."""
    c1 = get_coord(start=0.0, stop=10.0, step=1.0)
    c2 = get_coord(start=15.0, stop=25.0, step=1.0)
    return concat_coords(c1, c2)


@pytest.fixture(scope="session")
def time_gap_coord() -> CoordSegmented:
    """Two evenly sampled time blocks separated by a 3 second gap."""
    one_s = np.timedelta64(1, "s")
    t0 = np.datetime64("2020-01-01T00:00:00", "ns")
    c1 = get_coord(start=t0, stop=t0 + 10 * one_s, step=one_s)
    c2 = get_coord(start=t0 + 12 * one_s, stop=t0 + 22 * one_s, step=one_s)
    return concat_coords(c1, c2)


@pytest.fixture(scope="session")
def mixed_segment_coord() -> CoordSegmented:
    """A range segment followed by an irregular array segment."""
    c1 = get_coord(start=0.0, stop=10.0, step=1.0)
    c2 = get_coord(data=np.array([12.0, 12.1, 13.7, 20.0]))
    assert isinstance(c2, CoordMonotonicArray)
    return concat_coords(c1, c2)


@pytest.fixture(scope="session")
def reverse_gap_coord() -> CoordSegmented:
    """A reverse-sorted segmented coordinate."""
    c1 = get_coord(start=24.0, stop=14.0, step=-1.0)
    c2 = get_coord(start=9.0, stop=-1.0, step=-1.0)
    return concat_coords(c1, c2)


class TestConstruction:
    """Tests for building segmented coords and their normal form."""

    def test_concat_returns_segmented(self, float_gap_coord):
        """Two non-contiguous blocks make a segmented coord."""
        assert isinstance(float_gap_coord, CoordSegmented)
        assert float_gap_coord.segment_count == 2
        assert len(float_gap_coord) == 20

    def test_exactly_contiguous_fuse(self):
        """Blocks that continue exactly fuse back into one range."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        c2 = get_coord(start=10.0, stop=20.0, step=1.0)
        out = concat_coords(c1, c2)
        assert isinstance(out, CoordRange)
        assert out == get_coord(start=0.0, stop=20.0, step=1.0)

    def test_uniform_array_segments_promoted(self):
        """Exactly evenly sampled array segments become ranges."""
        c1 = get_coord(data=np.arange(5.0))
        c2 = get_coord(start=8.0, stop=12.0, step=1.0)
        out = concat_coords(c1, c2)
        assert all(isinstance(x, CoordRange) for x in out.segments)

    def test_canonical_across_construction_orders(self):
        """Equal values give equal coords regardless of how assembled."""
        a = concat_coords(
            get_coord(start=0.0, stop=5.0, step=1.0),
            get_coord(start=8.0, stop=12.0, step=1.0),
        )
        b = concat_coords(
            get_coord(data=np.array([8.0, 9.0, 10.0, 11.0])),
            get_coord(data=np.array([0.0, 1.0, 2.0, 3.0, 4.0])),
        )
        assert a == b
        assert a.fingerprint() == b.fingerprint()

    def test_out_of_order_inputs_sorted(self):
        """concat_coords orders inputs by their envelopes."""
        c1 = get_coord(start=15.0, stop=25.0, step=1.0)
        c2 = get_coord(start=0.0, stop=10.0, step=1.0)
        out = concat_coords(c1, c2)
        assert out.min() == 0.0 and out.max() == 24.0
        assert np.all(np.diff(out.values) > 0)

    def test_adjacent_arrays_fuse(self):
        """Adjacent irregular arrays fuse (their boundary has no meaning)."""
        c1 = get_coord(data=np.array([0.0, 0.1, 1.7]))
        c2 = get_coord(data=np.array([2.0, 3.3, 3.4]))
        out = concat_coords(c1, c2)
        assert isinstance(out, CoordMonotonicArray)

    def test_direct_construction_needs_two_segments(self):
        """The class itself requires >= 2 segments post normalization."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        with pytest.raises(ValidationError, match="fuse"):
            CoordSegmented(segments=(c1,))

    def test_fusing_inputs_rejected_by_class(self):
        """Directly constructing with fusable segments raises (use concat)."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        c2 = get_coord(start=10.0, stop=20.0, step=1.0)
        with pytest.raises(ValidationError, match="fuse"):
            CoordSegmented(segments=(c1, c2))

    def test_overlap_raises(self):
        """Overlapping segments are rejected."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        c2 = get_coord(start=5.0, stop=15.0, step=1.0)
        with pytest.raises(CoordError, match="overlap"):
            concat_coords(c1, c2)

    def test_shared_value_raises(self):
        """Segments sharing a boundary value are rejected (not strict)."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)  # max 9
        c2 = get_coord(data=np.array([9.0, 11.0, 12.0]))
        with pytest.raises(CoordError, match="overlap"):
            concat_coords(c1, c2)

    def test_mixed_direction_raises(self):
        """Segments sorted in different directions are rejected."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        c2 = get_coord(start=25.0, stop=15.0, step=-1.0)
        with pytest.raises(CoordError):
            concat_coords(c1, c2)

    def test_mixed_dtype_kind_raises(self):
        """Time and numeric segments cannot mix."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        t0 = np.datetime64("2020-01-01", "ns")
        c2 = get_coord(
            start=t0, stop=t0 + np.timedelta64(10, "s"), step=np.timedelta64(1, "s")
        )
        with pytest.raises(CoordError):
            concat_coords(c1, c2)

    def test_mixed_units_raise(self):
        """Segments with different units are rejected."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0, units="m")
        c2 = get_coord(start=15.0, stop=25.0, step=1.0, units="ft")
        with pytest.raises(CoordError, match="units"):
            concat_coords(c1, c2)

    def test_unsupported_coord_type_raises(self):
        """String and other unsupported coords are rejected."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        c2 = get_coord(data=np.array(["a", "b", "c"]))
        with pytest.raises(CoordError, match="only supports"):
            concat_coords(c1, c2)

    def test_non_coord_raises(self):
        """Raw arrays are not silently coerced."""
        with pytest.raises(CoordError, match="requires coordinates"):
            concat_coords(np.arange(10))

    def test_empty_input_raises(self):
        """At least one non-empty coordinate is required."""
        with pytest.raises(CoordError, match="at least one"):
            concat_coords()

    def test_degenerate_inputs_skipped(self):
        """Empty coordinates contribute nothing."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        empty = c1.empty()
        out = concat_coords(c1, empty)
        assert out == c1

    def test_single_input_returns_input(self):
        """A single coordinate comes back as itself."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        assert concat_coords(c1) == c1

    def test_segmented_inputs_flatten(self, float_gap_coord):
        """Segmented inputs contribute their segments."""
        c3 = get_coord(start=30.0, stop=40.0, step=1.0)
        out = concat_coords(float_gap_coord, c3)
        assert out.segment_count == 3

    def test_units_param_sets_units(self):
        """The units argument sets units on the result."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        c2 = get_coord(start=15.0, stop=25.0, step=1.0)
        out = concat_coords(c1, c2, units="m")
        assert get_quantity(out.units) == get_quantity("m")

    def test_get_coord_segments_kwarg(self, float_gap_coord):
        """get_coord(segments=...) dispatches to concat_coords."""
        out = get_coord(segments=float_gap_coord.segments)
        assert out == float_gap_coord

    def test_get_coord_segments_conflicts(self):
        """Segments cannot combine with other value inputs."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        with pytest.raises(CoordError, match="cannot be combined"):
            get_coord(segments=(c1,), start=0, stop=1, step=0.1)

    def test_container_units_mismatch_raises(self, float_gap_coord):
        """Explicit units that contradict segment units raise."""
        with pytest.raises(ValidationError, match="units"):
            CoordSegmented(segments=float_gap_coord.segments, units="m")


class TestProperties:
    """Tests for basic properties of segmented coords."""

    def test_step_is_none(self, float_gap_coord):
        """Segmented coords report no single step."""
        assert float_gap_coord.step is None

    def test_not_evenly_sampled(self, float_gap_coord):
        """Segmented coords are never evenly sampled."""
        assert not float_gap_coord.evenly_sampled

    def test_sorted_flags(self, float_gap_coord, reverse_gap_coord):
        """Sort direction is reported correctly."""
        assert float_gap_coord.sorted
        assert not float_gap_coord.reverse_sorted
        assert reverse_gap_coord.reverse_sorted
        assert not reverse_gap_coord.sorted

    def test_min_max_exact(self, float_gap_coord, reverse_gap_coord):
        """Min/max come from segment envelopes exactly."""
        assert float_gap_coord.min() == 0.0
        assert float_gap_coord.max() == 24.0
        assert reverse_gap_coord.min() == 0.0
        assert reverse_gap_coord.max() == 24.0

    def test_values_are_concatenation(self, float_gap_coord):
        """Values are exactly the concatenated segment values."""
        expected = np.concatenate([x.values for x in float_gap_coord.segments])
        assert np.array_equal(float_gap_coord.values, expected)

    def test_dtype_and_shape(self, time_gap_coord):
        """Dtype and shape derive from segments."""
        assert np.issubdtype(time_gap_coord.dtype, np.datetime64)
        assert time_gap_coord.shape == (20,)

    def test_not_degenerate(self, float_gap_coord):
        """A populated segmented coord is not degenerate."""
        assert not float_gap_coord.degenerate

    def test_str_and_rich(self, float_gap_coord):
        """String representations render."""
        assert str(float_gap_coord)
        assert float_gap_coord.__rich__() is not None

    def test_time_units_forced_to_seconds(self, time_gap_coord):
        """Time coords keep the standard second units."""
        assert get_quantity(time_gap_coord.units) == get_quantity("s")

    def test_coord_range_no_extend(self, float_gap_coord):
        """coord_range works without extension; extension needs a step."""
        assert float_gap_coord.coord_range(extend=False) == 24.0
        with pytest.raises(CoordError):
            float_gap_coord.coord_range(extend=True)

    def test_get_sample_count_raises(self, float_gap_coord):
        """No single sampling interval means no sample counts."""
        with pytest.raises(CoordError):
            float_gap_coord.get_sample_count(1.0)


class TestEquivalenceWithMonotonic:
    """Segmented coords must behave exactly like their materialized values."""

    def get_pair(self, coord):
        """Return (segmented, equivalent monotonic array coord)."""
        return coord, get_coord(data=coord.values, units=coord.units)

    @pytest.fixture(
        params=[
            "float_gap_coord",
            "time_gap_coord",
            "mixed_segment_coord",
            "reverse_gap_coord",
        ]
    )
    def coord_pair(self, request):
        """A segmented coord and its materialized twin."""
        return self.get_pair(request.getfixturevalue(request.param))

    def test_select_range_parity(self, coord_pair):
        """Range selects agree with the materialized coordinate."""
        seg, mono = coord_pair
        values = seg.values
        probes = [
            (values[2], values[-3]),
            (values[0], values[-1]),
            (None, values[5]),
            (values[5], None),
        ]
        for args in probes:
            c1, i1 = seg.select(args)
            c2, i2 = mono.select(args)
            assert np.array_equal(np.atleast_1d(c1.values), np.atleast_1d(c2.values))
            # Slices must resolve identically (literal form may differ, e.g.
            # slice(None, None) vs slice(None, len)).
            assert i1.indices(len(seg)) == i2.indices(len(mono))

    def test_getitem_parity(self, coord_pair):
        """Integer and slice indexing agree with the materialized coord."""
        seg, mono = coord_pair
        for ind in [0, 5, len(seg) - 1, -1]:
            assert seg[ind] == mono.values[ind]
        for slc in [slice(2, 15), slice(None), slice(None, None, 2), slice(15, 2, -1)]:
            assert np.array_equal(
                np.atleast_1d(seg[slc].values), np.atleast_1d(mono[slc].values)
            )

    def test_get_next_index_parity(self, coord_pair):
        """get_next_index matches the materialized coordinate."""
        seg, mono = coord_pair
        if seg.reverse_sorted:  # get_next_index requires sorted coords.
            return
        probe = seg.values[3]
        assert seg.get_next_index(probe) == mono.get_next_index(probe)

    def test_approx_equal(self, coord_pair):
        """A segmented coord approx-equals its materialized twin."""
        seg, mono = coord_pair
        assert seg.approx_equal(mono)


class TestSelect:
    """Tests for value-based selection."""

    def test_select_within_gap_is_empty(self, float_gap_coord):
        """A window entirely inside a gap selects nothing (no ghosts)."""
        out, indexer = float_gap_coord.select((11.0, 14.0))
        assert len(out) == 0
        assert indexer == slice(0, 0)

    def test_select_across_seam_keeps_structure(self, float_gap_coord):
        """A window spanning the seam returns a segmented coord."""
        out, indexer = float_gap_coord.select((5.0, 18.0))
        assert isinstance(out, CoordSegmented)
        assert out.segment_count == 2
        assert np.array_equal(out.values, np.array([5.0, 6, 7, 8, 9, 15, 16, 17, 18]))
        assert indexer == slice(5, 14)

    def test_select_single_segment_degrades(self, float_gap_coord):
        """A window inside one block returns a plain range."""
        out, _ = float_gap_coord.select((16.0, 20.0))
        assert isinstance(out, CoordRange)

    def test_select_samples(self, float_gap_coord):
        """Samples-based selection works via base machinery (stop exclusive)."""
        out, _ = float_gap_coord.select((2, 12), samples=True)
        assert len(out) == 10
        assert np.array_equal(out.values, float_gap_coord.values[2:12])

    def test_select_relative(self, float_gap_coord):
        """Relative selection resolves against min/max."""
        out, _ = float_gap_coord.select((1.0, -1.0), relative=True)
        assert out.min() == 1.0
        assert out.max() == 23.0

    def test_select_bool_array(self, float_gap_coord):
        """Boolean mask selection materializes correctly."""
        mask = float_gap_coord.values > 8.0
        out, _ = float_gap_coord.select(mask)
        assert np.array_equal(out.values, float_gap_coord.values[mask])

    def test_select_value_array(self, float_gap_coord):
        """Value-array selection keeps only matching values."""
        out, _ = float_gap_coord.select(np.array([1.0, 15.0, 99.0]))
        assert np.array_equal(out.values, np.array([1.0, 15.0]))

    def test_select_between_samples_of_segment(self):
        """A window inside a segment's envelope but between samples is empty."""
        coarse = CoordMonotonicArray(values=np.array([0.0, 10.0]))
        fine = get_coord(start=20.0, stop=25.0, step=1.0)
        coord = concat_coords(coarse, fine)
        out, indexer = coord.select((3.0, 7.0))
        assert len(out) == 0
        assert indexer == slice(0, 0)

    def test_select_none_returns_all(self, float_gap_coord):
        """A null select keeps everything."""
        out, _ = float_gap_coord.select(None)
        assert out == float_gap_coord


class TestGetItem:
    """Tests for indexing behavior."""

    def test_int_indexing(self, float_gap_coord):
        """Integer indexing crosses segments correctly."""
        assert float_gap_coord[0] == 0.0
        assert float_gap_coord[9] == 9.0
        assert float_gap_coord[10] == 15.0
        assert float_gap_coord[-1] == 24.0

    def test_out_of_bounds_raises(self, float_gap_coord):
        """Bad indices raise IndexError."""
        with pytest.raises(IndexError):
            _ = float_gap_coord[len(float_gap_coord)]

    def test_slice_across_seam(self, float_gap_coord):
        """Slices spanning the seam preserve structure."""
        out = float_gap_coord[8:12]
        assert isinstance(out, CoordSegmented)
        assert np.array_equal(out.values, np.array([8.0, 9.0, 15.0, 16.0]))

    def test_slice_single_segment(self, float_gap_coord):
        """Slices within one block degrade to that block's type."""
        out = float_gap_coord[2:5]
        assert isinstance(out, CoordRange)

    def test_empty_slice(self, float_gap_coord):
        """Empty slices produce an empty coordinate."""
        out = float_gap_coord[5:5]
        assert isinstance(out, CoordPartial)
        assert len(out) == 0

    def test_fancy_indexing_materializes(self, float_gap_coord):
        """Fancy indexing falls back to materialized values."""
        out = float_gap_coord[np.array([0, 10, 19])]
        assert np.array_equal(out.values, np.array([0.0, 15.0, 24.0]))


class TestSortAndShift:
    """Tests for sort, update_limits, and unit operations."""

    def test_sort_noop(self, float_gap_coord):
        """Sorting a sorted coord is a no-op."""
        out, indexer = float_gap_coord.sort()
        assert out is float_gap_coord
        assert indexer == slice(None)

    def test_reverse_round_trip(self, float_gap_coord):
        """Reversing twice returns the original."""
        rev, indexer = float_gap_coord.sort(reverse=True)
        assert rev.reverse_sorted
        assert indexer == slice(None, None, -1)
        back, _ = rev.sort()
        assert back == float_gap_coord

    def test_update_limits_min_shifts(self, float_gap_coord):
        """Setting min translates all segments."""
        out = float_gap_coord.update_limits(min=100.0)
        assert out.min() == 100.0
        assert np.array_equal(out.values, float_gap_coord.values + 100.0)
        assert isinstance(out, CoordSegmented)

    def test_update_limits_max_shifts(self, float_gap_coord):
        """Setting max translates all segments."""
        out = float_gap_coord.update_limits(max=0.0)
        assert out.max() == 0.0
        assert np.array_equal(out.values, float_gap_coord.values - 24.0)

    def test_update_limits_step_raises(self, float_gap_coord):
        """Setting a step on a segmented coord is an error."""
        with pytest.raises(ParameterError, match="no single step"):
            float_gap_coord.update_limits(step=2.0)

    def test_update_limits_min_and_max_raises(self, float_gap_coord):
        """Min and max cannot both be given."""
        with pytest.raises(ParameterError):
            float_gap_coord.update_limits(min=0.0, max=100.0)

    def test_time_shift(self, time_gap_coord):
        """Time coords shift with datetime min values."""
        new_min = time_gap_coord.min() + np.timedelta64(1, "D")
        out = time_gap_coord.update_limits(min=new_min)
        assert out.min() == new_min

    def test_set_units(self, float_gap_coord):
        """set_units applies to container and segments."""
        out = float_gap_coord.set_units("m")
        assert get_quantity(out.units) == get_quantity("m")
        assert all(get_quantity(x.units) == get_quantity("m") for x in out.segments)

    def test_convert_units(self, float_gap_coord):
        """convert_units scales all values consistently."""
        out = float_gap_coord.set_units("m").convert_units("ft")
        expected = float_gap_coord.values / 0.3048
        assert np.allclose(out.values, expected)
        assert isinstance(out, CoordSegmented)

    def test_convert_units_time_noop(self, time_gap_coord):
        """Time coords do not convert units."""
        assert time_gap_coord.convert_units("ft") is time_gap_coord


class TestSimplifyAndSnap:
    """Tests for tolerance-bounded simplification and snapping."""

    def test_simplify_zero_keeps_structure(self, float_gap_coord):
        """Zero tolerance cannot absorb a real gap."""
        out = float_gap_coord.simplify(0)
        assert out == float_gap_coord

    def test_simplify_absorbs_gap_within_tolerance(self, float_gap_coord):
        """A large enough tolerance collapses to a single range."""
        out = float_gap_coord.simplify(3.0)
        assert isinstance(out, CoordRange)
        assert len(out) == len(float_gap_coord)
        assert out.min() == float_gap_coord.min()
        assert out.max() == float_gap_coord.max()

    def test_simplify_error_bounded(self, float_gap_coord):
        """No value moves by more than the tolerance."""
        tol = 3.0
        out = float_gap_coord.simplify(tol)
        deviation = np.max(np.abs(out.values - float_gap_coord.values))
        assert deviation <= tol

    def test_simplify_insufficient_tolerance_noop(self, float_gap_coord):
        """A too-small tolerance leaves the coord unchanged."""
        out = float_gap_coord.simplify(0.5)
        assert out == float_gap_coord

    def test_simplify_idempotent(self, float_gap_coord):
        """Simplifying twice equals simplifying once."""
        once = float_gap_coord.simplify(3.0)
        assert once.simplify(3.0) == once

    def test_simplify_time_tolerance_seconds(self, time_gap_coord):
        """Numeric tolerances on time coords mean seconds."""
        out = time_gap_coord.simplify(2)
        assert isinstance(out, CoordRange)
        deviation = np.max(np.abs(out.values - time_gap_coord.values))
        assert deviation <= np.timedelta64(2, "s")

    def test_simplify_promotes_close_array_segment(self):
        """Nearly uniform array segments become ranges within tolerance."""
        values = np.array([0.0, 1.05, 2.0, 2.95, 4.0])
        coord = concat_coords(
            get_coord(data=values), get_coord(start=10.0, stop=14.0, step=1.0)
        )
        out = coord.simplify(0.1)
        assert all(isinstance(x, CoordRange) for x in out.segments)

    def test_negative_tolerance_raises(self, float_gap_coord):
        """Negative tolerances make no sense."""
        with pytest.raises(ParameterError):
            float_gap_coord.simplify(-1.0)

    def test_simplify_base_coord_noop(self):
        """Other coords return themselves from simplify."""
        coord = get_coord(start=0.0, stop=10.0, step=1.0)
        assert coord.simplify(10) is coord

    def test_snap_forces_range(self, float_gap_coord):
        """Snap always produces a range preserving min/max and length."""
        out = float_gap_coord.snap()
        assert isinstance(out, CoordRange)
        assert len(out) == len(float_gap_coord)
        assert out.min() == float_gap_coord.min()
        assert out.max() == float_gap_coord.max()

    def test_reverse_simplify(self, reverse_gap_coord):
        """Simplify works on reverse-sorted coords."""
        out = reverse_gap_coord.simplify(3.0)
        assert isinstance(out, CoordRange)
        assert out.reverse_sorted


class TestDiscontinuities:
    """Tests for gap/boundary introspection."""

    def test_all_boundaries(self, float_gap_coord):
        """Every segment boundary is reported."""
        df = float_gap_coord.get_discontinuities()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["index"] == 10
        assert row["before"] == 9.0
        assert row["after"] == 15.0
        assert row["delta"] == 6.0
        assert row["excess"] == 5.0

    def test_gaps_with_tolerance(self, time_gap_coord):
        """Gap filtering respects the tolerance."""
        assert len(time_gap_coord.get_discontinuities("gaps")) == 1
        assert len(time_gap_coord.get_discontinuities("gaps", tolerance=10)) == 0

    def test_bad_kind_raises(self, float_gap_coord):
        """Unknown kinds raise ParameterError."""
        with pytest.raises(ParameterError, match="kind"):
            float_gap_coord.get_discontinuities("overlaps")

    def test_base_coord_empty(self):
        """Coordinates without structure report no discontinuities."""
        coord = get_coord(start=0.0, stop=10.0, step=1.0)
        df = coord.get_discontinuities()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_reverse_gaps(self, reverse_gap_coord):
        """Gap detection works on reverse-sorted coords."""
        df = reverse_gap_coord.get_discontinuities("gaps")
        assert len(df) == 1


class TestRoundTrips:
    """Serialization and summary round trips."""

    def test_model_dump_round_trip(self, mixed_segment_coord):
        """model_dump payloads rebuild the same coordinate."""
        out = CoordSegmented(**mixed_segment_coord.model_dump())
        assert out == mixed_segment_coord

    def test_pickle_round_trip(self, time_gap_coord):
        """Pickling preserves the coordinate."""
        out = pickle.loads(pickle.dumps(time_gap_coord))
        assert out == time_gap_coord

    def test_to_summary(self, float_gap_coord):
        """Summaries carry exact min/max and a null step."""
        summary = float_gap_coord.to_summary()
        assert summary.min == 0.0
        assert summary.max == 24.0
        assert summary.step is None
        assert summary.len == 20
        assert summary.fingerprint == float_gap_coord.fingerprint()

    def test_get_attrs_dict_no_step(self, float_gap_coord):
        """The attrs dict omits the (null) step."""
        out = float_gap_coord.get_attrs_dict("time")
        assert "time_step" not in out
        assert out["time_min"] == 0.0
        assert out["time_max"] == 24.0

    def test_ns_precision_exact(self):
        """Nanosecond datetimes survive bit-exactly (no float pass)."""
        ns = np.timedelta64(1, "ns")
        t0 = np.datetime64("2020-01-01T00:00:00.123456789", "ns")
        v1 = t0 + np.arange(10) * 1000 * ns
        v2 = t0 + (np.arange(10) * 1000 + 10_501) * ns
        coord = concat_coords(get_coord(data=v1), get_coord(data=v2))
        assert np.array_equal(coord.values, np.concatenate([v1, v2]))

    def test_ns_precision_sort_order(self):
        """Segments closer than float epoch resolution still sort correctly.

        Casting ns datetimes to float collapses differences below ~128 ns
        at modern epochs, so the envelope sort must use native values.
        """
        ns = np.timedelta64(1, "ns")
        t0 = np.datetime64("2024-01-01T00:00:00", "ns")
        v1 = t0 + np.arange(3) * 10 * ns
        v2 = t0 + (np.arange(3) * 10 + 35) * ns
        assert float(v1[0]) == float(v2[0])  # the collapse this guards against
        coord = concat_coords(get_coord(data=v2), get_coord(data=v1))
        assert np.array_equal(coord.values, np.concatenate([v1, v2]))

    def test_empty_from_coord(self, float_gap_coord):
        """Emptying a segmented coord gives a zero-length coordinate."""
        out = float_gap_coord.empty()
        assert len(out) == 0
        assert np.dtype(out.dtype) == np.dtype(float_gap_coord.dtype)

    def test_new_with_segments(self, float_gap_coord):
        """new(segments=...) builds an updated coordinate."""
        segments = tuple(
            x.update_limits(min=x.min() + 100) for x in float_gap_coord.segments
        )
        out = float_gap_coord.new(segments=segments)
        assert out.min() == 100.0

    def test_new_with_data(self, float_gap_coord):
        """new(data=...) falls back to plain coord creation."""
        out = float_gap_coord.new(data=np.arange(10.0))
        assert isinstance(out, CoordRange)


class TestEdgeCases:
    """Edge and guard-path behaviors."""

    def test_promotion_requires_bit_exact_round_trip(self):
        """Equal float diffs alone don't promote; values must round-trip."""
        # [0, 0.1, 0.2] has bit-equal diffs but a range cannot reproduce
        # the values exactly, so it must stay an array segment.
        values = np.array([0.0, 0.1, 0.2])
        assert len(np.unique(np.diff(values))) == 1
        out = concat_coords(CoordMonotonicArray(values=values))
        assert isinstance(out, CoordMonotonicArray)

    def test_slice_can_promote_and_fuse(self):
        """Slicing an array segment uniform lets it fuse with a range."""
        a = get_coord(start=0.0, stop=10.0, step=1.0)
        b = CoordMonotonicArray(values=np.array([10.0, 11.0, 12.0, 13.5]))
        coord = concat_coords(a, b)
        assert coord.segment_count == 2
        out = coord[0:13]
        assert isinstance(out, CoordRange)
        assert np.array_equal(out.values, np.arange(13.0))

    def test_segments_must_be_coords(self):
        """Raw arrays passed as segments raise."""
        with pytest.raises(ValidationError, match="must be coordinates"):
            CoordSegmented(segments=(np.arange(3),))

    def test_unsupported_segment_class_raises(self):
        """Coord classes other than range/monotonic are rejected."""
        from dascore.core.coords import CoordArray

        bad = CoordArray(values=np.array([3.0, 1.0, 2.0]))
        good = get_coord(start=10.0, stop=20.0, step=1.0)
        with pytest.raises(ValidationError, match="CoordRange or CoordMonotonic"):
            CoordSegmented(segments=(bad, good))

    def test_empty_segment_raises(self):
        """Zero-length segments are rejected."""
        empty = CoordMonotonicArray(values=np.array([], dtype=np.float64))
        good = get_coord(start=10.0, stop=20.0, step=1.0)
        with pytest.raises(ValidationError, match="must not be empty"):
            CoordSegmented(segments=(empty, good))

    def test_no_segments_raises(self):
        """An empty segment tuple is rejected."""
        with pytest.raises(ValidationError, match="at least one segment"):
            CoordSegmented(segments=())

    def test_single_coord_instance_as_segments(self):
        """A bare coordinate as segments is wrapped, then fails the 2+ rule."""
        c1 = get_coord(start=0.0, stop=10.0, step=1.0)
        with pytest.raises(ValidationError, match="fuse"):
            CoordSegmented(segments=c1)

    def test_validator_passes_non_dict_through(self, float_gap_coord):
        """The before-validator leaves non-dict payloads alone."""
        out = CoordSegmented._validate_segments(float_gap_coord)
        assert out is float_gap_coord

    def test_eq_other_types(self, float_gap_coord):
        """Equality is False for other coord types and segment counts."""
        mono = get_coord(data=float_gap_coord.values)
        assert float_gap_coord != mono
        three = concat_coords(
            float_gap_coord, get_coord(start=30.0, stop=40.0, step=1.0)
        )
        assert float_gap_coord != three

    def test_zero_dim_index_returns_scalar(self, float_gap_coord):
        """A 0-d array index returns a scalar value."""
        out = float_gap_coord[np.array(10)]
        assert out == 15.0

    def test_shift_mixed_segments(self, mixed_segment_coord):
        """Shifting moves array segments as well as ranges."""
        out = mixed_segment_coord.update_limits(min=100.0)
        assert np.allclose(out.values, mixed_segment_coord.values + 100.0)

    def test_simplify_single_sample_segment(self):
        """Length-one segments cannot be fit and pass through simplify."""
        coord = concat_coords(
            get_coord(start=0.0, stop=10.0, step=1.0),
            CoordMonotonicArray(values=np.array([99.0])),
        )
        out = coord.simplify(0.5)
        assert out == coord

    def test_discontinuities_after_array_segment(self):
        """The expected step after an array segment comes from its diffs."""
        coord = concat_coords(
            CoordMonotonicArray(values=np.array([0.0, 1.0, 2.5])),
            get_coord(start=10.0, stop=20.0, step=1.0),
        )
        df = coord.get_discontinuities()
        assert len(df) == 1
        # expected local spacing is 2.5 - 1.0 = 1.5; delta is 10 - 2.5 = 7.5.
        assert df.iloc[0]["excess"] == 6.0

    def test_discontinuities_after_single_sample_segment(self):
        """No expected step after a length-one segment (excess is null)."""
        coord = concat_coords(
            CoordMonotonicArray(values=np.array([0.0])),
            get_coord(start=10.0, stop=20.0, step=1.0),
        )
        df = coord.get_discontinuities()
        assert len(df) == 1
        assert pd.isnull(df.iloc[0]["excess"])
        # boundaries with unknown spacing are never reported as gaps.
        assert coord.get_discontinuities("gaps").empty

    def test_base_get_discontinuities_bad_kind(self):
        """The BaseCoord implementation validates kind too."""
        coord = get_coord(start=0.0, stop=10.0, step=1.0)
        with pytest.raises(ParameterError, match="kind"):
            coord.get_discontinuities("overlaps")

    def test_rebuild_reraises_real_errors(self, float_gap_coord):
        """_rebuild only swallows errors caused by segments fusing to one."""
        t0 = np.datetime64("2020-01-01", "ns")
        time_seg = get_coord(
            start=t0, stop=t0 + np.timedelta64(10, "s"), step=np.timedelta64(1, "s")
        )
        bad = [float_gap_coord.segments[0], time_seg]
        with pytest.raises(ValidationError, match="compatible dtypes"):
            float_gap_coord._rebuild(bad)


class TestReviewFindings:
    """Regression tests for review findings on PR #749."""

    def test_mixed_int_float_segments_raise(self):
        """Mixing int and float segments could corrupt large ints; reject."""
        floats = get_coord(data=np.array([0.0, 1.0]))
        big = 2**53 + 1
        ints = get_coord(data=np.array([big, big + 2], dtype=np.int64))
        with pytest.raises(CoordError, match="compatible dtypes"):
            concat_coords(floats, ints)

    def test_int_width_promotion_lossless(self):
        """Different int widths share a kind and concatenate exactly."""
        # Non-uniform diffs keep these as array segments; a float pass
        # would corrupt values above 2**53.
        small = CoordMonotonicArray(values=np.array([0, 1, 3], dtype=np.int32))
        big_val = 2**53 + 1
        big = CoordMonotonicArray(
            values=np.array([big_val, big_val + 2, big_val + 3], dtype=np.int64)
        )
        coord = concat_coords(small, big)
        assert coord.values[-3] == big_val
        assert coord.values[-1] == big_val + 3

    def test_reverse_ns_select_monotonic(self):
        """Reverse ns-datetime lookup must not go through floats."""
        ns = np.timedelta64(1, "ns")
        t0 = np.datetime64("2024-01-01T00:00:00", "ns")
        values = t0 + np.array([95, 85, 75]) * ns
        coord = get_coord(data=values)
        assert coord.reverse_sorted
        probe = t0 + 75 * ns
        out, _ = coord.select((probe, probe))
        assert len(out) == 1
        assert out.values[0] == probe

    def test_reverse_ns_select_segmented(self):
        """The exact reverse-segmented scenario from review."""
        ns = np.timedelta64(1, "ns")
        t0 = np.datetime64("2024-01-01T00:00:00", "ns")
        c1 = get_coord(data=t0 + np.array([95, 85, 75]) * ns)
        c2 = get_coord(data=t0 + np.array([60, 50, 40]) * ns)
        coord = concat_coords(c1, c2)
        assert coord.reverse_sorted
        probe = t0 + 75 * ns
        out, _ = coord.select((probe, probe))
        assert len(out) == 1
        assert out.values[0] == probe

    def test_reverse_unsigned_select(self):
        """Unsigned reverse coords negate via float (would wrap natively)."""
        # Direct construction; get_coord inference wraps on descending uints.
        coord = CoordMonotonicArray(values=np.array([30, 20, 5], dtype=np.uint64))
        assert coord.reverse_sorted
        out, _ = coord.select((20, 20))
        assert len(out) == 1
        assert out.values[0] == 20

    def test_quantity_tolerance_numeric(self, float_gap_coord):
        """Unit-bearing tolerances convert to coordinate units."""
        coord = float_gap_coord.set_units("m")
        out = coord.simplify(get_quantity("3 m"))
        assert isinstance(out, CoordRange)
        # 300 cm is the same tolerance in other units.
        out2 = coord.simplify(get_quantity("300 cm"))
        assert out2 == out

    def test_quantity_tolerance_time(self, time_gap_coord):
        """Time coords accept time-dimensional quantity tolerances."""
        out = time_gap_coord.simplify(get_quantity("2000 ms"))
        assert isinstance(out, CoordRange)
        df = time_gap_coord.get_discontinuities("gaps", tolerance=get_quantity("10 s"))
        assert df.empty

    def test_quantity_tolerance_bad_dimensionality(self, time_gap_coord):
        """Dimensionality mismatches raise."""
        with pytest.raises(Exception, match=r"(?i)cannot convert|dimensionality"):
            time_gap_coord.simplify(get_quantity("1 m"))

    def test_select_never_materializes(self, monkeypatch):
        """Value selection must stay O(segments): no full-array pass.

        Locks in the second-review performance finding: range selects on a
        20M-sample segmented coord must not touch the concatenated values.
        """
        c1 = get_coord(start=0.0, stop=1e7, step=1.0)
        c2 = get_coord(start=2e7, stop=3e7, step=1.0)
        coord = concat_coords(c1, c2)

        def _boom(self):
            msg = "select must not materialize concatenated values"
            raise AssertionError(msg)

        monkeypatch.setattr(CoordSegmented, "values", property(_boom))
        out, indexer = coord.select((5.0, 2.5e7))
        assert isinstance(indexer, slice)
        assert out.min() == 5.0 and out.max() == 2.5e7
        # empty, full, and relative selects also stay off the values path.
        empty, _ = coord.select((1.2e7, 1.8e7))
        assert len(empty) == 0
        full, _ = coord.select(None)
        assert len(full) == len(coord)
        sub, _ = coord.select((10.0, -10.0), relative=True)
        assert len(sub)

    def test_patch_round_trip(self, float_gap_coord):
        """Segmented coords survive attachment to a Patch (CoordManager)."""
        time = dc.to_datetime64(np.arange(10))
        rng = np.random.default_rng(42)
        patch = dc.Patch(
            data=rng.random((len(float_gap_coord), 10)),
            coords={"distance": float_gap_coord, "time": time},
            dims=("distance", "time"),
        )
        attached = patch.get_coord("distance")
        assert isinstance(attached, CoordSegmented)
        assert np.array_equal(attached.values, float_gap_coord.values)
        # update_coords goes through the model_dump round trip.
        updated = patch.update_coords(distance=float_gap_coord)
        assert isinstance(updated.get_coord("distance"), CoordSegmented)
        # selection across the gap trims data consistently.
        sub = patch.select(distance=(5.0, 18.0))
        assert sub.shape == (9, 10)
        assert np.array_equal(
            sub.get_coord("distance").values,
            np.array([5.0, 6, 7, 8, 9, 15, 16, 17, 18]),
        )
        # equality of identically constructed patches works.
        patch2 = patch.new()
        assert patch == patch2


class TestFromArray:
    """Tests for CoordSegmented.from_array."""

    def test_gapped_uniform_array(self):
        """Uniform runs separated by a gap become two range segments."""
        values = np.array([0.0, 1, 2, 3, 10, 11, 12, 13])
        coord = CoordSegmented.from_array(values)
        assert isinstance(coord, CoordSegmented)
        assert coord.segment_count == 2
        assert all(isinstance(x, CoordRange) for x in coord.segments)
        assert np.array_equal(coord.values, values)
        gaps = coord.get_discontinuities("gaps")
        assert len(gaps) == 1
        assert gaps.iloc[0]["index"] == 4

    def test_fully_uniform_returns_range(self):
        """A gapless uniform array degrades to a plain range."""
        coord = CoordSegmented.from_array(np.arange(10.0))
        assert isinstance(coord, CoordRange)

    def test_irregular_returns_monotonic(self):
        """Arrays with no uniform runs come back as one monotonic coord."""
        values = np.array([0.0, 1.0, 2.1, 3.3, 4.0])
        coord = CoordSegmented.from_array(values)
        assert isinstance(coord, CoordMonotonicArray)
        assert np.array_equal(coord.values, values)

    def test_isolated_sample_between_gaps(self):
        """A lone sample between gaps becomes its own segment."""
        values = np.array([0.0, 1, 2, 10, 20, 21, 22])
        coord = CoordSegmented.from_array(values)
        assert coord.segment_count == 3
        assert np.array_equal(coord.values, values)
        assert len(coord.get_discontinuities()) == 2

    def test_datetime_gap(self):
        """Datetime arrays with internal gaps segment exactly."""
        one_s = np.timedelta64(1, "s")
        t0 = np.datetime64("2020-01-01", "ns")
        values = np.concatenate(
            [t0 + np.arange(5) * one_s, t0 + (np.arange(5) + 8) * one_s]
        )
        coord = CoordSegmented.from_array(values)
        assert coord.segment_count == 2
        assert np.array_equal(coord.values, values)
        assert len(coord.get_discontinuities("gaps")) == 1

    def test_tolerance_absorbs_gap(self):
        """A tolerance re-fits the result with bounded error."""
        values = np.array([0.0, 1, 2, 3, 6, 7, 8, 9])
        coord = CoordSegmented.from_array(values, tolerance=2.0)
        assert isinstance(coord, CoordRange)
        assert np.max(np.abs(coord.values - values)) <= 2.0

    def test_reverse_array(self):
        """Reverse-sorted arrays segment correctly."""
        values = np.array([13.0, 12, 11, 10, 3, 2, 1, 0])
        coord = CoordSegmented.from_array(values)
        assert coord.segment_count == 2
        assert coord.reverse_sorted
        assert np.array_equal(coord.values, values)

    def test_short_arrays_pass_through(self):
        """Arrays too short to segment build plain coords."""
        assert len(CoordSegmented.from_array(np.array([1.0, 2.0]))) == 2
        assert len(CoordSegmented.from_array(np.array([1.0]))) == 1

    def test_non_monotonic_raises(self):
        """Unsorted or duplicated values are rejected."""
        with pytest.raises(CoordError, match="monotonic"):
            CoordSegmented.from_array(np.array([0.0, 2.0, 1.0, 3.0]))
        with pytest.raises(CoordError, match="monotonic"):
            CoordSegmented.from_array(np.array([0.0, 1.0, 1.0, 2.0]))

    def test_nan_raises(self):
        """Missing values are rejected."""
        with pytest.raises(CoordError, match="missing"):
            CoordSegmented.from_array(np.array([0.0, np.nan, 2.0]))

    def test_2d_raises(self):
        """Only 1D arrays are supported."""
        with pytest.raises(CoordError, match="1D"):
            CoordSegmented.from_array(np.zeros((3, 3)))

    def test_units(self):
        """Units land on the result."""
        values = np.array([0.0, 1, 2, 10, 11, 12])
        coord = CoordSegmented.from_array(values, units="m")
        assert get_quantity(coord.units) == get_quantity("m")
