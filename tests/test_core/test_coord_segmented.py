"""Tests for segmented (piecewise) coordinates."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

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
            assert i1 == i2

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
        out, indexer = float_gap_coord.select((2, 12), samples=True)
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
        out, indexer = float_gap_coord.select(mask)
        assert np.array_equal(out.values, float_gap_coord.values[mask])

    def test_select_value_array(self, float_gap_coord):
        """Value-array selection keeps only matching values."""
        out, _ = float_gap_coord.select(np.array([1.0, 15.0, 99.0]))
        assert np.array_equal(out.values, np.array([1.0, 15.0]))

    def test_select_none_returns_all(self, float_gap_coord):
        """A null select keeps everything."""
        out, indexer = float_gap_coord.select(None)
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
