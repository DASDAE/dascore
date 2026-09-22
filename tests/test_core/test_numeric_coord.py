"""Tests for the run model a numeric coordinate is built from."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from dascore.core import coords as coords_module
from dascore.core.coords import (
    CoordString,
    Grid,
    Labels,
    NumericCoord,
    concat_coords,
    get_coord,
)

T0 = np.datetime64("2020-01-01T00:00:00")


@pytest.fixture
def gappy():
    """Two evenly sampled runs with four positions missing between them."""
    first = get_coord(start=0.0, step=1.0, shape=(5,))
    second = get_coord(start=9.0, step=1.0, shape=(5,))
    return concat_coords(first, second)


@pytest.fixture
def jittered():
    """One grid, then labels a grid cannot restate, then another grid."""
    values = np.array([0.0, 1.0, 2.0, 5.5, 7.1, 20.0, 21.0, 22.0])
    return get_coord(data=values, snap=False)


class TestRunsHoldTheLabels:
    """The runs a coordinate is built from describe the labels it has."""

    def test_one_grid_is_evenly_sampled(self):
        """A coordinate of one grid answers by arithmetic."""
        coord = get_coord(start=0, stop=10, step=1)
        assert coord.runs_count == 1
        assert isinstance(coord.runs[0], Grid)
        assert coord.evenly_sampled

    def test_a_float_run_folds_its_offset(self):
        """A float run's index offset moves its origin, as an exact one's does."""
        coord = get_coord(runs=(Grid(0.0, 1.0, 0, 3, k0=5),), dtype=np.dtype("float64"))
        assert np.array_equal(coord.values, [5.0, 6.0, 7.0])
        assert np.array_equal(coord.select((5.0, 6.0))[0].values, [5.0, 6.0])
        assert coord.runs[0].canonical() == (5.0, 1.0, 0, 3)
        assert coord.data_id != get_coord(start=0.0, step=1.0, shape=(3,)).data_id

    def test_stored_labels_are_one_run(self):
        """Labels no grid describes are held as they are."""
        coord = NumericCoord.from_labels(np.array([0.0, 1.0, 3.5]))
        assert coord.runs_count == 1
        assert isinstance(coord.runs[0], Labels)
        assert not coord.evenly_sampled

    def test_segments_are_the_runs(self, gappy):
        """Each run comes back as a coordinate of its own."""
        segments = gappy.segments
        assert len(segments) == gappy.runs_count == 2
        assert np.array_equal(
            np.concatenate([x.values for x in segments]), gappy.values
        )

    def test_dtype_comes_from_stored_runs(self):
        """A coordinate of stored labels need not be told its dtype."""
        coord = NumericCoord(runs=(np.arange(4.0),))
        assert np.dtype(coord.dtype) == np.dtype("float64")

    def test_grids_alone_must_state_their_dtype(self):
        """Ticks say nothing about whether they are integers or times."""
        with pytest.raises(ValidationError, match="must state its dtype"):
            NumericCoord(runs=(Grid(0, 1, 1, 4),))


class TestFuse:
    """Runs which continue each other exactly become one."""

    def test_a_hole_keeps_the_runs_apart(self, gappy):
        """A gap is a fact about the data, so the runs stay two."""
        assert gappy.runs_count == 2
        assert not gappy.evenly_sampled

    def test_two_stored_runs_never_fuse(self):
        """Joining stored labels would cost a copy and lose both ids."""
        first = NumericCoord.from_labels(np.array([0.0, 1.5, 3.0]))
        second = NumericCoord.from_labels(np.array([4.0, 5.5, 7.0]))
        assert concat_coords(first, second).runs_count == 2

    def test_fuse_absorbs_a_hole_within_tolerance(self, gappy):
        """A wide enough tolerance refits the runs as one grid."""
        out = gappy.fuse(10.0)
        assert out.runs_count == 1 and out.evenly_sampled
        assert out.min() == gappy.min() and out.max() == gappy.max()

    def test_fuse_keeps_a_hole_its_tolerance_cannot_reach(self, gappy):
        """No label may move further than the tolerance allows."""
        assert gappy.fuse(0.1).runs_count == 2

    def test_fuse_with_keep_step_keeps_missing_samples_missing(self, gappy):
        """A hole is data that is absent, not a slower sampling rate."""
        assert gappy.fuse(10.0, keep_step=True).runs_count == 2

    def test_fuse_leaves_an_evenly_sampled_coord_alone(self):
        """There is nothing simpler than one grid."""
        coord = get_coord(start=0.0, step=1.0, shape=(5,))
        assert coord.fuse(1.0) is coord

    def test_a_slice_landing_on_a_grid_becomes_one(self, gappy):
        """Trimming away the seam leaves an evenly sampled coordinate."""
        assert gappy[:5].evenly_sampled


class TestMultiRunSlicing:
    """Slicing maps flat sample positions back onto the runs."""

    def test_slice_inside_one_run(self, gappy):
        """A window wholly inside a run keeps that run's arithmetic."""
        out = gappy[1:4]
        assert out.evenly_sampled
        assert np.array_equal(out.values, gappy.values[1:4])

    def test_slice_across_the_seam(self, gappy):
        """A window over the seam keeps both runs and the hole between."""
        out = gappy[3:7]
        assert out.runs_count == 2
        assert np.array_equal(out.values, gappy.values[3:7])

    def test_slice_at_the_run_bounds_reuses_the_runs(self, gappy):
        """A slice on a boundary hands the run back rather than copying it."""
        assert gappy[5:].runs[0] is gappy.runs[1]

    def test_strided_slice_keeps_the_labels(self, jittered):
        """A stride across runs may not move a label."""
        out = jittered[::3]
        assert np.array_equal(out.values, jittered.values[::3])

    def test_reversed_slice_keeps_the_runs(self, gappy):
        """Reversing a coordinate with holes keeps them where they are."""
        out = gappy[::-1]
        assert out.runs_count == 2 and out.reverse_sorted
        assert np.array_equal(out.values, gappy.values[::-1])

    def test_index_a_single_sample(self, gappy):
        """An integer index evaluates only the sample it names."""
        assert gappy[7] == gappy.values[7]

    def test_array_index_keeps_the_labels(self, jittered):
        """Fancy indexing across runs may not move a label either."""
        indices = np.array([0, 3, 4, 6])
        assert np.array_equal(jittered[indices].values, jittered.values[indices])


class TestMultiRunSelect:
    """Selection asks each run in turn rather than concatenating them."""

    def test_window_inside_one_run(self, gappy):
        """A window one run covers selects inside it."""
        new, indexer = gappy.select((1.0, 3.0))
        assert np.array_equal(new.values, np.array([1.0, 2.0, 3.0]))
        assert gappy.values[indexer].tolist() == new.values.tolist()

    def test_open_window_keeps_everything(self, gappy):
        """A window bounded on neither side selects the whole coordinate."""
        new, _ = gappy.select((None, None))
        assert np.array_equal(new.values, gappy.values)

    def test_reverse_sorted_select(self, gappy):
        """A descending coordinate selects the same window."""
        flipped = gappy[::-1]
        new, _ = flipped.select((3.0, 10.0))
        assert np.array_equal(new.values, np.array([10.0, 9.0, 4.0, 3.0]))

    def test_get_index_on_descending_labels(self):
        """Descending stored labels are searched by negating them."""
        coord = NumericCoord.from_labels(np.array([9.0, 7.5, 3.0, 1.0]))
        assert coord.reverse_sorted
        new, _ = coord.select((3.0, 8.0))
        assert np.array_equal(new.values, np.array([7.5, 3.0]))

    def test_index_of_a_value_in_descending_labels(self):
        """Descending labels are searched by negating them, not by scanning."""
        coord = NumericCoord.from_labels(np.array([9.0, 7.5, 3.0, 1.0]))
        # forward is the coordinate's own direction, which here descends
        assert coord._get_index(7.5, forward=False) == 1
        assert coord._get_index(7.5, forward=True) == 2


class TestHoles:
    """Missing grid positions are read from the runs, not the labels."""

    def test_holes_between_runs(self, gappy):
        """The positions between two runs of one grid are missing."""
        missing = gappy.missing()
        assert missing.count == 4
        assert list(missing.iter_runs()) == [(5.0, 8.0)]

    def test_an_evenly_sampled_coord_is_complete(self):
        """One grid skips nothing."""
        assert get_coord(start=0, stop=10, step=1).missing().complete

    def test_holes_inside_a_stored_run(self):
        """Labels on a declared grid state the positions they skip."""
        coord = NumericCoord.from_labels(np.array([1, 3, 4]), step=1)
        assert coord.missing().count == 1

    def test_a_coordinate_without_a_step_misses_nothing(self, jittered):
        """Missing is relative to a grid, and these labels state none."""
        assert jittered.step is None and jittered.missing().complete

    def test_seams_report_every_run_boundary(self, gappy):
        """Each boundary is one row of the discontinuity frame."""
        assert len(gappy.get_discontinuities("all")) == 1
        assert len(gappy.get_discontinuities("gaps")) == 1

    def test_unordered_labels_have_no_seams(self):
        """A coordinate with no direction has no spacing to depart from."""
        coord = NumericCoord.from_labels(np.array([3.0, 1.0, 2.0]))
        assert not len(coord.get_discontinuities("all"))


class TestLabelsAreHeldOnce:
    """A stored array is frozen, copied, and hashed once, on the way in."""

    @pytest.fixture
    def stored(self):
        """A coordinate holding labels no grid describes."""
        return NumericCoord.from_labels(np.array([0.0, 1.5, 4.0, 7.0, 9.5, 20.0]))

    def test_sources_are_read_only(self, stored):
        """Nothing may write through a stored array."""
        with pytest.raises(ValueError, match="read-only"):
            next(iter(stored.sources.values()))[0] = 1.0

    def test_sources_are_copied(self):
        """A later write to the array a coordinate was given cannot reach it."""
        source = np.arange(5.0)
        coord = NumericCoord.from_labels(source)
        source[0] = 99.0
        assert coord.values[0] == 0.0

    def test_windowing_never_hashes(self, stored, monkeypatch):
        """A slice, a stride and a reversal only move the window."""
        values = stored.values
        monkeypatch.setattr(coords_module, "hash_array", _no_hashing)
        for item in (slice(1, None), slice(None, None, 2), slice(None, None, -1)):
            sliced = stored[item]
            assert isinstance(sliced.runs[0], Labels)
            assert np.array_equal(sliced.values, values[item])

    def test_striding_never_hashes(self, stored, monkeypatch):
        """A stride is window arithmetic, inside a run and across a seam."""
        second = NumericCoord.from_labels(np.array([30.0, 31.5, 34.0, 37.2, 40.0]))
        joined = concat_coords(stored, second)
        ids = {stored.runs[0].id, second.runs[0].id}
        values = joined.values
        monkeypatch.setattr(coords_module, "hash_array", _no_hashing)
        # wholly inside the first run, forwards and backwards
        assert joined[1:6:2].runs == (Labels(stored.runs[0].id, 3, 1, 2),)
        assert joined[5:0:-2].runs == (Labels(stored.runs[0].id, 3, 5, -2),)
        for item in (
            slice(1, 6, 2),
            slice(5, 0, -2),
            *(slice(None, None, x) for x in (2, -2)),
        ):
            out = joined[item]
            windows = [x for x in out.runs if isinstance(x, Labels)]
            assert windows and {x.id for x in windows} <= ids
            assert np.array_equal(out.values, values[item])

    def test_slicing_at_run_bounds_never_hashes(self, gappy, monkeypatch):
        """A run a slice keeps whole is the run it already was."""
        monkeypatch.setattr(coords_module, "hash_array", _no_hashing)
        assert gappy[5:].runs_count == 1

    def test_moving_between_coordinates_never_hashes(self, stored, monkeypatch):
        """Concatenation, fusion and metadata leave the store alone."""
        second = NumericCoord.from_labels(np.array([30.0, 31.5, 33.0]))
        monkeypatch.setattr(coords_module, "hash_array", _no_hashing)
        joined = concat_coords(stored, second)
        assert joined.runs_count == 2 and len(joined.sources) == 2
        assert joined.fuse(0.0).runs_count == 2
        assert joined.set_units("m").units is not None

    def test_abutting_windows_fuse(self, stored, monkeypatch):
        """Two windows reading on through one array become one window."""
        monkeypatch.setattr(coords_module, "hash_array", _no_hashing)
        joined = concat_coords(stored[:3], stored[3:])
        assert joined.runs_count == 1 and joined == stored

    def test_unused_entries_are_dropped(self, stored):
        """A store never outgrows the coordinate that carries it."""
        joined = concat_coords(stored, NumericCoord.from_labels(np.array([30.0])))
        assert len(joined.sources) == 2 and len(joined[:3].sources) == 1

    def test_a_run_needs_its_source(self, stored):
        """A window naming labels no source holds is refused."""
        with pytest.raises(ValidationError, match="no source holds"):
            NumericCoord(runs=stored.runs, dtype=stored.dtype)

    def test_converting_units_hashes_once(self, monkeypatch):
        """Scaled labels are a new array, so they enter a store of their own."""
        coord = NumericCoord.from_labels(np.array([0.0, 1.5, 3.0]), units="m")
        calls = []
        monkeypatch.setattr(
            coords_module,
            "hash_array",
            lambda values: calls.append(values) or "converted",
        )
        out = coord.convert_units("cm")
        assert len(calls) == 1 and np.allclose(out.values, coord.values * 100)
        assert coord.convert_units("m") is coord and len(calls) == 1

    def test_the_id_names_the_labels(self):
        """Two coordinates holding the same labels store them under one id."""
        labels = np.array([0.0, 1.5, 3.0])
        first, second = (NumericCoord.from_labels(labels) for _ in range(2))
        assert set(first.sources) == set(second.sources)


def _no_hashing(_):
    """Stand in for hash_array where nothing should be hashed."""
    raise AssertionError("the labels were hashed again")


class TestBaseCoordFallbacks:
    """The behaviour coords which hold no runs fall back on."""

    @pytest.fixture
    def partial(self):
        """A coordinate which states a shape and a step but no values."""
        return get_coord(shape=(5,), step=1)

    def test_set_units_is_a_no_op(self, partial):
        """A coordinate already carrying these units hands itself back."""
        assert partial.set_units(None) is partial

    def test_snap_returns_itself(self, partial):
        """There are no values to move onto a grid."""
        assert partial.snap() is partial

    def test_fuse_returns_itself(self, partial):
        """There are no runs to refit."""
        assert partial.fuse(1.0) is partial

    def test_no_discontinuities(self, partial):
        """A coordinate which states no values states no seams."""
        assert not len(partial.get_discontinuities("all"))

    def test_nothing_is_missing(self, partial):
        """A declared step with no values skips no position."""
        assert partial.missing().complete

    def test_string_coords_state_no_seams(self):
        """Text has no spacing to be discontinuous in."""
        coord = get_coord(data=np.array(["a", "b", "c"]))
        assert isinstance(coord, CoordString)
        assert not len(coord.get_discontinuities("all"))


class TestUpdates:
    """What `new` and `change_length` do to each shape of coordinate."""

    def test_new_without_arguments_is_a_no_op(self, gappy):
        """Nothing was asked for, so nothing changes."""
        assert gappy.new() is gappy

    def test_new_moves_stored_labels(self, jittered):
        """A stored coordinate is rebuilt from the labels it holds."""
        out = jittered.new(units="m")
        assert out.units is not None

    def test_new_declares_a_grid_on_stored_labels(self):
        """A step given to a stored coordinate is the grid its labels sit on."""
        coord = NumericCoord.from_labels(np.array([0.0, 1.0, 3.0]))
        out = coord.new(step=1.0)
        assert out.step == 1.0 and out.missing().count == 1

    def test_new_start_keeps_the_count(self):
        """Moving the start of a grid keeps its length and step."""
        coord = get_coord(start=0.0, step=1.0, shape=(5,))
        out = coord.new(start=10.0)
        assert len(out) == len(coord) and out.step == coord.step
        assert out.min() == 10.0

    def test_new_with_segments(self):
        """A coordinate may be rebuilt from coordinates to concatenate."""
        first = get_coord(start=0.0, step=1.0, shape=(5,))
        second = get_coord(start=9.0, step=1.0, shape=(5,))
        out = first.new(segments=(first, second))
        assert out.runs_count == 2

    def test_change_length_needs_a_grid(self, gappy):
        """Only an evenly sampled coordinate can say what comes next."""
        with pytest.raises(NotImplementedError, match="change_length"):
            gappy.change_length(20)

    def test_declared_step_needs_monotonic_labels(self):
        """A step is the grid the labels sit on, so they must climb it."""
        with pytest.raises(ValidationError, match="monotonic values"):
            NumericCoord.from_labels(np.array([3.0, 1.0, 2.0]), step=1.0)

    def test_a_dumped_coordinate_round_trips(self, gappy):
        """A coordinate rebuilt from its own dump is the same coordinate."""
        assert get_coord(**gappy.model_dump()) == gappy

    def test_concat_accepts_dumped_coordinates(self):
        """Concatenation takes the mappings a dump produces."""
        first = get_coord(start=0.0, step=1.0, shape=(5,))
        second = get_coord(start=9.0, step=1.0, shape=(5,))
        out = concat_coords(first.model_dump(), second.model_dump())
        assert out.runs_count == 2


class TestNDimensionalLabels:
    """Labels of more than one dimension are one stored run."""

    @pytest.fixture
    def two_d(self):
        """A coordinate over a two dimensional array of labels."""
        return NumericCoord.from_labels(np.arange(12.0).reshape(3, 4))

    def test_shape_comes_from_the_labels(self, two_d):
        """The run states the whole shape, not a sample count."""
        assert two_d.shape == (3, 4) and two_d.runs_count == 1

    def test_no_direction(self, two_d):
        """A coordinate of more than one dimension is not ordered."""
        assert not two_d.sorted and not two_d.reverse_sorted

    def test_no_seams(self, two_d):
        """There is no single axis for a spacing to change along."""
        assert not len(two_d.get_discontinuities("all"))

    def test_select_by_mask(self, two_d):
        """Unordered labels are selected by comparing every one of them."""
        new, mask = two_d.select((2.0, 5.0))
        assert np.array_equal(np.sort(two_d.values[mask]), np.arange(2.0, 6.0))
        assert new.size == 4
