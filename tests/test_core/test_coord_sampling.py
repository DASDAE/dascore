"""Tests for a coordinate's step, its jittered and gapped flags, and inference."""

from __future__ import annotations

import warnings
from fractions import Fraction

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

import dascore as dc
import dascore.utils.patch
from dascore.core.coords import NumericCoord, get_coord
from dascore.exceptions import CoordError
from dascore.io.dasdae.utils import _read_coord, _save_coord

T0 = np.datetime64("2024-01-01T00:00:00", "ns")
MS = np.timedelta64(1, "ms")


def _jittered_floats():
    """Unit-step labels, every other one 0.3 steps late."""
    values = np.arange(20.0)
    values[1::2] += 0.3
    return values


def _slightly_jittered_floats():
    """Unit-step labels within the released 0.1% snapping tolerance."""
    values = np.arange(100.0)
    values[1::2] += 1e-4
    return values


def _drifting_floats():
    """Labels whose spacing changes by 0.09% halfway, drifting 0.9 steps."""
    diffs = np.r_[np.full(2000, 1.0), np.full(2000, 1.0009)]
    return np.r_[0.0, np.cumsum(diffs)]


def _us_1024_hz(count=12_000):
    """1024 Hz labels stored as whole microseconds (issue #1419)."""
    ticks = (np.arange(count, dtype=np.int64) * 15625) // 16
    start = np.datetime64("2024-01-01T00:00:00", "us")
    return start + ticks.astype("timedelta64[us]")


def _no_warning(func, *args, **kwargs):
    """Call func, failing on any FutureWarning it emits."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        return func(*args, **kwargs)


class TestFlags:
    """A step and two independent flags describe how a coordinate is sampled."""

    def test_range(self):
        """A range is evenly sampled, neither jittered nor gapped."""
        coord = get_coord(start=0, stop=10, step=1)
        assert coord.evenly_sampled
        assert not coord.jittered and not coord.gapped

    def test_gapped(self):
        """Labels on a stated grid with skipped positions are gapped."""
        coord = get_coord(data=[1, 3, 4, 10, 11, 12], step=1)
        assert coord.gapped and not coord.jittered
        assert not coord.evenly_sampled
        assert coord.step == 1

    def test_jittered(self):
        """Labels within half a step of a grid keep it and are jittered."""
        values = _jittered_floats()
        coord = get_coord(data=values, snap=False)
        assert coord.jittered and not coord.gapped
        assert not coord.evenly_sampled
        assert coord.step == pytest.approx(1.0, rel=0.05)
        np.testing.assert_array_equal(coord.values, values)

    def test_jittered_and_gapped(self):
        """Jitter and holes are independent; a coordinate can have both."""
        coord = get_coord(data=[0.0, 1.1, 1.9, 5.05, 6.0], step=1.0)
        assert coord.jittered and coord.gapped
        assert coord.missing().count == 2

    def test_irregular(self):
        """A coordinate with no step is neither jittered nor gapped."""
        coord = get_coord(data=[0.0, 1.0, 5.0, 6.0])
        assert coord.step is None
        assert not coord.jittered and not coord.gapped
        assert not coord.evenly_sampled

    def test_string_coord(self):
        """Coordinates outside the sampling rules have neither flag."""
        coord = get_coord(data=np.array(["a", "b"]))
        assert not coord.jittered and not coord.gapped

    def test_stored_labels_on_their_step(self):
        """Stored labels sitting exactly on their step are evenly sampled."""
        coord = NumericCoord.from_labels(np.arange(10), step=1)
        assert coord.evenly_sampled


class TestInference:
    """get_coord(data=...) finds a grid or nothing (rule 2)."""

    @pytest.mark.parametrize("snap", [None, False])
    def test_skipped_position(self, snap):
        """An exact fit only with a skipped position gives no step."""
        assert get_coord(data=np.array([0, 1, 3]), snap=snap).step is None

    @pytest.mark.parametrize("snap", [None, False])
    def test_missing_integer(self, snap):
        """An integer array missing one value is not a fractional grid."""
        values = np.delete(np.arange(1000), 500)
        assert get_coord(data=values, snap=snap).step is None

    @pytest.mark.parametrize("snap", [None, False])
    def test_central_hole(self, snap):
        """A hole is never read as jitter on a longer step."""
        values = np.delete(np.arange(1001.0), 500)
        assert get_coord(data=values, snap=snap).step is None

    def test_jittered_floats(self):
        """Labels too jittered for the released snap get a step and the flag."""
        values = _jittered_floats()
        coord = _no_warning(get_coord, data=values)
        assert coord.jittered and coord.step is not None
        np.testing.assert_array_equal(coord.values, values)

    def test_jittered_times(self):
        """Time labels a nanosecond off their grid are jittered, judged exactly."""
        jitter = np.tile([0, 1, -1], 10)[:30].astype("timedelta64[ns]")
        values = T0 + np.arange(30) * MS + jitter
        coord = get_coord(data=values, snap=False)
        assert coord.jittered and coord.step == MS
        np.testing.assert_array_equal(coord.values, values)

    @pytest.mark.parametrize("snap", [None, False])
    def test_rounded_step_must_fit(self, snap):
        """Integer jitter keeps a whole-tick step only where it fits the labels."""
        values = np.floor(np.arange(1000) * 100.4).astype("int64")
        values[1::3] += 2
        coord = get_coord(data=values, snap=snap)
        assert coord.step is None
        np.testing.assert_array_equal(coord.values, values)

    def test_drift_has_no_step(self):
        """Drift is caught by checking every label, not just the spacings."""
        coord = get_coord(data=_drifting_floats(), snap=False)
        assert coord.step is None

    def test_far_from_zero(self):
        """Floats far from zero are on their grid within a few ulps."""
        values = 1.6e9 + np.arange(1000) * 1e-3
        coord = get_coord(data=values, snap=False)
        assert coord.evenly_sampled
        np.testing.assert_array_equal(coord.values, values)

    def test_float32_held_in_float64(self):
        """float64 labels holding float32 values are judged as float32."""
        values = np.linspace(0, 100, 1001, dtype=np.float32).astype(np.float64)
        coord = get_coord(data=values, snap=False)
        assert coord.evenly_sampled
        np.testing.assert_array_equal(coord.values, values)

    def test_descending(self):
        """A descending grid is judged as the ascending one is."""
        values = (1.6e9 + np.arange(1000) * 1e-3)[::-1]
        coord = get_coord(data=values, snap=False)
        assert coord.evenly_sampled and coord.step < 0

    def test_1024_hz_nanoseconds(self):
        """1024 Hz labels stored in nanoseconds are an exact grid."""
        ticks = (np.arange(4096, dtype=np.int64) * 1953125) // 2
        values = T0 + ticks.astype("timedelta64[ns]")
        coord = get_coord(data=values)
        assert coord.evenly_sampled
        assert coord.step_exact == Fraction(1, 1024)


class TestMicrosecondGrid:
    """Labels are found exactly in their own tick unit (issue #1419)."""

    @pytest.fixture(scope="class")
    def labels(self):
        """1024 Hz labels stored as microseconds, the first floored."""
        ticks = (np.arange(12_000, dtype=np.int64) * 15625 + 9) // 16
        start = np.datetime64("2024-01-01T00:00:00", "us")
        return start + ticks.astype("timedelta64[us]")

    @pytest.fixture(scope="class")
    def coord(self, labels):
        """The coordinate inferred from them."""
        return _no_warning(get_coord, data=labels)

    def test_exact_grid(self, coord, labels):
        """The grid is the exact rate, and no label moves."""
        assert coord.evenly_sampled
        assert coord.step_exact == Fraction(1, 1024)
        assert coord.dtype == labels.dtype
        np.testing.assert_array_equal(coord.values, labels)

    def test_snap_false(self, labels):
        """The exact read finds the same grid."""
        coord = get_coord(data=labels, snap=False)
        assert coord.evenly_sampled
        np.testing.assert_array_equal(coord.values, labels)

    def test_select(self, coord, labels):
        """Selecting by value lands on the stored labels."""
        out, _ = coord.select((labels[100], labels[200]))
        np.testing.assert_array_equal(out.values, labels[100:201])

    def test_summary_states_no_step(self, coord):
        """A floored coarse-unit grid is summarized without a range to rebuild."""
        summary = coord.to_summary()
        assert summary.step is None and summary.step_numerator is None

    def test_chunk_keeps_every_sample(self, coord):
        """Chunking reads the labels, so no sample falls between windows."""
        data = np.arange(len(coord) * 2).reshape(-1, 2)
        coords = {"time": coord, "distance": [0, 1]}
        patch = dc.Patch(data=data, coords=coords, dims=("time", "distance"))
        chunks = dc.spool([patch]).chunk(time=0.5, keep_partial=True)
        values = np.concatenate([x.get_coord("time").values for x in chunks])
        np.testing.assert_array_equal(values, coord.values)

    def test_dasdae_holes_keep_the_exact_step(self, tmp_path):
        """A fractional grid with a hole is stored and read back on its grid."""
        time = get_coord(start=T0, step=(1, 1024), shape=(4000,))
        holed = time[np.r_[0:2000, 2010:4000]]
        with h5py.File(tmp_path / "holed.h5", "w") as h5:
            _save_coord(holed, "_coord_time", h5, compact=True)
            back = _read_coord(h5["_coord_time"], "time", {}, snap=False)
        assert back.step_exact == Fraction(1, 1024) and not back.jittered
        assert back.gapped and np.array_equal(back.values, holed.values)

    def test_dasdae_strided_run_keeps_its_step(self, tmp_path):
        """A stored run one stride apart reads back on its declared step."""
        sparse = get_coord(data=np.arange(0, 2000, 2), step=1)
        dense = get_coord(start=3000, stop=3010, step=1)
        coord = dc.core.coords.concat_coords(sparse, dense)
        with h5py.File(tmp_path / "strided.h5", "w") as h5:
            _save_coord(coord, "_coord_x", h5, compact=True)
            back = _read_coord(h5["_coord_x"], "x", {}, snap=False)
        assert back.step == 1 and back.gapped
        np.testing.assert_array_equal(back.values, coord.values)

    def test_dasdae_round_trip(self, coord, labels, tmp_path):
        """DASDAE stores the grid and reads back the same labels."""
        data = np.zeros((len(coord), 2))
        coords = {"time": coord, "distance": [0, 1]}
        patch = dc.Patch(data=data, coords=coords, dims=("time", "distance"))
        back = dc.read(dc.write(patch, tmp_path / "us.h5", "dasdae"))[0]
        time = back.get_coord("time")
        assert time.evenly_sampled and time.dtype == labels.dtype
        np.testing.assert_array_equal(time.values, labels)

    def test_new_keeps_labels(self, coord, labels):
        """Rebuilding the grid with new units keeps the labels."""
        out = coord.new(shape=(100,))
        np.testing.assert_array_equal(out.values, labels[:100])

    def test_holes(self, labels):
        """Missing positions of a microsecond grid are microsecond labels."""
        coord = get_coord(data=labels)
        holed = coord[np.r_[0:10, 15:30]]
        missing = holed.missing()
        assert missing.count == 5
        np.testing.assert_array_equal(missing.positions(), labels[10:15])

    def test_month_labels(self):
        """Months have no length in ticks, so they find no fractional grid."""
        values = np.array(["2020-01", "2020-03", "2020-04"], dtype="M8[M]")
        coord = get_coord(data=values, snap=False)
        np.testing.assert_array_equal(coord.values, values)

    def test_coarse_grid_past_nanoseconds(self):
        """A seconds grid whose nanoseconds overflow int64 keeps its labels."""
        ticks = (np.arange(300, dtype=np.int64) * 1000) // 3
        values = np.datetime64("3000-01-01T00:00:00", "s") + ticks.astype("m8[s]")
        coord = get_coord(data=values, snap=False)
        assert coord._grid is None
        np.testing.assert_array_equal(coord.values, values)


class TestStatedStep:
    """With a stated step, labels are placed on that grid (rule 2)."""

    def test_holes(self):
        """Skipped positions are holes."""
        coord = get_coord(data=[0.0, 1.0, 2.0, 5.0, 6.0], step=1.0)
        assert coord.gapped and coord.missing().count == 2

    def test_jitter(self):
        """Labels off the stated grid by under half a step are kept, jittered."""
        values = np.array([0.0, 1.1, 1.95, 3.02])
        coord = get_coord(data=values, step=1.0)
        assert coord.jittered and not coord.gapped
        assert coord.step == 1.0
        np.testing.assert_array_equal(coord.values, values)

    def test_integer_jitter(self):
        """Integer labels off the stated grid are jittered, judged exactly."""
        coord = get_coord(data=[0, 10, 21, 30], step=10)
        assert coord.jittered and coord.step == 10

    def test_one_position_raises(self):
        """Two labels on one grid position raise."""
        with pytest.raises(CoordError, match="one position"):
            get_coord(data=[0.0, 1.0, 1.3, 2.0], step=1.0)

    def test_drift_raises(self):
        """Labels drifting half a step or more from the stated grid raise."""
        with pytest.raises(CoordError, match="half a step"):
            get_coord(data=np.arange(100) * 1.02, step=1.0)

    def test_unsigned_past_int64(self):
        """Unsigned labels past the int64 range are kept, with no step."""
        values = np.array([0, 2**64 - 1], dtype=np.uint64)
        coord = get_coord(data=values, snap=False)
        np.testing.assert_array_equal(coord.values, values)

    def test_drift_across_runs_drops_the_step(self):
        """Offsets accumulating over many joined runs leave no common grid."""
        pieces = [
            NumericCoord.from_labels(np.array([1.8 * k, 1.8 * k + 0.9]), step=1.0)
            for k in range(6)
        ]
        assert all(x.step == 1.0 for x in pieces)
        assert dc.core.coords.concat_coords(*pieces).step is None

    def test_stored_drift_raises(self):
        """Stored labels drifting from their step are refused when built too."""
        with pytest.raises(ValidationError, match="half a step"):
            NumericCoord.from_labels(np.arange(100) * 1.02, step=1.0)


class TestReleasedSnapping:
    """The released 0.1% snapping is kept for one release, with a warning."""

    def test_default_warns(self):
        """The default keeps the released grid and warns that it will change."""
        values = _slightly_jittered_floats()
        with pytest.warns(FutureWarning, match="snap=False"):
            coord = get_coord(data=values)
        assert coord.evenly_sampled

    def test_snap_true_is_silent(self):
        """Asking for the released snapping keeps it without a warning."""
        coord = _no_warning(get_coord, data=_slightly_jittered_floats(), snap=True)
        assert coord.evenly_sampled

    def test_snap_false_is_exact(self):
        """snap=False gives the new result now."""
        values = _slightly_jittered_floats()
        coord = _no_warning(get_coord, data=values, snap=False)
        assert coord.jittered
        np.testing.assert_array_equal(coord.values, values)

    def test_on_grid_is_silent(self):
        """Labels already on a grid give the same result and no warning."""
        values = np.arange(0, 10, 0.1)
        coord = _no_warning(get_coord, data=values)
        assert coord.evenly_sampled
        np.testing.assert_allclose(coord.values, values)

    def test_relabelling_is_not_kept(self):
        """A released grid moving labels half a step or more is not kept."""
        coord = _no_warning(get_coord, data=_drifting_floats())
        assert coord.step is None

    def test_relabelling_jitter_snaps_from_its_ends(self):
        """Jitter the median step would relabel is snapped from its end labels."""
        pattern = np.array([0.9999, 0.9999, 0.9999, 1.00015, 1.00015])
        values = np.r_[0.0, np.cumsum(np.tile(pattern, 6000))]
        with pytest.warns(FutureWarning, match="snap=False"):
            coord = get_coord(data=values)
        assert coord.evenly_sampled
        assert coord.step == pytest.approx(1.0)
        assert np.max(np.abs(coord.values - values)) < 0.01


class TestSelectionKeepsStep:
    """Removing samples keeps the step and leaves holes (rule 1)."""

    @pytest.fixture(scope="class")
    def coord(self):
        """Twenty samples one step apart."""
        return get_coord(start=0, stop=20, step=1)

    def test_mask(self, coord):
        """A mask removing a block leaves a hole."""
        mask = np.ones(20, dtype=bool)
        mask[5:10] = False
        out, _ = coord.select(mask)
        assert out.step == 1 and out.gapped
        assert out.missing().count == 5

    def test_regular_mask(self, coord):
        """Samples one stride apart, by mask or array, are that stride."""
        out = coord[np.arange(20) % 2 == 0]
        assert out == coord[::2]
        assert coord[np.array([1, 4, 7])] == coord[1:8:3]

    def test_stride(self, coord):
        """Only an explicit stride gives a coarser step."""
        out = coord[::2]
        assert out.step == 2 and out.evenly_sampled

    def test_jittered_holes_keep_their_positions(self):
        """A hole in jittered labels counts the samples removed, or states no step."""
        jittered = get_coord(data=[0.0, 1.1, 1.9, 3.05, 4.0], step=1.0)
        out = jittered[np.array([0, 3, 4])]
        assert out.step == 1.0 and out.missing().count == 2
        # 2.8 is nearer three steps than the two it was taken from
        wide = get_coord(data=[0.0, 1.4, 2.8, 3.6], step=1.0)
        assert wide[np.array([0, 2, 3])].step is None

    def test_stepless_take_stays_stepless(self):
        """Removing samples from labels with no step invents no grid."""
        coord = get_coord(data=[0, 1, 3, 4, 5], snap=False)
        assert coord.step is None
        assert coord[np.array([0, 1, 3, 4])].step is None

    def test_unsigned_descending_positions(self, coord):
        """Unsigned positions running backwards are a backwards stride."""
        out = coord[np.array([5, 3, 1], dtype=np.uint64)]
        assert out == coord[5:0:-2]

    def test_positions_past_the_end_raise(self, coord):
        """Positions outside the coordinate are refused, as numpy refuses them."""
        with pytest.raises(IndexError):
            coord[np.array([0, 50])]

    def test_stored_grid_stacks(self):
        """Tiles of labels on their step within rounding are placed by that step."""
        values = np.linspace(0, 100, 1001, dtype=np.float32)[:100]
        distance = get_coord(data=values, snap=False)
        assert distance.evenly_sampled and distance._grid is None
        patch = dc.get_example_patch().select(distance=(0, 100), samples=True)
        patch = patch.update_coords(distance=distance)
        out = patch.tile_apply(lambda x: x, mode="stack", distance=10, samples=True)
        offsets = out.get_coord("distance_offset").values
        np.testing.assert_allclose(offsets, np.arange(10) * 0.1, atol=1e-5)

    def test_reorder(self, coord):
        """Reordered samples have no step."""
        assert coord[np.array([3, 1, 2])].step is None

    def test_dropna(self, random_patch):
        """Dropping missing samples leaves holes."""
        data = np.array(random_patch.data)
        axis = random_patch.get_axis("time")
        index = [slice(None)] * data.ndim
        index[axis] = slice(10, 20)
        data[tuple(index)] = np.nan
        out = random_patch.new(data=data).dropna("time")
        time = out.get_coord("time")
        assert time.step == random_patch.get_coord("time").step
        assert time.missing().count == 10

    def test_isel_array(self, random_patch):
        """Picking positions leaves holes."""
        out = random_patch.isel(time=[0, 1, 2, 7, 8])
        time = out.get_coord("time")
        assert time.step == random_patch.get_coord("time").step
        assert time.missing().count == 4

    def test_isel_slice_of_gapped(self, holed_patch):
        """Slicing a gapped coordinate by position keeps its step."""
        out = holed_patch.isel(time=slice(50, 70))
        time = out.get_coord("time")
        assert time.step == 1.0 and time.gapped

    def test_sel_of_gapped(self, holed_patch):
        """Slicing a gapped coordinate by value keeps its step."""
        out = holed_patch.sel(time=slice(50.0, 110.0))
        assert out.get_coord("time").gapped


class TestRequireEvenlySampled:
    """require_evenly_sampled is the one requirement check (rule 4)."""

    @pytest.fixture(scope="class")
    def patches(self, holed_patch, stepless_seam_patch):
        """A gapped, a jittered and a stepless patch."""
        jittered = holed_patch.select(time=(0, 59), copy=True)
        values = np.arange(60.0)
        values[1::2] += 0.3
        time = get_coord(data=values, step=1.0)
        jittered = jittered.update_coords(time=time)
        return {
            "gapped": holed_patch,
            "jittered": jittered,
            "stepless": stepless_seam_patch,
        }

    @pytest.mark.parametrize("kind", ["gapped", "jittered", "stepless"])
    def test_refuses(self, patches, kind):
        """Gapped, jittered and stepless coordinates fail the check."""
        with pytest.raises(CoordError, match=r"fill_gaps.*snap_coords"):
            patches[kind].get_coord("time", require_evenly_sampled=True)

    def test_require_no_holes_removed(self):
        """The separate hole check is gone."""
        assert not hasattr(dascore.utils.patch, "require_no_holes")
