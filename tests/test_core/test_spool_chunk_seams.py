"""
Tests for how Spool.chunk joins patches: the seam test, overlaps, and the
one grid it fits to each stretch (rules 5 and 6 of the coordinate note).
"""

from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
import pytest

import dascore as dc
from dascore.config import config_context
from dascore.core.coords import get_coord
from dascore.exceptions import ParameterError
from dascore.units import s as seconds

T0 = np.datetime64("2020-01-01T00:00:00", "ns")
STEP = np.timedelta64(1_000_000, "ns")  # 1 ms
DIMS = ("distance", "time")
TOLERANCE_WARNING = "was used to force merging the patches"


def _patch(time):
    """A two-channel patch over the given time coordinate."""
    data = np.arange(2 * len(time), dtype=np.float64).reshape(2, len(time))
    coords = {"distance": np.arange(2), "time": time}
    return dc.Patch(data=data, coords=coords, dims=DIMS)


def _at(steps, count=10, step=STEP):
    """A patch whose first label is ``steps`` steps after T0."""
    ticks = int(step.astype("m8[ns]").astype(np.int64))
    start = T0 + np.timedelta64(round(steps * ticks), "ns")
    return _patch(get_coord(start=start, step=step, shape=(count,)))


def _seam_spool(width, count=10):
    """Two patches meeting at a seam ``width`` steps wide."""
    return dc.spool([_at(0, count), _at(count - 1 + width, count)])


def _lengths(spool):
    """The time length of each patch in a spool, in order."""
    return [len(x.get_coord("time")) for x in spool]


def _chunk(spool, **kwargs):
    """Chunk along time, returning the outputs and any tolerance warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = list(spool.chunk(time=None, **kwargs))
    gaps = [x for x in caught if TOLERANCE_WARNING in str(x.message)]
    return out, gaps


def _same(patch, coord, data):
    """Assert a patch holds exactly these time labels and this data."""
    assert patch.get_coord("time") == coord
    assert np.array_equal(patch.data, data)


def _joined(*patches, skip=0):
    """The data of patches joined along time, the later ones' first ``skip`` dropped."""
    later = [x.data[:, skip:] for x in patches[1:]]
    return np.concatenate([patches[0].data, *later], axis=1)


class TestSeamTest:
    """Two patches join when the later starts one step after the earlier ends."""

    @pytest.mark.parametrize(
        ("width", "tolerance", "count", "warns"),
        [
            (1.0, None, 1, False),
            (1.3, None, 2, False),
            (4.0, None, 2, False),
            (1.3, 1.5, 1, False),
            (4.0, 1.5, 2, False),
            (1.3, 10, 1, False),
            (4.0, 10, 1, True),
            (1.5, 2, 1, False),
            (1.6, 2, 1, True),
        ],
    )
    def test_seams(self, width, tolerance, count, warns):
        """Seams join by max_shift, or up to `tolerance` steps when given."""
        out, gaps = _chunk(_seam_spool(width), tolerance=tolerance)
        assert len(out) == count
        assert bool(gaps) == warns
        for patch in out:
            coord = patch.get_coord("time")
            assert coord.evenly_sampled
            assert not coord.jittered and not coord.gapped

    @pytest.mark.parametrize(
        ("width", "joins"), [(1.19, True), (1.21, False), (0.81, True), (0.79, False)]
    )
    def test_max_shift_threshold(self, width, joins):
        """A seam joins within 0.2 step of one step, on the earlier grid."""
        early, late = _at(0), _at(9 + width)
        out, _ = _chunk(dc.spool([late, early]))
        if joins:
            (patch,) = out
            grid = get_coord(start=T0, step=STEP, shape=(20,))
            _same(patch, grid, _joined(early, late))
        else:
            assert len(out) == 2
            for patch, source in zip(out, (early, late)):
                _same(patch, source.get_coord("time"), source.data)

    @pytest.mark.parametrize(("width", "joins"), [(2.0, True), (2.01, False)])
    def test_tolerance_threshold(self, width, joins):
        """A seam exactly `tolerance` steps wide joins; a wider one does not."""
        spool = _seam_spool(width)
        out, gaps = _chunk(spool, tolerance=2)
        assert len(out) == (1 if joins else 2)
        assert bool(gaps) == joins
        if joins:
            coord = out[0].get_coord("time")
            assert (coord.min(), coord.max()) == (T0, T0 + (18 + width) * STEP)
            assert np.array_equal(out[0].data, _joined(*spool))

    def test_note_example(self):
        """After 99 s with a 1 s step, 100.1 s joins and 101.3 s does not."""
        step = np.timedelta64(1, "s")

        def patch(start):
            return _at(start, count=100, step=step)

        assert len(dc.spool([patch(0), patch(100.1)]).chunk(time=None)) == 1
        assert len(dc.spool([patch(0), patch(101.3)]).chunk(time=None)) == 2
        late = dc.spool([patch(0), patch(101.3)])
        with pytest.warns(UserWarning, match="gap"):
            assert len(late.chunk(time=None, tolerance=2.3)) == 1

    def test_tolerance_join_is_spread(self):
        """A gap closed by tolerance is refitted evenly, ends unchanged."""
        spool = _seam_spool(4.0)
        with pytest.warns(UserWarning, match="gap"):
            (patch,) = spool.chunk(time=None, tolerance=10)
        coord = patch.get_coord("time")
        assert len(coord) == 20
        assert coord.min() == spool[0].get_coord("time").min()
        assert coord.max() == spool[1].get_coord("time").max()
        assert coord.evenly_sampled

    def test_quantity_tolerance_is_the_seam_width(self):
        """A quantity tolerance is the widest seam joined, not an excess."""
        spool = _seam_spool(1.3)
        joined = spool.chunk(time=None, tolerance=1.4 * STEP)
        assert len(joined) == 1
        # 1.3 steps of excess would have joined it as an excess
        assert len(spool.chunk(time=None, tolerance=1.25 * STEP)) == 2
        # a width under one step still leaves contiguous patches joined
        assert len(_seam_spool(1.0).chunk(time=None, tolerance=0.5 * STEP)) == 1

    def test_stepless_pieces_join_only_by_distance(self):
        """Labels with no step join only under a distance tolerance."""
        one = _patch(T0 + np.array([0, 3, 4]) * STEP)
        two = _patch(T0 + np.array([6, 9, 10]) * STEP)
        spool = dc.spool([one, two])
        assert len(spool.chunk(time=None)) == 2
        assert len(spool.chunk(time=None, tolerance=10)) == 2
        assert len(spool.chunk(time=None, tolerance=2 * STEP)) == 1

    def test_step_is_the_earlier_patch(self):
        """The seam is judged against the earlier patch's step."""
        early = _at(0, step=STEP)
        # 1.21 earlier steps is a gap, though only 1.16 of the later's own
        start = early.get_coord("time").max() + 1.21 * STEP
        late = _patch(get_coord(start=start, step=1.04 * STEP, shape=(10,)))
        assert len(dc.spool([early, late]).chunk(time=None)) == 2

    def test_stepless_stretch_keeps_its_labels(self):
        """Labels with no step never join a stepped stretch before them."""
        stepped = _at(0, 100)
        one = _patch(T0 + np.array([200.5, 201.2, 202.6, 203.5]) * STEP)
        two = _patch(T0 + np.array([203.0, 204.2, 205.1]) * STEP)
        first, second = dc.spool([stepped, one, two]).chunk(time=None)
        _same(first, stepped.get_coord("time"), stepped.data)
        labels = T0 + np.array([200.5, 201.2, 202.6, 203.5, 204.2, 205.1]) * STEP
        assert np.array_equal(second.get_array("time"), labels)
        assert np.array_equal(second.data, _joined(one, two, skip=1))


class TestMaxShift:
    """The max_shift argument bounds each seam."""

    def test_larger_shift_joins(self):
        """0.4 joins a 1.3 step seam, which the default 0.2 does not."""
        spool = _seam_spool(1.3)
        assert len(spool.chunk(time=None)) == 2
        (patch,) = spool.chunk(time=None, max_shift=0.4)
        assert patch.get_coord("time").step == STEP

    def test_none_is_config(self):
        """None means dascore.config.max_shift."""
        with config_context(max_shift=0.4):
            assert len(_seam_spool(1.3).chunk(time=None)) == 1

    @pytest.mark.parametrize("bad", [0.5, -0.1, 0.3 * seconds, True, "0.2", np.nan])
    def test_bad_values_raise(self, bad):
        """Only a plain number in [0, 0.5) is accepted."""
        spool = _seam_spool(1.0)
        for method in (spool.chunk, spool.chunk_plan):
            with pytest.raises(ParameterError, match="max_shift"):
                method(time=None, max_shift=bad)
        for method in (spool.get_gaps, spool.get_coverage):
            with pytest.raises(ParameterError, match="max_shift"):
                method(max_shift=bad)

    def test_zero_joins_exact_seams(self):
        """With no slack, exactly contiguous patches still join."""
        assert len(_seam_spool(1.0).chunk(time=None, max_shift=0)) == 1
        assert len(_seam_spool(1.1).chunk(time=None, max_shift=0)) == 2


class TestStretchGrid:
    """Each stretch gets one grid, fitted before windows are cut."""

    def test_drifting_files_spread_evenly(self):
        """Ten files each 0.15 step late spread evenly, ends unchanged."""
        spool = dc.spool([_at(i * 1000 + 0.15 * i, 1000) for i in range(10)])
        (patch,) = spool.chunk(time=None)
        coord = patch.get_coord("time")
        first, last = spool[0].get_coord("time"), spool[-1].get_coord("time")
        assert len(coord) == 10_000
        assert coord.evenly_sampled
        assert coord.min() == first.min()
        assert coord.max() == last.max()
        assert coord.step_exact == Fraction(
            int((last.max() - first.min()).astype(np.int64)), 9999 * 10**9
        )
        shift = spool.chunk_plan(time=None).members["shift"]
        # the labels moved, by less than a step anywhere
        assert 0 < shift.max() < 1

    def test_slightly_late_files_keep_step(self):
        """Three files each 0.05 step late keep the stated step exactly."""
        spool = dc.spool([_at(i * 10 + 0.05 * i) for i in range(3)])
        (patch,) = spool.chunk(time=None)
        coord = patch.get_coord("time")
        assert coord.step == STEP
        assert coord.min() == spool[0].get_coord("time").min()
        shift = spool.chunk_plan(time=None).members["shift"]
        assert shift.max() == pytest.approx(0.1)

    def test_roundest_stated_step(self):
        """Files stating 29.99999 Hz then 30 Hz give one output at 30 Hz."""
        first = get_coord(start=T0, step=1 / Fraction(2999999, 100000), shape=(300,))
        start = first.max() + np.timedelta64(33_333_333, "ns")
        second = get_coord(start=start, step=Fraction(1, 30), shape=(300,))
        spool = dc.spool([_patch(second), _patch(first)])
        (patch,) = spool.chunk(time=None)
        assert patch.get_coord("time").step_exact == Fraction(1, 30)
        assert patch.get_coord("time").min() == T0

    def test_equally_round_steps_go_to_the_earliest(self):
        """Of two equally round steps that both fit, the earlier patch's wins."""
        early = _at(0, 5, step=np.timedelta64(1020, "us"))
        start = T0 + 5 * np.timedelta64(1020, "us")
        late = _patch(
            get_coord(start=start, step=np.timedelta64(1010, "us"), shape=(5,))
        )
        (patch,) = dc.spool([late, early]).chunk(time=None)
        assert patch.get_coord("time").step == np.timedelta64(1020, "us")

    @pytest.mark.parametrize("max_shift", [None, 0])
    def test_fractional_rate_stays_exact(self, max_shift):
        """Three 1024 Hz files keep the exact 1024 Hz step when merged."""
        full = get_coord(start=T0, step=Fraction(1, 1024), shape=(3 * 61441,))
        pieces = [full[i * 61441 : (i + 1) * 61441] for i in range(3)]
        spool = dc.spool([_patch(x) for x in pieces])
        (patch,) = spool.chunk(time=None, max_shift=max_shift)
        coord = patch.get_coord("time")
        assert coord.step_exact == Fraction(1, 1024)
        assert coord == full

    def test_long_spread_stretch(self):
        """A long drifting stretch still gets one grid, and ordered windows."""
        step = np.timedelta64(1, "s")
        spool = dc.spool([_at(i * 40_000.15, 40_000, step) for i in range(3)])
        (patch,) = spool.chunk(time=None)
        coord = patch.get_coord("time")
        first, last = spool[0].get_coord("time").min(), spool[2].get_coord("time").max()
        assert len(coord) == 120_000 and coord.evenly_sampled
        assert coord.min() == first
        assert abs(coord.max() - last) < np.timedelta64(1, "us")
        starts = spool.chunk_plan(time=1000 * step).outputs["time_min"].to_numpy()
        assert (np.diff(starts) > np.timedelta64(0, "ns")).all()
        assert first <= starts.min() and starts.max() <= last

    def test_integer_spread_is_one_exact_grid(self):
        """Integers spread evenly hold the exact fractional grid, as planned."""

        def patch(start):
            coord = get_coord(data=start + np.arange(10) * 10)
            return dc.Patch(data=np.ones((1, 10)), coords={"x": coord}, dims=("y", "x"))

        spool = dc.spool([patch(i * 102) for i in range(10)])
        (whole,) = spool.chunk(x=None)
        coord = whole.get_coord("x")
        assert coord.dtype == np.int64 and len(coord.runs) == 1
        assert coord.step_exact == Fraction(1008, 99)
        assert np.array_equal(coord.values, np.arange(100) * 1008 // 99)
        windows = spool.chunk(x=250, keep_partial=True)
        labels = np.concatenate([x.get_array("x") for x in windows])
        assert np.array_equal(labels, coord.values)
        contents = windows.get_contents()
        assert contents["x_min"].tolist() == [x.get_coord("x").min() for x in windows]
        assert contents["x_max"].tolist() == [x.get_coord("x").max() for x in windows]

    @pytest.mark.parametrize(("late", "unit"), [(0.05, "us"), (0.15, "ns")])
    def test_microsecond_times_on_the_grid(self, late, unit):
        """Microsecond times take the grid, in their unit where it is whole."""
        step = np.timedelta64(10_000, "us")
        start = T0.astype("M8[us]")
        starts = [
            start + round(i * (100 + late) * 10_000) * step // 10_000 for i in range(5)
        ]
        spool = dc.spool([_patch(x + np.arange(100) * step) for x in starts])
        (whole,) = spool.chunk(time=None)
        coord = whole.get_coord("time")
        assert len(coord.runs) == 1 and coord.evenly_sampled
        assert np.datetime_data(coord.dtype)[0] == unit
        contents = spool.chunk(time=None).get_contents()
        assert contents["time_max"].iloc[0] == coord.max()
        windows = spool.chunk(time=250 * step, keep_partial=True)
        labels = np.concatenate([x.get_array("time") for x in windows])
        assert np.array_equal(labels, coord.values)

    def test_windows_start_from_first_label(self):
        """Windows start at start + k * (length - overlap) on the stretch."""
        spool = dc.spool([_at(i * 10 + 0.05 * i) for i in range(3)])
        length, overlap = 2.5 * STEP, 0.5 * STEP
        out = spool.chunk(time=length, overlap=overlap)
        starts = [x.get_coord("time").min() for x in out]
        whole = spool.chunk(time=None)[0].get_coord("time")
        for num, start in enumerate(starts):
            window = whole.min() + num * (length - overlap)
            expected = whole.values[whole.values >= window][0]
            assert start == expected


class TestOverlap:
    """Each sample is kept once and the earlier patch wins."""

    @pytest.mark.parametrize("tolerance", [None, 1.5])
    def test_overlap_on_grid(self, tolerance):
        """Labels 0..9 then 9..18 give 19 samples, one dropped."""
        spool = dc.spool([_at(0), _at(9)])
        (patch,) = spool.chunk(time=None, tolerance=tolerance)
        _same(
            patch, get_coord(start=T0, step=STEP, shape=(19,)), _joined(*spool, skip=1)
        )
        plan = spool.chunk_plan(time=None, tolerance=tolerance)
        assert plan.members["dropped"].tolist() == [0, 1]

    def test_overlap_off_grid(self):
        """A later patch half a step off the grid starts a new stretch."""
        early, late = _at(0), _at(8.5)
        one, two = dc.spool([early, late]).chunk(time=None)
        _same(one, early.get_coord("time"), early.data)
        _same(two, late.get_coord("time")[1:], late.data[:, 1:])

    def test_overlap_off_grid_tolerance(self):
        """Under tolerance=1.5 it continues from its sample 1.5 steps on."""
        spool = dc.spool([_at(0), _at(8.5)])
        (patch,) = spool.chunk(time=None, tolerance=1.5)
        coord = patch.get_coord("time")
        assert len(coord) == 18
        assert coord.evenly_sampled
        assert coord.max() == T0 + 17.5 * STEP
        assert np.array_equal(patch.data, _joined(*spool, skip=2))

    def test_nested_patch_dropped(self):
        """A patch wholly inside an earlier one adds nothing."""
        outer = _at(0, 20)
        (patch,) = dc.spool([outer, _at(5, 5)]).chunk(time=None)
        _same(patch, outer.get_coord("time"), outer.data)


@pytest.fixture(scope="module")
def late_spool():
    """Five slightly late files, so the stretch grid moves labels."""
    return dc.spool([_at(i * 10 + 0.15 * i) for i in range(5)])


class TestRejoin:
    """Chunk outputs are slices of one grid, so they rejoin exactly."""

    @pytest.mark.parametrize("loaded", [False, True])
    def test_rejoin(self, late_spool, loaded):
        """Rechunking the chunks, lazily or loaded, gives the spool's chunk."""
        whole = late_spool.chunk(time=None)[0]
        chunks = late_spool.chunk(time=2.5 * STEP)
        (rejoined,) = (dc.spool(list(chunks)) if loaded else chunks).chunk(time=None)
        _same(rejoined, whole.get_coord("time"), whole.data)

    def test_windows_are_slices(self, late_spool):
        """Every window's labels and data are a slice of the stretch's."""
        whole = late_spool.chunk(time=None)[0]
        labels = whole.get_coord("time").values
        for patch in late_spool.chunk(time=2.5 * STEP):
            values = patch.get_coord("time").values
            start = np.searchsorted(labels, values[0])
            assert np.array_equal(values, labels[start : start + len(values)])
            expected = whole.data[:, start : start + len(values)]
            assert np.array_equal(patch.data, expected)

    def test_float_coordinates(self):
        """Float labels rejoin bit for bit too."""

        def patch(start):
            return _patch(get_coord(start=start, step=0.1, shape=(10,)))

        spool = dc.spool([patch(0.0), patch(1.015), patch(2.03)])
        whole = spool.chunk(time=None)[0]
        rejoined = dc.spool(list(spool.chunk(time=0.25))).chunk(time=None)
        _same(rejoined[0], whole.get_coord("time"), whole.data)


class TestPlanRecords:
    """chunk_plan says why each output starts, and what the fit did."""

    def test_reasons(self):
        """Pieces record how they start or join."""
        spool = dc.spool([_at(0), _at(10), _at(30), _at(40.3)])
        members = spool.chunk_plan(time=None, tolerance=1.5).members
        assert members["reason"].tolist() == ["stretch", "seam", "gap", "tolerance"]
        windowed = spool.chunk_plan(time=5 * STEP).members
        assert "window" in set(windowed["reason"])

    def test_diagnostics_never_reach_patches(self):
        """The diagnostic columns are not attributes of what chunk yields."""
        spool = dc.spool([_at(0), _at(10)])
        patch = spool.chunk(time=None)[0]
        assert not {"reason", "shift", "dropped"} & set(patch.attrs.model_dump())
        rechunked = spool.chunk(time=None).chunk(time=None)
        assert len(rechunked) == 1


class TestSnapCoordsDeprecated:
    """chunk always fits one grid per stretch; snap_coords is ignored."""

    @pytest.mark.parametrize("value", [True, False])
    def test_warns(self, value):
        """Passing snap_coords at all warns."""
        spool = _seam_spool(1.1)
        with pytest.warns(DeprecationWarning, match="snap_coords"):
            spool.chunk(time=None, snap_coords=value)
        with pytest.warns(DeprecationWarning, match="snap_coords"):
            spool.chunk_plan(time=None, snap_coords=value)

    @pytest.mark.parametrize("value", [True, False])
    def test_changes_nothing(self, value):
        """Either value gives what the default gives, on files old snaps moved."""
        spool = dc.spool([_at(i * 10 + 0.15 * i) for i in range(3)])
        with pytest.warns(DeprecationWarning, match="snap_coords"):
            given = spool.chunk(time=None, snap_coords=value)
        assert list(given) == list(spool.chunk(time=None))
