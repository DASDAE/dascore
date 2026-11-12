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
    """Chunk along time, returning the outputs and any warnings raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = list(spool.chunk(time=None, **kwargs))
    gaps = [x for x in caught if "gap" in str(x.message)]
    return out, gaps


class TestSeamTest:
    """Two patches join when the later starts one step after the earlier ends."""

    @pytest.mark.parametrize(
        ("width", "tolerance", "count", "warns"),
        [
            (1.0, None, 1, False),
            (1.1, None, 1, False),
            (1.3, None, 2, False),
            (4.0, None, 2, False),
            (1.0, 1.5, 1, False),
            (1.1, 1.5, 1, False),
            (1.3, 1.5, 1, False),
            (4.0, 1.5, 2, False),
            (1.0, 10, 1, False),
            (1.1, 10, 1, False),
            (1.3, 10, 1, False),
            (4.0, 10, 1, True),
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
        late = _patch(
            get_coord(
                start=early.get_coord("time").max() + STEP,
                step=1.04 * STEP,
                shape=(10,),
            )
        )
        assert len(dc.spool([early, late]).chunk(time=None)) == 1


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

    def test_zero_joins_split_fractional_grid(self):
        """A 1024 Hz grid split in two joins back to itself exactly."""
        full = get_coord(start=T0, step=Fraction(1, 1024), shape=(4001,))
        spool = dc.spool([_patch(full[:1500]), _patch(full[1500:])])
        (patch,) = spool.chunk(time=None, max_shift=0)
        assert patch.get_coord("time") == full


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
        """Files stating 30 Hz and 29.99999 Hz give one output at 30 Hz."""
        first = get_coord(start=T0, step=Fraction(1, 30), shape=(300,))
        rate = Fraction(2999999, 100000)
        start = first.max() + np.timedelta64(33_333_333, "ns")
        second = get_coord(start=start, step=1 / rate, shape=(300,))
        spool = dc.spool([_patch(second), _patch(first)])
        (patch,) = spool.chunk(time=None)
        assert patch.get_coord("time").step_exact == Fraction(1, 30)

    def test_fractional_rate_stays_exact(self):
        """Three 1024 Hz files keep the exact 1024 Hz step when merged."""
        full = get_coord(start=T0, step=Fraction(1, 1024), shape=(3 * 61441,))
        pieces = [full[i * 61441 : (i + 1) * 61441] for i in range(3)]
        (patch,) = dc.spool([_patch(x) for x in pieces]).chunk(time=None)
        coord = patch.get_coord("time")
        assert coord.step_exact == Fraction(1, 1024)
        assert coord == full

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
        coord = patch.get_coord("time")
        assert len(coord) == 19
        assert coord.step == STEP
        plan = spool.chunk_plan(time=None, tolerance=tolerance)
        assert plan.members["dropped"].tolist() == [0, 1]

    def test_overlap_off_grid(self):
        """A later patch half a step off the grid starts a new stretch."""
        spool = dc.spool([_at(0), _at(8.5)])
        one, two = spool.chunk(time=None)
        assert len(one.get_coord("time")) == 10
        second = two.get_coord("time")
        assert second.min() == T0 + 9.5 * STEP
        assert len(second) == 9

    def test_overlap_off_grid_tolerance(self):
        """Under tolerance=1.5 it continues from its sample 1.5 steps on."""
        spool = dc.spool([_at(0), _at(8.5)])
        (patch,) = spool.chunk(time=None, tolerance=1.5)
        coord = patch.get_coord("time")
        assert len(coord) == 18
        assert coord.evenly_sampled
        assert coord.max() == T0 + 17.5 * STEP

    def test_nested_patch_dropped(self):
        """A patch wholly inside an earlier one adds nothing."""
        spool = dc.spool([_at(0, 20), _at(5, 5)])
        (patch,) = spool.chunk(time=None)
        assert len(patch.get_coord("time")) == 20


@pytest.fixture(scope="module")
def late_spool():
    """Five slightly late files, so the stretch grid moves labels."""
    return dc.spool([_at(i * 10 + 0.15 * i) for i in range(5)])


class TestRejoin:
    """Chunk outputs are slices of one grid, so they rejoin exactly."""

    def test_lazy(self, late_spool):
        """Rechunking the chunks gives what chunking the spool gives."""
        whole = late_spool.chunk(time=None)[0].get_coord("time")
        rejoined = late_spool.chunk(time=2.5 * STEP).chunk(time=None)
        assert len(rejoined) == 1
        assert rejoined[0].get_coord("time") == whole

    def test_loaded(self, late_spool):
        """The same holds once the chunks are in memory."""
        whole = late_spool.chunk(time=None)[0].get_coord("time")
        loaded = dc.spool(list(late_spool.chunk(time=2.5 * STEP)))
        rejoined = loaded.chunk(time=None)
        assert len(rejoined) == 1
        assert rejoined[0].get_coord("time") == whole

    def test_windows_are_slices(self, late_spool):
        """Every window's labels are a slice of the stretch's labels."""
        whole = late_spool.chunk(time=None)[0].get_coord("time").values
        for patch in late_spool.chunk(time=2.5 * STEP):
            values = patch.get_coord("time").values
            start = np.searchsorted(whole, values[0])
            assert np.array_equal(values, whole[start : start + len(values)])

    def test_float_coordinates(self):
        """Float labels rejoin bit for bit too."""

        def patch(start):
            return _patch(get_coord(start=start, step=0.1, shape=(10,)))

        spool = dc.spool([patch(0.0), patch(1.015), patch(2.03)])
        whole = spool.chunk(time=None)[0].get_coord("time")
        rejoined = dc.spool(list(spool.chunk(time=0.25))).chunk(time=None)
        assert rejoined[0].get_coord("time") == whole


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

    def test_false_is_default(self):
        """snap_coords=False gives what the default gives."""
        spool = dc.spool([_at(i * 10 + 0.15 * i) for i in range(3)])
        with pytest.warns(DeprecationWarning):
            exact = spool.chunk(time=None, snap_coords=False)
        assert list(exact) == list(spool.chunk(time=None))
