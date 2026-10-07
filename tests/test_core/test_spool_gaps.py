"""
Tests for reporting where a spool's data is missing.

Gaps are found with the rules chunk merges by, so the two must agree:
every gap is a boundary chunk refuses to close, and nothing else is.
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import concat_coords, get_coord
from dascore.examples import random_spool
from dascore.exceptions import ChunkError, ParameterError, UnitError

ONE_SECOND = np.timedelta64(1, "s")
ONE_NS = np.timedelta64(1, "ns")
ONE_MS = 1_000_000 * ONE_NS
T0 = np.datetime64("2020-01-01T00:00:00.000000000")
DIMS = ("distance", "time")


@pytest.fixture(scope="module")
def gappy_spool():
    """Four patches separated by one second holes."""
    return random_spool(time_gap=ONE_SECOND, length=4)


@pytest.fixture(scope="module")
def overlapping_spool():
    """Four patches which overlap by 10 ms."""
    return random_spool(time_gap=-np.timedelta64(10, "ms"), length=4)


@pytest.fixture(scope="module")
def distance_tiled_spool():
    """Two patches adjacent in distance, with a hole between them."""
    patch = dc.get_example_patch()
    step = patch.get_coord("distance").step
    shifted = patch.update_coords(
        distance_min=patch.get_coord("distance").max() + step * 20
    )
    return dc.spool([patch, shifted])


class TestGetGaps:
    """Spool.get_gaps."""

    def test_contiguous_spool_has_none(self):
        """A spool with no holes reports no gaps."""
        out = random_spool().get_gaps()
        assert out.empty
        assert {"time_min", "time_max", "time_step", "gap_size"} <= set(out.columns)

    def test_gap_per_hole(self, gappy_spool):
        """Each hole is one row, sized from the samples bracketing it."""
        out = gappy_spool.get_gaps()
        assert len(out) == len(gappy_spool) - 1
        # the bracketing convention makes gap_size one step wider than
        # the extent actually missing
        missing = out["gap_size"] - out["time_step"]
        assert (missing == ONE_SECOND).all()

    def test_bracketing_samples_are_real(self, gappy_spool):
        """The reported bounds are samples the spool actually holds."""
        contents = gappy_spool.get_contents()
        out = gappy_spool.get_gaps()
        assert set(out["time_min"]).issubset(set(contents["time_max"]))
        assert set(out["time_max"]).issubset(set(contents["time_min"]))

    def test_overlaps_are_not_gaps(self, overlapping_spool):
        """Overlapping patches leave nothing missing."""
        assert overlapping_spool.get_gaps().empty

    def test_agrees_with_chunk(self, gappy_spool, overlapping_spool):
        """Merging leaves one patch per group, plus one per gap it can't close."""
        for spool in (gappy_spool, overlapping_spool):
            merged = spool.chunk(time=...)
            assert len(merged) == len(spool.get_gaps()) + len(spool.get_coverage())

    def test_tolerance_closes_gap(self, gappy_spool):
        """A tolerance wide enough to span the hole reports no gap."""
        step = gappy_spool.get_contents()["time_step"].iloc[0]
        samples = ONE_SECOND / step
        assert gappy_spool.get_gaps(tolerance=samples + 2).empty

    def test_groups_never_bridge(self):
        """Two unrelated groups are not a gap in each other."""
        early = random_spool(tag="early")
        late = random_spool(time_min=np.datetime64("2030-01-01"), tag="late")
        spool = dc.spool([*early, *late])
        # collapsing the groups puts a decade-wide hole between them ...
        assert len(spool.get_gaps(group=[])) == 1
        # ... which the tags keep apart
        assert spool.get_gaps().empty
        assert len(spool.get_coverage()) == 2

    def test_other_dimension(self, distance_tiled_spool):
        """The dimension really is a parameter, not just time."""
        assert distance_tiled_spool.get_gaps().empty
        out = distance_tiled_spool.get_gaps("distance")
        assert len(out) == 1
        assert out["distance_max"].iloc[0] > out["distance_min"].iloc[0]

    def test_respects_select(self, gappy_spool):
        """Gaps are reported for the spool as currently selected."""
        contents = gappy_spool.get_contents()
        trimmed = gappy_spool.select(
            time=(None, contents["time_max"].iloc[1]),
        )
        assert len(trimmed.get_gaps()) == 1

    def test_on_missing_dim_dropped(self, spool_with_non_coords):
        """Patches without the dimension are excluded, the rest reported."""
        contents = spool_with_non_coords.get_contents()
        assert contents["time_min"].isna().any(), "fixture has no dimensionless patch"
        out = spool_with_non_coords.get_coverage()
        # the patches which do have time are still measured ...
        assert len(out) == 1
        assert out["time_max"].iloc[0] == contents["time_max"].max()
        assert spool_with_non_coords.get_gaps().empty
        # ... and the dropped ones can be made an error instead
        with pytest.raises(ChunkError, match="on_missing_dim='drop'"):
            spool_with_non_coords.get_gaps(on_missing_dim="raise")
        with pytest.raises(ChunkError, match="on_missing_dim='drop'"):
            spool_with_non_coords.get_coverage(on_missing_dim="raise")

    def test_plan_backed_spool(self, gappy_spool):
        """A report describes the patches the spool holds, not their sources."""
        merged = gappy_spool.concatenate(time=None)
        assert len(merged) == 1
        # concatenate ignores the coordinate values, so the holes are
        # inside the one patch it made and no boundary is left to report
        assert merged.get_gaps().empty
        assert merged.get_coverage()["coverage"].iloc[0] == 1
        # and the report agrees with rebuilding the spool from its patches
        assert merged.get_gaps().equals(dc.spool(list(merged)).get_gaps())

    def test_chunk_keeps_the_gaps_it_cannot_close(self, gappy_spool):
        """Merging does not close a real hole, so the report still sees it."""
        chunked = gappy_spool.chunk(time=...)
        assert len(chunked) == len(gappy_spool)
        assert len(chunked.get_gaps()) == len(gappy_spool.get_gaps())

    def test_samples_selection_is_measured(self, gappy_spool):
        """A samples window trims the envelopes the report reads."""
        trimmed = gappy_spool.select(time=(0, 200), samples=True)
        out = trimmed.get_gaps()
        assert len(out) == len(gappy_spool.get_gaps())
        # the trim shortened each patch, so the holes are wider than the
        # untrimmed spool's
        assert (out["gap_size"] > gappy_spool.get_gaps()["gap_size"]).all()

    def test_group_colliding_with_emitted_column(self, gappy_spool):
        """Grouping by a column the report emits is refused."""
        with pytest.raises(ParameterError, match="collide"):
            gappy_spool.get_gaps(group="time_step")

    def test_units_are_presented(self, gappy_spool):
        """The report says what unit its magnitudes are in."""
        out = gappy_spool.get_gaps()
        assert "time_units" in out.columns
        assert "_time_units" not in out.columns

    @pytest.mark.parametrize("method", ["get_gaps", "get_coverage"])
    def test_reports_are_public(self, gappy_spool, method):
        """Neither report hands back the index's own bookkeeping."""
        for spool in (gappy_spool, dc.spool([])):
            out = getattr(spool, method)()
            assert not [x for x in out.columns if str(x).startswith("_")]

    def test_irregular_numbers_report_none(self):
        """Numeric labels with no step state none, and report no gap."""
        time = np.array([0.0, 1.0, 2.5, 7.0, 8.0])
        patch = dc.Patch(data=np.zeros(5), coords={"time": time}, dims=("time",))
        assert dc.spool([patch]).get_gaps().empty

    def test_unknown_dim_raises(self, gappy_spool):
        """An unknown dimension names the ones which exist."""
        with pytest.raises(ParameterError, match="Cannot report on"):
            gappy_spool.get_gaps("not_a_dim")


class TestQuantityTolerance:
    """Reports whose tolerance is stated in the coordinate's own units."""

    def test_gaps_respect_absolute_tolerance(self, gappy_spool):
        """A tolerance wider than the holes leaves nothing to report."""
        assert len(gappy_spool.get_gaps()) == len(gappy_spool) - 1
        assert gappy_spool.get_gaps(tolerance=dc.get_quantity("2 s")).empty
        wide = gappy_spool.get_gaps(tolerance=dc.get_quantity("0.5 s"))
        assert len(wide) == len(gappy_spool) - 1

    def test_timedelta_says_the_same(self, gappy_spool):
        """A timedelta reports what the equivalent quantity reports."""
        delta = gappy_spool.get_gaps(tolerance=2 * ONE_SECOND)
        assert delta.empty
        tight = gappy_spool.get_gaps(tolerance=ONE_SECOND / 2)
        assert len(tight) == len(gappy_spool) - 1

    def test_coverage_respects_absolute_tolerance(self, gappy_spool):
        """Holes the tolerance closes are not missing coverage."""
        loose = gappy_spool.get_coverage(tolerance=dc.get_quantity("2 s"))
        assert (loose["coverage"] == 1).all()
        assert (gappy_spool.get_coverage()["coverage"] < 1).all()

    def test_distance_units_convert(self, distance_tiled_spool):
        """A distance report reads the tolerance in its own units, not as metres."""
        gaps = distance_tiled_spool.get_gaps("distance", tolerance=10 * dc.units.m)
        assert len(gaps) == 1
        hole = float(gaps["gap_size"].iloc[0])
        # a foot is 0.3048 m, so a magnitude between the hole in feet and
        # the hole in metres reports one way under each reading
        between = (hole + float(gaps["distance_step"].iloc[0])) * 2
        assert 0.3048 * between < hole < between
        feet = distance_tiled_spool.get_gaps(
            "distance", tolerance=between * dc.units.ft
        )
        assert len(feet) == 1
        metres = distance_tiled_spool.get_gaps(
            "distance", tolerance=between * dc.units.m
        )
        assert metres.empty

    def test_wrong_dimensionality_raises(self, gappy_spool):
        """A tolerance must measure the dimension it is applied to."""
        with pytest.raises(UnitError, match="must have units of time"):
            gappy_spool.get_gaps(tolerance=10 * dc.units.m)


class TestGetCoverage:
    """Spool.get_coverage."""

    def test_contiguous_is_complete(self):
        """A spool with no holes is fully covered."""
        out = random_spool().get_coverage()
        assert len(out) == 1
        assert out["coverage"].iloc[0] == 1

    def test_incomplete_with_gaps(self, gappy_spool):
        """Holes lower the coverage below one."""
        assert (gappy_spool.get_coverage()["coverage"] < 1).all()

    def test_totals_match_gap_frame(self, gappy_spool):
        """gap_total is the sum of the rows get_gaps reports."""
        out = gappy_spool.get_coverage()
        assert out["gap_total"].iloc[0] == gappy_spool.get_gaps()["gap_size"].sum()
        assert out["covered"].iloc[0] == out["span"].iloc[0] - out["gap_total"].iloc[0]

    def test_span_matches_contents(self, gappy_spool):
        """The span reaches from the first sample to the last."""
        contents = gappy_spool.get_contents()
        out = gappy_spool.get_coverage()
        assert out["time_min"].iloc[0] == contents["time_min"].min()
        assert out["time_max"].iloc[0] == contents["time_max"].max()

    def test_row_per_group(self, diverse_spool):
        """Each group gets its own row, keyed by group_id."""
        out = diverse_spool.get_coverage()
        assert len(out) > 1
        assert out["group_id"].is_unique
        # every gap belongs to a group the coverage frame names
        assert set(diverse_spool.get_gaps()["group_id"]) <= set(out["group_id"])

    def test_group_id_joins_the_frames(self, diverse_spool):
        """gap_total is the sum of that group's own gap rows."""
        coverage = diverse_spool.get_coverage().set_index("group_id")
        summed = diverse_spool.get_gaps().groupby("group_id")["gap_size"].sum()
        for group_id, total in summed.items():
            assert coverage.loc[group_id, "gap_total"] == total

    def test_cells_are_distinguishable(self):
        """Two cells alike in every shown attribute still get separate ids."""
        patch = dc.get_example_patch()
        moved = patch.update_coords(
            distance_min=patch.get_coord("distance").max() + 1000
        )
        out = dc.spool([patch, moved]).get_coverage()
        assert len(out) == 2
        assert out["group_id"].is_unique

    def test_empty_spool(self):
        """An empty spool reports the schema a populated one would."""
        empty = dc.spool([])
        assert empty.get_gaps().empty
        out = empty.get_coverage()
        assert out.empty
        expected = {"time_min", "time_max", "span", "gap_total", "covered", "coverage"}
        assert expected <= set(out.columns)
        # the measured columns keep their dtypes, so summing an empty
        # report works. String attrs come from the catalog's own empty
        # relation, which is where their dtype is decided.
        populated = random_spool().get_coverage()
        measured = ["time_min", "time_max", "span", "gap_total", "covered", "coverage"]
        assert out[measured].dtypes.to_dict() == populated[measured].dtypes.to_dict()


def _time_runs_patch(count, samples=10):
    """A patch whose time coordinate holds ``count`` runs a minute apart."""
    runs = [
        get_coord(start=T0 + 60 * i * ONE_SECOND, step=ONE_MS, shape=(samples,))
        for i in range(count)
    ]
    data = np.arange(3.0 * count * samples).reshape(3, -1)
    coords = {"distance": np.arange(3), "time": concat_coords(*runs)}
    return dc.Patch(data=data, coords=coords, dims=DIMS)


class TestGappedPatchRows:
    """A spool row never holds a hole: a gapped patch enters as its pieces."""

    @pytest.mark.parametrize("count", [2, 256, 257, 1440])
    def test_one_row_per_run(self, count):
        """Every run is a row, so gaps, coverage and chunk see every hole."""
        spool = dc.spool([_time_runs_patch(count)])
        assert len(spool) == count
        assert len(spool.get_gaps()) == count - 1
        assert spool.get_coverage()["coverage"].iloc[0] < 1
        assert len(spool.chunk(time=None)) == count

    def test_pieces_are_views(self):
        """The pieces share the gapped patch's data; a plain patch is itself."""
        patch = _time_runs_patch(3)
        pieces = list(dc.spool([patch]).sort("time"))
        assert len(pieces) == 3
        assert all(np.shares_memory(x.data, patch.data) for x in pieces)
        data = np.concatenate([x.data for x in pieces], axis=1)
        assert np.array_equal(data, patch.data)
        plain = _time_runs_patch(1)
        assert dc.spool([plain])[0] is plain

    def test_two_gapped_dims_give_the_product(self):
        """Holes in two dimensions give one piece per pair of runs."""
        distance = concat_coords(
            get_coord(start=0, step=1, shape=(2,)),
            get_coord(start=10, step=1, shape=(1,)),
        )
        patch = _time_runs_patch(3).update_coords(distance=distance)
        assert len(dc.spool([patch])) == 6
        assert len(patch.split_gaps()) == 6

    def test_identity_is_stable(self):
        """A gapped patch, or one of its pieces, given again is the same entry."""
        patch = _time_runs_patch(3)
        assert len(dc.spool([patch, patch])) == 3
        spool = dc.spool([patch])
        spool._catalog.add(patch)
        spool._catalog.add(spool[0])
        assert len(spool) == len(spool + dc.spool(list(spool))) == 3


class TestChunkKeepsHoles:
    """Without fill_value, chunk never yields an output holding a hole."""

    def test_only_fill_value_bridges(self):
        """A wide tolerance leaves a hole (#1217); fill_value closes it (#1216)."""
        spool = dc.spool([_time_runs_patch(2, samples=100)])
        kept = spool.chunk(time=None, tolerance=100_000)
        assert len(kept) == 2
        assert len(kept.get_gaps()) == len(dc.spool(list(kept)).get_gaps()) == 1
        filled = spool.chunk(time=None, tolerance=100_000, fill_value=np.nan)
        assert len(filled) == 1 and filled.get_gaps().empty
        assert filled[0].get_coord("time").evenly_sampled

    @pytest.mark.parametrize(
        ("runs", "order"),
        [
            ([(5.0, -1.0, 3), (0.0, -1.0, 2)], -1),
            ([(T0, ONE_MS, 3), (T0 + 6_370_000 * ONE_NS, ONE_MS, 3)], 1),
        ],
    )
    def test_pieces_untouched(self, runs, order):
        """Descending runs, or runs off each other's lattice, come back as given."""
        runs = [get_coord(start=x, step=y, shape=(n,)) for x, y, n in runs]
        time = concat_coords(*runs)
        data = np.arange(float(len(time)))[None]
        patch = dc.Patch(data=data, coords={"distance": [0], "time": time}, dims=DIMS)
        out = list(dc.spool([patch]).chunk(time=None, tolerance=10))
        assert len(out) == 2
        assert out == list(dc.spool([patch]))[::order]

    def test_overlapping_or_stepless_members_join(self):
        """Overlapping members, or labels with no step, name no lattice to break."""
        coords = {"distance": [0], "time": [0.0, 1, 2]}
        first = dc.Patch(data=np.zeros((1, 3)), coords=coords, dims=DIMS)
        overlapping = first.update_coords(time=np.array([1.0, 2, 3]))
        times = (T0 + np.array([0, 3, 4]) * ONE_MS, T0 + np.array([5, 9, 10]) * ONE_MS)
        uneven = [first.update_coords(time=x) for x in times]
        for patches in ([first, overlapping], uneven):
            assert len(dc.spool(patches).chunk(time=None, snap_coords=False)) == 1

    @pytest.mark.parametrize(
        ("step", "cuts", "count"),
        [
            ((1, 1024), (30, 30), 1),
            ((1, 1024), (30, 31), 2),
        ],
    )
    def test_exact_grids_join_on_their_lattice(self, step, cuts, count):
        """Without snapping, exact grids join only where no position is missing."""
        t0 = np.datetime64("2020-01-01T00:00:00")
        full = get_coord(start=t0, step=step, shape=(60,))
        pieces = full[: cuts[0]], full[cuts[1] :]
        patches = [
            dc.Patch(data=np.zeros(len(x)), coords={"time": x}, dims=("time",))
            for x in pieces
        ]
        chunked = dc.spool(patches).chunk(time=None, snap_coords=False)
        assert len(chunked) == count

    @pytest.mark.parametrize("snap", [True, False])
    @pytest.mark.parametrize("shift", [0.0, 0.3, 0.6])
    @pytest.mark.parametrize("seed", range(8))
    def test_rows_are_what_chunk_yields(self, seed, shift, snap):
        """Snapped outputs are one run per dim; the plan counts what it yields."""
        rng = np.random.default_rng(seed)
        count = int(rng.integers(1, 5))
        gaps = rng.integers(1, 8, size=count)
        starts = np.cumsum(np.r_[0, 20 + gaps[:-1]])
        pieces = [get_coord(start=float(x), step=1.0, shape=(20,)) for x in starts]
        coords = {"distance": [0], "time": concat_coords(*pieces)}
        data = rng.random((1, 20 * count))
        patch = dc.Patch(data=data, coords=coords, dims=DIMS)
        # a member shifted a fraction of a step lies on another lattice
        later = patch.update_coords(time_min=float(starts[-1]) + 20 + shift)
        length = [None, 5.0, 13.0][int(rng.integers(3))]
        tolerance = float(rng.choice([1.5, 10.0, 100.0]))
        chunked = dc.spool([patch, later]).chunk(
            time=length, tolerance=tolerance, snap_coords=snap
        )
        yielded = list(chunked)
        if snap:
            assert all(x.get_coord(d).runs_count == 1 for x in yielded for d in x.dims)
        assert len(chunked) == len(yielded) == len(dc.spool(yielded))
