"""
Tests for reporting where a spool's data is missing.

Gaps are found with the rules chunk merges by, so the two must agree:
every gap is a boundary chunk refuses to close, and nothing else is.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import concat_coords, get_coord
from dascore.examples import random_spool
from dascore.exceptions import ChunkError, CoordError, ParameterError, UnitError
from dascore.utils.misc import suppress_warnings

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
    """Four patches which overlap by two samples, on one another's grid."""
    return random_spool(time_gap=-np.timedelta64(12, "ms"), length=4)


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

    @pytest.mark.parametrize(
        ("tolerance", "max_shift", "split"),
        [(None, None, True), (1.5, None, False), (None, 0.4, False)],
    )
    def test_agrees_with_chunk_on_a_narrow_seam(self, tolerance, max_shift, split):
        """A seam 1.3 steps wide is a gap by default, and where chunk splits."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        later = first.update_coords(time_min=time.max() + 1.3 * time.step)
        spool = dc.spool([first, later])
        kwargs = dict(tolerance=tolerance, max_shift=max_shift)
        merged = spool.chunk(time=None, **kwargs)
        gaps = spool.get_gaps(**kwargs)
        coverage = spool.get_coverage(**kwargs)
        assert len(merged) == len(gaps) + len(coverage)
        assert len(gaps) == int(split)
        assert coverage["missing"].sum() == 0
        if split:
            # the gap's ends are where the outputs end and start
            assert gaps["time_min"].iloc[0] == merged[0].get_coord("time").max()
            assert gaps["time_max"].iloc[0] == merged[1].get_coord("time").min()
            assert gaps["missing"].iloc[0] == 0

    def test_missing_counts_samples(self, gappy_spool):
        """A gap reports how many samples it is missing."""
        step = gappy_spool.get_contents()["time_step"].iloc[0]
        gaps = gappy_spool.get_gaps()
        assert (gaps["missing"] == ONE_SECOND // step).all()
        coverage = gappy_spool.get_coverage()
        assert coverage["missing"].sum() == gaps["missing"].sum()

    def test_overlap_off_grid_is_a_gap(self):
        """A patch half a step off its overlapped neighbour is where chunk splits."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        later = first.update_coords(time_min=time.max() - 1.5 * time.step)
        spool = dc.spool([first, later])
        merged = spool.chunk(time=None)
        gaps = spool.get_gaps()
        assert len(merged) == 2 == len(gaps) + 1
        assert gaps["time_max"].iloc[0] == merged[1].get_coord("time").min()

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
        # concatenate keeps each patch a segment, so the holes between them
        # are where chunk would split the one patch it made
        assert len(merged.get_gaps()) == len(gappy_spool) - 1
        assert merged.get_coverage()["coverage"].iloc[0] < 1
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
    """A spool holds a patch with holes whole; its index states the holes."""

    @pytest.mark.parametrize("count", [2, 256, 257, 1440])
    def test_holes_are_seen_through_runs(self, count):
        """Gaps, coverage and chunk see every hole of the one row."""
        spool = dc.spool([_time_runs_patch(count)])
        assert len(spool) == 1
        assert len(spool.get_gaps()) == count - 1
        assert spool.get_coverage()["coverage"].iloc[0] < 1
        assert len(spool.chunk(time=None)) == count

    def test_split_gaps_gives_views(self):
        """split_gaps cuts the pieces as views; a plain patch is itself."""
        patch = _time_runs_patch(3)
        pieces = list(patch.split_gaps().sort("time"))
        assert len(pieces) == 3
        assert all(np.shares_memory(x.data, patch.data) for x in pieces)
        data = np.concatenate([x.data for x in pieces], axis=1)
        assert np.array_equal(data, patch.data)
        plain = _time_runs_patch(1)
        assert dc.spool([plain])[0] is plain
        assert dc.spool([patch])[0] is patch

    def test_two_gapped_dims_give_the_product(self):
        """Holes in two dimensions split into one piece per pair of runs."""
        distance = concat_coords(
            get_coord(start=0, step=1, shape=(2,)),
            get_coord(start=10, step=1, shape=(1,)),
        )
        patch = _time_runs_patch(3).update_coords(distance=distance)
        spool = dc.spool([patch])
        assert len(spool) == 1
        assert len(spool.get_gaps("distance")) == 1
        assert len(patch.split_gaps()) == 6

    def test_read_selection_with_holes_is_one_patch(self, tmp_path):
        """A read picking labels which leave holes gives one segmented patch."""
        patch = dc.get_example_patch()
        path = dc.write(patch, tmp_path / "one.h5", "dasdae")
        values = patch.get_coord("distance").values
        (out,) = dc.read(path, distance=values[[1, 2, 5]])
        assert out.get_coord("distance").runs_count == 2
        assert len(dc.spool([patch.select(distance=values[[1, 2, 5]])])) == 1

    def test_identity_is_stable(self):
        """A gapped patch given again is the same entry."""
        patch = _time_runs_patch(3)
        assert len(dc.spool([patch, patch])) == 1
        spool = dc.spool([patch])
        spool._catalog.add(patch)
        assert len(spool) == len(spool + dc.spool(list(spool))) == 1


class TestChunkKeepsHoles:
    """Without tolerance, chunk never joins across a hole; with it, it refits."""

    def test_tolerance_refits_and_fill_value_fills(self):
        """A wide tolerance spreads labels over a hole; fill_value fills it."""
        spool = dc.spool([_time_runs_patch(2, samples=100)])
        assert len(spool.chunk(time=None)) == 2
        with pytest.warns(UserWarning, match="gap"):
            joined = spool.chunk(time=None, tolerance=100_000)
        assert len(joined) == 1
        assert joined[0].get_coord("time").evenly_sampled
        assert len(joined[0].get_coord("time")) == 200
        assert joined.get_gaps().empty
        with warnings.catch_warnings():  # a fill spreads no labels, so no warning
            warnings.simplefilter("error")
            filled = spool.fill_gaps("time")
        assert len(filled) == 1 and filled.get_gaps().empty
        assert filled[0].get_coord("time").evenly_sampled
        assert len(filled[0].get_coord("time")) > 200

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
        out = list(dc.spool([patch]).chunk(time=None))
        assert len(out) == 2
        assert out == list(patch.split_gaps())[::order]
        # a tolerance spanning the hole joins the runs onto one grid
        with pytest.warns(UserWarning, match="gap"):
            (joined,) = dc.spool([patch]).chunk(time=None, tolerance=10)
        coord = joined.get_coord("time")
        assert coord.evenly_sampled and len(coord) == len(time)
        assert {coord.min(), coord.max()} == {time.min(), time.max()}

    def test_overlapping_members(self):
        """Overlapping members join."""
        coords = {"distance": [0], "time": [0.0, 1, 2]}
        first = dc.Patch(data=np.zeros((1, 3)), coords=coords, dims=DIMS)
        overlapping = first.update_coords(time=np.array([1.0, 2, 3]))
        assert len(dc.spool([first, overlapping]).chunk(time=None)) == 1

    @pytest.mark.parametrize(
        ("step", "cuts", "count"),
        [
            ((1, 1024), (30, 30), 1),
            ((1, 1024), (30, 31), 2),
        ],
    )
    def test_exact_grids_join_on_their_lattice(self, step, cuts, count):
        """Exact grids join only where no position is missing."""
        t0 = np.datetime64("2020-01-01T00:00:00")
        full = get_coord(start=t0, step=step, shape=(60,))
        pieces = full[: cuts[0]], full[cuts[1] :]
        patches = [
            dc.Patch(data=np.zeros(len(x)), coords={"time": x}, dims=("time",))
            for x in pieces
        ]
        chunked = dc.spool(patches).chunk(time=None, max_shift=0)
        assert len(chunked) == count
        if count == 1:
            assert chunked[0].get_coord("time") == full

    @pytest.mark.parametrize("shift", [0.0, 0.3, 0.6])
    @pytest.mark.parametrize("seed", range(8))
    def test_rows_are_what_chunk_yields(self, seed, shift):
        """Outputs are one run per dim; the plan counts what it yields."""
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # gaps tolerance joins
            chunked = dc.spool([patch, later]).chunk(time=length, tolerance=tolerance)
            yielded = list(chunked)
        assert all(x.get_coord(d).runs_count == 1 for x in yielded for d in x.dims)
        assert len(chunked) == len(yielded) == len(dc.spool(yielded))


class TestSpoolFillGaps:
    """Spool.fill_gaps fills lazily, and chunk and cut join across what it filled."""

    def test_fills_gaps_up_to_the_size(self, gappy_spool):
        """Gaps up to the size join into one patch; wider ones stay apart."""
        assert len(gappy_spool.fill_gaps(time=0.5)) == len(gappy_spool)
        (filled,) = gappy_spool.fill_gaps(time=2)
        step = gappy_spool[0].get_coord("time").step
        assert filled.get_coord("time").step == step
        holes = np.isnan(filled.data).all(axis=filled.get_axis("distance") or 0)
        assert holes.sum() == 3 * (ONE_SECOND // step)

    def test_chunk_joins_across_the_fill(self, gappy_spool):
        """A later chunk cuts the filled stretch into whole windows."""
        chunked = gappy_spool.fill_gaps("time").chunk(time=4)
        expected = dc.spool([gappy_spool.fill_gaps("time")[0]]).chunk(time=4)
        assert [x.shape for x in chunked] == [x.shape for x in expected]

    def test_window_reads_only_the_members_it_reaches(self, gappy_spool):
        """Loading a window reads the files it reaches, not the whole stretch."""
        filled = gappy_spool.fill_gaps("time")
        resolver = filled._catalog.resolver
        reads = []
        original = resolver._load_member
        resolver._load_member = lambda *a, **kw: (
            reads.append(1),
            original(*a, **kw),
        )[1]
        start = gappy_spool[0].get_coord("time").min()
        window = start + np.array([[1, 2]]) * ONE_SECOND
        (out,) = filled.cut(time=window)
        assert len(reads) == 1 and not np.isnan(out.data).any()

    def test_chunk_plans_once_onto_the_files(self, gappy_spool):
        """A chunk of a fill is one plan, its windows loading from the files."""
        filled = gappy_spool.fill_gaps("time")
        chunked = filled.chunk(time=4)
        assert filled._catalog._built is None  # the fill's own was never built
        paths = chunked._catalog.resolver.member_rows["source_path"].astype(str)
        assert not paths.str.startswith("plan://").any()
        assert len(chunked) == len(filled.chunk_plan(time=4).outputs)

    @pytest.mark.parametrize("start", [0.0, 1e9])
    def test_float_grid_windows(self, start):
        """Windows of a float grid whose step no float holds keep its samples."""
        full = get_coord(start=start, step=0.1, shape=(60,))

        def make(part):
            coords = {"distance": [0], "x": full[part]}
            return dc.Patch(
                data=np.arange(60.0)[part][None], coords=coords, dims=("distance", "x")
            )

        filled = dc.spool([make(slice(0, 20)), make(slice(25, 60))]).fill_gaps("x")
        chunks = filled.chunk(x=1.0, keep_partial=True)
        data = np.concatenate([x.data[0] for x in chunks])
        labels = np.concatenate([x.get_coord("x").values for x in chunks])
        np.testing.assert_allclose(labels, full.values, rtol=0, atol=1e-6)
        assert np.isnan(data[20:25]).all() and np.array_equal(
            data[25:], np.arange(25.0, 60)
        )
        (cut,) = filled.cut(x=np.array([[full[18], full[27]]]))
        assert cut.shape == (1, 10) and np.isnan(cut.data[0, 2:7]).all()

    @pytest.mark.parametrize(
        ("start", "step", "window"),
        [
            (1e9, 0.1, 3.7),
            (1e9, 0.001, 0.01),
            (1.7e9, 1.0209523, 10.209523),
            (1e4, 0.001, 0.037),
            (1e6, 0.001, 0.01),
            (0.0, 0.25, 2.5),
        ],
    )
    def test_float_windows_match_filled_patch(self, start, step, window):
        """Windows of a fill hold, sample for sample, those of the filled patch."""
        full = get_coord(start=start, step=step, shape=(600,))

        def make(part):
            coords = {"x": full[part], "distance": np.arange(2.0)}
            data = np.arange(600.0)[part][:, None] * np.ones((1, 2))
            return dc.Patch(data=data, coords=coords, dims=("x", "distance"))

        parts = (slice(0, 200), slice(207, 400), slice(450, 600))
        filled = dc.spool([make(x) for x in parts]).fill_gaps("x")
        expected = dc.spool([filled[0]]).chunk(x=window, keep_partial=True)
        out = filled.chunk(x=window, keep_partial=True)
        assert len(out) == len(expected)
        for got, want in zip(out, expected, strict=True):
            np.testing.assert_array_equal(got.data, want.data)
            assert got.get_coord("x") == want.get_coord("x")

    def test_gap_window_takes_its_own_stretch(self):
        """A window of only fill describes, and loads from, its own stretch."""

        def make(start, count, station, channels=2):
            coords = {
                "time": get_coord(start=float(start), step=0.01, shape=(count,)),
                "distance": np.arange(channels) * 1.0,
            }
            data = np.zeros((count, channels))
            attrs = {"station": station}
            return dc.Patch(data=data, coords=coords, dims=tuple(coords), attrs=attrs)

        spool = dc.spool([make(0, 100, "A"), make(10.0, 50, "A"), make(0, 300, "B", 3)])
        out = spool.fill_gaps("time").chunk(time=1)
        rows = out.get_contents()
        for station, patch in zip(rows["station"], out, strict=True):
            assert patch.attrs.station == station
            assert patch.shape[1] == (2 if station == "A" else 3)

    @pytest.mark.parametrize("offset", [0.0, 0.3, -0.3, 0.45])
    @pytest.mark.parametrize("missing", [0, 1, 2, 3])
    @pytest.mark.parametrize(
        "size", [None, {"x": 1}, {"x": 2}, {"x": 2.4}, {"x": 2, "samples": True}]
    )
    def test_matches_patch_fill(self, offset, missing, size):
        """A spool fills each seam exactly where the joined patch fills it."""
        first = get_coord(start=0.0, step=1.0, shape=(10,))
        second = get_coord(start=10.0 + missing + offset, step=1.0, shape=(6,))

        def make(coord):
            data = np.ones(len(coord))
            return dc.Patch(data=data, coords={"x": coord}, dims=("x",))

        kwargs = {"x": None} if size is None else size
        spool = dc.spool([make(first), make(second)])
        joined = make(concat_coords(first, second))
        with suppress_warnings(UserWarning):
            want = joined.fill_gaps(**kwargs)
            got = spool.fill_gaps(**kwargs)
        got_coords = [x.get_coord("x") for x in got]
        assert len(got_coords) == len(want.get_coord("x").segments)
        if len(got_coords) == 1:
            assert got_coords[0] == want.get_coord("x")
            assert len(got.get_contents()) == 1

    @pytest.mark.parametrize("labels", [[20.0, 21.2, 29.9, 31.0], [20, 21.2, 22.9, 24]])
    def test_labels_beside_an_unfilled_gap_load(self, labels):
        """A patch of labels no fill reaches loads, alone or in windows."""

        def make(coord):
            coords = {"time": coord, "distance": np.arange(2.0)}
            return dc.Patch(
                data=np.ones((len(coord), 2)), coords=coords, dims=tuple(coords)
            )

        labels = get_coord(data=np.array(labels))
        grid = get_coord(start=0.0, step=1.0, shape=(10,))
        spool = dc.spool([make(grid), make(labels)])
        assert [x.shape for x in spool.fill_gaps(time=2)] == [(10, 2), (4, 2)]
        assert [x.shape for x in dc.spool([make(labels)]).fill_gaps("time")] == [(4, 2)]
        windows = spool.fill_gaps(time=2).chunk(time=5, keep_partial=True)
        assert sum(x.shape[0] for x in windows) == 14

    def test_negative_sample_count_raises(self, gappy_spool):
        """A count of missing samples is never negative."""
        with pytest.raises(ParameterError, match="non-negative"):
            gappy_spool.fill_gaps(time=-1, samples=True)

    def test_step_change_across_a_fill_raises(self):
        """Samples on another step across a filled gap raise, as on a patch."""
        coords = [
            get_coord(start=0.0, step=1.0, shape=(10,)),
            get_coord(start=12.0, step=1.5, shape=(6,)),
        ]
        patches = [
            dc.Patch(data=np.ones(len(x)), coords={"x": x}, dims=("x",)) for x in coords
        ]
        joined = dc.Patch(
            data=np.ones(16), coords={"x": concat_coords(*coords)}, dims=("x",)
        )
        with pytest.raises(CoordError, match="Resample or snap"):
            joined.fill_gaps("x")
        with pytest.raises(CoordError, match="Resample or snap"):
            len(dc.spool(patches).fill_gaps("x"))  # it plans as it is asked

    @pytest.mark.parametrize("kwargs", [{}, {"time": 1, "distance": 1}])
    def test_one_dimension(self, gappy_spool, kwargs):
        """Filling names exactly one dimension."""
        with pytest.raises(ParameterError, match="one dimension"):
            gappy_spool.fill_gaps(**kwargs)

    def test_samples_must_be_a_count(self, gappy_spool):
        """A sample size is a whole number."""
        with pytest.raises(ParameterError, match="non-negative integer"):
            gappy_spool.fill_gaps(time=2.5, samples=True)

    @pytest.mark.parametrize("method", ["chunk", "chunk_plan"])
    def test_chunk_refuses_fill_value(self, gappy_spool, method):
        """Chunk never fills, and names the method which does."""
        with pytest.raises(ParameterError, match=r"Spool\.fill_gaps"):
            getattr(gappy_spool, method)(time=None, tolerance=10, fill_value=np.nan)


class TestSpoolSnapCoords:
    """Spool.snap_coords snaps each patch alone, as it loads."""

    @pytest.fixture(scope="class")
    def wacky(self):
        """A spool of one patch whose coordinates are uneven."""
        return dc.spool([dc.get_example_patch("wacky_dim_coords_patch")])

    def test_snaps_each_patch(self, wacky):
        """Each patch comes back as Patch.snap_coords gives it, its step stated."""
        expected = wacky[0].snap_coords("time")
        out = wacky.snap_coords("time")
        assert out[0].get_coord("time") == expected.get_coord("time")
        assert (
            out.get_contents()["time_step"].iloc[0] == expected.get_coord("time").step
        )

    @pytest.mark.parametrize(
        "kwargs", [{}, {"reverse": True}, {"step": 1.0}, {"step": 0.5}]
    )
    def test_lazy_matches_eager(self, kwargs):
        """Rows, chunks and gaps of a lazy snap are those of the snapped patches."""

        def make(values, start=0.0):
            coord = get_coord(data=np.asarray(values) + start)
            coords = {"time": coord, "distance": np.arange(2.0)}
            data = np.arange(2 * len(coord)).reshape(-1, 2)
            return dc.Patch(data=data, coords=coords, dims=tuple(coords))

        patches = [
            make([0.0, 1.1, 1.9, 5.05, 6.4]),
            make(np.arange(10.0), start=20.0),
            make([6.4, 5.05, 1.9, 1.1, 0.0], start=40.0),
        ]
        spool = dc.spool(patches)
        with suppress_warnings(UserWarning):
            lazy = spool.snap_coords("time", **kwargs)
            eager = dc.spool([x.snap_coords("time", **kwargs) for x in patches])
        cols = ["time_min", "time_max", "time_step"]
        assert lazy.get_contents()[cols].equals(eager.get_contents()[cols])
        labels = lambda x: [y.get_coord("time").values.tolist() for y in x]  # noqa
        assert labels(lazy) == labels(eager)
        assert labels(lazy.chunk(time=None)) == labels(eager.chunk(time=None))
        assert len(lazy.get_gaps()) == len(eager.get_gaps())

    def test_unknown_dimension_raises(self, wacky):
        """A dimension no patch has is refused by name."""
        with pytest.raises(dc.exceptions.CoordError, match="bob"):
            wacky.snap_coords("bob")

    def test_chunk_keeps_the_snap(self, wacky):
        """A later chunk plans over the snapped patches, not their sources."""
        (out,) = wacky.snap_coords("time").chunk(time=None)
        assert out.get_coord("time").evenly_sampled

    def test_max_shift_raises_when_planned(self, wacky):
        """A move past max_shift is refused before anything loads."""
        with pytest.raises(dc.exceptions.CoordError, match="max_shift"):
            wacky.snap_coords("time", max_shift=0.01)

    def test_step(self, wacky):
        """step= places each patch's labels on that grid, as a patch's are."""
        time = np.sort(wacky[0].get_coord("time").values)
        step = np.diff(time).min()
        expected = wacky[0].snap_coords("time", step=step).get_coord("time")
        assert wacky.snap_coords("time", step=step)[0].get_coord("time") == expected


class TestConcatenatedRuns:
    """A concatenation's index row states each member's runs as segments."""

    def test_members_with_runs(self, gappy_spool):
        """Members holding gaps of their own keep them in the joined row."""
        holed = [x.unselect(time=(10, 20), samples=True) for x in gappy_spool]
        merged = dc.spool(holed).concatenate(time=None)
        assert len(merged.get_gaps()) == 2 * len(holed) - 1
        assert merged.get_gaps().equals(dc.spool(list(merged)).get_gaps())

    def test_concatenating_chunks_keeps_each_sample_once(self):
        """Members a plan trimmed load trimmed, so no sample repeats."""
        spool = dc.get_example_spool("random_das")
        (out,) = spool.chunk(time=3).concatenate(time=None)
        assert out.shape[out.get_axis("time")] == sum(
            x.shape[x.get_axis("time")] for x in spool
        )
