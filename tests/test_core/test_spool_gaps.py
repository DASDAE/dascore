"""
Tests for reporting where a spool's data is missing.

Gaps are found with the rules chunk merges by, so the two must agree:
every gap is a boundary chunk refuses to close, and nothing else is.
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.examples import random_spool
from dascore.exceptions import ChunkError, ParameterError, UnitError

ONE_SECOND = np.timedelta64(1, "s")


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

    def test_missing_dim_dropped(self, spool_with_non_coords):
        """Patches without the dimension are excluded, the rest reported."""
        contents = spool_with_non_coords.get_contents()
        assert contents["time_min"].isna().any(), "fixture has no dimensionless patch"
        out = spool_with_non_coords.get_coverage()
        # the patches which do have time are still measured ...
        assert len(out) == 1
        assert out["time_max"].iloc[0] == contents["time_max"].max()
        assert spool_with_non_coords.get_gaps().empty
        # ... and the dropped ones can be made an error instead
        with pytest.raises(ChunkError, match="lack the dimension"):
            spool_with_non_coords.get_gaps(missing_dim="raise")

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

    def test_unknown_dim_raises(self, gappy_spool):
        """An unknown dimension names the ones which exist."""
        with pytest.raises(ParameterError, match="Cannot report on"):
            gappy_spool.get_gaps("not_a_dim")


class TestQuantityTolerance:
    """Reports whose tolerance is a distance rather than a sample count."""

    def test_gaps_respect_absolute_tolerance(self, gappy_spool):
        """A tolerance wider than the holes leaves nothing to report."""
        assert len(gappy_spool.get_gaps()) == len(gappy_spool) - 1
        assert gappy_spool.get_gaps(tolerance=dc.get_quantity("2 s")).empty
        wide = gappy_spool.get_gaps(tolerance=dc.get_quantity("0.5 s"))
        assert len(wide) == len(gappy_spool) - 1

    def test_coverage_respects_absolute_tolerance(self, gappy_spool):
        """Holes the tolerance closes are not missing coverage."""
        loose = gappy_spool.get_coverage(tolerance=dc.get_quantity("2 s"))
        assert (loose["coverage"] == 1).all()
        assert (gappy_spool.get_coverage()["coverage"] < 1).all()

    def test_distance_units_convert(self, distance_tiled_spool):
        """A distance report reads the tolerance in its own units."""
        gaps = distance_tiled_spool.get_gaps("distance", tolerance=10 * dc.units.m)
        assert len(gaps) == 1
        hole = float(gaps["gap_size"].iloc[0])
        feet = distance_tiled_spool.get_gaps(
            "distance", tolerance=(hole * 2) / 0.3048 * dc.units.ft
        )
        assert feet.empty

    def test_wrong_dimensionality_raises(self, gappy_spool):
        """A tolerance must measure the dimension it is applied to."""
        with pytest.raises(UnitError, match="time-like"):
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
