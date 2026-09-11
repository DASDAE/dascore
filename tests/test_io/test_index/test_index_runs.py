"""Tests for schema 17: segmented coordinates linked to their runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coords import (
    _MAX_SUMMARY_RUNS,
    CoordSegmented,
    concat_coords,
    get_coord,
)
from dascore.core.summary import PatchSummary
from dascore.io.index.backend import get_backend
from dascore.io.index.ingest import patch_record, summaries_to_records
from dascore.io.index.query import Query
from dascore.io.index.schema import PatchCoordRow

MS = np.timedelta64(1, "ms")
HOLE = pd.Timedelta(12, "ms")


@pytest.fixture(scope="module")
def gapped_patch():
    """The example patch with a 12 ms hole in its time coordinate."""
    patch = dc.get_example_patch()
    t0 = patch.get_coord("time").min()
    first = patch.select(time=(None, t0 + 1000 * MS))
    second = patch.select(time=(t0 + 1012 * MS, None))
    (out,) = dc.spool([first, second]).chunk(time=None, tolerance=5, snap_coords=False)
    assert isinstance(out.get_coord("time"), CoordSegmented)
    return out


@pytest.fixture(scope="module")
def gapped_directory(gapped_patch, tmp_path_factory):
    """A directory spool holding the gapped patch and a contiguous one after it."""
    path = tmp_path_factory.mktemp("runs")
    dc.write(gapped_patch, path / "gapped.h5", "dasdae")
    later = gapped_patch.get_coord("time").max() + 10 * 1000 * MS
    whole = dc.get_example_patch().update_coords(time_min=later)
    dc.write(whole, path / "whole.h5", "dasdae")
    return dc.spool(path).update()


@pytest.fixture(scope="module")
def crowded(gapped_patch):
    """The gapped patch among four contiguous ones, and its patch id."""
    later = gapped_patch.get_coord("time").max() + 10 * 1000 * MS
    others = [
        dc.get_example_patch().update_coords(time_min=later + i * 10_000 * MS)
        for i in range(4)
    ]
    spool = dc.spool([gapped_patch, *others])
    back = spool._catalog.backend
    (gapped_id,) = back._fetch_df(
        "SELECT DISTINCT patch_id FROM patch_coords WHERE run_index > 0"
    )["patch_id"]
    return back, int(gapped_id)


class TestSummaries:
    """A segmented coordinate's summary carries its runs."""

    def test_runs_in_order(self, gapped_patch):
        """Each run is summarized, first to last."""
        coord = gapped_patch.get_coord("time")
        summary = coord.to_summary(dims=("time",))
        assert len(summary.runs) == coord.segment_count
        assert summary.runs[0].min == coord.min()
        assert summary.runs[-1].max == coord.max()
        assert all(run.step == coord.segments[0].step for run in summary.runs)

    def test_other_coordinates_have_none(self):
        """Only segmented coordinates carry runs."""
        assert get_coord(start=0, stop=5, step=1).to_summary().runs is None
        assert get_coord(data=[1.0, 2.5, 7.0]).to_summary().runs is None

    @pytest.mark.parametrize("count", [_MAX_SUMMARY_RUNS, _MAX_SUMMARY_RUNS + 1])
    def test_run_cap(self, count):
        """Up to the cap every run is summarized; past it, none is."""
        runs = [get_coord(start=20.0 * i, step=1.0, shape=(10,)) for i in range(count)]
        coord = concat_coords(*runs)
        assert isinstance(coord, CoordSegmented)
        summary = coord.to_summary()
        expected = count if count <= _MAX_SUMMARY_RUNS else 0
        assert len(summary.runs or ()) == expected

    def test_flat_dump_omits_runs(self, gapped_patch):
        """Runs are structure, never a flat column."""
        flat = PatchSummary.from_patch(gapped_patch).flat_dump()
        assert "time_runs" not in flat

    def test_repr_omits_runs(self, gapped_patch):
        """A summary prints its envelope, not every run."""
        assert "runs" not in repr(gapped_patch.get_coord("time").to_summary())


class TestStorage:
    """The index links a segmented coordinate to each run."""

    def test_records(self, gapped_patch):
        """The whole coordinate is run 0; its runs follow, numbered from 1."""
        record = patch_record(PatchSummary.from_patch(gapped_patch))
        time = [c for c in record.coords if c.coord_name == "time"]
        assert [c.run_index for c in time] == [0, 1, 2]
        assert time[0].step_int is None
        assert time[1].step_int == time[2].step_int == 4_000_000
        assert {c.run_index for c in record.coords if c.coord_name == "distance"} == {0}

    def test_links(self, gapped_directory):
        """The link table holds the runs beside the coordinate."""
        back = gapped_directory._catalog.backend
        links = back._fetch_df("SELECT * FROM patch_coords")
        assert list(links.columns) == list(PatchCoordRow._fields)
        runs = links[links["run_index"] > 0]
        assert set(runs["coord_name"]) == {"time"}
        assert len(runs) == 2

    def test_flat_relation_is_one_row_per_patch(self, gapped_directory):
        """Every query about a patch's coordinate reads the coordinate whole."""
        contents = gapped_directory.get_contents()
        assert len(contents) == 2
        gapped = contents.sort_values("time_min").iloc[0]
        assert pd.isnull(gapped["time_step"])

    def test_export_keeps_runs(self, gapped_patch, tmp_path):
        """Records exported for a merge carry the run links."""
        summary = PatchSummary.from_patch(gapped_patch).model_copy(
            update={"source_path": "a.h5", "source_format": "DASDAE"}
        )
        back = get_backend(tmp_path / "runs.sqlite3")
        back.write_sources(summaries_to_records([summary]))
        (source,) = back.export_records()
        back.close()
        runs = [c.run_index for c in source.patches[0].coords if c.coord_name == "time"]
        assert runs == [0, 1, 2]


class TestLookups:
    """The backend finds runs by patch."""

    def test_few_ids_filter_in_sql(self, crowded):
        """Under a quarter of the patches, the ids go to the query."""
        back, gapped_id = crowded
        runs = back.coord_runs("time", [gapped_id])
        assert runs["run_index"].tolist() == [1, 2]
        assert back.coord_runs("time", [gapped_id + 1]).empty

    def test_patch_runs(self, crowded):
        """Runs come back as records, by patch."""
        back, gapped_id = crowded
        found = back.patch_runs([gapped_id, gapped_id + 1])
        assert list(found) == [gapped_id]
        assert [r.run_index for r in found[gapped_id]] == [1, 2]

    def test_patch_runs_without_any(self):
        """An index without runs answers empty."""
        back = dc.spool([dc.get_example_patch()])._catalog.backend
        assert back.patch_runs([1]) == {}


class TestReports:
    """Gap reports see holes inside a patch."""

    def test_directory_gaps(self, gapped_directory):
        """The hole inside the gapped patch and the space after it are both gaps."""
        gaps = gapped_directory.get_gaps().sort_values("time_min")
        assert len(gaps) == 2
        assert gaps["gap_size"].iloc[0] == HOLE

    def test_memory_gaps(self, gapped_patch):
        """An in-memory spool of the gapped patch reports its hole, once."""
        spool = dc.spool([gapped_patch])
        assert spool.get_gaps()["gap_size"].tolist() == [HOLE]
        (coverage,) = spool.get_coverage().to_dict("records")
        assert coverage["gap_total"] == HOLE
        assert coverage["covered"] == coverage["span"] - HOLE

    def test_selection_clips_runs(self, gapped_directory, gapped_patch):
        """A view trimmed past the hole reports no hole."""
        t0 = gapped_patch.get_coord("time").min()
        view = gapped_directory.select(time=(t0 + 1500 * MS, None))
        assert len(view.get_gaps()) == 1  # only the space between the files

    def test_selection_within_one_run(self, gapped_patch):
        """A view inside the first run keeps one row and no gap."""
        t0 = gapped_patch.get_coord("time").min()
        view = dc.spool([gapped_patch]).select(time=(t0, t0 + 500 * MS))
        assert view.get_gaps().empty
        assert len(view.get_coverage()) == 1

    def test_selection_inside_the_hole(self, gapped_patch):
        """A view holding no sample of the patch reports nothing of it."""
        t0 = gapped_patch.get_coord("time").min()
        view = dc.spool([gapped_patch]).select(time=(t0 + 1003 * MS, t0 + 1008 * MS))
        assert view.get_coverage().empty
        assert view.get_gaps().empty

    def test_other_dimension_unaffected(self, gapped_patch):
        """Runs of time do not touch the distance report."""
        assert dc.spool([gapped_patch]).get_gaps("distance").empty

    def test_runs_of_differing_steps_open_no_gap(self):
        """A run without the step of its neighbours does not read as a hole."""
        coord = concat_coords(
            get_coord(start=0.0, step=1.0, shape=(10,)),
            get_coord(data=np.array([10.0, 10.5, 12.0])),
            get_coord(start=13.0, step=1.0, shape=(10,)),
        )
        assert isinstance(coord, CoordSegmented)
        patch = dc.Patch(
            data=np.zeros((len(coord), 3)),
            coords={"distance": coord, "x": np.arange(3)},
            dims=("distance", "x"),
        )
        assert dc.spool([patch]).get_gaps("distance").empty

    def test_relative_runs_among_absolute_times(self, gapped_patch):
        """A segmented relative-time patch sits out an absolute report."""
        time = gapped_patch.get_coord("time")
        runs = [
            get_coord(
                start=x.min() - time.min(), step=x.step, shape=(len(x),), units=x.units
            )
            for x in time.segments
        ]
        relative = gapped_patch.update_coords(time=concat_coords(*runs))
        assert isinstance(relative.get_coord("time"), CoordSegmented)
        spool = dc.spool([gapped_patch, relative])
        assert spool.get_gaps()["gap_size"].tolist() == [HOLE]
        assert spool.get_coverage()["gap_total"].tolist() == [HOLE]

    def test_loose_tolerance_merges_across_runs(self, gapped_directory):
        """A tolerance wider than every gap merges the runs and the patches."""
        merged = gapped_directory.chunk(time=None, tolerance=10_000)
        assert len(merged) == 1

    def test_query_candidacy_unchanged(self, gapped_directory, gapped_patch):
        """A window inside the hole still selects the patch holding it."""
        t0 = gapped_patch.get_coord("time").min()
        window = (t0 + 1003 * MS, t0 + 1008 * MS)
        back = gapped_directory._catalog.backend
        assert len(back.query([Query(coords={"time": window})])) == 1


class TestDerived:
    """Plans carry their members' runs."""

    def test_chunked_view_reports_the_hole(self, gapped_patch):
        """A hole that chunk splits at is still a gap between its outputs."""
        chunked = dc.spool([gapped_patch]).chunk(time=None)
        assert chunked.get_gaps()["gap_size"].tolist() == [HOLE]
        assert chunked.get_coverage()["gap_total"].tolist() == [HOLE]

    def test_directory_chunked_view_keeps_the_hole(self, gapped_directory):
        """The same through a directory spool's plan."""
        chunked = gapped_directory.chunk(time=None)
        assert len(chunked.get_gaps()) == len(gapped_directory.get_gaps())

    def test_windows_report_what_they_hold(self, gapped_patch):
        """Windows end at the hole; a dropped partial window widens the gap.

        The first run's last sample begins a half-second window of its own,
        which ``keep_partial=False`` drops, so the gap spans one more step.
        """
        chunked = dc.spool([gapped_patch]).chunk(time=0.5)
        assert chunked.get_gaps()["gap_size"].tolist() == [HOLE + 4 * MS]

    def test_chunked_selection_keeps_the_hole(self, gapped_patch):
        """A view trimmed into the first run plans only what it holds."""
        t0 = gapped_patch.get_coord("time").min()
        view = dc.spool([gapped_patch]).select(time=(t0 + 500 * MS, None))
        chunked = view.chunk(time=None)
        first = chunked[0].get_coord("time")
        assert (first.min(), first.max()) == (t0 + 500 * MS, t0 + 1000 * MS)
        assert chunked.get_gaps()["gap_size"].tolist() == [HOLE]

    def test_slices_of_one_patch(self, gapped_patch):
        """Two members sharing one time coordinate each keep their own runs."""
        distance = gapped_patch.get_coord("distance")
        mid = distance.values[len(distance) // 2]
        slices = [
            gapped_patch.select(distance=(None, mid)),
            gapped_patch.select(distance=(mid + distance.step, None)),
        ]
        chunked = dc.spool(slices).chunk(time=None)
        assert chunked.get_gaps()["gap_size"].tolist() == [HOLE, HOLE]

    def test_merged_members_keep_shared_runs(self, gapped_patch):
        """Merged members pass on the runs of a coordinate they all share."""
        distance = gapped_patch.get_coord("distance")
        mid = distance.values[len(distance) // 2]
        slices = [
            gapped_patch.select(distance=(None, mid)),
            gapped_patch.select(distance=(mid + distance.step, None)),
        ]
        merged = dc.spool(slices).chunk(distance=None)
        assert len(merged) == 1
        assert merged.get_gaps()["gap_size"].tolist() == [HOLE]
        assert merged.get_gaps("distance").empty

    def test_rechunk_lends_no_runs(self, gapped_patch):
        """A re-chunk along the same dimension never lends one patch's runs.

        Its members are the first plan's, whose ids are not the parent
        index's, so no output takes runs and no patch drops out of the
        report.
        """
        later = dc.get_example_patch().update_coords(
            time_min=gapped_patch.get_coord("time").max() + 10_000 * MS
        )
        once = dc.spool([gapped_patch, later]).chunk(time=None)
        twice = once.chunk(time=None)
        assert once.get_gaps()["gap_size"].tolist() == [HOLE, pd.Timedelta(10, "s")]
        links = twice._catalog.backend._fetch_df("SELECT run_index FROM patch_coords")
        assert (links["run_index"] == 0).all()
        coverage = twice.get_coverage()
        assert coverage["time_max"].max() == later.get_coord("time").max()

    def test_concatenated_members_state_no_runs(self, gapped_patch):
        """An output joined along a dimension takes no member's runs of it."""
        later = gapped_patch.update_coords(
            time_min=gapped_patch.get_coord("time").max() + 4 * MS
        )
        joined = dc.spool([gapped_patch, later]).concatenate(time=None)
        (coverage,) = joined.get_coverage().to_dict("records")
        assert coverage["gap_total"] == pd.Timedelta(0)
        assert coverage["time_max"] == later.get_coord("time").max()

    def test_runs_in_other_units_are_dropped(self):
        """A run the plan restated in other units falls back to its envelope."""
        time = get_coord(start=0.0, step=1.0, shape=(4,), units="s")
        metres = get_coord(start=0.0, step=1.0, shape=(10,), units="m")
        centimetres = concat_coords(
            get_coord(start=2000.0, step=100.0, shape=(10,), units="cm"),
            get_coord(start=4000.0, step=100.0, shape=(10,), units="cm"),
        )
        patches = [
            dc.Patch(
                data=np.zeros((len(coord), 4)),
                coords={"distance": coord, "time": time},
                dims=("distance", "time"),
            )
            for coord in (metres, centimetres)
        ]
        coverage = dc.spool(patches).chunk(distance=None).get_coverage("distance")
        assert coverage["distance_max"].max() == 49.0


class TestChunkPlansRuns:
    """Chunk treats a hole inside a patch as a gap between patches."""

    @pytest.fixture(scope="class")
    def halves(self, gapped_patch):
        """The gapped patch's two runs, as selections."""
        t0 = gapped_patch.get_coord("time").min()
        first = gapped_patch.select(time=(None, t0 + 1000 * MS))
        second = gapped_patch.select(time=(t0 + 1012 * MS, None))
        return first, second

    def test_hole_ends_an_output(self, gapped_patch, halves):
        """Each run becomes an output holding exactly its samples."""
        chunked = dc.spool([gapped_patch]).chunk(time=None)
        assert len(chunked) == 2
        for out, half in zip(chunked, halves, strict=True):
            assert out.get_coord("time") == half.get_coord("time")
            np.testing.assert_array_equal(out.data, half.data)

    @pytest.mark.parametrize("snap_coords", [True, False])
    def test_tolerance_bridges_the_hole(self, gapped_patch, snap_coords):
        """Runs bridged into one output are the patch as stored, hole and all."""
        chunked = dc.spool([gapped_patch]).chunk(
            time=None, tolerance=5, snap_coords=snap_coords
        )
        (merged,) = chunked
        assert merged.equals(gapped_patch)
        assert chunked.get_gaps()["gap_size"].tolist() == [HOLE]

    def test_run_joins_a_contiguous_neighbour(self, gapped_patch, halves):
        """The run next to a patch merges with it, as patches do."""
        time = gapped_patch.get_coord("time")
        later = dc.get_example_patch().update_coords(time_min=time.max() + time.step)
        first, second = dc.spool([gapped_patch, later]).chunk(time=None)
        assert first.get_coord("time") == halves[0].get_coord("time")
        span = second.get_coord("time")
        assert (span.min(), span.max()) == (
            halves[1].get_coord("time").min(),
            later.get_coord("time").max(),
        )

    def test_windows_never_straddle_the_hole(self, gapped_patch):
        """No window holds samples from both sides of the hole."""
        chunked = dc.spool([gapped_patch]).chunk(time=1)
        assert len(chunked) == 7
        assert not any(isinstance(p.get_coord("time"), CoordSegmented) for p in chunked)

    def test_plan_members_are_runs(self, gapped_patch, halves):
        """Each run is a trimmed member; runs in one output are read once."""
        spool = dc.spool([gapped_patch])
        members = spool.chunk_plan(time=None).members
        assert members["_modified"].all()
        for (_, row), half in zip(members.iterrows(), halves, strict=True):
            coord = half.get_coord("time")
            assert (row["time_min"], row["time_max"]) == (coord.min(), coord.max())
        (bridged,) = spool.chunk_plan(time=None, tolerance=5).members.to_dict("records")
        time = gapped_patch.get_coord("time")
        assert (bridged["time_min"], bridged["time_max"]) == (time.min(), time.max())

    def test_directory_loads_runs(self, gapped_directory, halves):
        """Members read from files load as selections of their patch."""
        chunked = gapped_directory.chunk(time=None)
        assert len(chunked) == 3
        for out, half in zip(list(chunked)[:2], halves, strict=True):
            np.testing.assert_array_equal(out.data, half.data)

    def test_reports_agree_with_chunk(self, gapped_directory):
        """Every reported gap starts where a chunked output ends."""
        gaps = gapped_directory.get_gaps()
        ends = [p.get_coord("time").max() for p in gapped_directory.chunk(time=None)]
        assert len(ends) == len(gaps) + 1
        assert list(gaps["time_min"].sort_values()) == [
            pd.Timestamp(x) for x in ends[:-1]
        ]

    def test_runs_past_the_cap_plan_whole(self):
        """A coordinate with more runs than the index stores is planned whole."""
        runs = [
            get_coord(start=20.0 * i, step=1.0, shape=(10,), units="m")
            for i in range(_MAX_SUMMARY_RUNS + 1)
        ]
        coord = concat_coords(*runs)
        patch = dc.Patch(
            data=np.zeros((len(coord), 2)),
            coords={"distance": coord, "x": np.arange(2)},
            dims=("distance", "x"),
        )
        assert len(dc.spool([patch]).chunk(distance=None)) == 1

    def test_sample_selection_plans_whole(self, gapped_patch):
        """A sample selection resolves on the whole patch, so runs stay together."""
        view = dc.spool([gapped_patch]).select(time=(-10, None), samples=True)
        (out,) = view.chunk(time=None)
        assert out.get_coord("time").shape == (10,)

    def test_time_after_distance(self, gapped_patch):
        """Chunking time after distance still splits at the hole."""
        chunked = dc.spool([gapped_patch]).chunk(distance=None).chunk(time=None)
        assert len(chunked) == 2
        assert chunked.get_gaps()["gap_size"].tolist() == [HOLE]

    def test_chained_chunks_keep_neighbour_samples(self):
        """A run merged with a neighbour lends it no runs to be trimmed by."""
        x = np.arange(3)
        distance = concat_coords(
            get_coord(start=0.0, step=1.0, shape=(5,), units="m"),
            get_coord(start=10.0, step=1.0, shape=(5,), units="m"),
        )
        gapped = dc.Patch(
            data=np.zeros((10, 3)),
            coords={"distance": distance, "x": x},
            dims=("distance", "x"),
        )
        neighbour = dc.Patch(
            data=np.ones((5, 3)),
            coords={
                "distance": get_coord(start=15.0, step=1.0, shape=(5,), units="m"),
                "x": x,
            },
            dims=("distance", "x"),
        )
        spool = dc.spool([gapped, neighbour])
        chained = spool.chunk(distance=None).chunk(x=None).chunk(distance=None)
        ends = [p.get_coord("distance").max() for p in chained]
        assert ends == [4.0, 19.0]
