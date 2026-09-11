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
from dascore.io.index.schema import INDEX_VERSION, PatchCoordRow

MS = np.timedelta64(1, "ms")


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

    def test_too_many_runs_is_an_envelope(self):
        """Past the cap the summary is the envelope alone."""
        runs = [
            get_coord(start=20.0 * i, step=1.0, shape=(10,))
            for i in range(_MAX_SUMMARY_RUNS + 1)
        ]
        coord = concat_coords(*runs)
        assert isinstance(coord, CoordSegmented)
        assert coord.segment_count > _MAX_SUMMARY_RUNS
        assert coord.to_summary().runs is None

    def test_flat_dump_omits_runs(self, gapped_patch):
        """Runs are structure, never a flat column."""
        flat = PatchSummary.from_patch(gapped_patch).flat_dump()
        assert "time_runs" not in flat


class TestStorage:
    """The index links a segmented coordinate to each run."""

    def test_version(self):
        """Run links arrived with schema 17."""
        assert INDEX_VERSION == 17

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

    def test_ordering_reads_the_whole_coordinate(self, gapped_directory):
        """Ordering by a coordinate is by its whole minimum, once per patch."""
        back = gapped_directory._catalog.backend
        ids = back.query_ids(None, order_by=("coord", "time", False))
        assert len(ids) == 2

    def test_coord_runs_empty_without_segments(self, tmp_path):
        """An archive of contiguous patches states no runs."""
        spool = dc.spool([dc.get_example_patch()])
        back = spool._catalog.backend
        assert back.coord_runs("time", [1, 2, 3]).empty


class TestReports:
    """Gap reports see holes inside a patch."""

    def test_directory_gaps(self, gapped_directory, gapped_patch):
        """The hole inside the gapped patch and the space after it are both gaps."""
        gaps = gapped_directory.get_gaps().sort_values("time_min")
        assert len(gaps) == 2
        hole = gaps.iloc[0]
        assert hole["gap_size"] == pd_timedelta(12)

    def test_memory_gaps(self, gapped_patch):
        """An in-memory spool of the gapped patch reports its hole."""
        gaps = dc.spool([gapped_patch]).get_gaps()
        assert len(gaps) == 1
        assert gaps["gap_size"].iloc[0] == pd_timedelta(12)

    def test_coverage(self, gapped_patch):
        """Coverage counts the hole as missing."""
        (coverage,) = dc.spool([gapped_patch]).get_coverage()["coverage"]
        assert coverage < 1

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

    def test_other_dimension_unaffected(self, gapped_patch):
        """Runs of time do not touch the distance report."""
        assert dc.spool([gapped_patch]).get_gaps("distance").empty

    def test_chunk_plan_unchanged(self, gapped_directory):
        """Chunking plans whole patches, as before.

        The gapped patch states no step in the relation chunk reads, so it
        plans in a sampling group of its own and never merges with its
        contiguous neighbour, however loose the tolerance.
        """
        merged = gapped_directory.chunk(time=None, tolerance=10_000)
        assert len(merged) == 2

    def test_query_candidacy_unchanged(self, gapped_directory, gapped_patch):
        """A window inside the hole still selects the patch holding it."""
        t0 = gapped_patch.get_coord("time").min()
        window = (t0 + 1003 * MS, t0 + 1008 * MS)
        back = gapped_directory._catalog.backend
        assert len(back.query([Query(coords={"time": window})])) == 1


def pd_timedelta(milliseconds: int):
    """A gap size of this many milliseconds, as the report states it."""
    return pd.Timedelta(milliseconds, "ms")


class TestReviewFindings:
    """Cases the counterpart review found."""

    def test_relative_runs_among_absolute_times(self, gapped_patch):
        """A segmented relative-time patch among absolute ones is skipped."""
        time = gapped_patch.get_coord("time")
        relative = gapped_patch.update_coords(time=time.values - time.min())
        spool = dc.spool([gapped_patch, relative])
        assert len(spool.get_gaps()) >= 1
        assert len(spool.get_coverage()) >= 1

    def test_chunked_view_keeps_the_hole(self, gapped_patch):
        """A whole member keeps its identity, and so its runs, through a plan."""
        chunked = dc.spool([gapped_patch]).chunk(time=None)
        assert isinstance(chunked[0].get_coord("time"), CoordSegmented)
        assert len(chunked.get_gaps()) == 1
        assert chunked.get_coverage()["coverage"].iloc[0] < 1

    def test_directory_chunked_view_keeps_the_hole(self, gapped_directory):
        """The same through a directory spool's plan."""
        chunked = gapped_directory.chunk(time=None)
        assert len(chunked.get_gaps()) == len(gapped_directory.get_gaps())

    def test_selected_ids_filter_in_sql(self, gapped_directory):
        """Asking for a few patches reads only their runs."""
        back = gapped_directory._catalog.backend
        ids = sorted(back._fetch_df("SELECT patch_id FROM patches")["patch_id"])
        with_runs = back._fetch_df(
            "SELECT DISTINCT patch_id FROM patch_coords WHERE run_index > 0"
        )["patch_id"].tolist()
        without = [x for x in ids if x not in with_runs]
        # fewer than a quarter of the patches takes the SQL path
        many = [*without, *range(10_000, 10_020)]
        assert back.coord_runs("time", many).empty
        assert len(back.coord_runs("time", with_runs)) == 2

    def test_run_records_by_key(self, gapped_directory):
        """Runs are found by the whole coordinate's def key."""
        back = gapped_directory._catalog.backend
        keys = back._fetch_df(
            "SELECT cd.def_key FROM patch_coords pc JOIN coord_defs cd "
            "ON cd.coord_def_id = pc.coord_def_id "
            "WHERE pc.coord_name = 'time' AND pc.run_index = 0"
        )["def_key"].tolist()
        found = back.run_records("time", keys)
        assert len(found) == 1
        (runs,) = found.values()
        assert [r.run_index for r in runs] == [1, 2]
        assert back.run_records("time", []) == {}
