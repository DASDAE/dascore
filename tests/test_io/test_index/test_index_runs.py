"""Tests for the index's coordinate table and runs (rule 12 of the coordinate note)."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coords import CoordSummary, concat_coords, get_coord
from dascore.core.summary import PatchSummary
from dascore.io.dasdae.utils import _save_patch
from dascore.io.index.backend import get_backend
from dascore.io.index.ingest import (
    index_runs,
    patch_record,
    stored_runs,
    summaries_to_records,
)
from tests.conftest import join_patches

MS = np.timedelta64(1, "ms")
HOLE = pd.Timedelta(12, "ms")


@pytest.fixture(scope="module")
def gapped_patch():
    """The example patch with a 12 ms hole in its time coordinate."""
    patch = dc.get_example_patch()
    t0 = patch.get_coord("time").min()
    first = patch.select(time=(None, t0 + 1000 * MS))
    second = patch.select(time=(t0 + 1012 * MS, None))
    out = join_patches([first, second])
    assert out.get_coord("time").gapped
    return out


def _records(patch, path="a.h5"):
    """The source records of a patch as a file would give them."""
    summary = PatchSummary.from_patch(patch).model_copy(
        update={"source_path": path, "source_format": "DASDAE"}
    )
    return summaries_to_records([summary])


def _time_record(patch):
    """The time coordinate's record of a patch."""
    record = patch_record(PatchSummary.from_patch(patch))
    (time,) = [x for x in record.coords if x.coord_name == "time"]
    return time


class TestCoordinateRecords:
    """A coordinate record states the step, flags, offsets and runs."""

    def test_gapped(self, gapped_patch):
        """A gapped coordinate keeps its step and states its two runs."""
        time = _time_record(gapped_patch)
        assert time.step_int == 4_000_000 and not time.is_exact
        assert time.gapped and not time.jittered
        assert time.run_count == len(time.runs) == 2
        assert time.offset_min == time.offset_max == 0.0

    def test_jittered(self):
        """A jittered coordinate keeps its step and its offset bounds."""
        coord = get_coord(data=[0.0, 1.1, 1.95, 3.02], step=1.0)
        patch = dc.Patch(data=np.zeros(4), coords={"time": coord}, dims=("time",))
        time = _time_record(patch)
        assert time.step_float == 1.0 and time.jittered and not time.gapped
        assert time.offset_min == pytest.approx(-0.05)
        assert time.offset_max == pytest.approx(0.1)
        assert time.run_count == 1 and time.runs == ()

    def test_contiguous_rows_store_no_runs(self):
        """A coordinate of one run is its own row."""
        time = _time_record(dc.get_example_patch())
        assert time.is_exact and time.run_count == 1 and time.runs == ()

    @pytest.mark.parametrize(
        "coord",
        [
            concat_coords(
                get_coord(start=0.0, step=1.0, shape=(3,)),
                get_coord(start=3.0, step=1.5, shape=(3,)),
            ),
            get_coord(data=np.array([9, 8, 7, 4, 3, 1, 0]), step=-1),
            concat_coords(
                get_coord(start=np.datetime64("2020-01-01"), step=MS, shape=(3,)),
                get_coord(start=np.datetime64("2020-01-02"), step=MS, shape=(3,)),
            ),
            get_coord(data=np.array([0, 2, 3, 7]) * MS, step=MS),
        ],
    )
    def test_runs_round_trip(self, coord):
        """Stored runs read back as the records the coordinate states."""
        records = coord._run_records()
        kind = "time" if coord.dtype.kind in "mM" else "num"
        absolute = coord.dtype.kind != "m"
        stored = stored_runs(records, kind, absolute)
        assert index_runs(stored, kind, absolute) == records


class TestStoredRuns:
    """The coord_runs table holds a definition's runs and exports them."""

    def test_rows(self, gapped_patch):
        """The runs table holds the gapped coordinate's runs, in order."""
        backend = get_backend(":memory:")
        backend.write_sources(_records(gapped_patch))
        runs = backend._fetch_df("SELECT * FROM coord_runs ORDER BY run")
        assert runs["run"].tolist() == [0, 1]
        assert runs["position"].tolist() == [0, 253]
        defs = backend._fetch_df("SELECT * FROM coord_defs WHERE run_count = 2")
        assert defs["gapped"].tolist() == [1]

    def test_export_keeps_runs(self, gapped_patch):
        """Records exported for a union carry the runs and flags."""
        backend = get_backend(":memory:")
        backend.write_sources(_records(gapped_patch))
        (source,) = backend.export_records()
        (time,) = [x for x in source.patches[0].coords if x.coord_name == "time"]
        assert time == _time_record(gapped_patch)

    def test_relation_states_range_steps(self, gapped_patch):
        """The flat contents keep stating the step of a range only."""
        backend = get_backend(":memory:")
        backend.write_sources(_records(gapped_patch))
        contents = backend.query([])
        assert contents["time_step"].isna().all()
        assert contents["distance_step"].notna().all()


def _runs_patch(count, samples=10):
    """A patch whose time coordinate holds ``count`` runs a minute apart."""
    t0 = np.datetime64("2020-01-01T00:00:00")
    runs = [
        get_coord(start=t0 + 60_000 * i * MS, step=MS, shape=(samples,))
        for i in range(count)
    ]
    data = np.zeros((2, count * samples))
    coords = {"distance": np.arange(2), "time": concat_coords(*runs)}
    return dc.Patch(data=data, coords=coords, dims=("distance", "time"))


class TestRunCap:
    """Past max_index_runs a coordinate is stored as its envelope, with a warning."""

    def test_envelope_past_the_cap(self):
        """The file and the fix are named, and no run is stored."""
        patch = _runs_patch(257)
        with pytest.warns(UserWarning, match=r"a\.h5.*257 runs.*max_index_runs"):
            records = _records(patch)
        (time,) = [x for x in records[0].patches[0].coords if x.coord_name == "time"]
        assert time.run_count == 257 and time.runs == () and time.gapped

    def test_cap_is_configurable(self):
        """Raising the cap stores every run, silently."""
        with dc.config_context(max_index_runs=300):
            records = _records(_runs_patch(257))
        (time,) = [x for x in records[0].patches[0].coords if x.coord_name == "time"]
        assert len(time.runs) == 257


def _write_whole(patch, path):
    """Write a patch to a DASDAE file holes and all, which dc.write splits."""
    dc.write(patch.split_gaps()[0], path, "dasdae")
    with h5py.File(path, "a") as h5:
        del h5["waveforms"][next(iter(h5["waveforms"]))]
        _save_patch(patch, h5["waveforms"], "whole", compact=True)
    return path


@pytest.fixture(scope="module")
def gapped_directory(gapped_patch, tmp_path_factory):
    """A directory holding the gapped patch whole and a contiguous one after it."""
    path = tmp_path_factory.mktemp("runs")
    _write_whole(gapped_patch, path / "gapped.h5")
    later = gapped_patch.get_coord("time").max() + 10 * 1000 * MS
    whole = dc.get_example_patch().update_coords(time_min=later)
    dc.write(whole, path / "whole.h5", "dasdae")
    return dc.spool(path).update()


class TestReports:
    """Gap reports see holes inside a patch, from memory or from a directory."""

    def test_directory_gaps(self, gapped_directory):
        """The hole inside the gapped patch and the space after it are both gaps."""
        assert len(gapped_directory) == 2
        gaps = gapped_directory.get_gaps().sort_values("time_min")
        assert len(gaps) == 2
        assert gaps["gap_size"].iloc[0] == HOLE

    def test_memory_gaps(self, gapped_patch):
        """An in-memory spool of the gapped patch reports its hole, once."""
        spool = dc.spool([gapped_patch])
        assert len(spool) == 1
        assert spool.get_gaps()["gap_size"].tolist() == [HOLE]
        (coverage,) = spool.get_coverage().to_dict("records")
        assert coverage["gap_total"] == HOLE
        assert coverage["covered"] == coverage["span"] - HOLE

    def test_memory_and_directory_agree(self, gapped_directory):
        """A memory spool of a directory's patches reports what it does."""
        memory = dc.spool(list(gapped_directory))
        columns = ["time_min", "time_max", "time_step", "gap_size"]
        for name in ("get_gaps", "get_coverage"):
            disk, mem = (getattr(x, name)() for x in (gapped_directory, memory))
            shared = [x for x in disk.columns if x in columns or x == "coverage"]
            pd.testing.assert_frame_equal(disk[shared], mem[shared])
        contents = [x.get_contents() for x in (gapped_directory, memory)]
        shared = ["time_min", "time_max", "time_step", "data_size"]
        pd.testing.assert_frame_equal(contents[0][shared], contents[1][shared])

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


class TestPlans:
    """Chunk plans cut a patch at its holes and read each piece once."""

    def test_members_are_runs(self, gapped_directory):
        """Each run is a trimmed member; runs in one output are read once."""
        members = gapped_directory.chunk_plan(time=None).members
        assert len(members) == 3 and members["_modified"].sum() == 2
        bridged = gapped_directory.chunk_plan(time=None, tolerance=5, fill_value=np.nan)
        assert len(bridged.members) == 2

    def test_chunk_outputs(self, gapped_directory):
        """Chunk splits at the hole, and its outputs report it as a gap."""
        chunked = gapped_directory.chunk(time=None)
        assert len(chunked) == 3
        assert not any(x.get_coord("time").gapped for x in chunked)
        assert chunked.get_gaps().sort_values("time_min")["gap_size"].iloc[0] == HOLE
        joined = gapped_directory.chunk(time=None, tolerance=10_000, fill_value=np.nan)
        assert joined.get_gaps().empty

    def test_other_dimension_keeps_runs(self, gapped_patch):
        """Chunking along distance keeps each output's holes along time."""
        chunked = dc.spool([gapped_patch]).chunk(distance=100)
        assert len(chunked.get_gaps()) == len(chunked)


@pytest.fixture(scope="module")
def crowded(tmp_path_factory):
    """A directory whose one file holds a coordinate of 300 runs."""
    path = tmp_path_factory.mktemp("crowded")
    _write_whole(_runs_patch(300), path / "crowded.h5")
    with pytest.warns(UserWarning, match="300 runs"):
        return dc.spool(path).update()


class TestRunsPastTheCap:
    """A coordinate stored as its envelope is read when its runs are needed."""

    def test_reports_read_the_file(self, crowded):
        """Reports and plans see every hole the index did not store."""
        backend = crowded._catalog.backend
        assert backend._fetch_df("SELECT * FROM coord_runs").empty
        assert len(crowded.get_gaps()) == 299
        assert len(crowded.chunk(time=None)) == 300


class TestSelections:
    """A range selection sees a patch's holes, as its patch does."""

    def test_range_inside_a_hole(self, gapped_patch, gapped_directory):
        """A range holding no sample of a patch selects no row of it."""
        t0 = gapped_patch.get_coord("time").min()
        window = (t0 + 1003 * MS, t0 + 1008 * MS)
        for spool in (dc.spool([gapped_patch]), gapped_directory):
            view = spool.select(time=window)
            assert len(view) == len(view.get_contents()) == len(list(view)) == 0

    def test_range_ending_in_a_hole(self, gapped_patch):
        """A row's envelope ends at the last sample the range reaches."""
        t0 = gapped_patch.get_coord("time").min()
        view = dc.spool([gapped_patch]).select(time=(t0 + 500 * MS, t0 + 1005 * MS))
        (row,) = view.get_contents().to_dict("records")
        (patch,) = view
        assert row["time_min"] == patch.get_coord("time").min()
        assert row["time_max"] == patch.get_coord("time").max() == t0 + 1000 * MS


class TestReviewFindings:
    """Runs and decodes survive what the review found losing them."""

    def test_sample_selection_reports_the_loaded_runs(self):
        """A sample selection's report reads the runs the patch loads with."""
        time = get_coord(data=np.array([0, 1, 2, 6, 7, 8]), step=1)
        patch = dc.Patch(data=np.zeros(6), coords={"time": time}, dims=("time",))
        view = dc.spool([patch]).select(time=(3, 6), samples=True)
        (loaded,) = view
        assert not loaded.get_coord("time").gapped
        assert view.get_gaps().empty

    def test_replanning_keeps_runs_of_another_dim(self, gapped_patch):
        """A collapsed re-plan still carries the time runs of its members."""
        once = dc.spool([gapped_patch]).chunk(distance=100)
        twice = once.chunk(distance=100)
        assert len(twice.get_gaps()) == len(once.get_gaps()) == len(once)

    def test_cap_applies_at_ingest(self):
        """A summary taken under a larger cap still stores no more runs."""
        patch = _runs_patch(10)
        with dc.config_context(max_index_runs=20):
            assert len(patch.summary.coords["time"].runs) == 10
        with dc.config_context(max_index_runs=5):
            spool = dc.spool([patch])
            assert spool._catalog.backend._fetch_df("SELECT * FROM coord_runs").empty
            assert len(spool.get_gaps()) == 9

    def test_runs_survive_a_summary_dump(self, gapped_patch):
        """A summary rebuilt from its dump, even as text, states the same runs."""
        summary = gapped_patch.get_coord("time").to_summary()
        dumped = summary.model_dump()
        assert CoordSummary(**dumped).runs == summary.runs
        as_text = [{k: str(v) for k, v in x.items()} for x in dumped["runs"]]
        assert CoordSummary(**{**dumped, "runs": as_text}).runs == summary.runs

    def test_merged_outputs_state_no_parent_runs(self, gapped_directory):
        """Outputs along the planned dimension hold no hole, and say so."""
        chunked = gapped_directory.chunk(time=None)
        backend = chunked._catalog.backend
        assert backend._fetch_df("SELECT * FROM coord_runs").empty
        assert backend._fetch_df("SELECT * FROM coord_defs WHERE gapped = 1").empty
