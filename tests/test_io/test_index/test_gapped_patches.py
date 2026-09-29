"""Tests for spools of patches whose coordinates hold holes."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coords import concat_coords, get_coord
from dascore.core.summary import PatchSummary
from dascore.io.dasdae.utils import _save_patch
from dascore.io.index.backend import get_backend
from dascore.io.index.ingest import summaries_to_records
from dascore.io.index.query import Query
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
    assert out.get_coord("time").runs_count > 1
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
def indexed_runs(gapped_patch, tmp_path_factory):
    """A directory whose file holds the gapped patch whole, written around dc.write."""
    path = tmp_path_factory.mktemp("indexed_runs") / "gapped.h5"
    later = gapped_patch.get_coord("time").max() + 10 * 1000 * MS
    dc.write(dc.get_example_patch().update_coords(time_min=later), path, "dasdae")
    with h5py.File(path, "a") as h5:
        _save_patch(gapped_patch, h5["waveforms"], "gapped", compact=True)
    return dc.spool(path.parent).update()


class TestIndexedRuns:
    """A file-backed patch holding a hole is indexed by its envelope."""

    def test_reports_see_the_envelope(self, indexed_runs):
        """The hole is not a gap, and the patch plans as one member."""
        contents = indexed_runs.get_contents().sort_values("time_min")
        assert len(contents) == 2
        assert pd.isnull(contents["time_step"].iloc[0])
        assert indexed_runs.get_gaps().empty
        members = indexed_runs.chunk_plan(time=None).members
        assert len(members) == 2 and not members["_modified"].any()

    def test_query_candidacy_unchanged(self, indexed_runs, gapped_patch):
        """A window inside the hole still selects the patch holding it."""
        t0 = gapped_patch.get_coord("time").min()
        window = (t0 + 1003 * MS, t0 + 1008 * MS)
        back = indexed_runs._catalog.backend
        assert len(back.query([Query(coords={"time": window})])) == 1

    def test_export_skips_stored_run_links(self, gapped_patch, tmp_path):
        """Run links an earlier index holds are not exported as coordinates."""
        summary = PatchSummary.from_patch(gapped_patch).model_copy(
            update={"source_path": "a.h5", "source_format": "DASDAE"}
        )
        back = get_backend(tmp_path / "runs.sqlite3")
        back.write_sources(summaries_to_records([summary]))
        back._execute(
            "INSERT INTO patch_coords SELECT patch_row, coord_name, 1, "
            "coord_dims, coord_row, dtype FROM patch_coords WHERE coord_name = 'time'"
        )
        (source,) = back.export_records()
        back.close()
        names = [c.coord_name for c in source.patches[0].coords]
        assert names.count("time") == 1


class TestReports:
    """Gap reports see holes inside a patch."""

    def test_directory_gaps(self, gapped_directory):
        """The hole inside the gapped patch and the space after it are both gaps."""
        gaps = gapped_directory.get_gaps().sort_values("time_min")
        assert len(gaps) == 2
        assert gaps["gap_size"].iloc[0] == HOLE

    def test_written_pieces_are_rows(self, gapped_directory):
        """The gapped patch was written as its runs, each a row with a step."""
        contents = gapped_directory.get_contents()
        assert len(contents) == 3
        assert not contents["time_step"].isnull().any()

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
        assert coord.runs_count > 1
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
        assert relative.get_coord("time").runs_count > 1
        spool = dc.spool([gapped_patch, relative])
        assert spool.get_gaps()["gap_size"].tolist() == [HOLE]
        assert spool.get_coverage()["gap_total"].tolist() == [HOLE]

    def test_loose_tolerance_merges_across_runs(self, gapped_directory):
        """A tolerance wider than every gap merges the runs and the patches."""
        merged = gapped_directory.chunk(time=None, tolerance=10_000, fill_value=0)
        assert len(merged) == 1


class TestDerived:
    """Derived spools report the holes their sources held."""

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
        """Slices merge along distance once per run; the hole stays a gap."""
        distance = gapped_patch.get_coord("distance")
        mid = distance.values[len(distance) // 2]
        slices = [
            gapped_patch.select(distance=(None, mid)),
            gapped_patch.select(distance=(mid + distance.step, None)),
        ]
        merged = dc.spool(slices).chunk(distance=None)
        assert len(merged) == 2
        assert merged.get_gaps()["gap_size"].tolist() == [HOLE]
        assert merged.get_gaps("distance").empty

    def test_rechunk_reports_as_the_first_chunk(self, gapped_patch):
        """A re-chunk along the same dimension reports what the first did."""
        later = dc.get_example_patch().update_coords(
            time_min=gapped_patch.get_coord("time").max() + 10_000 * MS
        )
        once = dc.spool([gapped_patch, later]).chunk(time=None)
        twice = once.chunk(time=None)
        expected = [HOLE, pd.Timedelta(10, "s")]
        assert once.get_gaps()["gap_size"].tolist() == expected
        assert twice.get_gaps()["gap_size"].tolist() == expected
        assert len(twice.get_coverage()) == 1

    def test_concatenated_members_report_no_hole(self, gapped_patch):
        """An output joined along a dimension reports its span as covered."""
        later = gapped_patch.update_coords(
            time_min=gapped_patch.get_coord("time").max() + 4 * MS
        )
        joined = dc.spool([gapped_patch, later]).concatenate(time=None)
        (coverage,) = joined.get_coverage().to_dict("records")
        assert coverage["gap_total"] == pd.Timedelta(0)
        assert coverage["time_max"] == later.get_coord("time").max()

    def test_other_units_report_envelope(self):
        """A patch the plan restated in other units reports its envelope."""
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
    def test_tolerance_keeps_the_hole(self, gapped_patch, snap_coords):
        """Without fill_value a tolerance spanning the hole leaves it a gap."""
        chunked = dc.spool([gapped_patch]).chunk(
            time=None, tolerance=5, snap_coords=snap_coords
        )
        assert len(chunked) == 2
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
        assert not any(p.get_coord("time").runs_count > 1 for p in chunked)

    def test_plan_members_are_runs(self, gapped_patch, halves):
        """Each run is a whole member; bridged runs span the patch."""
        spool = dc.spool([gapped_patch])
        members = spool.chunk_plan(time=None).members
        assert not members["_modified"].any()
        for (_, row), half in zip(members.iterrows(), halves, strict=True):
            coord = half.get_coord("time")
            assert (row["time_min"], row["time_max"]) == (coord.min(), coord.max())
        bridged = spool.chunk_plan(time=None, tolerance=5).members
        time = gapped_patch.get_coord("time")
        span = (bridged["time_min"].min(), bridged["time_max"].max())
        assert span == (time.min(), time.max())

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

    def test_many_runs_plan_per_run(self):
        """A coordinate of hundreds of runs plans per run."""
        runs = [
            get_coord(start=20.0 * i, step=1.0, shape=(10,), units="m")
            for i in range(300)
        ]
        coord = concat_coords(*runs)
        patch = dc.Patch(
            data=np.zeros((len(coord), 2)),
            coords={"distance": coord, "x": np.arange(2)},
            dims=("distance", "x"),
        )
        assert len(dc.spool([patch]).chunk(distance=None)) == 300

    def test_sample_selection_applies_per_run(self, gapped_patch):
        """Each run is a row, so a sample selection applies to each."""
        view = dc.spool([gapped_patch]).select(time=(-10, None), samples=True)
        outs = view.chunk(time=None)
        assert [x.get_coord("time").shape for x in outs] == [(10,), (10,)]

    def test_time_after_distance(self, gapped_patch):
        """Chunking time after distance still splits at the hole."""
        chunked = dc.spool([gapped_patch]).chunk(distance=None).chunk(time=None)
        assert len(chunked) == 2
        assert chunked.get_gaps()["gap_size"].tolist() == [HOLE]

    def test_chained_chunks_keep_neighbour_samples(self):
        """A run merged with a neighbour keeps the neighbour's samples."""
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

    def test_rechunk_keeps_the_hole(self, gapped_patch):
        """A re-chunk of a chunk that kept the hole keeps it too."""
        kept = dc.spool([gapped_patch]).chunk(time=None, tolerance=5)
        assert len(kept.chunk(time=None, tolerance=5)) == 2
        windows = kept.chunk(time=1)
        assert not any(p.get_coord("time").runs_count > 1 for p in windows)
