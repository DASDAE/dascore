"""Tests for the index's coordinate table and runs (rule 12 of the coordinate note)."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import concat_coords, get_coord
from dascore.core.summary import PatchSummary
from dascore.io.index.backend import get_backend
from dascore.io.index.ingest import (
    index_runs,
    patch_record,
    stored_runs,
    summaries_to_records,
)
from tests.conftest import join_patches

MS = np.timedelta64(1, "ms")


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
