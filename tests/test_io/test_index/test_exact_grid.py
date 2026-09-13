"""Tests for schema 16: exact grids stored and rebuilt by the index."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coords import CoordRange, get_coord
from dascore.core.summary import PatchSummary
from dascore.io.index.backend import get_backend
from dascore.io.index.catalog import _coord_from_envelope
from dascore.io.index.ingest import _coord_record, summaries_to_records
from dascore.io.index.planned import _coord_record_from_row
from dascore.utils.patch_assembly import coord_from_row

T0 = np.datetime64("2020-01-01T00:00:00")


@pytest.fixture(scope="module")
def hz_1024_patch():
    """The example patch on an exact 1024 Hz grid."""
    patch = dc.get_example_patch()
    time = get_coord(start=T0, step=(1, 1024), shape=(patch.shape[1],))
    return patch.update_coords(time=time)


@pytest.fixture(scope="module")
def indexed(hz_1024_patch, tmp_path_factory):
    """A directory spool holding the 1024 Hz patch, indexed."""
    path = tmp_path_factory.mktemp("exact") / "hz.h5"
    dc.write(hz_1024_patch, path, "dasdae")
    return dc.spool(path.parent).update()


class TestSchema:
    """The stored rows carry the grid."""

    def test_record_carries_grid(self, hz_1024_patch):
        """A range summary's record is exact and holds its grid."""
        summary = PatchSummary.from_patch(hz_1024_patch).coords["time"]
        record = _coord_record("time", summary)
        assert record is not None
        assert record.is_exact
        assert (record.step_numerator, record.step_denominator) == (1953125, 2)
        assert record.origin_offset == 0
        # the envelope still holds the whole-tick step
        assert record.step_int == 976562

    def test_array_record_is_not_exact(self):
        """An array coordinate's row is only an envelope."""
        coord = get_coord(data=np.array([1.0, 2.5, 7.0]))
        record = _coord_record("x", coord.to_summary(dims=("x",)))
        assert record is not None
        assert not record.is_exact
        assert record.step_numerator is None

    def test_stored_columns(self, indexed):
        """The coord_defs row stores the grid."""
        back = indexed._catalog.backend
        defs = back._fetch_df("SELECT * FROM coord_defs")
        assert defs["is_exact"].all()
        time_def = defs[defs["step_denominator"] == 2]
        assert len(time_def) == 1
        assert int(time_def["step_numerator"].iloc[0]) == 1953125


class TestFlatRelation:
    """The spool's frame rebuilds exact coordinates from its rows."""

    def test_contents_step_is_whole_tick(self, indexed):
        """The public step column stays a plain duration."""
        step = indexed.get_contents()["time_step"].iloc[0]
        assert step == pd.Timedelta(976562, "ns")

    def test_grid_column(self, indexed):
        """The private grid column holds a fractional grid and its length.

        A whole-tick grid is what the envelope already states, so it is
        not carried.
        """
        df = indexed._catalog.to_df()
        grid = df["_time_grid"].iloc[0]
        assert grid == (1953125, 2, 0, 2000)
        assert df["_distance_grid"].iloc[0] is None

    def test_coord_from_row(self, indexed, hz_1024_patch):
        """A row rebuilds the exact coordinate, not the rounded one."""
        row = indexed._catalog.to_df().iloc[0].to_dict()
        coord = coord_from_row(row, "time", units="s")
        assert isinstance(coord, CoordRange)
        assert coord == hz_1024_patch.get_coord("time")
        distance = coord_from_row(row, "distance", units="m")
        assert distance == hz_1024_patch.get_coord("distance")

    def test_converted_units_drop_grid(self, indexed):
        """A grid in the file's units cannot describe a converted envelope."""
        row = indexed._catalog.to_df().iloc[0].to_dict()
        row["_time_units_source"] = "ms"
        coord = coord_from_row(row, "time", units="s")
        assert coord is not None
        assert coord.step_exact != Fraction(1, 1024)

    def test_envelope_rebuild(self, indexed, hz_1024_patch):
        """The catalog's stashed envelope rebuilds the exact coordinate."""
        row = indexed._catalog.to_df().iloc[0].to_dict()
        envelope = {k: row[k] for k in ("time_min", "time_max", "time_step")}
        assert _coord_from_envelope(envelope, "time", "s") != (
            hz_1024_patch.get_coord("time")
        )
        envelope["_time_grid"] = row["_time_grid"]
        coord = _coord_from_envelope(envelope, "time", "s")
        assert coord == hz_1024_patch.get_coord("time")

    def test_select_matches_memory(self, indexed, hz_1024_patch):
        """A reader-hinted selection lands on the same samples as in memory.

        The bound sits where the whole-tick and exact grids disagree, and
        the recorded selection (the processing id) shows the replay saw
        the same coordinate the file holds.
        """
        window = (T0 + np.timedelta64(1, "s"), T0 + np.timedelta64(1952148000, "ns"))
        out, expected = (
            indexed.select(time=window)[0],
            hz_1024_patch.select(time=window),
        )
        assert out == expected
        assert out.attrs.processing_id == expected.attrs.processing_id

    def test_merge_through_index(self, hz_1024_patch, tmp_path):
        """Two files merged by their rows alone keep the grid.

        A two-member merge reads arrays and builds coordinates from the
        index rows, the path a whole-tick rebuild would send off the grid.
        """
        halves = (
            hz_1024_patch.select(time=(None, 1000), samples=True),
            hz_1024_patch.select(time=(1000, None), samples=True),
        )
        for num, half in enumerate(halves):
            half.update_attrs(history=[]).io.write(tmp_path / f"{num}.h5", "dasdae")
        (merged,) = dc.spool(tmp_path).update().chunk(time=None)
        assert merged.get_coord("time").step_exact == Fraction(1, 1024)
        assert merged.get_coord("time") == hz_1024_patch.get_coord("time")

    def test_chunk_outputs_carry_grid(self, indexed):
        """A chunk plan keeps the grid of the dimensions it leaves whole."""
        chunked = indexed.chunk(distance=100)
        rows = chunked._catalog.to_df()
        assert rows["_time_grid"].notna().all()
        defs = chunked._catalog.backend._fetch_df("SELECT * FROM coord_defs")
        assert (defs["step_denominator"] == 2).any()

    def test_descending_whole_ticks_carry_grid(self, tmp_path):
        """A descending range needs its grid: the envelope does not name its start."""
        patch = dc.get_example_patch()
        reversed_patch = patch.update_coords(time=patch.get_coord("time")[::-1])
        dc.write(reversed_patch, tmp_path / "rev.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        row = spool._catalog.to_df().iloc[0].to_dict()
        assert row["_time_grid"] is not None
        coord = coord_from_row(row, "time", units="s")
        assert coord == reversed_patch.get_coord("time")

    def test_descending_grid(self, indexed, hz_1024_patch):
        """A descending grid rebuilds from its maximum, which the row states."""
        row = indexed._catalog.to_df().iloc[0].to_dict()
        reversed_time = hz_1024_patch.get_coord("time")[::-1]
        num, den, offset = (
            reversed_time.step_numerator,
            reversed_time.step_denominator,
            reversed_time.origin_offset,
        )
        row["time_step"] = -row["time_step"]
        row["_time_grid"] = (num, den, offset, len(reversed_time))
        assert coord_from_row(row, "time", units="s") == reversed_time
        # without the grid the row cannot say where a descending range starts
        row["_time_grid"] = None
        assert coord_from_row(row, "time", units="s") is None

    def test_float_row_skips_grid(self):
        """A plan row stating a float placeholder dtype cannot use an integer grid."""
        row = {"x_min": 0.0, "x_max": 9.0, "x_step": 1.0, "_x_grid": (1, 1, 0, 10)}
        coord = coord_from_row(row, "x")
        assert coord is not None
        assert coord.dtype == np.dtype("float64")
        assert coord.step_numerator is None


class TestPlannedRows:
    """Plan outputs keep the grid only while the coordinate's identity holds."""

    def test_grid_with_identity(self, indexed):
        """A kept def key brings the grid into the output record."""
        row = indexed._catalog.to_df().iloc[0].to_dict()
        record = _coord_record_from_row(row, "time")
        assert record is not None
        assert record.step_denominator == 2
        assert record.is_exact

    def test_grid_without_identity(self, indexed):
        """A re-planned output states a rounded step and no grid."""
        row = indexed._catalog.to_df().iloc[0].to_dict()
        row["_time_def_key"] = None
        record = _coord_record_from_row(row, "time")
        assert record is not None
        assert record.step_numerator is None


class TestRecordsRoundTrip:
    """Summaries written and read back through the backend agree."""

    def test_grid_survives_write_and_export(self, hz_1024_patch, tmp_path):
        """The exported record equals the ingested one."""
        summary = PatchSummary.from_patch(hz_1024_patch).model_copy(
            update={"source_path": "a.h5", "source_format": "DASDAE"}
        )
        records = summaries_to_records([summary])
        back = get_backend(tmp_path / "grid.sqlite3")
        back.write_sources(records)
        exported = back.export_records()
        back.close()
        before = {c.coord_name: c for c in records[0].patches[0].coords}
        after = {c.coord_name: c for c in exported[0].patches[0].coords}
        assert before == after
