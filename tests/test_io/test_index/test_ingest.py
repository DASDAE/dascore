"""Tests for turning stored coordinate rows back into summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coords import get_coord
from dascore.io.index.ingest import _as_dtype, coord_summary


class TestCoordSummaryFromRow:
    """A stored coordinate row states everything its summary carries."""

    @pytest.fixture()
    def indexed(self):
        """A spool whose patch has a numeric, a time and a label coord."""
        patch = dc.get_example_patch()
        n = patch.shape[patch.get_axis("distance")]
        labels = np.array([f"s{i:03d}" for i in range(n)])
        patch = patch.update_coords(
            latitude=("distance", np.arange(n) * 1.0),
            station=("distance", labels),
        )
        return patch, dc.spool([patch])

    def _rows(self, spool):
        """Every stored coordinate row of the spool's one patch."""
        backend = spool._catalog.backend
        frame = backend._fetch_df("SELECT patch_id FROM patches")
        ids = [int(x) for x in frame["patch_id"]]
        return backend.coord_frame(ids).to_dict("records")

    def test_describes_every_coordinate(self, indexed):
        """Each row rebuilds the coordinate the patch holds."""
        patch, spool = indexed
        rows = {x["coord_name"]: x for x in self._rows(spool)}
        assert set(rows) == set(patch.coords.coord_map)
        for name, row in rows.items():
            summary = coord_summary(row)
            coord = patch.get_coord(name)
            assert summary.min == coord.min()
            assert summary.max == coord.max()
            assert summary.len == len(coord)
            assert summary.dims == patch.coords.dim_map[name]
            assert summary.fingerprint == coord.fingerprint()

    def test_range_coords_rebuild_exactly(self, indexed):
        """A sampled coordinate rebuilds into the coordinate it describes."""
        patch, spool = indexed
        for row in self._rows(spool):
            summary = coord_summary(row)
            if not summary.is_range_like:
                continue
            rebuilt = summary.to_coord(on_grid=True)
            coord = patch.get_coord(row["coord_name"])
            assert rebuilt.fingerprint() == coord.fingerprint()
            assert np.array_equal(rebuilt.values, coord.values)

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint16"])
    def test_every_numeric_dtype_rebuilds_as_itself(self, dtype):
        """A stored envelope comes back as the kind of number it was."""
        patch = dc.get_example_patch()
        n = patch.shape[patch.get_axis("distance")]
        coord = get_coord(values=np.arange(n, dtype=dtype))
        spool = dc.spool([patch.update_coords(distance=coord)])
        row = next(x for x in self._rows(spool) if x["coord_name"] == "distance")
        rebuilt = coord_summary(row).to_coord(on_grid=True)
        assert rebuilt.dtype == coord.dtype
        assert rebuilt.fingerprint() == coord.fingerprint()

    def test_a_cast_which_would_change_a_value_is_not_made(self):
        """Restoring a dtype is a change of type, never of value."""
        # a fractional value cannot be an integer, so it stays as it is
        assert _as_dtype(1.5, np.dtype("int64")) == 1.5
        # a dtype which is not a number is left alone
        assert _as_dtype(1.5, np.dtype("str")) == 1.5

    def test_relative_time_is_a_duration(self):
        """A relative time coordinate rebuilds as a duration, not a date."""
        row = {
            "value_kind": "time",
            "is_relative": True,
            "min_ns": 0,
            "max_ns": 1_000_000_000,
            "step_ns": 1_000_000,
            "dtype": "timedelta64",
            "coord_dims": "time",
            "length": 1001,
            "units": None,
            "fingerprint": None,
        }
        summary = coord_summary(row)
        assert summary.max == np.timedelta64(1, "s")
        assert summary.step == np.timedelta64(1, "ms")

    def test_a_row_stating_no_values_is_described(self):
        """A null envelope gives a summary which claims nothing."""
        row = {
            "value_kind": "num",
            "min_num": np.nan,
            "max_num": np.nan,
            "step_num": None,
            "dtype": "float64",
            "coord_dims": "rank",
            "length": None,
            "units": None,
            "fingerprint": "abc",
        }
        summary = coord_summary(row)
        assert not summary.is_range_like
        assert summary.fingerprint == "abc"

    @pytest.mark.parametrize("relative", [True, False])
    def test_a_time_row_without_values(self, relative):
        """A time coordinate stating no values summarizes as NaT."""
        row = {
            "value_kind": "time",
            "is_relative": relative,
            "min_ns": None,
            "max_ns": None,
            "step_ns": None,
            "dtype": "timedelta64" if relative else "datetime64",
            "coord_dims": "time",
            "length": None,
            "units": None,
            "fingerprint": None,
        }
        summary = coord_summary(row)
        assert pd.isnull(summary.min) and pd.isnull(summary.max)
        assert not summary.is_range_like

    def test_unknown_kind_is_skipped(self):
        """A value kind the index does not represent describes nothing."""
        assert coord_summary({"value_kind": "something_else"}) is None
