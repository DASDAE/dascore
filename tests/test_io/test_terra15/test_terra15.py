"""Misc. tests for Terra15."""

from __future__ import annotations

import shutil
from typing import ClassVar

import h5py
import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.io.core import FiberIO
from dascore.io.terra15.core import Terra15FormatterV4
from dascore.io.terra15.utils import _get_version_data_node


class TestTerra15:
    """Misc tests for Terra15."""

    @pytest.fixture(scope="class")
    def missing_gps_terra15_hdf5(self, terra15_v5_path, tmp_path_factory):
        """Creates a terra15 file with missing GPS Time."""
        new = tmp_path_factory.mktemp("missing_gps") / "missing.hdf5"
        shutil.copy(terra15_v5_path, new)
        with h5py.File(new, "a") as fi:
            del fi["data_product/gps_time"]
        return new

    def test_missing_gps_time(self, missing_gps_terra15_hdf5):
        """Tests for when GPS time isn't found."""
        patch = dc.read(missing_gps_terra15_hdf5)[0]
        assert isinstance(patch, dc.Patch)
        assert not np.any(pd.isnull(patch.coords.get_array("time")))

    def test_time_slice_no_snap(self, terra15_v6_path):
        """Ensure no snapping returns raw time."""
        info = dc.scan_to_df(terra15_v6_path).iloc[0]
        file_t1, file_t2 = info["time_min"], info["time_max"]
        dur = file_t2 - file_t1
        new_dur = dur / 4
        t1, t2 = file_t1 + new_dur, file_t1 + 2 * new_dur
        out = dc.read(terra15_v6_path, time=(t1, t2), snap_dims=False)[0]
        assert isinstance(out, dc.Patch)
        time_summary = out.summary.get_coord_summary("time")
        assert time_summary.min >= t1
        assert time_summary.max <= t2

    def test_scan_payload_snap_contract(self, terra15_v6_path):
        """Raw payload scans should expose exact stored times on request."""
        snapped = dc.scan_payloads(terra15_v6_path, snap=True)[0]["coords"]
        exact = dc.scan_payloads(terra15_v6_path, snap=False)[0]["coords"]
        read_exact = dc.read(terra15_v6_path, snap_dims=False)[0].coords

        assert snapped.get_coord("time").evenly_sampled
        np.testing.assert_array_equal(
            exact.get_coord("time").values,
            read_exact.get_coord("time").values,
        )

    def test_units(self, terra15_das_patch):
        """All units should be defined on terra15 patch."""
        patch = terra15_das_patch
        assert patch.attrs.data_units is not None
        assert patch.get_coord("distance").units is not None
        assert patch.get_coord("time").units is not None
        assert (
            patch.get_coord("time").units
            == patch.summary.get_coord_summary("time").units
        )
        assert (
            patch.get_coord("distance").units
            == patch.summary.get_coord_summary("distance").units
        )

    def test_unsupported_version_error(self):
        """Test that unsupported Terra15 version raises NotImplementedError."""

        # Create a mock HDF5 root object with unsupported version
        class MockRoot:
            attrs: ClassVar = {"file_version": "999"}  # Unsupported version

        mock_root = MockRoot()

        # Test that it raises NotImplementedError
        with pytest.raises(NotImplementedError, match="Unknown Terra15 version"):
            _get_version_data_node(mock_root)


class TestTerra15Unfinished:
    """Test for reading files with zeroes filled at the end."""

    @pytest.fixture(scope="class")
    def patch_unfinished(self, terra15_das_unfinished_path):
        """Return the patch with zeroes at the end."""
        out = dc.spool(terra15_das_unfinished_path)[0]
        return out

    def test_zeros_gone(self, patch_unfinished):
        """No zeros should exist in the data."""
        data = patch_unfinished.data
        all_zero_rows = np.all(data == 0, axis=1)
        assert not np.any(all_zero_rows)

    def test_monotonic_time(self, patch_unfinished):
        """Ensure the time is increasing."""
        time = patch_unfinished.coords.get_array("time")
        assert np.all(np.diff(time) >= np.timedelta64(0, "s"))


class TestReadArray:
    """Tests for slicing the data node directly."""

    def test_unfinished_file_stops_at_written_samples(
        self, terra15_das_unfinished_path
    ):
        """Zero-filled rows past the last written sample are never returned."""
        io = Terra15FormatterV4()
        patch = dc.spool(terra15_das_unfinished_path)[0]
        out = io.read_array(terra15_das_unfinished_path, {})
        assert out.shape == patch.shape
        assert np.array_equal(out, patch.data)
        # a window past the end clips to the written samples, as select does
        tail = io.read_array(terra15_das_unfinished_path, {"time": (-3, 10**6)})
        assert np.array_equal(tail, patch.data[-3:])

    def test_raw_grid_counts_every_row(self, terra15_das_unfinished_path):
        """With snap_dims=False the window lives on the raw grid, as in read."""
        io = Terra15FormatterV4()
        path = terra15_das_unfinished_path
        out = io.read_array(path, {"time": (-5, None)}, snap_dims=False)
        expected = FiberIO.read_array(io, path, {"time": (-5, None)}, snap_dims=False)
        assert np.array_equal(out, expected)
        assert len(out) == 5
        raw = io.read_array(path, {}, snap_dims=False)
        assert len(raw) > len(io.read_array(path, {}))

    def test_both_spellings_agree(self, terra15_das_unfinished_path):
        """`scan` calls it snap and `read` snap_dims; both are taken here.

        The unfinished file is the one where the option changes how many
        rows there are, so the two grids differ and a mix-up shows.
        """
        io = Terra15FormatterV4()
        path = terra15_das_unfinished_path
        snapped = io.read_array(path, {})
        raw = io.read_array(path, {}, snap=False)
        assert len(raw) > len(snapped)
        # either spelling alone selects the same grid
        assert len(io.read_array(path, {}, snap_dims=False)) == len(raw)
        assert len(io.read_array(path, {}, snap=True)) == len(snapped)
        # given both, snap wins, in read_array and in read alike
        assert len(io.read_array(path, {}, snap=True, snap_dims=False)) == len(snapped)
        both = io.read(path, snap=True, snap_dims=False)[0]
        assert len(both.get_coord("time")) == len(snapped)

    def test_reads_only_the_window(self, terra15_v6_path, monkeypatch):
        """The data node is sliced in the file, not read whole then trimmed."""
        seen = []
        original = h5py.Dataset.__getitem__

        def spy(self, index):
            if self.name.endswith("/data"):
                seen.append(index)
            return original(self, index)

        monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
        Terra15FormatterV4().read_array(terra15_v6_path, {"time": (2, 6)})
        assert len(seen) == 1
        assert seen[0][0] == slice(2, 6)
