"""Tests for the DASHDF5 format."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import dascore as dc


class TestSnap:
    """Read and scan must agree on whether coordinates are snapped."""

    @pytest.fixture(scope="class")
    def jittered_path(self, tmp_path_factory):
        """
        A minimal DASHDF5 file whose time samples shift by a microsecond
        halfway through, so snapping changes the values.
        """
        path = tmp_path_factory.mktemp("dashdf5") / "jitter.h5"
        time = np.arange(20) * 0.001 + 1e9
        time[10:] += 1e-6
        with h5py.File(path, "w") as h5:
            h5.attrs["Conventions"] = np.array(
                ["CF-1.7", "DAS-HDF5-1.0"], dtype=h5py.string_dtype()
            )
            h5["channel"] = np.arange(4)
            h5["trace"] = np.arange(20)
            h5["t"] = time
            for name in "xyz":
                h5[name] = np.arange(4, dtype=float)
                h5[name].attrs["units"] = "m"
            h5["das"] = np.zeros((4, 20), dtype="float32")
        return path

    def test_read_honours_snap(self, jittered_path):
        """
        With snap=False read returns the file's own time values, as scan
        does; read used to snap them anyway.
        """
        scanned = dc.scan_payloads(jittered_path, snap=False)[0]["coords"]
        patch = dc.read(jittered_path, snap=False)[0]
        np.testing.assert_array_equal(
            scanned.get_coord("time").values, patch.get_coord("time").values
        )
