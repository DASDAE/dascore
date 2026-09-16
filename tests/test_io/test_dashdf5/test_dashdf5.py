"""Tests for the DASHDF5 format."""

from __future__ import annotations

import shutil

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
            h5["das"].attrs["long_name"] = "strain_rate"
        return path

    def test_read_honours_snap(self, jittered_path):
        """
        With snap=False read returns the file's own time values, as scan
        does; read used to snap them anyway.
        """
        scanned = dc.scan_payloads(jittered_path, snap=False)[0].coords
        patch = dc.read(jittered_path, snap=False)[0]
        np.testing.assert_array_equal(
            scanned.get_coord("time").values, patch.get_coord("time").values
        )

    def test_long_name_sets_data_type(self, jittered_path):
        """The CF signal description supplies the canonical data type."""
        assert dc.read(jittered_path)[0].attrs.data_type == "strain_rate"
        assert dc.scan(jittered_path)[0].attrs.data_type == "strain_rate"

    @pytest.mark.parametrize("reader", [dc.read, dc.scan], ids=["read", "scan"])
    @pytest.mark.parametrize(
        ("long_name", "expected"),
        [
            (None, ""),
            (np.bytes_(b"strain_rate"), "strain_rate"),
            ("Axial Strain Rate (nm/m/s)", "Axial Strain Rate (nm/m/s)"),
        ],
        ids=["missing", "bytes", "free_text"],
    )
    def test_long_name_representation(
        self, jittered_path, tmp_path, reader, long_name, expected
    ):
        """Signal descriptions decode bytes and preserve optional free text."""
        path = tmp_path / "description.h5"
        shutil.copyfile(jittered_path, path)
        with h5py.File(path, "r+") as h5:
            attrs = h5["das"].attrs
            if long_name is None:
                del attrs["long_name"]
            else:
                attrs["long_name"] = long_name
        assert reader(path)[0].attrs.data_type == expected
