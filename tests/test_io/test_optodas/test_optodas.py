"""
Tests for optoDAS files.
"""

import shutil

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.io.optodas import OptoDASV8
from dascore.utils.downloader import fetch


class TestOptoDASIssues:
    """Test case related to issues in OptoDAS parser."""

    def test_read_decimated_patch(self):
        """Tests for reading spatially decimated patch (#419)"""
        path = fetch("decimated_optodas.hdf5")
        fiber_io = OptoDASV8()

        fmt_str, version_str = fiber_io.get_format(path)
        assert (fmt_str, version_str) == (fiber_io.name, fiber_io.version)

        spool = fiber_io.read(path)
        patch = spool[0]
        assert isinstance(patch, dc.Patch)
        assert patch.data.shape


class TestDataScale:
    """Storage scaling must produce values matching the declared units."""

    @pytest.fixture
    def scaled_file(self, tmp_path):
        """Make a writable copy of the existing OptoDAS fixture."""
        path = tmp_path / "scaled.hdf5"
        shutil.copyfile(fetch("decimated_optodas.hdf5"), path)
        return path

    @pytest.mark.parametrize("version", [8, 9, 10, 11])
    @pytest.mark.parametrize("dtype", ["int16", "float32", "float64"])
    @pytest.mark.parametrize("scale", [0.004693713039159775, 1.0, None])
    def test_scaled_values(self, scaled_file, dtype, scale, version):
        """Scale all storage dtypes, preserving legacy files without a scale."""
        with h5py.File(scaled_file, "r+") as fi:
            shape = fi["data"].shape
            values = np.resize(np.array([-12, 0, 7, 103], dtype=dtype), shape)
            del fi["data"]
            fi["data"] = values
            del fi["fileVersion"]
            fi["fileVersion"] = version
            if "dataScale" in fi["header"]:
                del fi["header/dataScale"]
            if scale is not None:
                fi["header/dataScale"] = scale
        result = dc.read(scaled_file)[0]
        expected = values.astype("float64") * (1.0 if scale is None else scale)
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)
        expected_dtype = dtype if scale is None else np.result_type(dtype, np.float32)
        assert result.data.dtype == expected_dtype
        assert result.attrs.data_units == dc.scan(scaled_file)[0].data_units

    def test_subset(self, scaled_file):
        """Reading a subset scales exactly once and keeps coordinates aligned."""
        with h5py.File(scaled_file, "r+") as fi:
            fi["header/dataScale"][...] = 0.25
            stored = fi["data"][:]
        full = dc.read(scaled_file)[0]
        np.testing.assert_allclose(full.data, stored.astype("float64") * 0.25)
        bounds = {
            dim: (full.get_array(dim)[1], full.get_array(dim)[-2]) for dim in full.dims
        }
        subset = dc.read(scaled_file, **bounds)[0]
        expected = full.select(**bounds)
        np.testing.assert_array_equal(subset.data, expected.data)
        assert subset.coords == expected.coords
