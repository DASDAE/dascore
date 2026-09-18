"""
Tests for optoDAS files.
"""

import shutil

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.io.optodas import OptoDASV8
from dascore.io.optodas.utils import _get_coord_manager
from dascore.utils.downloader import fetch


class TestOptoDASIssues:
    """Test case related to issues in OptoDAS parser."""

    def test_scan_distance_units_preserved(self):
        """Snapped and exact scan coordinates should retain header units."""
        path = fetch("decimated_optodas.hdf5")
        fiber_io = OptoDASV8()

        for snap in (True, False):
            payload = fiber_io.scan(path, snap=snap)[0]
            distance = payload["coords"].get_coord("distance")
            assert distance.units == dc.get_quantity("m")


class TestReadArray:
    """Tests for slicing the data dataset directly."""

    @pytest.fixture(scope="class")
    def transposed_path(self, tmp_path_factory):
        """An OptoDAS file whose header states the other dimension order."""
        source = fetch("opto_das_1.hdf5")
        path = tmp_path_factory.mktemp("optodas_transposed") / "transposed.hdf5"
        with h5py.File(source, "r") as src, h5py.File(path, "w") as dest:
            for name in src:
                src.copy(name, dest)
            names = [x.decode() for x in dest["header"]["dimensionNames"][:]]
            del dest["header"]["dimensionNames"]
            dest["header"]["dimensionNames"] = np.array(
                [x.encode() for x in names[::-1]]
            )
            data = dest["data"][:]
            del dest["data"]
            dest["data"] = data.T
        return path

    def test_dimension_order_follows_the_header(self, transposed_path):
        """The header names the stored order; a hardcoded one would transpose."""
        io = OptoDASV8()
        with h5py.File(transposed_path, "r") as h5:
            stored = h5["data"][:] * h5["header/dataScale"][()]
        out = io.read_array(transposed_path, {"time": (1, 4)})
        # the header now calls the first axis distance, so a time window
        # takes columns rather than rows
        np.testing.assert_allclose(out, stored[:, 1:4], rtol=1e-6)


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
        summary = dc.scan(scaled_file)[0]
        assert summary.dtype == str(expected_dtype)
        assert result.attrs.data_units == summary.attrs.data_units

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

    def test_read_array_scaled(self, scaled_file):
        """read_array applies the same scale as read, and scan reports its dtype."""
        with h5py.File(scaled_file, "r+") as fi:
            fi["header/dataScale"][...] = 0.25
        windows = {"time": (1, 4)}
        out = OptoDASV8().read_array(scaled_file, windows)
        expected = dc.read(scaled_file)[0].select(samples=True, **windows)
        np.testing.assert_array_equal(out, expected.data)
        assert str(out.dtype) == dc.scan(scaled_file)[0].dtype


class TestChannelMaps:
    """The channel numbers a file lists become its distance labels."""

    @pytest.fixture(scope="class")
    def descending_channels(self, tmp_path_factory):
        """A file whose unsigned channel numbers count down."""
        path = tmp_path_factory.mktemp("optodas") / "descending.h5"
        with h5py.File(path, "w") as fi:
            header = fi.create_group("header")
            header["channels"] = np.asarray([3, 2, 1], dtype="uint16")
            header["dimensionNames"] = np.asarray([b"time", b"distance"])
            header["dimensionUnits"] = np.asarray([b"s", b"m"])
            header["time"] = 1.0
            ranges = header.create_group("dimensionRanges")
            for index, (low, high) in enumerate([(0, 9), (0, 2)]):
                group = ranges.create_group(f"dimension{index}")
                group["min"] = np.asarray([low])
                group["max"] = np.asarray([high])
                group["unitScale"] = np.asarray([1.0])
        return path

    @pytest.mark.parametrize("snap", [True, False])
    def test_descending_unsigned_channels(self, descending_channels, snap):
        """A stride read from unsigned numbers must not wrap into a huge step."""
        with h5py.File(descending_channels) as fi:
            coords = _get_coord_manager(fi, snap=snap)
        np.testing.assert_array_equal(coords.coord_map["distance"].values, [3, 2, 1])
