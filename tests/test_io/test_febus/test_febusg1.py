"""
Tests for the G1 csv files.

Note: These can't just be includes in the common_io tests because these
files only have one timestep.
"""

import shutil
from io import BytesIO, StringIO
from pathlib import Path

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.io.febus.core import FebusG1CSV1, FebusG1MTXH5V1
from dascore.io.febus.g1utils import _is_g1_file
from dascore.utils.downloader import fetch

g1_files = (
    "febg1_C1_2023-05-10T12.25.03+0000.bsl",
    "febg1_C1_2023-05-10T12.27.33+0000.bsl",
    # "febus-g1-specta_C1_2023-11-30T15.13.14+0000.mtx"
)
g1_mtx_h5_file = "febus-g1-spectra_C2_2026-06-03T17.28.13+0200.mtx.h5"


@pytest.fixture(scope="module", params=g1_files)
def g1_path(request):
    """
    The paths to the g1 times series files.
    """
    return fetch(request.param)


@pytest.fixture(scope="function")
def g1_mtx_buffer():
    """Get a buffer with an mtx file"""
    text = "\n".join(
        [
            "Param;sampling resolution;0.1",
            "Param;fiberFrom;0",
            "Param;fiberTo;1",
            "Param;start time;1683721503.0;2023-05-10T12:25:03+0000",
            "Param;end time;1683721504.0;2023-05-10T12:25:04+0000",
            "Param;mode;strain",
            "Param;channel;1",
            "0.0",
        ]
    )
    resource = StringIO(text)
    resource.name = "febg1_C1_2023-05-10T12.25.03+0000.mtx"
    return resource


@pytest.fixture(scope="module")
def g1_mtx_h5_path():
    """Return the path to a G1 Brillouin spectra HDF5 file."""
    return fetch(g1_mtx_h5_file)


def _write_mtx_h5(path, format_version=1):
    """Write a minimal 3D Febus G1 MTX HDF5 test file."""
    with h5py.File(path, "w") as h5:
        h5.create_dataset("distances", data=np.array([1.0, 2.0]))
        h5.create_dataset("start_times", data=np.array([1_780_500_493.0]))
        h5.create_dataset("end_times", data=np.array([1_780_500_494.0]))
        h5.create_dataset("temperatures", data=np.array([24.0], dtype=np.float32))
        h5.create_dataset("mtx", data=np.ones((1, 2, 3), dtype=np.float32))
        h5.attrs["freq_offset_abs"] = np.array([10750.0], dtype=np.float32)
        h5.attrs["freq_step"] = np.array([-3.90625], dtype=np.float32)
        if format_version is not None:
            h5.attrs["formatVersion"] = np.asarray(format_version)


class TestG1GetFormat:
    """Tests for determining g1 file formats."""

    def test_is_format(self, g1_path):
        """Ensure the g1 format can be auto-detected."""
        g1 = FebusG1CSV1()
        format_tuple = g1.get_format(g1_path)
        assert format_tuple == (g1.name, g1.version)

    def test_bad_name_format(self):
        """Ensure names with wrong underscore pattern are rejected."""
        bad_file = BytesIO(b"Param;a\nParam;b\nParam;c\n")
        bad_file.name = "febg1_2023-05-10T12:25:03+00:00.bsl"
        assert not _is_g1_file(bad_file)

    def test_bad_iso8601_datetime(self):
        """Ensure non-ISO8601 datetimes in file names are rejected."""
        bad_file = BytesIO(b"Param;a\nParam;b\nParam;c\n")
        bad_file.name = "febg1_C1_2023-1-10T12:25:03.bsl"
        assert not _is_g1_file(bad_file)

    def test_missing_param_lines(self):
        """Ensure files without Param; headers are rejected."""
        bad_file = BytesIO(b"NotParam;a\nNotParam;b\nNotParam;c\n")
        bad_file.name = "febg1_C1_2023-05-10T12:25:03+00:00.bsl"
        assert not _is_g1_file(bad_file)


class TestG1Scan:
    """Tests for determining g1 coords attributes."""

    def test_scan(self, g1_path):
        """Ensure attrs can be extracted from G1 files."""
        g1 = FebusG1CSV1()
        attrs = g1.scan(g1_path)
        assert len(attrs) == 1
        attr = attrs[0]
        assert isinstance(attr, dc.PatchAttrs)
        assert str(g1_path) == attr.path
        assert attr.file_format == g1.name
        assert attr.file_version == g1.version


class TestG1Read:
    """Tests for reading G1 files into patches."""

    def test_read(self, g1_path):
        """Ensure a G1 file is read into a Patch with expected data."""
        spool = dc.read(g1_path)
        assert len(spool) == 1
        patch = spool[0]
        assert isinstance(patch, dc.Patch)


class TestG1MTXH5:
    """Tests for G1 Brillouin spectrum HDF5 files."""

    def test_get_format(self, g1_mtx_h5_path):
        """Ensure the MTX HDF5 format can be auto-detected."""
        fiber = FebusG1MTXH5V1()
        assert fiber.get_format(g1_mtx_h5_path) == (fiber.name, fiber.version)
        assert dc.get_format(g1_mtx_h5_path) == (fiber.name, fiber.version)

    def test_read(self, g1_mtx_h5_path):
        """Ensure MTX HDF5 data are read into a 3D patch."""
        patch = dc.read(g1_mtx_h5_path)[0]
        assert patch.dims == ("time", "distance", "frequency")
        assert patch.shape == (68, 100, 128)
        assert patch.attrs.data_category == "DSS"
        assert patch.attrs.febus_data_kind == "brillouin_spectrum"
        assert patch.attrs.format_version == 1
        assert patch.attrs.fiber_from == 50
        assert "temperature" in patch.coords.coord_map
        assert patch.coords.dim_map["temperature"] == ("time",)

    def test_coords(self, g1_mtx_h5_path):
        """Coordinates should come from Febus datasets and attributes."""
        patch = dc.read(g1_mtx_h5_path)[0]
        freq = patch.get_coord("frequency")
        time = patch.get_coord("time")
        dist = patch.get_coord("distance")
        temperature = patch.get_coord("temperature")
        assert freq.units == dc.get_quantity("MHz")
        assert dist.units == dc.get_quantity("m")
        assert temperature.units == dc.get_quantity("degC")
        assert freq.step == -3.90625
        assert freq.max() == 10750.0
        assert freq.min() == 10253.90625
        assert dist.min() == 50
        assert dist.max() == 149
        assert dist.step == 1
        assert str(time.min()) == "2026-06-03T15:28:13.647423488"
        np.testing.assert_allclose(temperature.values[[0, -1]], [24.67, 24.69])

    def test_scan_matches_read_attrs(self, g1_mtx_h5_path):
        """Scan and read should return matching coord summaries."""
        attrs = dc.scan(g1_mtx_h5_path)[0]
        patch_attrs = dc.read(g1_mtx_h5_path)[0].attrs
        assert attrs.dims == patch_attrs.dims
        assert attrs.coords == patch_attrs.coords
        assert attrs.file_format == FebusG1MTXH5V1.name
        assert attrs.file_version == FebusG1MTXH5V1.version

    def test_selects(self, g1_mtx_h5_path):
        """Read supports selecting along all three dimensions."""
        fiber = FebusG1MTXH5V1()
        assert fiber.read(g1_mtx_h5_path, frequency=(10300, 10400))[0].shape[2] < 128
        time_selected = fiber.read(
            g1_mtx_h5_path,
            time=("2026-06-03T15:29:00", ...),
        )[0]
        assert time_selected.dims == ("time", "distance", "frequency")
        assert time_selected.shape[0] < 68
        assert len(time_selected.get_coord("temperature")) == time_selected.shape[0]
        assert fiber.read(g1_mtx_h5_path, distance=(60, 70))[0].shape[1] == 11
        assert len(fiber.read(g1_mtx_h5_path, frequency=(99999, ...))) == 0

    @pytest.mark.parametrize("format_version", [2, [1, 2], None])
    def test_unsupported_format_version_not_detected(self, tmp_path, format_version):
        """The v1 reader should not claim unsupported MTX HDF5 versions."""
        path = tmp_path / "bad_version_C1.mtx.h5"
        _write_mtx_h5(path, format_version=format_version)
        assert not FebusG1MTXH5V1().get_format(path)

    def test_three_dimensional_file(self, tmp_path):
        """MTX HDF5 files should produce patches in native HDF5 dim order."""
        path = tmp_path / "three_dimensional_C1.mtx.h5"
        _write_mtx_h5(path)
        patch = dc.read(path)[0]
        assert patch.dims == ("time", "distance", "frequency")
        assert patch.shape == (1, 2, 3)
        np.testing.assert_allclose(patch.get_coord("temperature").values, [24.0])


class TestMisc:
    """Misc integration tests for G1 files."""

    mtx_text = "cannot yet parse spectra"

    @pytest.fixture(scope="class")
    def g1_two_file_directory(self, tmp_path_factory):
        """Create a directory with only the two tracked G1 files."""
        out_dir = Path(tmp_path_factory.mktemp("g1_two_file_directory"))
        for name in g1_files[:2]:
            src = fetch(name)
            shutil.copy2(src, out_dir / Path(src).name)
        return out_dir

    def test_chunk_all_time_merges_to_single_patch(self, g1_two_file_directory):
        """Ensure chunk(time=None) merges both g1 files into one patch."""
        spool = dc.spool(g1_two_file_directory)
        # These weren't directly adjacent files so we adjust the tolerance.
        match = "There is a gap in the patch along dimension time"
        with pytest.warns(UserWarning, match=match):
            merged = spool.chunk(time=None, tolerance=3)
        assert len(merged) == 1

    def test_mtx_read_raises(self, g1_mtx_buffer):
        """
        Ensure reading g1 spectra (.mtx) files Raises NotImplementedError with read
        """
        fiber = FebusG1CSV1()
        with pytest.raises(NotImplementedError, match=self.mtx_text):
            fiber.read(g1_mtx_buffer)

    def test_mtx_scan_warns(self, g1_mtx_buffer):
        """Mtx scan should just issue a warning and return empty."""
        fiber = FebusG1CSV1()
        with pytest.warns(UserWarning, match=self.mtx_text):
            fiber.scan(g1_mtx_buffer)

    def test_directory_spool(self, two_patch_directory):
        """Ensure a directory spool works and can read files."""
        spool = dc.spool(two_patch_directory).update()
        patch = spool[0]
        assert isinstance(patch, dc.Patch)
