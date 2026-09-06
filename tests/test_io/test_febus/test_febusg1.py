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
from dascore.io.febus.core import FebusG1CSV1, FebusMTXAttrs, FebusMTXH5V1
from dascore.io.febus.g1utils import _get_mtx_patch, _is_g1_file
from dascore.utils.downloader import fetch

g1_files = (
    "febg1_C1_2023-05-10T12.25.03+0000.bsl",
    "febg1_C1_2023-05-10T12.27.33+0000.bsl",
    # "febus-g1-specta_C1_2023-11-30T15.13.14+0000.mtx"
)
mtx_h5_file = "febus-g1-spectra_C2_2026-06-03T17.28.13+0200.mtx.h5"


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
def mtx_h5_path():
    """Return the path to a Brillouin spectra HDF5 file."""
    return fetch(mtx_h5_file)


def _write_mtx_h5(path, format_version=1):
    """Write a minimal 3D Febus MTX HDF5 test file."""
    with h5py.File(path, "w") as h5:
        h5.create_dataset("distances", data=np.array([1.0, 2.0]))
        h5.create_dataset("start_times", data=np.array([1_780_500_493.0]))
        h5.create_dataset("end_times", data=np.array([1_780_500_494.0]))
        h5.create_dataset("temperatures", data=np.array([24.0], dtype=np.float32))
        h5.create_dataset("mtx", data=np.ones((1, 2, 3), dtype=np.float32))
        h5.attrs["freq_offset_abs"] = np.array([10750.0], dtype=np.float32)
        h5.attrs["freq_step"] = np.array([-3.90625], dtype=np.float32)
        h5.attrs["febusDataKind"] = "brillouin_spectrum"
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

    def test_raw_scan_excludes_source_metadata(self, g1_path):
        """Direct G1 scan should return a structured payload without source metadata."""
        g1 = FebusG1CSV1()
        attrs = g1.scan(g1_path)
        assert len(attrs) == 1
        attr = attrs[0]
        assert isinstance(attr, dict)
        assert "source_path" not in attr
        assert "source_format" not in attr
        assert "source_version" not in attr

    def test_public_scan_adds_source_metadata(self, g1_path):
        """Public scan should attach reload metadata for G1 files."""
        g1 = FebusG1CSV1()
        attrs = dc.scan(g1_path)
        assert len(attrs) == 1
        attr = attrs[0]
        assert isinstance(attr, dc.PatchSummary)
        assert str(g1_path) == str(attr.source_path)
        assert attr.source_format == g1.name
        assert attr.source_version == g1.version


class TestG1MTXH5:
    """Tests for Brillouin spectrum HDF5 files."""

    def test_read(self, mtx_h5_path):
        """Ensure MTX HDF5 data are read into a 3D patch."""
        patch = dc.read(mtx_h5_path)[0]
        assert patch.dims == ("time", "distance", "frequency")
        assert patch.shape == (68, 100, 128)
        assert patch.attrs.data_category == "DSS"
        assert patch.attrs.data_type == "brillouin_spectrum"
        assert patch.attrs.fiber_from == 50
        assert "temperature" in patch.coords.coord_map
        assert patch.coords.dim_map["temperature"] == ("time",)
        assert patch.coords.dim_map["sample_span"] == ("time",)

    def test_format_version_not_duplicated_in_attrs(self, mtx_h5_path):
        """The file version is reported as the format version, not as an attr."""
        patch = dc.read(mtx_h5_path)[0]
        assert "format_version" not in dict(patch.attrs)
        assert dc.scan(mtx_h5_path)[0].source_version == FebusMTXH5V1.version

    def test_read_preserves_mtx_array_order(self, tmp_path):
        """The patch data should match the MTX array stored in the file."""
        path = tmp_path / "order_C1.mtx.h5"
        _write_mtx_h5(path)
        stored = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
        with h5py.File(path, "a") as h5:
            del h5["mtx"]
            h5.create_dataset("mtx", data=stored)
        patch = FebusMTXH5V1().read(path)[0]
        frequency = patch.get_coord("frequency")
        expected_frequency = 10750.0 + np.arange(3) * -3.90625
        np.testing.assert_array_equal(patch.data, stored)
        np.testing.assert_allclose(frequency.values, expected_frequency)

    def test_selects(self, mtx_h5_path):
        """Read supports selecting along all three dimensions."""
        fiber = FebusMTXH5V1()
        assert fiber.read(mtx_h5_path, frequency=(10300, 10400))[0].shape[2] < 128
        time_selected = fiber.read(
            mtx_h5_path,
            time=("2026-06-03T15:29:00", ...),
        )[0]
        assert time_selected.dims == ("time", "distance", "frequency")
        assert time_selected.shape[0] < 68
        assert len(time_selected.get_coord("temperature")) == time_selected.shape[0]
        assert fiber.read(mtx_h5_path, distance=(60, 70))[0].shape[1] == 11
        assert len(fiber.read(mtx_h5_path, frequency=(99999, ...))) == 0

    def test_selection_does_not_read_the_whole_array(self, mtx_h5_path):
        """A trimmed read touches only the samples it keeps."""
        read = []

        class _CountingDataset:
            """Stand in for the stored array and record what is asked of it."""

            def __init__(self, dataset):
                self._dataset = dataset

            def __getattr__(self, name):
                return getattr(self._dataset, name)

            def __getitem__(self, item):
                out = self._dataset[item]
                read.append(np.size(out))
                return out

            def __array__(self, *args, **kwargs):
                read.append(self._dataset.size)
                return np.asarray(self._dataset, *args, **kwargs)

        class _CountingFile:
            """An open file which hands out the counting array."""

            def __init__(self, h5):
                self._h5 = h5

            def __getattr__(self, name):
                return getattr(self._h5, name)

            def __getitem__(self, item):
                out = self._h5[item]
                return _CountingDataset(out) if item == "mtx" else out

        with h5py.File(mtx_h5_path, "r") as h5:
            stored_size = h5["mtx"].size
            patch = _get_mtx_patch(
                _CountingFile(h5),
                attr_cls=FebusMTXAttrs,
                select_kwargs={"distance": (60, 70)},
            )
        assert max(read) == patch.data.size < stored_size

    def test_mtx_must_be_three_dimensional(self, tmp_path):
        """Non-3D MTX datasets should fail with explicit validation."""
        path = tmp_path / "two_dimensional_C1.mtx.h5"
        _write_mtx_h5(path)
        with h5py.File(path, "a") as h5:
            del h5["mtx"]
            h5.create_dataset("mtx", data=np.ones((2, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="expected 'mtx' to be 3D, got 2D"):
            FebusMTXH5V1().read(path)


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
        """Ensure chunk(time=None) merges both g1 files into one patch.

        The files carry per-file attrs (temperature, freqoffset) that
        differ, so conflict="keep_first" is required — matching what an
        in-memory spool of the same patches requires. (The old HDF5 index
        silently dropped these attrs, which masked the conflict for
        directory spools.)
        """
        spool = dc.spool(g1_two_file_directory)
        # These weren't directly adjacent files so we adjust the tolerance.
        match = "There is a gap in the patch along dimension time"
        with pytest.warns(UserWarning, match=match):
            merged = spool.chunk(time=None, tolerance=3, conflict="keep_first")
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
