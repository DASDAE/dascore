"""
Tests for the G1 csv files.

Note: These can't just be includes in the common_io tests because these
files only have one timestep.
"""

import shutil
from io import BytesIO, StringIO
from pathlib import Path

import pytest

import dascore as dc
from dascore.io.febus.core import FebusG1CSV1
from dascore.io.febus.g1utils import _get_g1_coords_and_attrs, _is_g1_file
from dascore.utils.downloader import fetch

g1_files = (
    "febg1_C1_2023-05-10T12.25.03+0000.bsl",
    "febg1_C1_2023-05-10T12.27.33+0000.bsl",
    # "febus-g1-specta_C1_2023-11-30T15.13.14+0000.mtx"
)


@pytest.fixture(scope="module", params=g1_files)
def g1_path(request):
    """
    The paths to the g1 times series files.
    """
    return fetch(request.param)


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
        """Ensure attrs can be extracted from G1 files."""
        spool = dc.read(g1_path)
        assert len(spool) == 1
        patch = spool[0]
        assert isinstance(patch, dc.Patch)

    def test_read_mtx_raises_not_implemented(self):
        """Ensure reading g1 spectra (.mtx) files raises NotImplementedError."""
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
        with pytest.raises(NotImplementedError, match="cannot yet parse spectra"):
            _get_g1_coords_and_attrs(resource)


class TestMisc:
    """Misc integration tests for G1 files."""

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
        merged = spool.chunk(time=None, tolerance=3)
        assert len(merged) == 1
