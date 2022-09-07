"""
Tests for reading silixa TDMS file format.
"""

from pathlib import Path

import pytest

import dascore
from dascore.constants import REQUIRED_DAS_ATTRS
from dascore.core.schema import PatchFileSummary
from dascore.io.tdms.core import TDMSFormatterV4713
from dascore.utils.downloader import fetch
from dascore.utils.misc import register_func

TDMS_PATHS = []


@pytest.fixture()
@register_func(TDMS_PATHS)
def TDMS_das_example_path1():
    """Get path for test file"""
    file_path = fetch("sample_tdms_file_v4713.tdms")
    return Path(file_path)


@pytest.fixture()
@register_func(TDMS_PATHS)
def TDMS_das_example_path2():
    """Get path for test file"""
    file_path = fetch("iDAS005_tdms_example.626.tdms")
    return Path(file_path)


@pytest.fixture(params=TDMS_PATHS)
def tdms_das_example_path(request):
    """Get path for test file"""
    return request.getfixturevalue(request.param)


@pytest.fixture()
def tdms_das_patch(tdms_das_example_path):
    """Make patch for test data"""
    ft = TDMSFormatterV4713()
    stream_data = ft.read(tdms_das_example_path)
    return stream_data[0]


class TestReadTDMS:
    """Tests for reading the TDMS format."""

    def test_type(self, tdms_das_patch):
        """Ensure the expected type is returned."""
        assert isinstance(tdms_das_patch, dascore.Patch)

    def test_attributes(self, tdms_das_patch):
        """Ensure a few of the expected attrs exist in array."""
        attrs = tdms_das_patch.attrs
        expected_attrs = {"time_min", "time_max", "distance_min", "data_units"}
        assert set(expected_attrs).issubset(set(attrs))

    def test_has_required_attrs(self, tdms_das_patch):
        """ "Ensure the required das attrs are found"""
        assert set(REQUIRED_DAS_ATTRS).issubset(set(tdms_das_patch.attrs))

    def test_coord_attr_time_equal(self, tdms_das_patch):
        """The time reported in the attrs and coords should match"""
        attr_time = tdms_das_patch.attrs["time_max"]
        coord_time = tdms_das_patch.coords["time"].max()
        assert attr_time == coord_time

    def test_time_dist_slice(self, tdms_das_patch, tdms_das_example_path):
        """Ensure slicing distance and time works from read func."""
        time_array = tdms_das_patch.coords["time"]
        dist_array = tdms_das_patch.coords["distance"]
        t1, t2 = time_array[10], time_array[40]
        d1, d2 = dist_array[10], dist_array[40]

        patch = TDMSFormatterV4713().read(
            tdms_das_example_path, time=(t1, t2), distance=(d1, d2)
        )[0]
        attrs, coords = patch.attrs, patch.coords
        assert attrs["time_min"] == coords["time"].min() == t1
        assert attrs["time_max"] == coords["time"].max() == t2
        assert attrs["distance_min"] == coords["distance"].min() == d1
        assert attrs["distance_max"] == coords["distance"].max() == d2


class TestGetFormatTDMS:
    """Tests for function to determine if a file is a silixa file."""

    def test_not_silixa_not_tdms(self, dummy_text_file):
        """Test for not a silixa tdms file."""
        parser = TDMSFormatterV4713()
        assert not parser.get_format(dummy_text_file)
        assert not parser.get_format(dummy_text_file.parent)

    def test_silixa_get_format(self, tdms_das_example_path):
        """Test for a silixa tdms file."""
        parser = TDMSFormatterV4713()
        assert parser.get_format(tdms_das_example_path)
        format_name, format_version = parser.get_format(tdms_das_example_path)
        assert format_name == parser.name


class TestScanTDMS:
    """Tests for scanning silixa file."""

    @pytest.fixture
    def tdms_scan(self, tdms_das_example_path):
        """Scan test tdms file, return summary info."""
        out = TDMSFormatterV4713().scan(tdms_das_example_path)
        return out

    def test_scanning(self, tdms_scan):
        """Tests for getting summary info from silixa data."""
        assert isinstance(tdms_scan, list)
        assert len(tdms_scan) == 1
        assert isinstance(tdms_scan[0], PatchFileSummary)

    def test_dims(self, tdms_scan):
        """Ensure dims are populated."""
        for tdms in tdms_scan:
            assert tdms.dims
