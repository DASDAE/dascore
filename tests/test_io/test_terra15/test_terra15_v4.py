"""
Tests for reading terra15 format, version 4.
"""
import numpy as np
import pytest

import dascore
from dascore.constants import REQUIRED_DAS_ATTRS
from dascore.core.schema import PatchFileSummary
from dascore.io.terra15.core import Terra15FormatterV4


class TestReadTerra15:
    """Tests for reading the terra15 format."""

    def test_type(self, terra15_das_patch):
        """Ensure the expected type is returned."""
        assert isinstance(terra15_das_patch, dascore.Patch)

    def test_attributes(self, terra15_das_patch):
        """Ensure a few of the expected attrs exist in array."""
        attrs = terra15_das_patch.attrs
        expected_attrs = {"time_min", "time_max", "distance_min", "data_units"}
        assert set(expected_attrs).issubset(set(attrs))

    def test_has_required_attrs(self, terra15_das_patch):
        """ "Ensure the required das attrs are found"""
        assert set(REQUIRED_DAS_ATTRS).issubset(set(terra15_das_patch.attrs))

    def test_coord_attr_time_equal(self, terra15_das_patch):
        """The time reported in the attrs and coords should match"""
        attr_time = terra15_das_patch.attrs["time_max"]
        coord_time = terra15_das_patch.coords["time"].max()
        assert attr_time == coord_time

    def test_time_dist_slice(self, terra15_das_patch, terra15_das_example_path):
        """Ensure slicing distance and time works from read func."""
        time_array = terra15_das_patch.coords["time"]
        dist_array = terra15_das_patch.coords["distance"]
        t1, t2 = time_array[10], time_array[40]
        d1, d2 = dist_array[10], dist_array[40]

        patch = Terra15FormatterV4().read(
            terra15_das_example_path, time=(t1, t2), distance=(d1, d2)
        )[0]
        attrs, coords = patch.attrs, patch.coords
        assert attrs["time_min"] == coords["time"].min() == t1
        assert attrs["time_max"] == coords["time"].max() == t2
        assert attrs["distance_min"] == coords["distance"].min() == d1
        assert attrs["distance_max"] == coords["distance"].max() == d2

    def test_no_arrays_in_attrs(self, terra15_das_patch):
        """
        Ensure that the attributes are not arrays.
        Originally, attrs like time_min can be arrays with empty shapes.
        """
        for key, val in terra15_das_patch.attrs.items():
            assert not isinstance(val, np.ndarray)


class TestIsTerra15:
    """Tests for function to determine if a file is a terra15 file."""

    def test_format_and_version(self, terra15_das_example_path):
        """Ensure version two is recognized."""
        format, version = Terra15FormatterV4().get_format(terra15_das_example_path)
        assert format == Terra15FormatterV4.name
        assert version == Terra15FormatterV4.version

    def test_not_terra15_not_df5(self, dummy_text_file):
        """Test for not even a hdf5 file."""
        parser = Terra15FormatterV4()
        assert not parser.get_format(dummy_text_file)
        assert not parser.get_format(dummy_text_file.parent)

    def test_hdf5file_not_terra15(self, generic_hdf5):
        """Assert that the generic hdf5 file is not a terra15."""
        parser = Terra15FormatterV4()
        assert not parser.get_format(generic_hdf5)


class TestScanTerra15:
    """Tests for scanning terra15 file."""

    def test_scanning(self, terra15_das_patch, terra15_das_example_path):
        """Tests for getting summary info from terra15 data."""
        parser = Terra15FormatterV4()
        out = parser.scan(terra15_das_example_path)
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], PatchFileSummary)

        data = out[0]
        assert data.file_format == parser.name
        assert data.file_version == parser.version


class TestTerra15Unfinished:
    """Test for reading files with zeroes filled at the end."""

    @pytest.fixture(scope="class")
    def patch_unfinished(self, terra15_das_unfinished_path):
        """Return the patch with zeroes at the end."""
        parser = Terra15FormatterV4()
        out = parser.read(terra15_das_unfinished_path)[0]
        return out

    def test_zeros_gone(self, patch_unfinished):
        """No zeros should exist in the data."""
        data = patch_unfinished.data
        all_zero_rows = np.all(data == 0, axis=1)
        assert not np.any(all_zero_rows)

    def test_monotonic_time(self, patch_unfinished):
        """Ensure the time is increasing."""
        time = patch_unfinished.coords["time"]
        assert np.all(np.diff(time) >= np.timedelta64(0, "s"))
