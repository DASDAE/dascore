"""
Tests for OptasenseV2 format
"""
import numpy as np
import pytest

import dascore as dc
from dascore.constants import REQUIRED_DAS_ATTRS
from dascore.core.schema import PatchFileSummary
from dascore.io.core import read
from dascore.io.optasense.core import OptasenseV2
from dascore.utils.downloader import fetch
from dascore.utils.misc import register_func

PATCH_FIXTURES = []


@pytest.fixture(scope="session")
def optasense_v2_example_path():
    """Return the path to the example OptasenseV2 file."""
    out = fetch("opta_sense_quantx_v2.h5")
    assert out.exists()
    return out


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def optasense_v2_das_patch(optasense_v2_example_path):
    """Read the OptasenseV2 data, return contained DataArray"""
    out = read(optasense_v2_example_path, "optasense")[0]
    attr_time = out.attrs["time_max"]
    coord_time = out.coords["time"].max()
    assert attr_time == coord_time
    return out


class TestReadOptasenseV2:
    """Tests for reading the OptasenseV2 format."""

    def test_type(self, optasense_v2_das_patch):
        """Ensure the expected type is returned."""
        assert isinstance(optasense_v2_das_patch, dc.Patch)

    def test_attributes(self, optasense_v2_das_patch):
        """Ensure a few of the expected attrs exist in array."""
        attrs = dict(optasense_v2_das_patch.attrs)
        expected_attrs = {"time_min", "time_max", "distance_min", "data_units"}
        assert set(expected_attrs).issubset(set(attrs))

    def test_has_required_attrs(self, optasense_v2_das_patch):
        """Ensure the required das attrs are found."""
        assert set(REQUIRED_DAS_ATTRS).issubset(set(dict(optasense_v2_das_patch.attrs)))

    def test_coord_attr_time_equal(self, optasense_v2_das_patch):
        """The time reported in the attrs and coords should match."""
        attr_time = optasense_v2_das_patch.attrs["time_max"]
        coord_time = optasense_v2_das_patch.coords["time"].max()
        assert attr_time == coord_time

    def test_time_dist_slice(self, optasense_v2_das_patch, optasense_v2_example_path):
        """Ensure slicing distance and time works from read func."""
        time_array = optasense_v2_das_patch.coords["time"]
        dist_array = optasense_v2_das_patch.coords["channel_number"]
        t1, t2 = time_array[10], time_array[40]
        d1, d2 = dist_array[10], dist_array[40]
        patch = OptasenseV2().read(
            optasense_v2_example_path, time=(t1, t2), distance=(d1, d2)
        )[0]
        attrs, coords = patch.attrs, patch.coords
        assert attrs["time_min"] == coords["time"].min() == t1
        assert (attrs["time_max"] - t2) < (attrs["d_time"] / 4)
        assert attrs["distance_min"] == coords["channel_number"].min() == d1
        assert attrs["distance_max"] == coords["channel_number"].max() == d2

    def test_no_arrays_in_attrs(self, optasense_v2_das_patch):
        """
        Ensure that the attributes are not arrays.
        Originally, attrs like time_min can be arrays with empty shapes.
        """
        for key, val in optasense_v2_das_patch.attrs.items():
            assert not isinstance(val, np.ndarray)


class TestIsOptasenseV2:
    """Tests for function to determine if a file is an OptasenseV2 file."""

    def test_format_and_version(self, optasense_v2_example_path):
        """Ensure version two is recognized."""
        format, version = OptasenseV2().get_format(optasense_v2_example_path)
        assert format == OptasenseV2.name
        assert version == OptasenseV2.version

    def test_not_optasensev2_not_df5(self, dummy_text_file):
        """Test for not even a hdf5 file."""
        parser = OptasenseV2()
        assert not parser.get_format(dummy_text_file)
        assert not parser.get_format(dummy_text_file.parent)

    def test_hdf5file_not_optasensev2(self, generic_hdf5):
        """Assert that the generic hdf5 file is not a OptasenseV2."""
        parser = OptasenseV2()
        assert not parser.get_format(generic_hdf5)


class TestScanOptasenseV2:
    """Tests for scanning OptasenseV2 file."""

    def test_scanning(self, optasense_v2_das_patch, optasense_v2_example_path):
        """Tests for getting summary info from OptasenseV2 data."""
        parser = OptasenseV2()
        out = parser.scan(optasense_v2_example_path)
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], PatchFileSummary)

        data = out[0]
        assert data.file_format == parser.name
        assert data.file_version == parser.version
