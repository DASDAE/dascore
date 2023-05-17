"""
Tests for QuantXV2 format
"""
import numpy as np
import pytest

import dascore as dc
from dascore.constants import REQUIRED_DAS_ATTRS
from dascore.core.schema import PatchFileSummary
from dascore.io.core import read
from dascore.io.quantx.core import QuantXV2
from dascore.utils.downloader import fetch
from dascore.utils.misc import register_func

PATCH_FIXTURES = []


@pytest.fixture(scope="session")
def quantx_v2_example_path():
    """Return the path to the example QuantXV2 file."""
    out = fetch("opta_sense_quantx_v2.h5")
    assert out.exists()
    return out


@pytest.fixture(scope="session")
@register_func(PATCH_FIXTURES)
def quantx_v2_das_patch(quantx_v2_example_path):
    """Read the QuantXV2 data, return contained DataArray"""
    out = read(quantx_v2_example_path, "quantx")[0]
    attr_time = out.attrs["time_max"]
    coord_time = out.coords["time"].max()
    assert attr_time == coord_time
    return out


class TestReadQuantXV2:
    """Tests for reading the QuantXV2 format."""

    def test_type(self, quantx_v2_das_patch):
        """Ensure the expected type is returned."""
        assert isinstance(quantx_v2_das_patch, dc.Patch)

    def test_attributes(self, quantx_v2_das_patch):
        """Ensure a few of the expected attrs exist in array."""
        attrs = dict(quantx_v2_das_patch.attrs)
        expected_attrs = {"time_min", "time_max", "distance_min", "data_units"}
        assert set(expected_attrs).issubset(set(attrs))

    def test_has_required_attrs(self, quantx_v2_das_patch):
        """Ensure the required das attrs are found."""
        assert set(REQUIRED_DAS_ATTRS).issubset(set(dict(quantx_v2_das_patch.attrs)))

    def test_coord_attr_time_equal(self, quantx_v2_das_patch):
        """The time reported in the attrs and coords should match."""
        attr_time_max = quantx_v2_das_patch.attrs["time_max"]
        coord_time_max = quantx_v2_das_patch.coords["time"].max()
        assert attr_time_max == coord_time_max
        attr_time_min = quantx_v2_das_patch.attrs["time_min"]
        coord_time_min = quantx_v2_das_patch.coords["time"].min()
        assert attr_time_min == coord_time_min

    def test_precision_of_time_array(self, quantx_v2_das_patch):
        """
        Ensure the time array is in ns, not native ms, in order to
        be consistent with other patches.
        """
        time = quantx_v2_das_patch.coords["time"]
        dtype = time.dtype
        assert "[ns]" in str(dtype)

    def test_time_dist_slice(self, quantx_v2_das_patch, quantx_v2_example_path):
        """Ensure slicing distance and time works from read func."""
        time_array = quantx_v2_das_patch.coords["time"]
        dist_array = quantx_v2_das_patch.coords["channel_number"]
        t1, t2 = time_array[10], time_array[40]
        d1, d2 = dist_array[10], dist_array[40]
        patch = QuantXV2().read(
            quantx_v2_example_path, time=(t1, t2), distance=(d1, d2)
        )[0]
        attrs, coords = patch.attrs, patch.coords
        assert attrs["time_min"] == coords["time"].min() == t1
        assert (attrs["time_max"] - t2) < (attrs["d_time"] / 4)
        assert attrs["distance_min"] == coords["channel_number"].min() == d1
        assert attrs["distance_max"] == coords["channel_number"].max() == d2

    def test_no_arrays_in_attrs(self, quantx_v2_das_patch):
        """
        Ensure that the attributes are not arrays.
        Originally, attrs like time_min can be arrays with empty shapes.
        """
        for key, val in quantx_v2_das_patch.attrs.items():
            assert not isinstance(val, np.ndarray)


class TestIsQuantXV2:
    """Tests for function to determine if a file is an QuantXV2 file."""

    def test_format_and_version(self, quantx_v2_example_path):
        """Ensure version two is recognized."""
        format, version = QuantXV2().get_format(quantx_v2_example_path)
        assert format == QuantXV2.name
        assert version == QuantXV2.version

    def test_not_QuantXV2_not_df5(self, dummy_text_file):
        """Test for not even a hdf5 file."""
        parser = QuantXV2()
        assert not parser.get_format(dummy_text_file)
        assert not parser.get_format(dummy_text_file.parent)

    def test_hdf5file_not_quantx(self, generic_hdf5):
        """Assert that the generic hdf5 file is not a QuantXV2."""
        parser = QuantXV2()
        assert not parser.get_format(generic_hdf5)


class TestScanQuantXV2:
    """Tests for scanning QuantXV2 file."""

    def test_scanning(self, quantx_v2_example_path):
        """Tests for getting summary info from QuantXV2 data."""
        parser = QuantXV2()
        out = parser.scan(quantx_v2_example_path)
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], PatchFileSummary)

        data = out[0]
        assert data.file_format == parser.name
        assert data.file_version == parser.version
