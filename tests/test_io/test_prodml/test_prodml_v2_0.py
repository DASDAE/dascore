"""
Tests for ProdML Version 2.0
"""

import numpy as np

import dascore as dc
from dascore.constants import REQUIRED_DAS_ATTRS
from dascore.core.schema import PatchFileSummary
from dascore.io.prodml import ProdMLV2_0


class TestReadProdML2_0Patch:
    """Tests for reading the prodML format."""

    def test_type(self, prodml_v2_0_patch):
        """Ensure the expected type is returned."""
        assert isinstance(prodml_v2_0_patch, dc.Patch)

    def test_attributes(self, prodml_v2_0_patch):
        """Ensure a few of the expected attrs exist in array."""
        attrs = dict(prodml_v2_0_patch.attrs)
        expected_attrs = {"time_min", "time_max", "distance_min", "data_units"}
        assert set(expected_attrs).issubset(set(attrs))

    def test_has_required_attrs(self, prodml_v2_0_patch):
        """ "Ensure the required das attrs are found"""
        assert set(REQUIRED_DAS_ATTRS).issubset(set(dict(prodml_v2_0_patch.attrs)))

    def test_coord_attr_time_equal(self, prodml_v2_0_patch):
        """The time reported in the attrs and coords should match"""
        attr_time = prodml_v2_0_patch.attrs["time_max"]
        coord_time = prodml_v2_0_patch.coords["time"].max()
        assert attr_time == coord_time

    def test_read_with_limits(self, prodml_v2_0_patch, prodml_v2_0_example_path):
        """If start/end time sare select the same patch ought to be returned."""
        attrs = prodml_v2_0_patch.attrs
        time = (attrs["time_min"], attrs["time_max"])
        dist = (attrs["distance_min"], attrs["distance_max"])
        patch = ProdMLV2_0().read(
            prodml_v2_0_example_path,
            time=time,
            distance=dist,
        )[0]
        assert attrs["time_max"] == patch.attrs["time_max"]

    def test_time_dist_slice(self, prodml_v2_0_patch, prodml_v2_0_example_path):
        """Ensure slicing distance and time works from read func."""
        time_array = prodml_v2_0_patch.coords["time"]
        dist_array = prodml_v2_0_patch.coords["distance"]
        t1, t2 = time_array[10], time_array[40]
        d1, d2 = dist_array[10], dist_array[40]
        patch = ProdMLV2_0().read(
            prodml_v2_0_example_path, time=(t1, t2), distance=(d1, d2)
        )[0]
        attrs, coords = patch.attrs, patch.coords
        assert attrs["time_min"] == coords["time"].min() == t1
        assert attrs["time_max"] == coords["time"].max()
        # since we use floats sometimes this are a little off.
        assert (attrs["time_max"] - t2) < (attrs["d_time"] / 4)
        assert attrs["distance_min"] == coords["distance"].min() == d1
        assert attrs["distance_max"] == coords["distance"].max() == d2

    def test_one_sided_slice(self, prodml_v2_0_patch, prodml_v2_0_example_path):
        """Ensure slice can specify only one side."""
        time_array = prodml_v2_0_patch.coords["time"]
        dist_array = prodml_v2_0_patch.coords["distance"]
        t1 = time_array[10]
        d2 = dist_array[40]
        patch = ProdMLV2_0().read(
            prodml_v2_0_example_path, time=(t1, None), distance=(None, d2)
        )[0]
        attrs, coords = patch.attrs, patch.coords
        assert attrs["time_min"] == coords["time"].min() == t1
        assert attrs["time_max"] == coords["time"].max()
        assert attrs["time_min"] >= t1
        assert attrs["distance_max"] <= d2
        assert attrs["distance_max"] == coords["distance"].max() == d2

    def test_no_arrays_in_attrs(self, prodml_v2_0_patch):
        """
        Ensure that the attributes are not arrays.
        Originally, attrs like time_min can be arrays with empty shapes.
        """
        for key, val in prodml_v2_0_patch.attrs.items():
            assert not isinstance(val, np.ndarray)


class TestIsProdMLv2_0:
    """Tests for function to determine if a file is a ProdML v2.0 file."""

    def test_format_and_version(self, prodml_v2_0_example_path):
        """Ensure version returns correct information."""
        name, version = ProdMLV2_0().get_format(prodml_v2_0_example_path)
        assert (name, version) == (ProdMLV2_0.name, ProdMLV2_0.version)

    def test_not_prodML_not_h5(self, dummy_text_file):
        """Test for not even a hdf5 file."""
        parser = ProdMLV2_0()
        assert not parser.get_format(dummy_text_file)
        assert not parser.get_format(dummy_text_file.parent)

    def test_hdf5file_not_prodml(self, terra15_v5_path):
        """Assert that the terra15 hdf5 file is not a prodml."""
        parser = ProdMLV2_0()
        assert not parser.get_format(terra15_v5_path)


class TestScanProdMLV2_0:
    """Tests for scanning prodML file."""

    def test_basic_scan(self, prodml_v2_0_example_path):
        """Tests for getting summary info from ProdML data."""
        parser = ProdMLV2_0()
        out = parser.scan(prodml_v2_0_example_path)
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], PatchFileSummary)

        data = out[0]
        assert data.file_format == parser.name
        assert data.file_version == parser.version
