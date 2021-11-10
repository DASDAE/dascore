"""
Tests for reading terra15 file formats.
"""
import numpy as np
import pytest
import tables as tb

import fios
from fios.constants import REQUIRED_DAS_ATTRS
from fios.io.terra15.ver2 import _is_terra15_v2, _scan_terra15_v2
from fios.utils.downloader import fetch


class TestReadTerra15:
    """Tests for reading the terra15 format."""

    def test_type(self, terra15_das_array):
        """Ensure the expected type is returned."""
        assert isinstance(terra15_das_array, fios.Patch)

    def test_attributes(self, terra15_das_array):
        """Ensure the expected attrs exist in array."""
        attrs = terra15_das_array.attrs
        expected_attrs = {"dT", "dx", "nx", "nT", "recorder_id"}
        assert set(expected_attrs).issubset(set(attrs))

    def test_has_required_attrs(self, terra15_das_array):
        """ "Ensure the required das attrs are found"""
        assert set(REQUIRED_DAS_ATTRS).issubset(set(terra15_das_array.attrs))


class TestIsTerra15:
    """Tests for function to determine if a file is a terra15 file."""

    @pytest.fixture
    def generic_hdf5(self, tmp_path):
        """Create a generic df5 file."""
        parent = tmp_path / "sum"
        parent.mkdir()
        path = parent / "simple.hdf5"

        with tb.open_file(str(path), "w") as fi:
            group = fi.create_group("/", "bob")
            fi.create_carray(group, "data", obj=np.random.rand(10))
        return path

    def test_version_2(self):
        """Ensure version two is recognized."""
        path = fetch("terra15_v2_das_1_trimmed.hdf5")
        assert _is_terra15_v2(path)

    def test_not_terra15_not_df5(self, dummy_text_file):
        """Test for not even a hdf5 file."""
        assert not _is_terra15_v2(dummy_text_file)
        assert not _is_terra15_v2(dummy_text_file.parent)

    def test_hdf5file_not_terra15(self, generic_hdf5):
        """Assert that the generic hdf5 file is not a terra15."""
        _is_terra15_v2(generic_hdf5)
        assert not _is_terra15_v2(generic_hdf5)


class TestScanTerra15:
    """Tests for scanning terra15 file."""

    def test_scanning(self, terra15_das_array, terra15_path):
        """Tests for getting summary info from terra15 data."""
        out = _scan_terra15_v2(terra15_path)
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], dict)
