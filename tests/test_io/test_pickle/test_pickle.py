"""Tests for reading/writing pickles."""

from __future__ import annotations

import pickle
from io import BytesIO

import pytest

import dascore as dc
from dascore.io.pickle.core import PickleIO


@pytest.fixture(scope="session")
def pickle_patch_path(tmp_path_factory, random_patch):
    """Pickle a patch and return the path."""
    path = tmp_path_factory.mktemp("pickle_test") / "test.pkl"
    random_patch.io.write(path, "pickle")
    return path


class TestGetFormat:
    """Test detecting pickle format."""

    def test_detect_file(self, pickle_patch_path):
        """Simple test on output of is pickle."""
        parser = PickleIO()
        out = parser.get_format(pickle_patch_path)
        assert out
        assert out[0] == "PICKLE"

    def test_not_pickle(self, generic_hdf5):
        """Ensure non-pickle file returns false."""
        parser = PickleIO()
        assert not parser.get_format(generic_hdf5)

    def test_read_pickle(self, pickle_patch_path, random_patch):
        """Ensure a pickle file can be read."""
        out = dc.read(pickle_patch_path)
        assert isinstance(out, dc.BaseSpool)
        assert len(out) == 1
        assert isinstance(out[0], dc.Patch)
        assert random_patch == out[0]

    def test_file_not_there(self):
        """Get format should return false if the file doesn't exist."""
        parser = PickleIO()
        assert not parser.get_format("surely_not_a_file_that_exists")

    def test_get_format_binary_file_too_small(self):
        """Ensure a file which is too small returns false."""
        bio = BytesIO()
        bio.write(b"one")
        fio = PickleIO()
        bio.seek(0)
        assert not fio.get_format(bio)

    def test_has_dascore_not_pickle(self):
        """Test a buffer which has dascore but isnt a DC pickle."""
        bio = BytesIO()
        bio.write(b"dascore.core Spool")
        fio = PickleIO()
        bio.seek(0)
        assert not fio.get_format(bio)

    def test_bytes_io_valid_pickle(self, random_patch):
        """Test a valid pickle in bytes io."""
        fio = PickleIO()
        bio = BytesIO()
        pickle.dump(random_patch, bio)
        bio.seek(0)
        fmt = fio.get_format(bio)
        assert "PICKLE" in fmt
        spool = fio.read(bio)
        assert spool[0] == random_patch


class TestScan:
    """Tests for scanning pickle files/."""

    comp_attrs = (
        "data_type",
        "data_units",
        "time_step",
        "time_min",
        "time_max",
        "distance_min",
        "distance_max",
        "distance_step",
        "tag",
        "network",
    )

    def test_scan_attrs_eq_read_attrs(self, pickle_patch_path):
        """Ensure read/scan produce the same attrs."""
        scan_list = dc.scan(pickle_patch_path)
        patch_attrs_list = [x.attrs for x in dc.read(pickle_patch_path)]

        for scan_attrs, patch_attrs in zip(scan_list, patch_attrs_list):
            for attr in self.comp_attrs:
                scan_attr = getattr(scan_attrs, attr)
                patch_attr = getattr(patch_attrs, attr)
                assert scan_attr == patch_attr
