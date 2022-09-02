"""
Tests for hdf5 utils.
"""
from pathlib import Path

import pytest
import tables

from dascore.exceptions import InvalidFileHandler
from dascore.utils.hdf5 import open_hdf5_file


class TestGetHDF5Handlder:
    """Tests for opening an HDF5 file from various inputs."""

    @pytest.fixture()
    def simple_hdf_path(self, tmp_path_factory):
        """Create a hdf5 file in a temporary directory."""
        new = tmp_path_factory.mktemp("dummy_hdf5") / "test.h5"
        with tables.open_file(str(new), mode="w") as fi:
            bob = fi.create_group(fi.root, name="bob")
            bob._v_attrs["lightening"] = 1
        return Path(new)

    @pytest.fixture()
    def simple_hdf_file_handler_read(self, simple_hdf_path):
        """Return a tables file handler in read mode"""
        with tables.open_file(simple_hdf_path, mode="r") as fi:
            yield fi

    @pytest.fixture()
    def simple_hdf_file_handler_append(self, simple_hdf_path):
        """Return a tables file handler in append mode"""
        with tables.open_file(simple_hdf_path, mode="a") as fi:
            yield fi

    def test_path_read(self, simple_hdf_path):
        """Ensure passing a path works."""
        with open_hdf5_file(simple_hdf_path) as fi:
            assert isinstance(fi, tables.File)

    def test_table_file_read(self, simple_hdf_file_handler_read):
        """Ensure a tables file also works."""
        with open_hdf5_file(simple_hdf_file_handler_read) as fi:
            assert isinstance(fi, tables.File)

    def test_read_only_filehandle_raises(self, simple_hdf_file_handler_read):
        """
        If write is requested but read handler is provided an error should raise.
        """
        with pytest.raises(InvalidFileHandler, match="but mode"):
            with open_hdf5_file(simple_hdf_file_handler_read, mode="w"):
                pass

    def test_read_with_write_filehandler(self, simple_hdf_file_handler_append):
        """
        Ensure a file handler is returned if read mode is requested but write
        mode is provided. This works because write is a superset of read
        functionality.
        """
        with open_hdf5_file(simple_hdf_file_handler_append, mode="r") as fi:
            assert isinstance(fi, tables.File)
