"""
Tests for hdf5 utils.
"""
from pathlib import Path

import pandas as pd
import pytest
import tables

import dascore as dc
from dascore.exceptions import InvalidFileHandler
from dascore.utils.hdf5 import HDFPatchIndexManager, open_hdf5_file


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


class TestHDFPatchIndexManager:
    """Tests for the HDF5 index manager."""

    @pytest.fixture
    def index_manager(self, tmp_path_factory):
        """Create a new index."""
        path = Path(tmp_path_factory.mktemp("example")) / ".index"
        return HDFPatchIndexManager(path)

    @pytest.fixture
    def index_manager_with_content(self, index_manager, random_spool):
        """Add content to the index manager."""
        spool_df = dc.scan_to_df(random_spool)
        index_manager.write_update(spool_df)
        return index_manager

    def test_extra_columns(self, index_manager, random_spool):
        """
        Only the columns used for indexing should be kept, extras discarded.

        Here we include a column with types that can't be serialized. If the
        write_update works the test passes.
        """
        df = dc.scan_to_df(random_spool).assign(
            bad_cols=[[] for _ in range(len(random_spool))]
        )
        index_manager.write_update(df)

    def test_empty_tuple(self, index_manager, random_spool):
        """
        Empty dims should convert to empty string.
        """
        df = dc.scan_to_df(random_spool).assign(
            dims=[() for _ in range(len(random_spool))],
        )
        index_manager.write_update(df)

    def test_has_content(self, index_manager_with_content, tmp_path):
        """`has_index` should return True if data have been writen else False"""
        assert index_manager_with_content.has_index
        # create hdf5 file with no index
        path = tmp_path / "empty.h5"
        df = pd.DataFrame([1, 2, 3], columns=["first"])
        df.to_hdf(str(path), "df")
        # assert it doesn't have an index
        assert not HDFPatchIndexManager(path).has_index
