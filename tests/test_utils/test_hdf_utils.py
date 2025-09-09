"""Tests for hdf5 utils."""

from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
import pytest
import tables
from tables.exceptions import ClosedNodeError

import dascore as dc
from dascore.exceptions import InvalidFileHandlerError
from dascore.utils.downloader import fetch
from dascore.utils.hdf5 import (
    HDFPatchIndexManager,
    PyTablesWriter,
    extract_h5_attrs,
    h5_matches_structure,
    open_hdf5_file,
)


@pytest.fixture(scope="class")
def h5_example_file():
    """Get an example file."""
    path = fetch("gdr_1.h5")
    with h5py.File(path, "r") as fi:
        yield fi
    fi.close()


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
        """Return a tables file handler in read mode."""
        with tables.open_file(simple_hdf_path, mode="r") as fi:
            yield fi

    @pytest.fixture()
    def simple_hdf_file_handler_append(self, simple_hdf_path):
        """Return a tables file handler in append mode."""
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
        """If write is requested but read handler is provided an error should raise."""
        with pytest.raises(InvalidFileHandlerError, match="but mode"):
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
        """Empty dims should convert to empty string."""
        df = dc.scan_to_df(random_spool).assign(
            dims=[() for _ in range(len(random_spool))],
        )
        index_manager.write_update(df)

    def test_has_content(self, index_manager_with_content, tmp_path):
        """`has_index` should return True if data have been written else False."""
        assert index_manager_with_content.has_index
        # create hdf5 file with no index
        path = tmp_path / "empty.h5"
        df = pd.DataFrame([1, 2, 3], columns=["first"])
        df.to_hdf(str(path), key="df")
        # assert it doesn't have an index
        assert not HDFPatchIndexManager(path).has_index

    def test_closed_node_error(self, index_manager_with_content, monkeypatch):
        """
        Test for when the file fails to open. This is a bit contrived but the
        closed node issues does happen sometimes in multiple thread environments.
        """
        failed_count = 0
        old_func = pd.read_hdf

        def _new_read(*args, **kwargs):
            nonlocal failed_count
            if failed_count < 1:
                failed_count += 1
                raise ClosedNodeError("Simulated failed node opening")
            else:
                return old_func(*args, **kwargs)

        monkeypatch.setattr(pd, "read_hdf", _new_read)

        df = index_manager_with_content.get_index()
        assert len(df)

        # now insure the exception propagates
        failed_count = 0
        monkeypatch.setattr(index_manager_with_content, "_max_retries", -1)

        with pytest.raises(ClosedNodeError):
            index_manager_with_content.get_index()

    def test_metadata_created(self, tmp_path_factory):
        """Tests for getting info from a index that doesnt yet exist."""
        path = tmp_path_factory.mktemp("non_existent_index") / "index.hdf5"
        with tables.open_file(path, "w"):
            pass
        index = HDFPatchIndexManager(path)
        meta = index._read_metadata()
        assert meta is not None


class TestHDFReaders:
    """Tests for HDF5 readers."""

    def test_get_handle(self, tmp_path_factory):
        """Ensure we can get a handle with the class."""
        path = tmp_path_factory.mktemp("hdf_handle_test") / "test_file.h5"
        handle = PyTablesWriter.get_handle(path)
        assert isinstance(handle, tables.File)
        handle_2 = PyTablesWriter.get_handle(handle)
        assert isinstance(handle_2, tables.File)


class TestH5MatchesStructure:
    """Tests for the h5 matches structure function."""

    def test_has_structure(self, h5_example_file):
        """Ensure the simple structure match works."""
        struct = (
            "DasMetadata",
            "DasMetadata/Cable/Fiber",
            "DasMetadata/Cable/Fiber.FiberComment",
        )
        assert h5_matches_structure(h5_example_file, struct)

    def test_missing_group(self, h5_example_file):
        """Ensure false is returned when a group is missing."""
        struct = ("DasBonkersData",)
        assert not h5_matches_structure(h5_example_file, struct)

    def test_missing_attribute(self, h5_example_file):
        """Ensure false is returned when an attribute is missing."""
        struct = ("DasMetadata/Cable.HasPolkaDots",)
        assert not h5_matches_structure(h5_example_file, struct)


class TestExtractH5Attrs:
    """Extract H5 attributes."""

    def test_extract_existing_attrs(self, h5_example_file):
        """Test extracting existing attributes."""
        acq = "DasMetadata/Interrogator/Acquisition"
        map_names = {
            "DasMetadata.Country": "country",
            f"{acq}.AcquisitionSampleRate": "sample_rate",
        }
        out = extract_h5_attrs(h5_example_file, map_names)
        assert "sample_rate" in out and "country" in out

    def test_bad_attr_raises(self, h5_example_file):
        """A bad attibute should raise a KeyError."""
        map_name = {
            "DasMetadata.CountryBumpkins": "country",
        }
        with pytest.raises(KeyError):
            extract_h5_attrs(h5_example_file, map_name)
