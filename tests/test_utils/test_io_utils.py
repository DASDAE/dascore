"""Tests for IO utilities."""
from __future__ import annotations

from io import BufferedReader, BufferedWriter
from pathlib import Path

import pytest
from tables import File

from dascore.utils.io import (
    BinaryReader,
    BinaryWriter,
    HDF5Reader,
    HDF5Writer,
    IOResourceManager,
    get_handle_from_resource,
)


class _BadType:
    """A dummy type for testing."""


def _dummy_func(arg: str, arg2: _BadType) -> int:
    """A dummy function."""


class TestGetHandleFromResource:
    """Tests for getting the file handle from specific resources."""

    def test_bad_type(self):
        """
        In order to not break anything, unsupported types should just
        return the original argument.
        """
        out = get_handle_from_resource("here", _BadType)
        assert out == "here"

    def test_path_to_buffered_reader(self, tmp_path):
        """Ensure we get a reader from tmp path reader."""
        path = tmp_path / "test_read_buffer.txt"
        path.touch()
        handle = get_handle_from_resource(path, BinaryReader)
        assert isinstance(handle, BufferedReader)
        handle.close()

    def test_path_to_buffered_writer(self, tmp_path):
        """Ensure we get a reader from tmp path reader."""
        path = tmp_path / "test_buffered_writer.txt"
        handle = get_handle_from_resource(path, BinaryWriter)
        assert isinstance(handle, BufferedWriter)
        handle.close()

    def test_path_to_hdf5_reader(self, generic_hdf5):
        """Ensure we get a reader from tmp path reader."""
        handle = get_handle_from_resource(generic_hdf5, HDF5Reader)
        assert isinstance(handle, File)
        handle.close()

    def test_path_to_hdf5_writer(self, tmp_path):
        """Ensure we get a reader from tmp path reader."""
        path = tmp_path / "test_hdf_writer.h5"
        handle = get_handle_from_resource(path, HDF5Writer)
        assert isinstance(handle, File)

    def test_get_path(self, tmp_path):
        """Ensure we can get a path."""
        path = get_handle_from_resource(tmp_path, Path)
        assert isinstance(path, Path)

    def test_get_str(self, tmp_path):
        """Ensure we can get a string."""
        my_str = get_handle_from_resource(tmp_path, str)
        assert isinstance(my_str, str)

    def test_already_file_handle(self, tmp_path):
        """Ensure an input that is already the requested type works."""
        path = tmp_path / "pass_back.txt"
        with open(path, "wb") as fi:
            out = get_handle_from_resource(fi, BinaryWriter)
            assert out is fi

    def test_not_implemented(self):
        """Tests for raising not implemented errors for types not supported."""
        bad_instance = _BadType()
        with pytest.raises(NotImplementedError):
            get_handle_from_resource(bad_instance, BinaryReader)
        with pytest.raises(NotImplementedError):
            get_handle_from_resource(bad_instance, BinaryWriter)
        with pytest.raises(NotImplementedError):
            get_handle_from_resource(bad_instance, HDF5Writer)
        with pytest.raises(NotImplementedError):
            get_handle_from_resource(bad_instance, HDF5Reader)


class TestIOResourceManager:
    """Tests for the IO resource manager."""

    def test_basic_context_manager(self, tmp_path):
        """Ensure it works as a context manager."""
        write_path = tmp_path / "io_writer"

        with IOResourceManager(write_path) as man:
            my_str = man.get_resource(_dummy_func)
            assert isinstance(my_str, str)
            path = man.get_resource(Path)
            assert isinstance(path, Path)
            hf = man.get_resource(HDF5Writer)
            fi = man.get_resource(BinaryWriter)
            # Why didn't pytables implement the stream like pythons?
            assert hf.isopen
            assert not fi.closed
        # after the context manager exists everything should be closed.
        assert not hf.isopen
        assert fi.closed

    def test_nested_context(self, tmp_path):
        """Ensure nested context works as well."""
        write_path = tmp_path / "io_writer"
        with IOResourceManager(write_path) as man:
            fi1 = man.get_resource(BinaryWriter)
            with IOResourceManager(man):
                fi2 = man.get_resource(BinaryWriter)
                # nested IOManager should just return value from previous
                assert fi1 is fi2
            # on first exist the resource should remain open
            assert not fi2.closed
        # then closed.
        assert fi2.closed

    def test_closed_after_exception(self, tmp_path):
        """Ensure the file resources are closed after an exception."""
        path = tmp_path / "closed_resource_test.txt"
        path.touch()
        try:
            with IOResourceManager(path) as man:
                fi = man.get_resource(BinaryReader)
                raise ValueError("Waaagh!")
        except ValueError:
            assert fi.closed
