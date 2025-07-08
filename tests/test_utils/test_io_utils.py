"""Tests for IO utilities."""

from __future__ import annotations

from io import BufferedReader, BufferedWriter
from pathlib import Path

import pytest
from tables import File

import dascore as dc
from dascore.exceptions import PatchConversionError
from dascore.utils.hdf5 import HDF5Reader, HDF5Writer
from dascore.utils.io import (
    BinaryReader,
    BinaryWriter,
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


class TestXarray:
    """Tests for xarray conversions."""

    @pytest.fixture
    def data_array_from_patch(self, random_patch):
        """Get a data array from a patch."""
        pytest.importorskip("xarray")
        return random_patch.io.to_xarray()

    def test_convert_to_xarray(self, data_array_from_patch):
        """Tests for converting to xarray object."""
        import xarray as xr

        assert isinstance(data_array_from_patch, xr.DataArray)

    def test_convert_from_xarray(self, data_array_from_patch):
        """Ensure xarray data arrays can be converted back."""
        out = dc.utils.io.xarray_to_patch(data_array_from_patch)
        assert isinstance(out, dc.Patch)

    def test_round_trip(self, random_patch, data_array_from_patch):
        """Converting to xarray should be lossless."""
        out = dc.utils.io.xarray_to_patch(data_array_from_patch)
        assert out == random_patch

    def test_convert_non_coord(self, random_patch):
        """Ensure a patch with non-coord can still be converted."""
        xr = pytest.importorskip("xarray")
        patch = random_patch.sum("time")
        dar = patch.io.to_xarray()
        assert isinstance(dar, xr.DataArray)
        # Ensure it round-trips
        patch2 = dc.utils.io.xarray_to_patch(dar)
        assert isinstance(patch2, dc.Patch)


class TestObsPy:
    """Tests for converting patches to/from ObsPy streams."""

    @pytest.fixture
    def short_patch(self, random_patch):
        """Just shorten the patch distance dim to speed up these tests."""
        return random_patch.select(distance=(0, 10), samples=True)

    @pytest.fixture
    def stream_from_patch(self, short_patch):
        """Get a stream from a patch."""
        pytest.importorskip("obspy")
        st = short_patch.io.to_obspy()
        return st

    def test_convert_to_obspy(self, stream_from_patch):
        """Ensure a patch can be converted to a stream."""
        import obspy

        assert isinstance(stream_from_patch, obspy.Stream)

    def test_obspy_to_patch(self, stream_from_patch):
        """Ensure we can convert back to patch from stream."""
        out = dc.io.obspy_to_patch(stream_from_patch)
        assert isinstance(out, dc.Patch)

    def test_patch_no_time_raises(self, random_patch):
        """Ensure a patch without time dimension raises."""
        pytest.importorskip("obspy")
        patch = random_patch.rename_coords(time="not_time")
        with pytest.raises(PatchConversionError):
            patch.io.to_obspy()

    def test_bad_stream_raises(self):
        """Ensure a stream without even length or require param raises."""
        obspy = pytest.importorskip("obspy")
        st = obspy.read()
        # since st doesn't have a value of "distance" in each of its traces
        # attrs dict this should raise.
        with pytest.raises(PatchConversionError):
            dc.io.obspy_to_patch(st)

    def test_empty_stream(self):
        """An empty Stream should return an empty Patch."""
        obspy = pytest.importorskip("obspy")
        st = obspy.Stream([])
        patch = dc.io.obspy_to_patch(st)
        assert not patch.dims

    def test_example_event(self, event_patch_2):
        """Ensure example event can be converted to stream."""
        obspy = pytest.importorskip("obspy")
        # make patch smaller to make test faster
        patch = event_patch_2.select(distance=(500, 550))
        st = patch.io.to_obspy()
        assert isinstance(st, obspy.Stream)
        assert len(st) == len(patch.get_coord("distance"))
