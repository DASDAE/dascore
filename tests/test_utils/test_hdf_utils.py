"""Tests for hdf5 utils."""

from __future__ import annotations

from contextlib import closing

import h5py
import pytest

from dascore.utils.downloader import fetch
from dascore.utils.hdf5 import (
    H5Reader,
    H5Writer,
    extract_h5_attrs,
    get_h5py_file,
    h5_matches_structure,
)


@pytest.fixture(scope="class")
def h5_example_file():
    """Get an example file."""
    path = fetch("gdr_1.h5")
    with h5py.File(path, "r") as fi:
        yield fi
    fi.close()


class TestH5Readers:
    """Tests for the h5py-based reader/writer handles."""

    def test_writer_get_handle(self, tmp_path_factory):
        """Ensure we can get a writable handle from a path or handle."""
        path = tmp_path_factory.mktemp("hdf_handle_test") / "test_file.h5"
        with closing(H5Writer.get_handle(path)) as handle:
            handle.create_group("waveforms")
            handle_2 = H5Writer.get_handle(handle)
            assert handle_2 is handle

    def test_reader_get_handle(self, tmp_path):
        """Ensure the reader opens existing files."""
        path = tmp_path / "local_reader.h5"
        with h5py.File(path, "w") as h5:
            h5.create_group("waveforms")
        with closing(H5Reader.get_handle(path)) as handle:
            assert "waveforms" in handle


class TestGetH5pyFile:
    """Tests for unwrapping managed h5py handles."""

    def test_unwraps_managed_handle(self, tmp_path):
        """A managed h5py handle should unwrap to a real h5py.File."""
        path = tmp_path / "managed.h5"
        with h5py.File(path, "w"):
            pass
        handle = H5Reader.get_handle(path)
        try:
            out = get_h5py_file(handle)
            assert isinstance(out, h5py.File)
        finally:
            handle.close()

    def test_passthrough_raw_handle(self, tmp_path):
        """A raw (non-managed) handle should pass through unchanged."""
        path = tmp_path / "raw.h5"
        with h5py.File(path, "w"):
            pass
        with h5py.File(path, "r") as raw:
            assert get_h5py_file(raw) is raw


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
