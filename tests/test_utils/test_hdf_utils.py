"""Tests for hdf5 utils."""

from __future__ import annotations

import io
from contextlib import closing

import h5py
import numpy as np
import pytest

from dascore.utils.downloader import fetch
from dascore.utils.hdf5 import (
    H5Reader,
    H5Writer,
    encode_h5_strings,
    extract_h5_attrs,
    get_h5py_file,
    h5_matches_structure,
    stored_byte_layout,
    unpack_scalar_h5_dataset,
)


@pytest.mark.parametrize(
    ("value", "expected"), (("one", [b"one"]), (["one", "two"], [b"one", b"two"]))
)
def test_encode_h5_strings(value, expected):
    """String encoding should consistently return a byte array."""
    out = encode_h5_strings(value)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == expected


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

    def test_missing_reader_does_not_create_parent(self, tmp_path):
        """Opening a missing file for reading does not mutate the filesystem."""
        path = tmp_path / "missing_parent" / "missing.h5"
        with pytest.raises(OSError):
            H5Reader.get_handle(path)
        assert not path.parent.exists()


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
        """A bad attribute should raise a KeyError."""
        map_name = {
            "DasMetadata.CountryBumpkins": "country",
        }
        with pytest.raises(KeyError):
            extract_h5_attrs(h5_example_file, map_name)


class TestUnpackScalarH5Dataset:
    """The old scalar reader warns and keeps working until it is removed."""

    def test_warns_and_reads(self, tmp_path):
        """It still reads a scalar, and says where to go instead."""
        path = tmp_path / "scalar.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("value", data=5.0)
            h5.create_dataset("wrapped", data=[5.0])
        with h5py.File(path, "r") as h5:
            with pytest.warns(DeprecationWarning, match="unpack_scalar_h5_dataset"):
                assert unpack_scalar_h5_dataset(h5["value"]) == 5.0
            with pytest.warns(DeprecationWarning, match="unpack_scalar_h5_dataset"):
                assert unpack_scalar_h5_dataset(h5["wrapped"]) == 5.0


class TestStoredByteLayout:
    """Where plain file reads find a dataset's rows, if they can."""

    @staticmethod
    def _layout(path, name="data"):
        with h5py.File(path, "r") as h5:
            return stored_byte_layout(h5[name])

    @pytest.mark.parametrize("dtype", ["<f4", ">f4", ">i2", "?"])
    def test_contiguous(self, tmp_path, dtype):
        """A contiguous dataset is one run of rows, in any byte order."""
        path, data = tmp_path / "c.h5", np.arange(40).reshape(10, 4).astype(dtype)
        with h5py.File(path, "w") as h5:
            h5["data"] = data
        offsets, rows = self._layout(path)
        assert len(offsets) == 1 and rows == 10

    def test_chunked_full_width(self, tmp_path):
        """Full-width chunks are listed in row order, the last one partial."""
        path, data = tmp_path / "k.h5", np.arange(40.0).reshape(10, 4)
        with h5py.File(path, "w") as h5:
            ds = h5.create_dataset("data", shape=(10, 4), dtype="f8", chunks=(4, 4))
            # Written last chunk first, so file order is not row order.
            for start in (8, 4, 0):
                ds[start : start + 4] = data[start : start + 4]
        offsets, per = self._layout(path)
        assert per == 4 and len(offsets) == 3 and offsets != tuple(sorted(offsets))

    @pytest.mark.parametrize(("rows", "found"), [(1024, True), (1025, False)])
    def test_chunk_count_cap(self, tmp_path, rows, found):
        """Past 1024 chunks no layout is recorded, so scans and rows stay small."""
        path = tmp_path / "many.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("data", data=np.ones((rows, 2)), chunks=(1, 2))
        assert (self._layout(path) is not None) is found

    def test_stored_type_differs(self, tmp_path):
        """A stored type HDF5 converts on read gives no layout."""
        path = tmp_path / "t.h5"
        short = h5py.h5t.STD_I16LE.copy()
        short.set_precision(12)
        short.set_offset(4)
        small = h5py.h5t.IEEE_F32LE.copy()
        small.set_fields(23, 15, 8, 0, 15)
        small.set_size(3)
        small.set_precision(24)
        small.set_ebias(127)
        with h5py.File(path, "w") as h5:
            for name, kind, dtype in ((b"short", short, "i2"), (b"small", small, "f4")):
                space = h5py.h5s.create_simple((10, 4))
                stored = h5py.h5d.create(h5.id, name, kind, space)
                data = np.arange(40).reshape(10, 4).astype(dtype)
                stored.write(h5py.h5s.ALL, h5py.h5s.ALL, data)
        assert self._layout(path, "short") is None
        assert self._layout(path, "small") is None

    def test_ndim_three(self, tmp_path):
        """Chunks spanning every trailing axis of a cube hold whole rows."""
        path = tmp_path / "cube.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("data", data=np.ones((6, 2, 3)), chunks=(2, 2, 3))
        assert self._layout(path)[1] == 2

    @pytest.mark.parametrize(
        "options",
        [
            {"chunks": (4, 2)},
            {"chunks": (4, 4), "compression": "gzip"},
            {"chunks": (4, 4), "shuffle": True},
            {"chunks": (4, 8), "maxshape": (None, 8)},
        ],
    )
    def test_not_plain(self, tmp_path, options):
        """Partial-width chunks and filtered storage give no layout."""
        path = tmp_path / "f.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("data", data=np.ones((10, 4)), **options)
        assert self._layout(path) is None

    @pytest.mark.parametrize("chunks", [None, (4, 4)])
    def test_unallocated(self, tmp_path, chunks):
        """A dataset with storage never written gives no layout."""
        path = tmp_path / "u.h5"
        with h5py.File(path, "w") as h5:
            ds = h5.create_dataset("data", shape=(10, 4), chunks=chunks)
            if chunks:
                ds[:4] = 1  # one chunk of three
        assert self._layout(path) is None

    def test_scalar_and_strings(self, tmp_path):
        """A scalar, or variable-length strings, give no layout."""
        path = tmp_path / "s.h5"
        with h5py.File(path, "w") as h5:
            h5["scalar"] = 1.0
            h5["text"] = encode_h5_strings(["a", "b"]).astype(object)
        assert self._layout(path, "scalar") is None
        assert self._layout(path, "text") is None

    def test_external_and_virtual(self, tmp_path):
        """Storage outside the file, or mapped from other datasets, gives none."""
        path, raw = tmp_path / "e.h5", tmp_path / "raw.bin"
        np.ones((10, 4)).tofile(raw)
        with h5py.File(path, "w") as h5:
            h5.create_dataset(
                "external", shape=(10, 4), dtype="f8", external=[(raw, 0, 320)]
            )
            h5["base"] = np.ones((10, 4))
            layout = h5py.VirtualLayout(shape=(10, 4), dtype="f8")
            layout[:] = h5py.VirtualSource(h5["base"])
            h5.create_virtual_dataset("virtual", layout)
        assert self._layout(path, "external") is None
        assert self._layout(path, "virtual") is None

    def test_file_object(self):
        """A file opened on a python file object is not one plain reads find."""
        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as h5:
            h5["data"] = np.ones((10, 4))
            assert stored_byte_layout(h5["data"]) is None
