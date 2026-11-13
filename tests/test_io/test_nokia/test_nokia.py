"""Tests for the Nokia multi-span (OFDR) formats."""

from __future__ import annotations

import shutil
import struct

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import UnknownFiberFormatError
from dascore.io.nokia import NokiaOFDRH5V1, NokiaOFDRV1
from dascore.utils.downloader import fetch

msgpack = pytest.importorskip("msgpack")

MAGIC = b"OFDR1MAGIC"


def _read_header(path):
    """Return the header objects and array start, parsed independently."""
    raw = path.read_bytes()
    _, header_bytes, array_start = struct.unpack("<BII", raw[10:19])
    unpacker = msgpack.Unpacker(raw=False)
    unpacker.feed(raw[19 : 19 + header_bytes])
    return list(unpacker), array_start


def _raw_rows(path, nchan=1559, nchunks=10):
    """Return the stored (channel, time) array without the reader."""
    _, array_start = _read_header(path)
    data = np.fromfile(path, dtype="<f4", count=nchunks * nchan * 32, offset=4096)
    assert array_start == 4096
    return data.reshape(nchunks, nchan, 32).transpose(1, 0, 2).reshape(nchan, -1)


@pytest.fixture(scope="module")
def binary_path():
    """The masked binary test file."""
    return fetch("nokia_ofdr_1")


@pytest.fixture(scope="module")
def binary_spool(binary_path):
    """The masked binary test file read into a spool."""
    return dc.read(binary_path)


class TestBinary:
    """Tests for the binary variant."""

    def test_one_patch_per_span(self, binary_spool, binary_path):
        """Each mask span is a patch over its own stored rows."""
        dx = 105.85775068689657
        raw = _raw_rows(binary_path)
        assert len(binary_spool) == 3
        first_channels, row = (0, 1605, 2597), 0
        for patch, first in zip(binary_spool, first_channels):
            assert patch.dims == ("distance", "time")
            distance = patch.get_coord("distance")
            assert distance.min() == pytest.approx(first * dx)
            count = len(distance)
            np.testing.assert_array_equal(patch.data, raw[row : row + count])
            row += count
        assert row == 1559
        assert [len(x.get_coord("distance")) for x in binary_spool] == [1039, 284, 236]

    def test_steps_match_stored_fields(self, binary_spool, binary_path):
        """The computed time and distance steps equal the stored fs2 and dx."""
        (_, _, meta), _ = _read_header(binary_path)
        patch = binary_spool[0]
        time_step = patch.get_coord("time").step / dc.to_timedelta64(1)
        assert time_step == pytest.approx(1 / meta["fs2"])
        assert patch.get_coord("distance").step == pytest.approx(meta["dx"])
        assert patch.get_coord("time").min() == np.datetime64(
            "2025-12-29T01:05:27.661054"
        )
        assert patch.attrs.data_units == dc.get_quantity("rad")
        assert patch.attrs.gauge_length == pytest.approx(104.5817, rel=1e-5)

    def test_unapplied_mask_is_one_patch(self, binary_path, tmp_path):
        """A mask whose spans don't add up to the channels gives one patch."""
        (array_info, host, meta), array_start = _read_header(binary_path)
        meta["masked"] = [[0.0, 110000.0]]
        header = b"".join(msgpack.packb(x) for x in (array_info, host, meta))
        raw = bytearray(binary_path.read_bytes())
        raw[10:19] = struct.pack("<BII", 1, len(header), array_start)
        raw[19 : 19 + len(header)] = header
        path = tmp_path / "unapplied_mask"
        path.write_bytes(bytes(raw))
        spool = dc.read(path)
        assert len(spool) == 1
        distance = spool[0].get_coord("distance")
        assert distance.min() == 0 and len(distance) == 1559
        np.testing.assert_array_equal(spool[0].data, _raw_rows(path))

    def test_incomplete_file_not_recognized(self, binary_path, tmp_path):
        """A file still being written (no end magic) is not this format."""
        path = tmp_path / "incomplete"
        shutil.copyfile(binary_path, path)
        with path.open("r+b") as fi:
            fi.truncate(path.stat().st_size - len(MAGIC))
        with pytest.raises(UnknownFiberFormatError):
            dc.get_format(path)

    def test_empty_distance_window(self, binary_path):
        """An empty distance window returns no rows rather than raising."""
        out = NokiaOFDRV1().read_array(binary_path, ((10, 10), (5, 40)), key="1")
        assert out.shape == (0, 35)

    def test_scan_without_range_requests(self, to_http_path):
        """A server ignoring Range headers scans through a local copy."""
        with dc.config_context(allow_remote_cache_for_metadata=True):
            spool = dc.spool(to_http_path("nokia_ofdr_1"))
        assert [x.shape[0] for x in spool] == [1039, 284, 236]

    def test_window_across_chunks(self, binary_path, binary_spool):
        """A window spanning a chunk boundary equals that slice of the patch."""
        part = NokiaOFDRV1().read(
            binary_path,
            samples=True,
            source_patch_key="1",
            time=(40, 100),
            distance=(10, 20),
        )
        expected = binary_spool[1].data[10:20, 40:100]
        np.testing.assert_array_equal(part[0].data, expected)


class TestH5:
    """Tests for the downsampled HDF5 variant."""

    def test_shape_and_step(self):
        """The data are (distance, time) with the stored decimated rate."""
        path = fetch("nokia_ofdr_downsampled_1.h5")
        assert dc.get_format(path) == (NokiaOFDRH5V1.name, "1")
        patch = dc.spool(path)[0]
        assert patch.dims == ("distance", "time")
        assert patch.shape == (500, 256)
        step = patch.get_coord("time").step / dc.to_timedelta64(1)
        assert step == pytest.approx(1 / 7.450580596923827)

    def test_masked_spans(self, tmp_path):
        """A masked file's spans read their own stored rows."""
        source = fetch("nokia_ofdr_downsampled_1.h5")
        path = tmp_path / "masked.h5"
        shutil.copyfile(source, path)
        with h5py.File(path, "r+") as h5:
            # 105.86 m channels: spans of 300 and 200 rows.
            masked = [[0.0, 31757.33], [52928.88, 74100.43]]
            h5["data"].attrs["masked"] = np.array(masked)
            stored = h5["data"][:]
        spool = dc.spool(path)
        assert [x.shape[0] for x in spool] == [300, 200]
        np.testing.assert_array_equal(spool[0].data, stored[:300])
        np.testing.assert_array_equal(spool[1].data, stored[300:])
        assert spool[1].get_coord("distance").min() == pytest.approx(500 * 105.8577)
