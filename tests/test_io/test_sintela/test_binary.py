"""
Tests for sintela binary format.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.sintela import SintelaBinaryV3
from dascore.io.sintela.utils import _get_complete_header
from dascore.utils.downloader import fetch


class TestScanSintelaBinary:
    """Tests for scanning a binary file."""

    @pytest.fixture(scope="class")
    def extra_bytes_file(self, tmp_path_factory):
        """Create a sintela binary file with extra bytes."""
        tmp = tmp_path_factory.mktemp("sintela_binary")
        binary_path = Path(fetch("sintela_binary_v3_test_1.raw"))
        new = tmp / "extra_bytes.raw"
        shutil.copy(binary_path, new)

        with open(new, "ab") as fi:
            fi.write(b"some_bytes_des_is")

        return new

    def test_extra_bytes_raises(self, extra_bytes_file):
        """Ensure a file with extra bytes raises an exception."""
        fiber_io = SintelaBinaryV3()
        with pytest.raises(InvalidFiberFileError):
            fiber_io.scan(extra_bytes_file)


class TestReadArray:
    """Tests for slicing the mapped packet payloads."""

    @pytest.fixture(scope="class")
    def binary_path(self):
        """The example Sintela binary recording."""
        return fetch("sintela_binary_v3_test_1.raw")

    def test_matches_default(self, binary_path):
        """The override returns what the read-and-trim default returns."""
        io = SintelaBinaryV3()
        patch = dc.spool(binary_path)[0]
        out = io.read_array(binary_path, ((3, 11), (2, 6)))
        expected = (
            io.read(binary_path, source_patch_key="")[0]
            .select(samples=True, time=(3, 11), distance=(2, 6))
            .data
        )
        assert out.dtype == expected.dtype
        assert np.array_equal(out, expected)
        assert np.array_equal(out, patch.data[3:11, 2:6])

    def test_reads_only_touched_packets(self, binary_path):
        """A window inside one packet copies that packet, not the file."""
        io = SintelaBinaryV3()
        with open(binary_path, "rb") as fi:
            header = _get_complete_header(fi)
        samples, channels = header["num_samples"], header["num_channels"]
        packets = header["num_packets"]
        if packets < 2:
            pytest.skip("file holds one packet")
        # a window wholly inside the last packet, so a whole-file read is
        # visible in what the returned view is backed by
        start = (packets - 1) * samples
        out = io.read_array(binary_path, ((start, start + 3),))
        assert out.shape == (3, channels)
        assert out.base is not None
        whole = packets * samples * channels * np.dtype(header["dtype"]).itemsize
        assert out.base.nbytes < whole
        # and it is the right packet
        expected = (
            io.read(binary_path, source_patch_key="")[0]
            .select(samples=True, time=(start, start + 3))
            .data
        )
        assert np.array_equal(out, expected)

    def test_empty_window(self, binary_path):
        """A window past the end is empty, with the right width and dtype."""
        io = SintelaBinaryV3()
        whole = io.read_array(binary_path, ())
        out = io.read_array(binary_path, ((10**9, 10**9 + 5),))
        assert out.shape == (0, whole.shape[1])
        assert out.dtype == whole.dtype


class TestPacketContinuity:
    """Tests for files whose packets are not one contiguous run."""

    @pytest.fixture(scope="class")
    def packets(self):
        """The example file's raw bytes split into its packets."""
        path = Path(fetch("sintela_binary_v3_test_1.raw"))
        with open(path, "rb") as fi:
            header = _get_complete_header(fi)
        raw = path.read_bytes()
        size = len(raw) // int(header["num_packets"])
        return [raw[i : i + size] for i in range(0, len(raw), size)]

    @staticmethod
    def _shift_time(packet, shift_ns):
        """Return the packet with its start_time_ns moved by shift_ns."""
        start = int(np.frombuffer(packet[20:28], dtype="<u8")[0]) + shift_ns
        return packet[:20] + np.uint64(start).tobytes() + packet[28:]

    def test_dropped_packet_raises(self, packets, tmp_path):
        """A missing middle packet raises rather than shifting later samples."""
        path = tmp_path / "dropped.raw"
        path.write_bytes(b"".join(packets[:2] + packets[3:]))
        with pytest.raises(InvalidFiberFileError, match="packet 2"):
            SintelaBinaryV3().scan(path)
        with pytest.raises(InvalidFiberFileError, match="contiguous"):
            SintelaBinaryV3().read_array(path)

    def test_time_gap_raises(self, packets, tmp_path):
        """A packet whose start time jumps raises despite contiguous counts."""
        path = tmp_path / "time_gap.raw"
        tail = [self._shift_time(x, 60 * 10**9) for x in packets[3:]]
        path.write_bytes(b"".join(packets[:3] + tail))
        with pytest.raises(InvalidFiberFileError, match="packet 3"):
            SintelaBinaryV3().scan(path)

    def test_time_jitter_allowed(self, packets, tmp_path):
        """Start times a few samples off the nominal grid still read."""
        path = tmp_path / "jitter.raw"
        jittered = [packets[0], self._shift_time(packets[1], 3 * 10**6)]
        path.write_bytes(b"".join(jittered + packets[2:]))
        patch = dc.spool(path)[0]
        assert patch.shape[0] == len(packets) * 500
