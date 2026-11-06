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
from dascore.io.sintela.utils import _check_packet_continuity, _get_complete_header
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

    # A packet is 500 samples at 1000 Hz, so half a packet is 0.25 s.
    half_packet_ns = 250_000_000

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
    def _shift(packet, field, delta):
        """Return the packet with sample_count or start_time_ns moved by delta."""
        start = {"count": 12, "time": 20}[field]
        value = int(np.frombuffer(packet[start : start + 8], dtype="<u8")[0])
        return packet[:start] + np.uint64(value + delta).tobytes() + packet[start + 8 :]

    def test_dropped_packet_raises(self, packets, tmp_path):
        """A missing middle packet raises rather than shifting later samples."""
        path = tmp_path / "dropped.raw"
        path.write_bytes(b"".join(packets[:2] + packets[3:]))
        with pytest.raises(InvalidFiberFileError, match=r"packet 3 .* sample count"):
            SintelaBinaryV3().scan(path)
        with pytest.raises(InvalidFiberFileError, match=r"packet 2 .* sample count"):
            SintelaBinaryV3().read_array(path)

    def test_time_jump_raises(self, packets, tmp_path):
        """Packets whose start time jumps raise despite contiguous counts."""
        path = tmp_path / "time_jump.raw"
        tail = [self._shift(x, "time", 60 * 10**9) for x in packets[3:]]
        path.write_bytes(b"".join(packets[:3] + tail))
        with pytest.raises(InvalidFiberFileError, match=r"packet 4 .* start time"):
            SintelaBinaryV3().scan(path)

    def test_inner_count_checked_on_read(self, packets, tmp_path):
        """A bad count between good endpoints passes scan but fails read."""
        path = tmp_path / "inner_count.raw"
        bad = self._shift(packets[2], "count", 1)
        path.write_bytes(b"".join([*packets[:2], bad, *packets[3:]]))
        assert len(SintelaBinaryV3().scan(path)) == 1
        with pytest.raises(InvalidFiberFileError, match=r"packet 2 .* sample count"):
            SintelaBinaryV3().read_array(path)

    @pytest.mark.parametrize("shift", [-1, 1])
    def test_time_tolerance(self, packets, tmp_path, shift):
        """A start time within half a packet of the grid reads; beyond raises."""
        near = self._shift(packets[2], "time", shift * (self.half_packet_ns - 1))
        far = self._shift(packets[2], "time", shift * (self.half_packet_ns + 1))
        near_path, far_path = tmp_path / "near.raw", tmp_path / "far.raw"
        near_path.write_bytes(b"".join([*packets[:2], near, *packets[3:]]))
        far_path.write_bytes(b"".join([*packets[:2], far, *packets[3:]]))
        assert dc.read(near_path)[0].shape[0] == len(packets) * 500
        with pytest.raises(InvalidFiberFileError, match=r"packet 2 .* start time"):
            dc.read(far_path)

    def test_no_drift_over_long_files(self):
        """A packet duration that is not whole ns does not drift into a raise."""
        header = {"num_samples": 1, "sample_rate": 3000.0}
        # 500001 samples at 3 kHz end 166.667 s in, exactly; rounding the
        # packet duration first would put it a whole tolerance away.
        stamps = np.array([[0, 0], [500_001, 166_667_000_000]], dtype="<u8")
        _check_packet_continuity(np.array([0, 500_001]), stamps, header)
