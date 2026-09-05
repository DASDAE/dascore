"""Tests for TDMS utilities to improve coverage."""

from __future__ import annotations

import datetime
import io
import struct
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
from dascore.io.core import FiberIO
from dascore.io.silixah5.utils import _ATTR_MAP as _SILIXA_ATTR_MAP
from dascore.io.tdms import utils as tdms_utils
from dascore.io.tdms.core import TDMSFormatterV4713
from dascore.io.tdms.utils import parse_time_stamp, type_not_supported
from dascore.utils.downloader import fetch


class _FakeTDMSFile(io.BytesIO):
    """A BytesIO object with the minimum file API TDMS utils expect."""

    name = "fake.tdms"

    def fileno(self):
        """Return a dummy file descriptor for monkeypatched mmap."""
        return 0


class TestTDMSUtils:
    """Tests for TDMS utility functions."""

    def test_type_not_supported(self):
        """Test that type_not_supported raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Reading of this tdsDataType is not implemented"
        ):
            type_not_supported("any_input")

    def test_parse_time_stamp_none(self):
        """Test that parse_time_stamp returns None for invalid input."""
        # Test with invalid timestamp values that should return None
        result = parse_time_stamp(0, 0)  # epoch start should return None
        assert result is None

        # Test with None values
        result = parse_time_stamp(None, 100)
        assert result is None

        result = parse_time_stamp(100, None)
        assert result is None

        result = parse_time_stamp(None, None)
        assert result is None

    def test_parse_time_stamp_valid(self):
        """Test that parse_time_stamp works with valid input."""
        # Test with valid timestamp - using a reasonable epoch timestamp
        # LabVIEW epoch starts at 1904-01-01, so we need a positive value

        # Use a timestamp that represents a valid date after 1904
        seconds = 365 * 24 * 3600 * 100  # 100 years after 1904
        fractions = 0.5  # Some fractional seconds

        result = parse_time_stamp(fractions, seconds)
        assert isinstance(result, datetime.datetime)
        assert result.year >= 1904

    def test_get_all_attrs_unsupported_data_type(self, monkeypatch):
        """Unsupported TDMS channel types should raise clearly."""
        lead_in = struct.pack("<4siiQQ", b"TDSm", 0, 4713, 0, 0)
        payload = b"".join(
            [
                struct.pack("<i", 3),  # object count -> 1 channel after adjustment
                struct.pack("<i", 0),  # object path len
                struct.pack("<i", 0),  # raw data index len
                struct.pack("<i", 0),  # property count
                struct.pack("<i", 0),  # group info len
                b"\x00" * 8,  # skipped group info bytes
                struct.pack("<i", 0),  # first channel path len
                struct.pack("<i", 0),  # index len
                struct.pack("<i", 0x21),  # bool -> unsupported
            ]
        )
        fake = _FakeTDMSFile(lead_in + payload)
        monkeypatch.setattr(
            tdms_utils,
            "get_buffer_size",
            lambda _: len(lead_in + payload),
        )
        monkeypatch.setattr(
            tdms_utils,
            "_get_distance_coord",
            lambda _: tdms_utils.get_coord(start=0, stop=1, step=1, units="m"),
        )
        with pytest.raises(Exception, match="Unsupported TDMS data type"):
            tdms_utils._get_all_attrs(fake)

    def test_get_data_decimated_multi_segment(self, monkeypatch):
        """Decimated multi-segment data should use the append/update path."""
        fileinfo = {
            "decimated": True,
            "chunk_size": 2,
            "data_type": "float32",
            "file_size": 52,
            "raw_data_offset": 0,
            "n_channels": 1,
            "next_segment_offset": 12,
        }
        attrs = {"tag": "example"}
        data = bytearray(52)
        data[0:12] = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
        data[24:40] = struct.pack("<qq", 12, 0)
        data[40:52] = np.array([4.0, 5.0, 6.0], dtype=np.float32).tobytes()
        fake = _FakeTDMSFile(bytes(data))
        monkeypatch.setattr(tdms_utils, "_get_fileinfo", lambda _: (fileinfo, attrs))
        monkeypatch.setattr(tdms_utils.mmap, "mmap", lambda *args, **kwargs: data)
        out_data, channel_length, out_attrs = tdms_utils._get_data(fake)
        # Three samples on one channel from each of the two segments.
        assert out_data.shape == (6, 1)
        assert channel_length == 6
        assert out_attrs == attrs

    def test_get_data_decimated_whole_chunks(self, monkeypatch):
        """A decimated segment whose chunks are all full still has its data."""
        fileinfo = {
            "decimated": True,
            "chunk_size": 2,
            "data_type": "float32",
            "file_size": 8,
            "raw_data_offset": 0,
            "n_channels": 1,
            "next_segment_offset": 8,
        }
        attrs = {"tag": "example"}
        data = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        fake = _FakeTDMSFile(data)
        monkeypatch.setattr(tdms_utils, "_get_fileinfo", lambda _: (fileinfo, attrs))
        monkeypatch.setattr(tdms_utils.mmap, "mmap", lambda *args, **kwargs: data)
        out_data, channel_length, _ = tdms_utils._get_data(fake)
        assert out_data.shape == (2, 1)
        assert channel_length == 2


class TestSampleOrder:
    """Raw chunks must become chronological samples, including windowed reads."""

    @pytest.mark.parametrize("shape", [(6, 1), (6, 2), (7, 2)])
    @pytest.mark.parametrize("decimated", [True, False])
    @pytest.mark.parametrize("bounds", [(0, None), (1, 5)])
    def test_chunk_order(self, tmp_path, shape, decimated, bounds):
        """Full chunks and a partial tail retain time and channel identities."""
        expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        chunk_size = 2
        if decimated:
            # Storage writes each channel's samples in one chunk before
            # moving to the next channel, then starts the next time chunk.
            payload = b"".join(
                expected[start : start + chunk_size, channel].tobytes()
                for start in range(0, shape[0], chunk_size)
                for channel in range(shape[1])
            )
        else:
            payload = expected.tobytes()
        path = tmp_path / "samples.raw"
        path.write_bytes(payload)
        fileinfo = {
            "decimated": decimated,
            "chunk_size": chunk_size,
            "data_type": "float32",
            "n_channels": shape[1],
            "raw_data_offset": 0,
            "next_segment_offset": len(payload),
            "file_size": len(payload),
        }
        start, stop = bounds
        with path.open("rb") as resource:
            actual = tdms_utils._read_sample_range(resource, fileinfo, start, stop)
        np.testing.assert_array_equal(actual, expected[start:stop])
        assert actual.dtype == expected.dtype


class TestMultiSegment:
    """A TDMS file can hold several segments, each a stretch of time."""

    @pytest.fixture(scope="class")
    def two_segment_path(self, tmp_path_factory):
        """The example file's segment, written twice.

        Appending segments is how a TDMS file grows, and this example
        declares its segment length exactly, so the second copy reads as a
        second segment. sample_tdms_file_v4713.tdms cannot stand in for it:
        its lead-in claims 146 MB for a 1 MB file, so anything after it is
        read as part of the first segment.
        """
        raw = Path(fetch("iDAS005_tdms_example.626.tdms")).read_bytes()
        path = tmp_path_factory.mktemp("tdms_multi_segment") / "two_segment.tdms"
        path.write_bytes(raw + raw)
        return path

    def test_segments_stack_along_time(self, two_segment_path):
        """Both segments are read, and they stack along time, not distance."""
        with open(two_segment_path, "rb") as fi:
            data, channel_length, _ = tdms_utils._get_data(fi)
        assert data.shape == (2000, 1152)
        assert channel_length == 2000
        # One segment written twice, so the halves are the same samples.
        assert np.array_equal(data[:1000], data[1000:])

    def test_read(self, two_segment_path):
        """A multi-segment file reads as one patch holding every segment."""
        patch = dc.spool(two_segment_path)[0]
        single = dc.spool(fetch("iDAS005_tdms_example.626.tdms"))[0]
        assert patch.shape == (2000, 1152)
        assert np.array_equal(patch.data[:1000], single.data)
        assert len(patch.get_coord("time")) == 2000

    def test_unclosed_last_segment_runs_to_the_end_of_the_file(self):
        """All ones for a segment's length means the rest of the file.

        TDMS writes that when a file was never closed -- the recording was
        cut short -- and the offsets are unsigned, so reading them as signed
        would make the last segment a backwards one.
        """
        fileinfo = {
            "raw_data_offset": 28,
            "next_segment_offset": 40,
            "file_size": 200,
            "n_channels": 1,
            "data_type": "float32",
        }
        buffer = bytearray(200)
        buffer[52:68] = struct.pack("<QQ", 0xFFFFFFFFFFFFFFFF, 0)
        fake = _FakeTDMSFile(bytes(buffer))
        bounds = list(tdms_utils._iter_segment_bounds(fake, fileinfo))
        assert bounds == [(28, 40), (68, 200)]
        assert tdms_utils._get_sample_count(fake, fileinfo) == 3 + 33

    def test_scan_agrees_with_read(self, two_segment_path):
        """Counting bytes to the end of the file would over-count the time.

        A scan never reads the data, so nothing else would catch it saying
        the file covers more time than it holds.
        """
        summary = dc.scan(two_segment_path)[0]
        time = dc.spool(two_segment_path)[0].get_coord("time")
        assert summary.coords["time"].len == 2000
        assert summary.coords["time"].min == time.min()
        assert summary.coords["time"].max == time.max()


class TestReadArray:
    """Tests for reading only the segments a window touches."""

    @pytest.fixture(scope="class")
    def two_segment_path(self, tmp_path_factory):
        """The example file's segment written twice (see TestMultiSegment)."""
        raw = Path(fetch("iDAS005_tdms_example.626.tdms")).read_bytes()
        path = tmp_path_factory.mktemp("tdms_read_array") / "two_segment.tdms"
        path.write_bytes(raw + raw)
        return path

    def test_matches_default_across_segments(self, two_segment_path):
        """A window spanning the segment boundary matches the default."""
        io = TDMSFormatterV4713()
        windows = {"time": (990, 1010), "distance": (5, 9)}
        out = io.read_array(two_segment_path, windows)
        expected = FiberIO.read_array(io, two_segment_path, windows)
        assert out.dtype == expected.dtype
        assert np.array_equal(out, expected)
        assert out.shape == (20, 4)

    def test_only_touched_segments_are_decoded(self, two_segment_path, monkeypatch):
        """A window inside the second segment decodes that segment alone."""
        from dascore.io.tdms import utils as tdms_utils  # noqa: PLC0415

        decoded = []
        original = tdms_utils._get_segment_data

        def spy(fileinfo, nch, dmap, nso, rdo):
            decoded.append(rdo)
            return original(fileinfo, nch, dmap, nso, rdo)

        single = dc.spool(fetch("iDAS005_tdms_example.626.tdms"))[0]
        with open(two_segment_path, "rb") as fi:
            fileinfo, _ = tdms_utils._get_fileinfo(fi)
            first, second = (
                rdo for rdo, _ in tdms_utils._iter_segment_bounds(fi, fileinfo)
            )
        monkeypatch.setattr(tdms_utils, "_get_segment_data", spy)
        io = TDMSFormatterV4713()
        # windows and the segment each should decode; the boundary is 1000
        cases = {
            (1200, 1300): [second],
            (100, 200): [first],
            (0, 1000): [first],
            (1000, 1100): [second],
            (990, 1010): [first, second],
        }
        for (start, stop), expected in cases.items():
            decoded.clear()
            out = io.read_array(two_segment_path, {"time": (start, stop)})
            assert decoded == expected, (start, stop)
            # both segments hold the example's samples
            rows = np.concatenate([single.data, single.data])[start:stop]
            assert np.array_equal(out, rows)

    def test_empty_window(self, two_segment_path):
        """A window past the end is empty with the right width and dtype."""
        io = TDMSFormatterV4713()
        out = io.read_array(two_segment_path, {"time": (5000, 6000)})
        assert out.shape == (0, 1152)
        assert out.dtype == FiberIO.read_array(io, two_segment_path, {}).dtype


class TestTDMSInterrogator:
    """A TDMS file names the interrogator by its host name."""

    @pytest.fixture(
        scope="class",
        params=["sample_tdms_file_v4713.tdms", "iDAS005_tdms_example.626.tdms"],
    )
    def tdms_path(self, request):
        """Paths to each TDMS test file."""
        return fetch(request.param)

    @pytest.fixture(scope="class")
    def tdms_attrs(self, tdms_path):
        """Attrs read back from a TDMS file."""
        return dict(dc.spool(tdms_path)[0].attrs)

    @pytest.fixture(scope="class")
    def raw_host_name(self, tdms_path):
        """The HostName property as the TDMS header states it."""
        with open(tdms_path, "rb") as fi:
            header, _ = tdms_utils._get_all_attrs(fi)
        return header["SystemInfomation.OS.HostName"]

    def test_name_is_host_name(self, tdms_attrs, raw_host_name):
        """The name is exactly the HostName property, eg "iDAS005"."""
        assert raw_host_name
        assert tdms_attrs["interrogator.name"] == raw_host_name

    def test_no_component_serial_as_interrogator(self, tdms_attrs, tdms_path):
        """No card or crate serial is passed off as the interrogator's."""
        with open(tdms_path, "rb") as fi:
            header, _ = tdms_utils._get_all_attrs(fi)
        serials = {
            v
            for k, v in header.items()
            if k.startswith("SystemInfomation.") and k.endswith(".SerialNum") and v
        }
        stated = {v for k, v in tdms_attrs.items() if k.startswith("interrogator.")}
        assert serials
        assert not (stated & serials)

    def test_no_serial_claimed(self, tdms_attrs):
        """The Devices and Chassis serials name parts, not the interrogator."""
        assert "interrogator.serial_number" not in tdms_attrs

    def test_agrees_with_silixa_h5(self, tdms_attrs, raw_host_name):
        """Both Silixa readers key the interrogator off the same attr."""
        key = "SystemInfomation.OS.HostName"
        assert _SILIXA_ATTR_MAP[key] == "interrogator.name"
        assert tdms_attrs["interrogator.name"] == raw_host_name
