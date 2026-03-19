"""Tests for TDMS utilities to improve coverage."""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from dascore.io.tdms import utils as tdms_utils
from dascore.io.tdms.utils import parse_time_stamp, type_not_supported


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
        import datetime

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
            tdms_utils.os.path,
            "getsize",
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
        assert out_data.shape == (3, 2)
        assert channel_length == 6
        assert out_attrs == attrs
