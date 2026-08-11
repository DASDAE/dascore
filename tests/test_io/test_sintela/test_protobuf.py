"""
Tests for Sintela protobuf format.
"""

from __future__ import annotations

import gc
import struct
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError, MissingOptionalDependencyError
from dascore.io.core import _scan_payload_to_summary
from dascore.io.sintela import SintelaProtobufV1
from dascore.io.sintela import protobuf_utils as sintela_utils
from dascore.units import get_quantity
from dascore.utils.downloader import fetch

# protobuf is an optional dependency and is not in the test extra, so the
# min-deps job runs without it. Skipping here (rather than installing it
# everywhere) keeps one CI environment genuinely protobuf-free, which is what
# catches an accidental eager `import google.protobuf` in the reader.
pytest.importorskip("google.protobuf")


@pytest.fixture(scope="module")
def sintela_protobuf_path():
    """Return the registered Sintela protobuf test file."""
    return fetch("sintela_protobuf_1.pb")


@pytest.fixture()
def fiber_io():
    """Return the Sintela protobuf FiberIO instance."""
    return SintelaProtobufV1()


def _envelope(record):
    """Build an EnvelopeRecord from a (tag, payload) tuple."""
    tag, payload = record
    return sintela_utils.EnvelopeRecord(tag=tag, payload=payload)


def _write_record(handle, tag: str, payload: bytes):
    """Write one MTLV record to a binary handle."""
    handle.write(struct.pack("<I", sintela_utils.PBUF_MAGIC))
    handle.write(tag.encode("utf-8")[:4].ljust(4, b"\x00"))
    handle.write(struct.pack("<I", len(payload)))
    handle.write(payload)


def _set_timestamp(field, seconds: int, nanos: int = 0):
    """Populate a protobuf timestamp field."""
    field.seconds = seconds
    field.nanos = nanos


def _build_meta_payload():
    """Create a META payload with selected provenance fields populated."""
    cls = sintela_utils._get_meta_message_class()
    msg = cls()
    msg.recorder_namespace = "manualRecord/recorder"
    _set_timestamp(msg.metadata_recording_time, 1_700_000_000, 123)
    msg.identification.manufacturer = "Sintela"
    msg.identification.model = "Onyx"
    msg.identification.serial_number = "SN123"
    msg.acquisition_stats.fiber_id = 2
    return msg.SerializeToString()


def _build_meta_payload_without_fiber_id():
    """Create a META payload with the optional fiber_id field unset."""
    cls = sintela_utils._get_meta_message_class()
    msg = cls()
    msg.recorder_namespace = "manualRecord/recorder"
    _set_timestamp(msg.metadata_recording_time, 1_700_000_000, 123)
    msg.identification.manufacturer = "Sintela"
    msg.identification.model = "Onyx"
    msg.identification.serial_number = "SN123"
    return msg.SerializeToString()


def _payload_to_summary(payload):
    """Convert a raw FiberIO scan payload using the production scan path."""
    return _scan_payload_to_summary(payload)


@cache
def _get_test_proto_messages():
    """Build local protobuf classes for Sintela synthetic test payloads."""
    return sintela_utils._build_proto_messages(
        include_sample_fields=True,
        package_name="test_sintela_common",
        file_name="test_sintela_common.proto",
    )


def _get_num_samples(payload: bytes) -> int:
    """Return the num_samples header field of a serialized TS packet."""
    msg = _get_test_proto_messages()["TimeseriesPacket"]()
    msg.ParseFromString(payload)
    return int(msg.header.num_samples)


def _build_ts_payloads(n_packets: int = 2):
    """
    Create a run of contiguous timeseries packets.

    Packet stamps advance by the packet's own duration (3 samples at 2 Hz =
    1.5 s), so the timestamps and the sample counters tell the same story --
    which is what a real recording does, and what the scan shortcut checks.
    """
    packet_cls = _get_test_proto_messages()["TimeseriesPacket"]
    packets = []
    for offset in range(n_packets):
        sample_count = offset * 3
        elapsed_ns = offset * 3 * 1_000_000_000 // 2
        msg = packet_cls()
        hdr = msg.header
        common = hdr.common_header
        _set_timestamp(
            common.time,
            1_700_000_000 + elapsed_ns // 1_000_000_000,
            elapsed_ns % 1_000_000_000,
        )
        common.num_channels = 2
        common.sample_rate = 2.0
        common.channel_spacing = 10.0
        common.gauge_length = 5.0
        common.start_channel = 3
        common.timeseries_data_type = 4
        hdr.sample_count = sample_count
        hdr.num_samples = 3
        hdr.channel_step = 1
        msg.samples.extend(
            np.arange(offset * 6, offset * 6 + 6, dtype=np.float32).tolist()
        )
        packets.append(("TS05", msg.SerializeToString()))
    return packets


def _build_band_payloads():
    """Create two band packets."""
    packet_cls = _get_test_proto_messages()["BandPacket"]
    packets = []
    for offset in range(2):
        msg = packet_cls()
        hdr = msg.header
        common = hdr.common_header
        _set_timestamp(common.time, 1_700_000_100 + offset)
        common.num_channels = 2
        common.channel_spacing = 5.0
        common.gauge_length = 5.0
        common.start_channel = 1
        info = hdr.band_data_info.add()
        info.band_data_type = 13
        info.start = 10.0
        info.end = 20.0
        info.description = "phase"
        info.source = "sensor"
        info = hdr.band_data_info.add()
        info.band_data_type = 10
        info.start = 20.0
        info.end = 30.0
        info.description = "temp"
        info.source = "sensor"
        msg.samples.extend(
            np.arange(offset * 4, offset * 4 + 4, dtype=np.float32).tolist()
        )
        packets.append(("BAND", msg.SerializeToString()))
    return packets


def _build_real_fft_payloads():
    """Create two real-valued FFT packets."""
    return _build_fft_payloads(complex_data=False)


def _build_fft_payloads(*, complex_data: bool):
    """Create two FFT packets."""
    packet_cls = _get_test_proto_messages()["FFTPacket"]
    packets = []
    for offset in range(2):
        msg = packet_cls()
        hdr = msg.header
        common = hdr.common_header
        _set_timestamp(common.time, 1_700_000_200 + offset)
        common.num_channels = 2
        common.channel_spacing = 4.0
        common.gauge_length = 6.0
        common.start_channel = 0
        hdr.num_bins = 3
        hdr.bin_res = 0.5
        hdr.channel_step = 1
        hdr.has_complex_data = complex_data
        if complex_data:
            samples = np.arange(offset * 12, offset * 12 + 12, dtype=np.float32)
        else:
            samples = np.arange(offset * 6, offset * 6 + 6, dtype=np.float32)
        msg.samples.extend(samples.tolist())
        packets.append(("FFT-", msg.SerializeToString()))
    return packets


def _write_records(path: Path, records):
    """Write a synthetic protobuf recording."""
    with path.open("wb") as handle:
        for tag, payload in records:
            _write_record(handle, tag, payload)


def _mutate_record(records, index: int, message_type: str, mutator):
    """Return a copy of records with one protobuf payload mutated."""
    out = list(records)
    packet_cls = _get_test_proto_messages()[message_type]
    msg = packet_cls()
    tag, payload = out[index]
    msg.ParseFromString(payload)
    mutator(msg)
    out[index] = (tag, msg.SerializeToString())
    return out


def _mutate_all(records, message_type: str, mutator):
    """Return a copy of records with every protobuf payload mutated."""
    out = records
    for index in range(len(records)):
        out = _mutate_record(out, index, message_type, mutator)
    return out


class _BytesReader:
    """A minimal seekable binary reader over a bytes object."""

    def __init__(self, data):
        self._data = data
        self._pos = 0

    def seek(self, pos, whence=0):
        """Seek from the start, or from the end when whence is 2."""
        self._pos = len(self._data) + pos if whence == 2 else pos

    def tell(self):
        """Return the current position."""
        return self._pos

    def read(self, size=-1):
        """Read up to size bytes, or the remainder when size is negative."""
        end = len(self._data) if size < 0 else self._pos + size
        out = self._data[self._pos : end]
        self._pos = min(end, len(self._data))
        return out


class _CountingReader:
    """
    A binary handle that records how many bytes were actually read.

    Wrapping the handle (rather than sampling process memory) makes the cost
    of a scan directly observable: protobuf holds sample data in C++ memory
    that Python's allocation tracing never sees, but every byte still has to
    come through a ``read`` call first.
    """

    def __init__(self, handle):
        self._handle = handle
        self.bytes_read = 0

    def read(self, size=-1):
        """Read from the wrapped handle, accumulating the byte count."""
        out = self._handle.read(size)
        self.bytes_read += len(out)
        return out

    def __getattr__(self, name):
        """Delegate seek/tell and friends to the wrapped handle."""
        return getattr(self._handle, name)


def _bytes_read_by(func, path) -> int:
    """Return how many bytes ``func`` pulls off disk for ``path``."""
    with path.open("rb") as handle:
        reader = _CountingReader(handle)
        func(reader)
        return reader.bytes_read


def del_samples_beyond(msg, keep: int):
    """Trim a packet's sample array down to ``keep`` values."""
    del msg.samples[keep:]


def _pad_samples(records, message_type: str, pad: int):
    """Return records whose packets each carry ``pad`` extra samples."""
    return _mutate_all(
        records, message_type, lambda msg: msg.samples.extend([0.0] * pad)
    )


@pytest.fixture()
def ts_records():
    """Return baseline timeseries records."""
    return _build_ts_payloads()


@pytest.fixture()
def band_records():
    """Return baseline band records."""
    return _build_band_payloads()


@pytest.fixture()
def fft_records():
    """Return baseline real FFT records."""
    return _build_fft_payloads(complex_data=False)


@pytest.fixture()
def complex_fft_records():
    """Return baseline complex FFT records."""
    return _build_fft_payloads(complex_data=True)


@pytest.fixture()
def write_sintela_file(tmp_path):
    """Write one synthetic Sintela protobuf file and return its path."""

    def _write(name: str, records) -> Path:
        path = tmp_path / name
        _write_records(path, records)
        return path

    return _write


class TestSintelaProtobuf:
    """Tests for Sintela protobuf IO support."""

    def test_get_format(self, fiber_io, sintela_protobuf_path):
        """A registered Sintela protobuf file should be detected."""
        assert fiber_io.get_format(sintela_protobuf_path) == (
            "Sintela_Protobuf",
            "1",
        )

    def test_scan_matches_read_summary(self, fiber_io, sintela_protobuf_path):
        """Scan metadata should match the loaded patch summary."""
        summary = _payload_to_summary(fiber_io.scan(sintela_protobuf_path)[0])
        patch_summary = fiber_io.read(sintela_protobuf_path)[0].summary
        assert summary == patch_summary

    def test_ts_read_promotes_selected_meta_attrs(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """META should supplement stable provenance attrs."""
        path = write_sintela_file(
            "ts.pb", [("META", _build_meta_payload()), *ts_records]
        )
        patch = fiber_io.read(path)[0]
        assert patch.dims == ("time", "distance")
        assert patch.attrs.recorder_namespace == "manualRecord/recorder"
        assert patch.attrs.get("interrogator.serial_number") == "SN123"
        assert patch.attrs.get("interrogator.manufacturer") == "Sintela"
        assert patch.attrs.get("interrogator.model") == "Onyx"
        assert patch.attrs.fiber_id == 2
        assert patch.attrs.data_type == "strain"

    def test_channel_step_attr_survives_indexing(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """
        The vendor's channel_step attr keeps its name through the spool index.

        Attrs and coords are independent, so a coord-shaped attr name is an
        ordinary attr; the patch has no "channel" coord, so nothing collides
        and no warning should be emitted.
        """
        path = write_sintela_file("ts.pb", ts_records)
        patch = fiber_io.read(path)[0]
        assert patch.attrs.channel_step == 1
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            contents = dc.spool([patch]).get_contents()
        assert contents["channel_step"].iloc[0] == 1

    def test_band_read_returns_expected_dims(
        self, fiber_io, write_sintela_file, band_records
    ):
        """Band recordings should load into a 3D patch."""
        path = write_sintela_file("band.pb", band_records)
        patch = fiber_io.read(path)[0]
        assert patch.dims == ("time", "distance", "band")
        assert patch.shape == (2, 2, 2)
        assert "band_start_frequency" in patch.coords.coord_map
        assert patch.attrs.data_type == "frequency_band_energy"
        assert patch.attrs.data_units in (None, "")

    def test_band_read_preserves_attrs_for_single_semantic_type(
        self, fiber_io, write_sintela_file, band_records
    ):
        """Uniform BAND semantics should populate patch-level data attrs."""
        records = _mutate_all(
            band_records,
            "BandPacket",
            lambda msg: setattr(msg.header.band_data_info[1], "band_data_type", 13),
        )
        path = write_sintela_file("band_uniform.pb", records)
        patch = fiber_io.read(path)[0]
        assert patch.attrs.data_type == "frequency_band_energy"
        assert patch.attrs.data_units == get_quantity("rad")

    def test_fft_read_returns_expected_dims(
        self, fiber_io, write_sintela_file, fft_records
    ):
        """FFT recordings should load into a frequency-domain patch."""
        path = write_sintela_file("fft.pb", fft_records)
        patch = fiber_io.read(path)[0]
        assert patch.dims == ("time", "distance", "frequency")
        assert patch.shape == (2, 2, 3)

    def test_band_scan_matches_read_summary(
        self, fiber_io, write_sintela_file, band_records
    ):
        """Band scan should exercise the metadata-only summary path."""
        path = write_sintela_file("band.pb", band_records)
        scan_summary = _payload_to_summary(fiber_io.scan(path)[0])
        assert scan_summary == fiber_io.read(path)[0].summary

    def test_fft_scan_matches_read_summary(
        self, fiber_io, write_sintela_file, complex_fft_records
    ):
        """FFT scan should exercise the metadata-only summary path."""
        path = write_sintela_file("fft.pb", complex_fft_records)
        scan_summary = _payload_to_summary(fiber_io.scan(path)[0])
        assert scan_summary == fiber_io.read(path)[0].summary

    def test_complex_fft_not_labeled_power_spectral_density(
        self, fiber_io, write_sintela_file, complex_fft_records, fft_records
    ):
        """Complex coefficients are not real power per frequency."""
        complex_path = write_sintela_file("fft_complex.pb", complex_fft_records)
        real_path = write_sintela_file("fft_real.pb", fft_records)
        assert fiber_io.read(complex_path)[0].attrs.data_type == "fourier_transform"
        # real packets keep the power-spectral-density label
        assert fiber_io.read(real_path)[0].attrs.data_type == "power_spectral_density"

    def test_raw_frame_only_tags_are_not_detected(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """
        RF01 must not be advertised while raw_frames cannot be decoded.

        Header-derived coords would let such a file scan cleanly and only fail
        at read, so it is better left undetected.
        """
        records = [("RF01", payload) for _tag, payload in ts_records]
        path = write_sintela_file("rf01.pb", records)
        assert fiber_io.get_format(path) is False
        assert "RF01" not in sintela_utils._TAG_TO_PACKET

    def test_raw_frame_packets_raise_clear_error(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """Samples carried in raw_frames get a specific, actionable error."""
        packet_cls = _get_test_proto_messages()["TimeseriesPacket"]
        records = []
        for tag, payload in ts_records:
            msg = packet_cls()
            msg.ParseFromString(payload)
            msg.ClearField("samples")
            msg.raw_frames = b"\x00\x01\x02\x03"
            records.append((tag, msg.SerializeToString()))
        path = write_sintela_file("ts_raw_frames.pb", records)
        with pytest.raises(InvalidFiberFileError, match="raw_frames"):
            fiber_io.read(path)

    def test_sample_count_rollover_is_contiguous(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """A uint32 sample_count wrap is not a gap."""
        packet_cls = _get_test_proto_messages()["TimeseriesPacket"]
        modulus = sintela_utils._SAMPLE_COUNT_MODULUS
        records = []
        for index, (tag, payload) in enumerate(ts_records):
            msg = packet_cls()
            msg.ParseFromString(payload)
            # first packet ends exactly on the boundary; the next wraps to 0
            msg.header.sample_count = (
                modulus - msg.header.num_samples if index == 0 else 0
            )
            records.append((tag, msg.SerializeToString()))
        path = write_sintela_file("ts_rollover.pb", records)
        patch = fiber_io.read(path)[0]
        assert patch.shape[0] == sum(
            _get_num_samples(payload) for _tag, payload in ts_records
        )

    def test_time_coords_keep_datetime_dtype(
        self, fiber_io, write_sintela_file, ts_records, fft_records
    ):
        """Datetime coordinates should remain time-like after Sintela parsing."""
        ts_path = write_sintela_file("ts_time_units.pb", ts_records)
        fft_path = write_sintela_file("fft_time_units.pb", fft_records)

        assert "datetime64" in str(fiber_io.read(ts_path)[0].get_coord("time").dtype)
        assert "datetime64" in str(fiber_io.read(fft_path)[0].get_coord("time").dtype)

    def test_fft_attrs_do_not_default_to_time_series_units(
        self, fiber_io, write_sintela_file, fft_records
    ):
        """FFT packets should not inherit time-series phase units by default."""
        path = write_sintela_file("fft_attrs.pb", fft_records)
        patch = fiber_io.read(path)[0]
        assert patch.attrs.data_type == "power_spectral_density"
        assert patch.attrs.data_units in (None, "")

    def test_complex_fft_is_complex_dtype(
        self, fiber_io, write_sintela_file, complex_fft_records
    ):
        """Complex FFT packets should decode into a complex dtype."""
        path = write_sintela_file("fft_complex.pb", complex_fft_records)
        patch = fiber_io.read(path)[0]
        assert np.issubdtype(patch.data.dtype, np.complexfloating)

    def test_read_applies_distance_selection(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """Read should apply coord selectors through the core wrapper."""
        path = write_sintela_file("ts_select.pb", ts_records)
        patch = fiber_io.read(path, distance=(30, 31))[0]
        assert patch.shape == (6, 1)
        assert patch.coords.dims == ("time", "distance")

    def test_read_returns_empty_spool_for_empty_selection(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """Selectors that remove all samples should return an empty spool."""
        path = write_sintela_file("ts_empty_select.pb", ts_records)
        spool = fiber_io.read(path, distance=(999, 1000))
        assert len(spool) == 0

    def test_mixed_families_raise(
        self, fiber_io, write_sintela_file, ts_records, band_records
    ):
        """Mixed data packet families are not supported."""
        records = [
            ("META", _build_meta_payload()),
            *ts_records,
            band_records[0],
        ]
        path = write_sintela_file("mixed.pb", records)
        with pytest.raises(InvalidFiberFileError, match="Mixed Sintela protobuf"):
            fiber_io.scan(path)

    def test_non_contiguous_timeseries_caught_on_read_not_scan(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """
        A self-consistent gap is reported by read, not by scan.

        A paused and resumed acquisition looks, from the endpoints alone, like
        a longer continuous one: the timestamps corroborate the counters. The
        summary spans the gap and read raises. A gap whose timestamps do *not*
        corroborate is caught earlier; see the concatenation test.
        """
        records = _mutate_record(
            ts_records,
            1,
            "TimeseriesPacket",
            # 10 samples precede this packet rather than 3, and its stamp moves
            # out to match: 10 samples at 2 Hz is 5 s after the first.
            lambda msg: (
                setattr(msg.header, "sample_count", 10),
                _set_timestamp(msg.header.common_header.time, 1_700_000_005),
            ),
        )
        path = write_sintela_file("non_contiguous.pb", records)
        # The last packet reports 10 samples before it and adds 3 of its own.
        assert _payload_to_summary(fiber_io.scan(path)[0]).shape[0] == 13
        with pytest.raises(InvalidFiberFileError, match="Non-contiguous"):
            fiber_io.read(path)

    def test_bad_magic_returns_false(self, fiber_io, tmp_path):
        """Invalid magic bytes should not identify as the format."""
        path = tmp_path / "bad.pb"
        path.write_bytes(b"NOPE")
        assert not fiber_io.get_format(path)

    def test_scan_rejects_invalid_or_truncated_envelope_headers(
        self, fiber_io, tmp_path
    ):
        """Strict scan mode should raise for malformed envelope headers."""
        bad_magic = tmp_path / "bad_magic.pb"
        bad_magic.write_bytes(b"NOPE")
        with pytest.raises(
            InvalidFiberFileError, match="Invalid Sintela protobuf magic"
        ):
            fiber_io.scan(bad_magic)

        short_magic = tmp_path / "short_magic.pb"
        short_magic.write_bytes(b"\x01\x02")
        with pytest.raises(
            InvalidFiberFileError, match="Truncated Sintela protobuf magic"
        ):
            fiber_io.scan(short_magic)

        short_header = tmp_path / "short_header.pb"
        with short_header.open("wb") as handle:
            handle.write(struct.pack("<I", sintela_utils.PBUF_MAGIC))
            handle.write(b"TS05")
        with pytest.raises(
            InvalidFiberFileError, match="Truncated Sintela protobuf record header"
        ):
            fiber_io.scan(short_header)

    def test_get_format_rejects_unknown_meta_only_and_truncated_records(
        self, fiber_io, write_sintela_file, tmp_path
    ):
        """Envelope-only detection should reject malformed or unsupported files."""
        unknown_path = write_sintela_file("unknown.pb", [("ABCD", b"\x01")])
        assert not fiber_io.get_format(unknown_path)

        meta_only_path = write_sintela_file(
            "meta_only.pb", [("META", _build_meta_payload())]
        )
        assert not fiber_io.get_format(meta_only_path)
        with pytest.raises(
            InvalidFiberFileError, match="No supported Sintela protobuf"
        ):
            fiber_io.scan(meta_only_path)

        trunc_magic_path = tmp_path / "trunc_magic.pb"
        trunc_magic_path.write_bytes(b"\x01\x02")
        assert not fiber_io.get_format(trunc_magic_path)

        trunc_header_path = tmp_path / "trunc_header.pb"
        with trunc_header_path.open("wb") as handle:
            handle.write(struct.pack("<I", sintela_utils.PBUF_MAGIC))
            handle.write(b"TS05")
        assert not fiber_io.get_format(trunc_header_path)

        with pytest.raises(
            InvalidFiberFileError, match="Unsupported Sintela protobuf tag"
        ):
            fiber_io.scan(unknown_path)

    def test_get_format_tolerates_unknown_tags_before_supported_data(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """Format detection should keep scanning past unknown non-data records."""
        path = write_sintela_file(
            "unknown_then_ts.pb", [("ABCD", b"\x01"), *ts_records]
        )
        assert fiber_io.get_format(path) == ("Sintela_Protobuf", "1")
        assert fiber_io.scan(path)
        assert fiber_io.read(path)

    def test_get_format_preserves_reader_position(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """Format detection should not consume a reusable binary reader."""
        path = write_sintela_file("position.pb", [("META", b""), *ts_records])
        with path.open("rb") as handle:
            assert handle.read(4)
            before = handle.tell()
            assert fiber_io.get_format.func(fiber_io, handle) == (
                "Sintela_Protobuf",
                "1",
            )
            assert handle.tell() == before

    def test_truncated_payload_raises(self, fiber_io, tmp_path):
        """Truncated payloads should fail in scan."""
        path = tmp_path / "truncated.pb"
        with path.open("wb") as handle:
            handle.write(struct.pack("<I", sintela_utils.PBUF_MAGIC))
            handle.write(b"TS05")
            handle.write(struct.pack("<I", 10))
            handle.write(b"\x00")
        assert not fiber_io.get_format(path)
        with pytest.raises(InvalidFiberFileError, match="Truncated"):
            fiber_io.scan(path)
        # read takes the full-payload path, which detects the short tail too
        with pytest.raises(InvalidFiberFileError, match="Truncated"):
            fiber_io.read(path)

    def test_bad_protobuf_payloads_raise_invalid_fiber_file_error(
        self, fiber_io, write_sintela_file
    ):
        """Malformed protobuf payloads should be normalized to InvalidFiberFileError."""
        meta_path = write_sintela_file(
            "bad_meta.pb",
            [("META", b"\xff"), ("TS05", b"\x08")],
        )
        with pytest.raises(
            InvalidFiberFileError, match="Failed to parse Sintela protobuf META payload"
        ):
            fiber_io.scan(meta_path)

        data_path = write_sintela_file("bad_data.pb", [("TS05", b"\xff")])
        with pytest.raises(
            InvalidFiberFileError, match="Failed to parse Sintela protobuf TS05 payload"
        ):
            fiber_io.scan(data_path)

    def test_read_rejects_family_change_between_endpoints(
        self, fiber_io, write_sintela_file, ts_records, band_records
    ):
        """
        A foreign packet between matching endpoints is caught on read.

        Both endpoints are timeseries, so the scan shortcut accepts the file;
        the streaming read visits every packet and rejects the BAND one.
        """
        records = [ts_records[0], band_records[0], ts_records[1]]
        path = write_sintela_file("ts_band_ts.pb", records)
        with pytest.raises(InvalidFiberFileError, match="Mixed Sintela protobuf"):
            fiber_io.read(path)

    def test_read_rejects_more_samples_than_endpoints_imply(
        self, fiber_io, write_sintela_file
    ):
        """
        Extra samples between the endpoints are caught before they overrun.

        The output array is sized from the endpoints, so a middle packet that
        pushes past that total has to fail rather than write out of bounds.
        """
        records = _build_ts_payloads(3)
        # Renumber the last packet so the endpoints imply two packets' worth
        # of samples while three packets are actually present.
        records = _mutate_record(
            records,
            2,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header, "sample_count", 3),
        )
        path = write_sintela_file("ts_overrun.pb", records)
        with pytest.raises(InvalidFiberFileError, match="Non-contiguous"):
            fiber_io.read(path)

    def test_short_final_packet_is_measured_from_the_last_endpoint(
        self, fiber_io, write_sintela_file
    ):
        """
        A recording ending in a partial packet reports its true length.

        Recorders close a file whenever the acquisition stops, so the final
        packet is routinely shorter than the rest. The endpoint total has to
        add *that* packet's length to the counter difference; adding the first
        packet's length instead happens to give the same answer for a uniform
        file, which is what every other fixture here is.
        """
        records = _build_ts_payloads(3)
        # 3 + 3 + 1 samples: counts stay 0, 3, 6 and the tail is short.
        records = _mutate_record(
            records,
            2,
            "TimeseriesPacket",
            lambda msg: (
                setattr(msg.header, "num_samples", 1),
                del_samples_beyond(msg, msg.header.common_header.num_channels),
            ),
        )
        path = write_sintela_file("ts_short_tail.pb", records)
        # Asserted on the shortcut itself, not just the final answer: adding
        # the wrong packet's length makes the endpoint total disagree with the
        # timestamps, and the fallback then returns the right shape anyway.
        with path.open("rb") as handle:
            endpoints = sintela_utils._get_endpoint_metadata(handle)
        assert endpoints is not None
        assert endpoints[0].shape[0] == 7
        assert _payload_to_summary(fiber_io.scan(path)[0]).shape[0] == 7
        assert fiber_io.read(path)[0].shape[0] == 7

    def test_reset_sample_counter_falls_back_instead_of_trusting_endpoints(
        self, fiber_io, write_sintela_file
    ):
        """
        A counter that restarts mid-file must not be read as billions of samples.

        `sample_count` is a uint32, so the endpoint difference is taken modulo
        2**32 to survive a wrap. A counter that *resets* instead looks like an
        enormous backwards jump, which would size the patch from a number the
        file cannot possibly hold. The shortcut declines and the full path
        reports the gap.
        """
        records = _build_ts_payloads(3)
        # Start high so the counter appears to restart: the last packet's
        # count is far below the first, and the modular difference is enormous.
        records = _mutate_record(
            records,
            0,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header, "sample_count", 1000),
        )
        path = write_sintela_file("ts_counter_reset.pb", records)
        with pytest.raises(InvalidFiberFileError, match="Non-contiguous"):
            fiber_io.scan(path)
        with pytest.raises(InvalidFiberFileError, match="Non-contiguous"):
            fiber_io.read(path)

    def test_read_picks_up_meta_after_the_first_data_packet(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """
        Read honours a META record wherever it appears, as it always has.

        The endpoint scan only walks as far as the first data packet, so a
        trailing META is invisible to it; the streaming read visits every
        record and must not lose metadata the old full-decode path collected.
        """
        records = [ts_records[0], ("META", _build_meta_payload()), ts_records[1]]
        path = write_sintela_file("ts_trailing_meta.pb", records)
        patch = fiber_io.read(path)[0]
        assert patch.attrs.instrument_manufacturer == "Sintela"
        assert patch.attrs.serial_number == "SN123"

    def test_timeseries_read_rejects_dropped_samples(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """Dropped-sample flags should fail on read."""
        records = _mutate_record(
            ts_records,
            0,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header.common_header, "has_dropped_samples", True),
        )
        path = write_sintela_file("dropped_read.pb", records)
        with pytest.raises(InvalidFiberFileError, match="Dropped samples"):
            fiber_io.read(path)

    def test_timeseries_scan_rejects_dropped_samples(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """Dropped-sample flags should fail on scan as well."""
        records = _mutate_record(
            ts_records,
            0,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header.common_header, "has_dropped_samples", True),
        )
        path = write_sintela_file("dropped_scan.pb", records)
        with pytest.raises(InvalidFiberFileError, match="Dropped samples"):
            fiber_io.scan(path)

    def test_timeseries_scan_rejects_missing_time(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """Timeseries scan should reject packets without a start time."""
        records = _mutate_record(
            ts_records,
            0,
            "TimeseriesPacket",
            lambda msg: msg.header.common_header.ClearField("time"),
        )
        path = write_sintela_file("ts_missing_time.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Missing Sintela protobuf start time"
        ):
            fiber_io.scan(path)

    @pytest.mark.parametrize("bad_sample_rate", [0.0, -1.0, np.nan, np.inf])
    def test_timeseries_scan_rejects_invalid_sample_rate(
        self, fiber_io, write_sintela_file, ts_records, bad_sample_rate
    ):
        """Timeseries scan should reject invalid sample rates."""
        records = _mutate_record(
            ts_records,
            0,
            "TimeseriesPacket",
            lambda msg: setattr(
                msg.header.common_header, "sample_rate", bad_sample_rate
            ),
        )
        path = write_sintela_file(f"ts_bad_rate_{bad_sample_rate}.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Invalid Sintela protobuf sample_rate"
        ):
            fiber_io.scan(path)

    @pytest.mark.parametrize("bad_spacing", [0.0, -1.0, np.nan, np.inf])
    def test_scan_rejects_invalid_channel_spacing(
        self, fiber_io, write_sintela_file, ts_records, bad_spacing
    ):
        """Invalid channel spacing should raise a format-specific error."""
        records = _mutate_all(
            ts_records,
            "TimeseriesPacket",
            lambda msg: setattr(
                msg.header.common_header, "channel_spacing", bad_spacing
            ),
        )
        path = write_sintela_file(f"ts_bad_spacing_{bad_spacing}.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Invalid Sintela protobuf channel_spacing"
        ):
            fiber_io.scan(path)

    @pytest.mark.parametrize("bad_step", [0, -1])
    def test_scan_rejects_invalid_channel_step(
        self, fiber_io, write_sintela_file, ts_records, bad_step
    ):
        """Invalid channel steps should raise a format-specific error."""
        records = _mutate_all(
            ts_records,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header, "channel_step", bad_step),
        )
        path = write_sintela_file(f"ts_bad_step_{bad_step}.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Invalid Sintela protobuf channel_step"
        ):
            fiber_io.scan(path)

    @pytest.mark.parametrize("bad_count", [0, -1])
    def test_scan_rejects_invalid_num_channels(
        self, fiber_io, write_sintela_file, ts_records, bad_count
    ):
        """Common headers with non-positive channel counts should fail early."""
        records = _mutate_all(
            ts_records,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header.common_header, "num_channels", bad_count),
        )
        path = write_sintela_file(f"ts_bad_channels_{bad_count}.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Invalid Sintela protobuf num_channels"
        ):
            fiber_io.scan(path)

    def test_timeseries_read_rejects_bad_size_and_inconsistent_headers(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """Timeseries read should reject malformed payloads and inconsistent headers."""
        records = _mutate_record(
            ts_records,
            0,
            "TimeseriesPacket",
            lambda msg: msg.samples.pop(),
        )
        bad_size = write_sintela_file("ts_bad_size.pb", records)
        with pytest.raises(InvalidFiberFileError, match="TS sample payload size"):
            fiber_io.read(bad_size)

        records = _mutate_record(
            ts_records,
            1,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header.common_header, "sample_rate", 3.0),
        )
        bad_rate = write_sintela_file("ts_bad_rate.pb", records)
        with pytest.raises(InvalidFiberFileError, match="Inconsistent sample_rate"):
            fiber_io.scan(bad_rate)

        records = _mutate_record(
            ts_records,
            1,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header.common_header, "num_channels", 3),
        )
        bad_channels = write_sintela_file("ts_bad_channels.pb", records)
        with pytest.raises(InvalidFiberFileError, match="Inconsistent num_channels"):
            fiber_io.read(bad_channels)

    @pytest.mark.parametrize("bad_num_samples", [0, -1])
    def test_timeseries_scan_rejects_invalid_num_samples(
        self, fiber_io, write_sintela_file, ts_records, bad_num_samples
    ):
        """Timeseries packet sample counts must be positive before scan/read."""
        records = _mutate_record(
            ts_records,
            1,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header, "num_samples", bad_num_samples),
        )
        path = write_sintela_file(f"ts_bad_num_samples_{bad_num_samples}.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Invalid Sintela protobuf num_samples"
        ):
            fiber_io.scan(path)
        with pytest.raises(
            InvalidFiberFileError, match="Invalid Sintela protobuf num_samples"
        ):
            fiber_io.read(path)

    def test_timeseries_read_rejects_non_contiguous_and_missing_time(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """Timeseries read should reject gaps and missing timestamps."""
        records = _mutate_record(
            ts_records,
            1,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header, "sample_count", 10),
        )
        non_contiguous = write_sintela_file("ts_gap.pb", records)
        with pytest.raises(InvalidFiberFileError, match="Non-contiguous"):
            fiber_io.read(non_contiguous)

        records = _mutate_record(
            ts_records,
            0,
            "TimeseriesPacket",
            lambda msg: msg.header.common_header.ClearField("time"),
        )
        missing_time = write_sintela_file("ts_missing_time_read.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Missing Sintela protobuf start time"
        ):
            fiber_io.read(missing_time)

    def test_band_scan_and_read_reject_malformed_packets(
        self, fiber_io, write_sintela_file, band_records
    ):
        """Band scan/read should reject missing defs, missing time, and bad sizes."""
        records = _mutate_all(
            band_records,
            "BandPacket",
            lambda msg: msg.header.ClearField("band_data_info"),
        )
        no_defs = write_sintela_file("band_no_defs.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Band packets missing band definitions"
        ):
            fiber_io.scan(no_defs)
        with pytest.raises(
            InvalidFiberFileError, match="Band packets missing band definitions"
        ):
            fiber_io.read(no_defs)

        records = _mutate_record(
            band_records,
            0,
            "BandPacket",
            lambda msg: msg.header.common_header.ClearField("time"),
        )
        missing_time = write_sintela_file("band_missing_time.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Missing time in Sintela BAND packet"
        ):
            fiber_io.scan(missing_time)
        with pytest.raises(
            InvalidFiberFileError, match="Missing time in Sintela BAND packet"
        ):
            fiber_io.read(missing_time)

        records = _mutate_record(
            band_records,
            0,
            "BandPacket",
            lambda msg: msg.samples.pop(),
        )
        bad_size = write_sintela_file("band_bad_size.pb", records)
        with pytest.raises(InvalidFiberFileError, match="BAND payload size"):
            fiber_io.read(bad_size)

    @pytest.mark.parametrize("complex_data", [False, True])
    def test_fft_read_rejects_bad_sizes(
        self, fiber_io, write_sintela_file, complex_data
    ):
        """FFT read should reject malformed real and complex sample payloads."""
        records = _build_fft_payloads(complex_data=complex_data)
        records = _mutate_record(records, 0, "FFTPacket", lambda msg: msg.samples.pop())
        path = write_sintela_file(f"fft_bad_size_{complex_data}.pb", records)
        with pytest.raises(InvalidFiberFileError, match="FFT payload size"):
            fiber_io.read(path)

    @pytest.mark.parametrize("bad_bin_res", [0.0, -1.0, np.nan, np.inf])
    def test_fft_scan_rejects_invalid_bin_res(
        self, fiber_io, write_sintela_file, fft_records, bad_bin_res
    ):
        """FFT bin resolution should be finite and positive."""
        records = _mutate_all(
            fft_records,
            "FFTPacket",
            lambda msg: setattr(msg.header, "bin_res", bad_bin_res),
        )
        path = write_sintela_file(f"fft_bad_bin_res_{bad_bin_res}.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Invalid Sintela protobuf bin_res"
        ):
            fiber_io.scan(path)

    @pytest.mark.parametrize("bad_num_bins", [0, -1])
    def test_fft_scan_rejects_invalid_num_bins(
        self, fiber_io, write_sintela_file, fft_records, bad_num_bins
    ):
        """FFT headers with non-positive bin counts should fail early."""
        records = _mutate_all(
            fft_records,
            "FFTPacket",
            lambda msg: setattr(msg.header, "num_bins", bad_num_bins),
        )
        path = write_sintela_file(f"fft_bad_num_bins_{bad_num_bins}.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Invalid Sintela protobuf num_bins"
        ):
            fiber_io.scan(path)

    def test_fft_scan_and_read_reject_missing_time(
        self, fiber_io, write_sintela_file, fft_records
    ):
        """FFT scan/read should reject packets without timestamps."""
        records = _mutate_record(
            fft_records,
            0,
            "FFTPacket",
            lambda msg: msg.header.common_header.ClearField("time"),
        )
        path = write_sintela_file("fft_missing_time.pb", records)
        with pytest.raises(
            InvalidFiberFileError, match="Missing time in Sintela FFT packet"
        ):
            fiber_io.scan(path)
        with pytest.raises(
            InvalidFiberFileError, match="Missing time in Sintela FFT packet"
        ):
            fiber_io.read(path)

    def test_missing_protobuf_only_affects_scan_and_read(
        self, sintela_protobuf_path, monkeypatch
    ):
        """Detection should stay available without protobuf support."""
        path = sintela_protobuf_path

        def _raise(*args, **kwargs):
            raise MissingOptionalDependencyError("protobuf missing")

        monkeypatch.setattr(sintela_utils, "_get_proto_messages", _raise)
        monkeypatch.setattr(sintela_utils, "_get_meta_message_class", _raise)

        fiber_io = SintelaProtobufV1()
        assert fiber_io.get_format(path) == ("Sintela_Protobuf", "1")
        with pytest.raises(MissingOptionalDependencyError, match="protobuf missing"):
            fiber_io.scan(path)
        with pytest.raises(MissingOptionalDependencyError, match="protobuf missing"):
            fiber_io.read(path)


_FAMILY_BUILDERS = [
    (_build_ts_payloads, "TimeseriesPacket"),
    (_build_band_payloads, "BandPacket"),
    (_build_real_fft_payloads, "FFTPacket"),
]


class TestSintelaProtobufScanCost:
    """Scanning must not pay for the sample data it does not report."""

    def test_scan_reads_do_not_grow_with_recording_length(self, write_sintela_file):
        """
        A longer timeseries recording must not cost a longer scan.

        This is what makes indexing a directory of large recordings tractable:
        the summary is derived from the first and last packets, so a file with
        sixteen times as many packets is read no more than a short one. The
        packets are padded past the header prefix so the reads being compared
        are the bounded ones, not an artifact of tiny fixtures.
        """

        def _write(name, n_packets):
            records = _pad_samples(
                _build_ts_payloads(n_packets), "TimeseriesPacket", 20_000
            )
            return write_sintela_file(name, records)

        short = _write("ts_short.pb", 4)
        long = _write("ts_long.pb", 64)
        assert long.stat().st_size > 8 * short.stat().st_size

        short_bytes = _bytes_read_by(sintela_utils.scan_payload, short)
        long_bytes = _bytes_read_by(sintela_utils.scan_payload, long)
        assert short_bytes == long_bytes
        assert long_bytes < long.stat().st_size / 4

    def test_read_and_scan_take_the_endpoint_shortcut(
        self, monkeypatch, write_sintela_file, ts_records, sintela_protobuf_path
    ):
        """
        Both entry points reach the fast path, for synthetic and real files.

        Every behavioural test here passes whether or not the shortcut engages,
        because the full path reports the same errors and returns the same
        data. Only the cost differs, so the call sites are checked directly:
        without this, `read` could quietly revert to decoding every packet up
        front and nothing would notice.
        """
        path = write_sintela_file("ts_shortcut.pb", ts_records)
        for target in (path, Path(sintela_protobuf_path)):
            with target.open("rb") as handle:
                assert sintela_utils._get_endpoint_metadata(handle) is not None

        used = []
        original = sintela_utils.TimeseriesMetadata.decode_stream

        def _spy(self, resource, meta):
            used.append(True)
            return original(self, resource, meta)

        monkeypatch.setattr(
            sintela_utils.TimeseriesMetadata, "decode_stream", _spy, raising=True
        )
        with path.open("rb") as handle:
            sintela_utils.read_payload(handle)
        assert used, "read_payload did not take the streaming endpoint path"

    def test_sample_counter_rollover_keeps_the_shortcut(
        self, write_sintela_file, ts_records
    ):
        """
        A counter that wraps mid-file is still summarized from its endpoints.

        Dropping the modulus would make the difference negative, which the
        file-size bound rejects: the answer stays right but arrives by the slow
        route, so only a cost assertion catches it.
        """
        packet_cls = _get_test_proto_messages()["TimeseriesPacket"]
        modulus = sintela_utils._SAMPLE_COUNT_MODULUS
        records = []
        for index, (tag, payload) in enumerate(ts_records):
            msg = packet_cls()
            msg.ParseFromString(payload)
            msg.header.sample_count = (
                modulus - msg.header.num_samples if index == 0 else 0
            )
            records.append((tag, msg.SerializeToString()))
        path = write_sintela_file("ts_rollover_shortcut.pb", records)
        with path.open("rb") as handle:
            endpoints = sintela_utils._get_endpoint_metadata(handle)
        assert endpoints is not None
        assert endpoints[0].shape[0] == 6

    def test_find_last_record_ignores_a_decoy_frame_inside_the_payload(self):
        """
        Only the frame whose extent ends on EOF is the final record.

        Sample bytes can spell out a magic word and a tag; anchoring on EOF is
        what distinguishes the real framing from that noise, and the backwards
        search reaches the decoy first.
        """
        magic = struct.pack("<I", sintela_utils.PBUF_MAGIC)
        # A last record whose 20-byte payload embeds a plausible-looking frame.
        decoy = magic + b"TS05" + struct.pack("<I", 99)
        payload = b"AAAA" + decoy + b"BBBB"
        buf = magic + b"TS05" + struct.pack("<I", len(payload)) + payload

        found = sintela_utils._find_last_record(_BytesReader(buf), len(buf), len(buf))
        assert found is not None
        assert found[0] == "TS05"
        # The real record's size, not the decoy's 99.
        assert found[1] == len(payload)

    def test_scan_hands_the_parser_headers_not_payloads(
        self, monkeypatch, write_sintela_file, band_records
    ):
        """
        The non-timeseries scan path asks the iterator for header-only records.

        Testing the iterator directly leaves the call site untested, and bytes
        read cannot tell the difference: a payload streamed past and dropped
        costs the same I/O as one that is kept. What changes is what reaches
        the parser, so that is what this measures.
        """
        records = _pad_samples(band_records, "BandPacket", 200_000)
        path = write_sintela_file("band_wiring.pb", records)
        payload_sizes = []
        original = sintela_utils._parse_records

        def _spy(records, **kwargs):
            records = list(records)
            payload_sizes.extend(len(record.payload) for record in records)
            return original(records, **kwargs)

        monkeypatch.setattr(sintela_utils, "_parse_records", _spy)
        with path.open("rb") as handle:
            sintela_utils.scan_payload(handle)
        assert payload_sizes
        # A padded BAND payload is ~1 MB; a header is a few hundred bytes.
        assert max(payload_sizes) < 4096

    def test_detection_does_not_pull_whole_payloads(
        self, write_sintela_file, band_records
    ):
        """
        Format detection reads tags, and stops at the first one it recognizes.

        It needs 12 bytes of framing per record and nothing behind them, so it
        must not stream a payload it will never look at -- over a directory of
        large recordings that is the difference between megabytes and
        gigabytes of pointless I/O.
        """
        records = _pad_samples(band_records, "BandPacket", 200_000)
        path = write_sintela_file("band_detect.pb", records)
        one_payload = path.stat().st_size / len(records)
        detect_bytes = _bytes_read_by(sintela_utils.get_supported_family_tag, path)
        assert detect_bytes < one_payload

    def test_records_after_a_large_data_packet_stay_aligned(
        self, write_sintela_file, band_records
    ):
        """
        A non-data record following a truncated payload is still read correctly.

        Header-only reads leave the rest of a data payload to be stepped over
        on the next iteration. Every record here is a data record in the other
        fixtures, so the deferred amount is always overwritten before it can
        matter; a META record behind a large packet is what would expose a
        stream left in the wrong place.
        """
        big = _pad_samples(band_records, "BandPacket", 200_000)
        records = [big[0], ("META", _build_meta_payload()), big[1]]
        path = write_sintela_file("band_then_meta.pb", records)
        with path.open("rb") as handle:
            seen = [
                (record.tag, len(record.payload))
                for record in sintela_utils._iter_envelope_records(
                    handle, strict=True, headers_only=True
                )
            ]
        assert [tag for tag, _size in seen] == ["BAND", "META", "BAND"]
        # The META payload is whole; the BAND payloads are trimmed to headers.
        assert seen[1][1] == len(_build_meta_payload())
        assert seen[0][1] < 4096 and seen[2][1] < 4096

    def test_short_first_packet_still_finds_the_final_record(self, write_sintela_file):
        """
        A recording opening with a small packet is still summarized cheaply.

        The tail-search window is sized from the first packet on the assumption
        that the last one is about as big. A recorder that ramps up breaks
        that: one file in the 10,308-recording archive this was written for
        opens with a 204 KB warm-up packet and ends with a 526 KB one. Without
        the wider second attempt such a file silently costs a full read.
        """
        records = _build_ts_payloads(3)
        for index in (1, 2):
            records = _mutate_record(
                records,
                index,
                "TimeseriesPacket",
                lambda msg: msg.samples.extend([0.0] * 20_000),
            )
        path = write_sintela_file("ts_warmup_first.pb", records)
        # The first packet is a few dozen bytes; the last is ~80 KB.
        with path.open("rb") as handle:
            endpoints = sintela_utils._get_endpoint_metadata(handle)
        assert endpoints is not None
        assert endpoints[0].shape[0] == 9

    def test_endpoint_packet_length_must_fit_its_own_record(self, write_sintela_file):
        """
        A packet cannot claim more samples than its record has room for.

        The last packet's length cancels out of the timestamp comparison, so
        it is the one field the time check cannot see. The file is built so
        this is the *only* check that can reject it: a padded middle packet
        keeps the whole-file bound satisfied and the counters are untouched,
        so the timestamps agree.
        """
        records = _build_ts_payloads(3)
        # Pad the middle packet so the file is comfortably larger than the
        # claimed sample count needs; the whole-file bound then cannot object.
        records = _mutate_record(
            records,
            1,
            "TimeseriesPacket",
            lambda msg: msg.samples.extend([0.0] * 50_000),
        )
        claimed = 10_000
        for index in (0, 2):
            bad = _mutate_record(
                records,
                index,
                "TimeseriesPacket",
                lambda msg: setattr(msg.header, "num_samples", claimed),
            )
            path = write_sintela_file(f"ts_impossible_{index}.pb", bad)
            with path.open("rb") as handle:
                assert sintela_utils._get_endpoint_metadata(handle) is None, (
                    f"endpoint {index} claimed {claimed} samples its record "
                    "cannot hold and was believed"
                )

    def test_streaming_read_keeps_only_one_decoded_packet(
        self, monkeypatch, write_sintela_file
    ):
        """
        Only one decoded packet may be alive at a time during a streaming read.

        Protobuf releases a packet's arena only when the whole message dies, so
        one that outlives its turn keeps owning its samples whatever is done to
        its fields. Process memory is too noisy to assert on at fixture size;
        the object graph is exact.
        """
        sample_cls = sintela_utils._get_proto_messages(include_sample_fields=True)[
            "TimeseriesPacket"
        ]
        path = write_sintela_file("ts_stream_mem.pb", _build_ts_payloads(6))
        live, kept = [], []
        # Whatever the rest of the session holds is not this test's business;
        # only the growth across the read is.
        baseline = sum(1 for obj in gc.get_objects() if type(obj) is sample_cls)
        original_parse = sintela_utils._parse_packet
        original_from_parsed = sintela_utils.TimeseriesMetadata.from_parsed.__func__

        def _spy_parse(tag, payload, messages, decode_error):
            live.append(sum(1 for obj in gc.get_objects() if type(obj) is sample_cls))
            return original_parse(tag, payload, messages, decode_error)

        def _spy_from_parsed(cls, parsed, meta, total_samples=None):
            kept.append([type(msg) for _tag, msg in parsed])
            return original_from_parsed(cls, parsed, meta, total_samples=total_samples)

        monkeypatch.setattr(sintela_utils, "_parse_packet", _spy_parse)
        monkeypatch.setattr(
            sintela_utils.TimeseriesMetadata,
            "from_parsed",
            classmethod(_spy_from_parsed),
        )
        with path.open("rb") as handle:
            sintela_utils.read_payload(handle)

        assert len(live) >= 6, "the streaming path did not decode packet by packet"
        assert max(live) == baseline, (
            "a previously decoded packet was still alive when the next was "
            f"decoded (live counts: {live}, baseline: {baseline})"
        )
        held = [cls for group in kept for cls in group]
        assert held, "nothing was handed to from_parsed"
        assert not any(
            any(field.name == "samples" for field in cls.DESCRIPTOR.fields)
            for cls in held
        ), "the read retains messages that are able to hold samples"

    @pytest.mark.parametrize(("builder", "message_type"), _FAMILY_BUILDERS)
    def test_header_only_records_exclude_samples(
        self, write_sintela_file, builder, message_type
    ):
        """
        Every family's scan path is handed headers, never sample payloads.

        The families that cannot use the endpoint shortcut still have to visit
        each packet; they must do it without pulling the samples along, so the
        payload handed to the parser stays the same size as the recording's
        samples grow.
        """

        def _payload_bytes(pad: int) -> int:
            records = _pad_samples(builder(), message_type, pad)
            path = write_sintela_file(f"headers_{message_type}_{pad}.pb", records)
            with path.open("rb") as handle:
                return sum(
                    len(record.payload)
                    for record in sintela_utils._iter_envelope_records(
                        handle, strict=True, headers_only=True
                    )
                )

        assert _payload_bytes(0) == _payload_bytes(50_000)

    def test_large_payloads_are_skipped_not_streamed(self, write_sintela_file):
        """
        A payload large enough to be worth a seek is stepped over, not read.

        Small remainders are streamed and discarded because a seek costs a
        platter rotation on spinning media, which buys back more than the
        transfer it saves. Past the threshold the seek wins, and this pins that
        the large-payload branch really does skip the bytes.
        """
        # Fixed size, not derived from the threshold: a fixture that scales
        # with the constant would keep passing however the constant is retuned.
        n_floats = 1_000_000
        records = _pad_samples(_build_band_payloads(), "BandPacket", n_floats)
        path = write_sintela_file("band_large.pb", records)
        size = path.stat().st_size
        # Guard on the payload bytes the fixture actually produces, so this
        # cannot silently turn into a permanent skip.
        payload_bytes = size / len(records)
        if payload_bytes <= sintela_utils._SEEK_SKIP_THRESHOLD:
            pytest.skip("seek threshold retuned above this fixture's payload size")

        def _walk(handle):
            for _record in sintela_utils._iter_envelope_records(
                handle, strict=True, headers_only=True
            ):
                pass

        assert _bytes_read_by(_walk, path) < size / 4

    @pytest.mark.parametrize(("builder", "message_type"), _FAMILY_BUILDERS)
    def test_scan_memory_independent_of_sample_count(self, builder, message_type):
        """
        Metadata-only scans must not retain the sample bytes.

        Protobuf keeps undeclared wire fields as unknown fields, so omitting
        the sample declarations is not by itself enough. The invariant is that
        scan-retained size does not grow with the sample payload.
        """

        def _retained(pad: int) -> int:
            records = [
                _envelope(record)
                for record in _pad_samples(builder(), message_type, pad)
            ]
            parsed, _ = sintela_utils._parse_records(records, scan_mode=True)
            return sum(msg.ByteSize() for _tag, msg in parsed)

        assert _retained(0) == _retained(5_000)


class TestSintelaProtobufWireHelpers:
    """
    Tests for the hand-rolled wire-format helpers.

    These read protobuf framing directly so a packet's header can be taken
    from a short prefix. Each returns None rather than raising when a payload
    does not fit that shape, which sends the caller to the read-everything
    path; the fallbacks are exercised here since a well-formed file never
    reaches them.
    """

    def test_read_varint_spans_multiple_bytes(self):
        """Varints longer than one byte decode, and a truncated one is None."""
        # 300 encodes as two bytes; the continuation bit drives the loop.
        assert sintela_utils._read_varint(b"\xac\x02", 0) == (300, 2)
        assert sintela_utils._read_varint(b"\x05", 0) == (5, 1)
        # A varint whose continuation bit never terminates is unusable.
        assert sintela_utils._read_varint(b"\xac", 0) == (None, 1)

    def test_leading_header_bytes_rejects_unusable_prefixes(self):
        """Only a complete field-1 submessage at the front is usable."""
        header = b"\x0a\x03abc"
        assert sintela_utils._leading_header_bytes(header) == header
        # Empty, or not starting with field 1 (here field 2).
        assert sintela_utils._leading_header_bytes(b"") is None
        assert sintela_utils._leading_header_bytes(b"\x12\x03abc") is None
        # Field 1 declared longer than the prefix actually holds.
        assert sintela_utils._leading_header_bytes(b"\x0a\x7fab") is None

    def test_parse_frame_rejects_short_and_wrong_magic(self):
        """Framing needs twelve bytes opening with the magic word."""
        frame = (
            struct.pack("<I", sintela_utils.PBUF_MAGIC) + b"TS05" + struct.pack("<I", 7)
        )
        assert sintela_utils._parse_frame(frame) == ("TS05", 7)
        assert sintela_utils._parse_frame(b"short") is None
        assert sintela_utils._parse_frame(b"NOPE" + b"TS05" + b"\x00" * 4) is None

    def test_parse_packet_header_returns_none_for_undecodable_header(self):
        """A header submessage that will not decode defers to the slow path."""
        # Field 1 holding two bytes that are not a valid submessage.
        assert sintela_utils._parse_packet_header("TS05", b"\x0a\x02\xff\xff") is None

    def test_decode_stream_refuses_to_return_an_underfilled_array(
        self, write_sintela_file, ts_records
    ):
        """
        A stream that does not fill the expected shape raises, never returns.

        A real file cannot reach this: the packets that fill the array are the
        same ones whose contiguity is validated, so a short fill implies a gap
        that is reported first. The check is unconditional (not an assertion)
        because the alternative to raising is handing back the uninitialized
        tail of the output array, so it is driven directly here.
        """
        path = write_sintela_file("ts_underfill.pb", ts_records)
        parsed, meta = sintela_utils._parse_records(
            [_envelope(record) for record in ts_records], scan_mode=True
        )
        # Claim far more samples than the two packets actually carry.
        metadata = sintela_utils.TimeseriesMetadata.from_parsed(
            parsed, meta, total_samples=999
        )
        with path.open("rb") as handle:
            with pytest.raises(InvalidFiberFileError, match="expected sample count"):
                metadata.decode_stream(handle, meta)

    def test_endpoint_time_disagreement_rejects_the_shortcut(
        self, fiber_io, write_sintela_file
    ):
        """
        Two recordings concatenated are not summarized as one.

        Both halves restart their sample counter, so the endpoints alone look
        like a short contiguous file and the file-size bound passes -- the
        derived length is too *small*, not too large. The timestamps are the
        only witness that disagrees, and without them the index would advertise
        half the real duration with no error anywhere.
        """
        first = _build_ts_payloads(4)
        second = _build_ts_payloads(4)
        # Second half continues in time but restarts its counter at zero.
        second = _mutate_all(
            second,
            "TimeseriesPacket",
            lambda msg: _set_timestamp(
                msg.header.common_header.time,
                1_700_000_100 + int(msg.header.sample_count),
            ),
        )
        path = write_sintela_file("ts_concat.pb", [*first, *second])
        with path.open("rb") as handle:
            assert sintela_utils._get_endpoint_metadata(handle) is None
        # The full path still reports the real problem rather than a summary.
        with pytest.raises(InvalidFiberFileError, match="Non-contiguous"):
            fiber_io.scan(path)

    def test_endpoint_validation_failure_falls_back(self, write_sintela_file):
        """
        Endpoints that fail validation defer to the full read, never raise here.

        A disagreement between the two endpoint headers may be a genuinely bad
        file or a bogus tail match. The shortcut cannot tell them apart, so it
        declines and lets the path that sees every packet decide.
        """
        records = _mutate_record(
            _build_ts_payloads(3),
            2,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header.common_header, "num_channels", 7),
        )
        path = write_sintela_file("ts_endpoint_disagree.pb", records)
        with path.open("rb") as handle:
            assert sintela_utils._get_endpoint_metadata(handle) is None

    def test_record_head_rejects_a_length_the_file_cannot_hold(
        self, write_sintela_file, tmp_path
    ):
        """
        A corrupt size field must not be trusted to size a read.

        Every later step -- the tail-search window, the META payload read --
        is sized from this number, so one bad field would otherwise be enough
        to pull an entire file into a single buffer.
        """
        path = tmp_path / "bad_size.pb"
        with path.open("wb") as handle:
            handle.write(struct.pack("<I", sintela_utils.PBUF_MAGIC))
            handle.write(b"TS05")
            handle.write(struct.pack("<I", 2**31))  # far beyond the file
            handle.write(b"\x00" * 32)
        with path.open("rb") as handle:
            size = sintela_utils._stream_size(handle)
            assert sintela_utils._read_record_head(handle, 0, size) is None
            assert sintela_utils._find_first_data_record(handle, size) is None

    def test_read_varint_is_bounded(self):
        """
        An unterminated varint gives up instead of building a huge integer.

        Bounds are literals, not the constant under test: comparing the result
        against ``_MAX_VARINT_BYTES`` would keep passing however far that is
        raised, which is exactly the failure being guarded against.
        """
        # Far more continuation bytes than any real varint carries.
        value, pos = sintela_utils._read_varint(b"\xff" * 4096, 0)
        assert value is None
        # A base-128 varint encodes at most 64 bits, so it never reads past 10.
        assert pos <= 10

    def test_record_head_boundary_is_exact(self, tmp_path):
        """A record filling the file exactly is valid; one byte more is not."""
        path = tmp_path / "exact.pb"
        payload = b"\x00" * 16
        with path.open("wb") as handle:
            handle.write(struct.pack("<I", sintela_utils.PBUF_MAGIC))
            handle.write(b"TS05")
            handle.write(struct.pack("<I", len(payload)))
            handle.write(payload)
        with path.open("rb") as handle:
            size = sintela_utils._stream_size(handle)
            assert sintela_utils._read_record_head(handle, 0, size) is not None
            # The same file judged one byte shorter must reject that record.
            assert sintela_utils._read_record_head(handle, 0, size - 1) is None

    def test_find_last_record_returns_none_without_framing(self):
        """A tail holding no valid framing yields no candidate."""
        buf = b"\x01\x02\x03\x04" * 32
        assert (
            sintela_utils._find_last_record(_BytesReader(buf), len(buf), len(buf))
            is None
        )

    def test_fits_in_file_bounds_the_endpoint_length(self):
        """A derived length larger than the bytes could encode is rejected."""
        fits = sintela_utils._fits_in_file
        # 10 samples over 2 channels needs at least 80 wire bytes.
        assert fits(10, 2, 80)
        assert not fits(10, 2, 79)
        # A counter reset yields a nonsensical length; so do empty dimensions.
        assert not fits(2**32 - 5, 1052, 505_115_555)
        assert not fits(0, 2, 1000)
        assert not fits(10, 0, 1000)


class TestSintelaProtobufUtils:
    """Tests for lower-level Sintela protobuf helpers."""

    def test_helper_functions(self, ts_records):
        """Helper utilities should normalize metadata and coordinates."""
        records = [
            _envelope(("META", _build_meta_payload())),
            _envelope(ts_records[0]),
        ]
        parsed, meta = sintela_utils._parse_records(records, scan_mode=True)

        assert meta.serial_number == "SN123"
        assert sintela_utils._validate_single_family(parsed) == "timeseries"
        assert sintela_utils._assert_equal("value", [1, 1]) == 1
        assert sintela_utils._assert_float_equal("value", [1.0, 1.0]) == 1.0
        with pytest.raises(InvalidFiberFileError, match="Cannot validate value"):
            sintela_utils._assert_equal("value", [])
        with pytest.raises(InvalidFiberFileError, match="Cannot validate value"):
            sintela_utils._assert_float_equal("value", [])

        time = sintela_utils._get_time_coord_from_samples(
            np.datetime64("2024-01-01"), 2.0, 4
        )
        with pytest.raises(
            InvalidFiberFileError, match="Invalid Sintela protobuf sample_rate"
        ):
            sintela_utils._get_time_coord_from_samples(
                np.datetime64("2024-01-01"), 0.0, 4
            )
        packet_times = sintela_utils._get_times(
            [
                np.datetime64("2024-01-01T00:00:00"),
                np.datetime64("2024-01-01T00:00:01"),
            ]
        )
        assert len(time) == 4
        assert len(packet_times) == 2

        attrs = sintela_utils._base_attrs(
            parsed[0][1].header.common_header,
            packet_type="TS05",
            meta=meta,
            extra={"data_type": "strain"},
        )
        assert attrs.data_type == "strain"
        with pytest.raises(
            InvalidFiberFileError, match="Invalid Sintela protobuf channel_spacing"
        ):
            sintela_utils._get_distance_coord(0, np.nan, 1)
        assert sintela_utils._get_band_attr_data_type(((999, 0, 1, "", ""),)) == (
            "frequency_band_energy",
            "",
        )

        with pytest.raises(
            InvalidFiberFileError, match="No supported Sintela protobuf"
        ):
            sintela_utils._parse_records(
                [_envelope(("META", _build_meta_payload()))],
                scan_mode=True,
            )

    def test_parse_meta_without_optional_fiber_id(self):
        """META parsing should keep fiber_id unset when omitted."""
        meta = sintela_utils._parse_meta(_build_meta_payload_without_fiber_id())
        assert meta.fiber_id is None

    def test_decode_and_scan_helpers_cover_all_families(
        self,
        ts_records,
        band_records,
        fft_records,
        complex_fft_records,
        write_sintela_file,
    ):
        """Direct helper entry points should decode and summarize each family."""
        ts_records_with_meta = [
            _envelope(("META", _build_meta_payload())),
            *[_envelope(record) for record in ts_records],
        ]
        ts_parsed, ts_meta = sintela_utils._parse_records(
            ts_records_with_meta,
            scan_mode=False,
        )
        ts_data, ts_coords, _ = sintela_utils.TimeseriesMetadata.from_parsed(
            ts_parsed, ts_meta
        ).decode(ts_parsed)
        ts_scan = sintela_utils.TimeseriesMetadata.from_parsed(
            *sintela_utils._parse_records(ts_records_with_meta, scan_mode=True)
        ).scan()
        ts_shape, ts_scan_coords, _, ts_dtype = ts_scan
        assert ts_data.shape == ts_shape
        assert ts_coords.dims == ts_scan_coords.dims == ("time", "distance")
        assert ts_dtype == str(np.dtype(np.float32))

        band_path = write_sintela_file("band_helper.pb", band_records)
        with band_path.open("rb") as handle:
            band_data, band_coords, _ = sintela_utils.read_payload(handle)
        with band_path.open("rb") as handle:
            band_summary = _payload_to_summary(sintela_utils.scan_payload(handle)[0])
        decoded_band = sintela_utils._decode_family(
            *sintela_utils._parse_records(
                [_envelope(record) for record in band_records],
                scan_mode=False,
            )
        )
        band_scan = sintela_utils.BandMetadata.from_parsed(
            *sintela_utils._parse_records(
                [_envelope(record) for record in band_records],
                scan_mode=True,
            )
        ).scan()
        band_shape, scan_band_coords, _, band_dtype = band_scan
        assert band_data.shape == band_shape
        assert decoded_band[0].shape == band_shape
        assert band_coords.dims == scan_band_coords.dims == ("time", "distance", "band")
        assert band_summary.shape == band_shape
        assert band_dtype == str(np.dtype(np.float32))

        fft_path = write_sintela_file("fft_helper.pb", complex_fft_records)
        with fft_path.open("rb") as handle:
            fft_data, fft_coords, _ = sintela_utils.read_payload(handle)
        with fft_path.open("rb") as handle:
            fft_summary = _payload_to_summary(sintela_utils.scan_payload(handle)[0])
        decoded_fft = sintela_utils._decode_family(
            *sintela_utils._parse_records(
                [_envelope(record) for record in complex_fft_records],
                scan_mode=False,
            )
        )
        fft_scan = sintela_utils.FFTMetadata.from_parsed(
            *sintela_utils._parse_records(
                [_envelope(record) for record in fft_records],
                scan_mode=True,
            )
        ).scan()
        fft_shape, scan_fft_coords, _, fft_dtype = fft_scan
        assert np.iscomplexobj(fft_data)
        assert np.iscomplexobj(decoded_fft[0])
        assert fft_coords.dims == ("time", "distance", "frequency")
        assert scan_fft_coords.dims == ("time", "distance", "frequency")
        assert fft_summary.shape == fft_data.shape
        assert fft_shape == (len(fft_records), 2, 3)
        assert fft_dtype == str(np.dtype(np.float32))
