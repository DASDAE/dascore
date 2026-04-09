"""
Tests for Sintela protobuf format.
"""

from __future__ import annotations

import struct
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from dascore.exceptions import InvalidFiberFileError, MissingOptionalDependencyError
from dascore.io.sintela_protobuf import SintelaProtobufV1
from dascore.io.sintela_protobuf import utils as sintela_utils
from dascore.utils.downloader import fetch


@pytest.fixture(scope="module")
def sintela_protobuf_path():
    """Return the registered Sintela protobuf test file."""
    return fetch("sintela_protobuf_1.pb")


@pytest.fixture()
def fiber_io():
    """Return the Sintela protobuf FiberIO instance."""
    return SintelaProtobufV1()


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
    cls = _get_test_meta_message_class()
    msg = cls()
    msg.recorder_namespace = "manualRecord/recorder"
    _set_timestamp(msg.metadata_recording_time, 1_700_000_000, 123)
    msg.identification.manufacturer = "Sintela"
    msg.identification.model = "Onyx"
    msg.identification.serial_number = "SN123"
    msg.acquisition_stats.fiber_id = 2
    return msg.SerializeToString()


def _payload_to_summary(payload):
    """Convert a raw FiberIO scan payload using the production scan path."""
    from dascore.io.core import _scan_payload_to_summary

    return _scan_payload_to_summary(payload)


@cache
def _get_test_proto_messages():
    """Build local protobuf classes for Sintela synthetic test payloads."""
    from google.protobuf import (
        descriptor_pb2,
        descriptor_pool,
        message_factory,
        timestamp_pb2,
    )

    return sintela_utils._build_proto_messages(
        descriptor_pb2=descriptor_pb2,
        descriptor_pool=descriptor_pool,
        message_factory=message_factory,
        timestamp_pb2=timestamp_pb2,
        include_sample_fields=True,
        package_name="test_sintela_common",
        file_name="test_sintela_common.proto",
    )


@cache
def _get_test_meta_message_class():
    """Build a local RecordingMetadata class for META payload tests."""
    from google.protobuf import (
        descriptor_pb2,
        descriptor_pool,
        message_factory,
        timestamp_pb2,
    )

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "test_sintela_meta.proto"
    file_proto.package = "test_sintela_meta"
    file_proto.dependency.append("google/protobuf/timestamp.proto")

    identification = file_proto.message_type.add()
    identification.name = "IdentificationResponse"
    for number, name in (
        (1, "manufacturer"),
        (2, "system_type"),
        (3, "model"),
        (4, "serial_number"),
    ):
        field = identification.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    acquisition = file_proto.message_type.add()
    acquisition.name = "AcquisitionStatsResponse"
    field = acquisition.field.add()
    field.name = "fiber_id"
    field.number = 8
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT32

    recording = file_proto.message_type.add()
    recording.name = "RecordingMetadata"
    for name, number, type_, type_name in (
        ("recorder_namespace", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING, ""),
        (
            "metadata_recording_time",
            2,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            ".google.protobuf.Timestamp",
        ),
        (
            "identification",
            3,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            ".test_sintela_meta.IdentificationResponse",
        ),
        (
            "acquisition_stats",
            7,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            ".test_sintela_meta.AcquisitionStatsResponse",
        ),
    ):
        field = recording.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = type_
        if type_name:
            field.type_name = type_name

    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
    pool.Add(file_proto)
    descriptor = pool.FindMessageTypeByName("test_sintela_meta.RecordingMetadata")
    return message_factory.GetMessageClass(descriptor)


def _build_ts_payloads():
    """Create two contiguous timeseries packets."""
    packet_cls = _get_test_proto_messages()["TimeseriesPacket"]
    packets = []
    for offset, sample_count in enumerate((0, 3)):
        msg = packet_cls()
        hdr = msg.header
        common = hdr.common_header
        _set_timestamp(common.time, 1_700_000_000 + offset)
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
        assert patch.attrs.instrument_id == "SN123"
        assert patch.attrs.instrument_manufacturer == "Sintela"
        assert patch.attrs.instrument_model == "Onyx"
        assert patch.attrs.fiber_id == 2
        assert patch.attrs.data_type == "strain"

    def test_band_read_returns_expected_dims(
        self, fiber_io, write_sintela_file, band_records
    ):
        """Band recordings should load into a 3D patch."""
        path = write_sintela_file("band.pb", band_records)
        patch = fiber_io.read(path)[0]
        assert patch.dims == ("time", "distance", "band")
        assert patch.shape == (2, 2, 2)
        assert "band_start_frequency" in patch.coords.coord_map

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
        assert patch.attrs.data_type == ""
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

    def test_non_contiguous_timeseries_raises(
        self, fiber_io, write_sintela_file, ts_records
    ):
        """Timeseries packets with gaps or reordering should fail."""
        records = _mutate_record(
            ts_records,
            1,
            "TimeseriesPacket",
            lambda msg: setattr(msg.header, "sample_count", 10),
        )
        path = write_sintela_file("non_contiguous.pb", records)
        with pytest.raises(InvalidFiberFileError, match="Non-contiguous"):
            fiber_io.scan(path)

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
        records = _mutate_record(
            band_records,
            0,
            "BandPacket",
            lambda msg: msg.header.ClearField("band_data_info"),
        )
        records = _mutate_record(
            records,
            1,
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

    def test_optional_dependency_error_message(self):
        """The missing-protobuf helper should keep a stable message."""
        err = sintela_utils._optional_dependency_error()
        assert isinstance(err, MissingOptionalDependencyError)
        assert "protobuf is not installed" in str(err)

    def test_missing_protobuf_only_affects_scan_and_read(self, tmp_path, monkeypatch):
        """Detection should stay available without protobuf support."""
        path = tmp_path / "ts.pb"
        _write_records(path, [("META", _build_meta_payload()), *_build_ts_payloads()])

        def _raise():
            raise MissingOptionalDependencyError("protobuf missing")

        monkeypatch.setattr(sintela_utils, "_get_proto_messages", _raise)
        monkeypatch.setattr(sintela_utils, "_get_scan_proto_messages", _raise)
        monkeypatch.setattr(sintela_utils, "_get_meta_message_class", _raise)

        fiber_io = SintelaProtobufV1()
        assert fiber_io.get_format(path) == ("Sintela_Protobuf", "1")
        with pytest.raises(MissingOptionalDependencyError, match="protobuf missing"):
            fiber_io.scan(path)
        with pytest.raises(MissingOptionalDependencyError, match="protobuf missing"):
            fiber_io.read(path)


class TestSintelaProtobufUtils:
    """Tests for lower-level Sintela protobuf helpers."""

    def test_helper_functions(self, ts_records):
        """Helper utilities should normalize metadata and coordinates."""
        records = [
            sintela_utils.EnvelopeRecord("META", _build_meta_payload()),
            sintela_utils.EnvelopeRecord(*ts_records[0]),
        ]
        parsed, meta = sintela_utils._parse_records(records, scan_mode=True)

        assert meta.serial_number == "SN123"
        assert sintela_utils._validate_single_family(parsed) == "timeseries"
        assert sintela_utils._assert_equal("value", [1, 1]) == 1
        assert sintela_utils._assert_float_equal("value", [1.0, 1.0]) == 1.0
        assert sintela_utils._normalize_data_type("strain") == "strain"
        assert sintela_utils._normalize_data_type("not-real") == ""
        with pytest.raises(InvalidFiberFileError, match="Cannot validate value"):
            sintela_utils._assert_equal("value", [])
        with pytest.raises(InvalidFiberFileError, match="Cannot validate value"):
            sintela_utils._assert_float_equal("value", [])

        time = sintela_utils._get_time_coord_from_samples(
            np.datetime64("2024-01-01"), 2.0, 4
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
            InvalidFiberFileError, match="No supported Sintela protobuf"
        ):
            sintela_utils._parse_records(
                [sintela_utils.EnvelopeRecord("META", _build_meta_payload())],
                scan_mode=True,
            )

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
            sintela_utils.EnvelopeRecord("META", _build_meta_payload()),
            *[sintela_utils.EnvelopeRecord(*record) for record in ts_records],
        ]
        ts_parsed, ts_meta = sintela_utils._parse_records(
            ts_records_with_meta,
            scan_mode=False,
        )
        ts_data, ts_coords, _ = sintela_utils._decode_timeseries(ts_parsed, ts_meta)
        ts_shape, ts_scan_coords, _, ts_dtype = sintela_utils._scan_timeseries(
            *sintela_utils._parse_records(ts_records_with_meta, scan_mode=True)
        )
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
                [sintela_utils.EnvelopeRecord(*record) for record in band_records],
                scan_mode=False,
            )
        )
        band_shape, scan_band_coords, _, band_dtype = sintela_utils._scan_band(
            *sintela_utils._parse_records(
                [sintela_utils.EnvelopeRecord(*record) for record in band_records],
                scan_mode=True,
            )
        )
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
                [
                    sintela_utils.EnvelopeRecord(*record)
                    for record in complex_fft_records
                ],
                scan_mode=False,
            )
        )
        fft_shape, scan_fft_coords, _, fft_dtype = sintela_utils._scan_fft(
            *sintela_utils._parse_records(
                [sintela_utils.EnvelopeRecord(*record) for record in fft_records],
                scan_mode=True,
            )
        )
        assert np.iscomplexobj(fft_data)
        assert np.iscomplexobj(decoded_fft[0])
        assert fft_coords.dims == ("time", "distance", "frequency")
        assert scan_fft_coords.dims == ("time", "distance", "frequency")
        assert fft_summary.shape == fft_data.shape
        assert fft_shape == (len(fft_records), 2, 3)
        assert fft_dtype == str(np.dtype(np.float32))
