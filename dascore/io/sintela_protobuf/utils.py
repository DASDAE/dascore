"""
Utilities for reading Sintela protobuf MTLV recordings.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

import dascore as dc
from dascore.constants import VALID_DATA_TYPES
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.core.summary import PatchSummary
from dascore.exceptions import InvalidFiberFileError, MissingOptionalDependencyError
from dascore.utils.misc import suppress_warnings

PBUF_MAGIC = 0x46554250
META_TAG = "META"
TS_TAGS = frozenset({"TS05", "RF01"})
FFT_TAGS = frozenset({"FFT", "FFT-"})
BAND_TAGS = frozenset({"BAND"})
DIMS_TS = ("time", "distance")
DIMS_BAND = ("time", "distance", "band")
DIMS_FFT = ("time", "distance", "frequency")

_TIMESERIES_DATA_TYPE_MAP = {
    0: ("phase", "radians"),
    1: ("phase", "radians"),
    2: ("phase_difference", "radians"),
    3: ("phase_rate", "radians/s"),
    4: ("strain", "microstrain"),
    5: ("strain_rate", "microstrain/s"),
}
_BAND_DATA_TYPE_MAP = {
    10: ("temperature", ""),
    13: ("phase", "radians"),
}


class SintelaProtobufAttrs(PatchAttrs):
    """Patch attributes for Sintela protobuf recordings."""

    gauge_length: float = np.nan
    gauge_length_units: str = "m"
    packet_type: str = ""
    recorder_namespace: str = ""
    metadata_recording_time: np.datetime64 | None = None
    instrument_manufacturer: str = ""
    instrument_model: str = ""
    fiber_id: int | None = None
    serial_number: str = ""
    start_channel: int | None = None
    channel_spacing: float = np.nan
    channel_spacing_units: str = "m"
    channel_step: int | None = None
    sample_rate: float = np.nan
    demod_data_type: str = ""


@dataclass(frozen=True)
class EnvelopeRecord:
    """The envelope information for one MTLV record."""

    tag: str
    payload: bytes


@dataclass(frozen=True)
class ParsedMeta:
    """Selected metadata fields promoted from META packets."""

    recorder_namespace: str = ""
    metadata_recording_time: np.datetime64 | None = None
    instrument_manufacturer: str = ""
    instrument_model: str = ""
    serial_number: str = ""
    fiber_id: int | None = None


def _timestamp_to_dt64(timestamp) -> np.datetime64 | None:
    """Convert a protobuf timestamp into datetime64[ns]."""
    seconds = int(getattr(timestamp, "seconds", 0))
    nanos = int(getattr(timestamp, "nanos", 0))
    return np.datetime64(seconds, "s") + np.timedelta64(nanos, "ns")


def _iter_envelope_records(resource, *, strict: bool) -> list[EnvelopeRecord]:
    """Read all MTLV envelope records from a binary stream."""
    resource.seek(0)
    out: list[EnvelopeRecord] = []
    while True:
        magic = resource.read(4)
        if not magic:
            break
        if len(magic) < 4:
            if strict:
                raise InvalidFiberFileError("Truncated Sintela protobuf magic header.")
            return []
        if struct.unpack("<I", magic)[0] != PBUF_MAGIC:
            if strict:
                raise InvalidFiberFileError("Invalid Sintela protobuf magic header.")
            return []
        header = resource.read(8)
        if len(header) < 8:
            if strict:
                raise InvalidFiberFileError("Truncated Sintela protobuf record header.")
            return []
        tag = header[:4].rstrip(b"\x00").decode("utf-8", errors="ignore")
        size = struct.unpack("<I", header[4:8])[0]
        payload = resource.read(size)
        if len(payload) < size:
            if strict:
                raise InvalidFiberFileError("Truncated Sintela protobuf payload.")
            return []
        out.append(EnvelopeRecord(tag=tag, payload=payload))
    return out


def get_supported_family_tag(resource) -> str | None:
    """Return the first supported data tag in a file without using protobuf."""
    for record in _iter_envelope_records(resource, strict=False):
        if record.tag == META_TAG:
            continue
        if record.tag in TS_TAGS | BAND_TAGS | FFT_TAGS:
            return record.tag
        return None
    return None


def _optional_dependency_error() -> MissingOptionalDependencyError:
    """Return the standardized missing dependency error."""
    msg = (
        "protobuf is not installed but is required for Sintela protobuf scan/read "
        "operations."
    )
    return MissingOptionalDependencyError(msg)


@cache
def _get_proto_messages():
    """Build lightweight protobuf messages for supported Sintela packet types."""
    try:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
        from google.protobuf import timestamp_pb2
    except Exception as exc:  # pragma: no cover - import failure path
        raise _optional_dependency_error() from exc

    return _build_proto_messages(
        descriptor_pb2=descriptor_pb2,
        descriptor_pool=descriptor_pool,
        message_factory=message_factory,
        timestamp_pb2=timestamp_pb2,
        include_sample_fields=True,
        package_name="sintela_common",
        file_name="sintela_common_lite.proto",
    )


@cache
def _get_scan_proto_messages():
    """Build scan-only protobuf messages which omit sample payload fields."""
    try:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
        from google.protobuf import timestamp_pb2
    except Exception as exc:  # pragma: no cover - import failure path
        raise _optional_dependency_error() from exc

    return _build_proto_messages(
        descriptor_pb2=descriptor_pb2,
        descriptor_pool=descriptor_pool,
        message_factory=message_factory,
        timestamp_pb2=timestamp_pb2,
        include_sample_fields=False,
        package_name="sintela_common_scan",
        file_name="sintela_common_scan.proto",
    )


def _build_proto_messages(
    *,
    descriptor_pb2,
    descriptor_pool,
    message_factory,
    timestamp_pb2,
    include_sample_fields: bool,
    package_name: str,
    file_name: str,
):
    """Build lightweight protobuf message classes for data packets."""

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = file_name
    file_proto.package = package_name
    file_proto.dependency.append("google/protobuf/timestamp.proto")

    def add_field(message, name, number, type_, *, label=None, type_name=""):
        field = message.field.add()
        field.name = name
        field.number = number
        field.label = (
            label
            if label is not None
            else descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        )
        field.type = type_
        if type_name:
            field.type_name = type_name
        return field

    common = file_proto.message_type.add()
    common.name = "CommonHeader"
    add_field(
        common,
        "time",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".google.protobuf.Timestamp",
    )
    for number, name, type_ in (
        (2, "num_channels", descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        (3, "sample_rate", descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
        (4, "channel_spacing", descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
        (5, "gauge_length", descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
        (6, "start_channel", descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        (7, "end_of_replay", descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        (8, "fiber_flipped", descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        (9, "loop_removed", descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        (10, "has_dropped_samples", descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        (11, "timeseries_data_type", descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        (12, "demod_data_type", descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
    ):
        add_field(common, name, number, type_)

    timeseries_header = file_proto.message_type.add()
    timeseries_header.name = "TimeseriesHeader"
    add_field(
        timeseries_header,
        "common_header",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{package_name}.CommonHeader",
    )
    add_field(
        timeseries_header, "sample_count", 2, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32
    )
    add_field(
        timeseries_header, "num_samples", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT32
    )
    add_field(
        timeseries_header, "channel_step", 4, descriptor_pb2.FieldDescriptorProto.TYPE_INT32
    )

    timeseries_packet = file_proto.message_type.add()
    timeseries_packet.name = "TimeseriesPacket"
    add_field(
        timeseries_packet,
        "header",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{package_name}.TimeseriesHeader",
    )
    if include_sample_fields:
        add_field(
            timeseries_packet,
            "samples",
            3,
            descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        )
        add_field(
            timeseries_packet,
            "raw_frames",
            4,
            descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
        )

    band_info = file_proto.message_type.add()
    band_info.name = "BandDataInfo"
    for number, name, type_ in (
        (1, "band_data_type", descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        (2, "start", descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
        (3, "end", descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
        (4, "averaging_type", descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        (5, "description", descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
        (6, "source", descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ):
        add_field(band_info, name, number, type_)

    band_header = file_proto.message_type.add()
    band_header.name = "BandHeader"
    add_field(
        band_header,
        "common_header",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{package_name}.CommonHeader",
    )
    add_field(
        band_header,
        "band_data_info",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type_name=f".{package_name}.BandDataInfo",
    )

    band_packet = file_proto.message_type.add()
    band_packet.name = "BandPacket"
    add_field(
        band_packet,
        "header",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{package_name}.BandHeader",
    )
    if include_sample_fields:
        add_field(
            band_packet,
            "samples",
            2,
            descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        )

    fft_header = file_proto.message_type.add()
    fft_header.name = "FFTHeader"
    add_field(
        fft_header,
        "common_header",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{package_name}.CommonHeader",
    )
    for number, name, type_ in (
        (2, "num_bins", descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        (3, "bin_res", descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
        (4, "averaging_type", descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        (5, "channel_step", descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        (6, "normalised", descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        (7, "has_power_data", descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        (8, "has_complex_data", descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
    ):
        add_field(fft_header, name, number, type_)

    fft_packet = file_proto.message_type.add()
    fft_packet.name = "FFTPacket"
    add_field(
        fft_packet,
        "header",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{package_name}.FFTHeader",
    )
    if include_sample_fields:
        add_field(
            fft_packet,
            "samples",
            2,
            descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        )

    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
    pool.Add(file_proto)
    out = {}
    for name in ("TimeseriesPacket", "BandPacket", "FFTPacket"):
        descriptor = pool.FindMessageTypeByName(f"{package_name}.{name}")
        out[name] = message_factory.GetMessageClass(descriptor)
    return out


@cache
def _get_meta_message_class():
    """Build a lightweight RecordingMetadata parser for selected fields."""
    try:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
        from google.protobuf import timestamp_pb2
    except Exception as exc:  # pragma: no cover - import failure path
        raise _optional_dependency_error() from exc

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "sintela_meta_lite.proto"
    file_proto.package = "sintela_meta"
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
    fields = (
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
            ".sintela_meta.IdentificationResponse",
        ),
        (
            "acquisition_stats",
            7,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            ".sintela_meta.AcquisitionStatsResponse",
        ),
    )
    for name, number, type_, type_name in fields:
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
    descriptor = pool.FindMessageTypeByName("sintela_meta.RecordingMetadata")
    return message_factory.GetMessageClass(descriptor)


def _parse_meta(payload: bytes) -> ParsedMeta:
    """Parse selected fields from a META payload."""
    message_cls = _get_meta_message_class()
    msg = message_cls()
    with suppress_warnings():
        msg.ParseFromString(payload)
    identification = msg.identification if msg.HasField("identification") else None
    acquisition = msg.acquisition_stats if msg.HasField("acquisition_stats") else None
    return ParsedMeta(
        recorder_namespace=str(getattr(msg, "recorder_namespace", "") or ""),
        metadata_recording_time=(
            _timestamp_to_dt64(msg.metadata_recording_time)
            if msg.HasField("metadata_recording_time")
            else None
        ),
        instrument_manufacturer=str(
            getattr(identification, "manufacturer", "") or ""
        ),
        instrument_model=str(getattr(identification, "model", "") or ""),
        serial_number=str(getattr(identification, "serial_number", "") or ""),
        fiber_id=(
            int(getattr(acquisition, "fiber_id"))
            if acquisition is not None and getattr(acquisition, "fiber_id", None) is not None
            else None
        ),
    )


def _common_header_time(common_header) -> np.datetime64 | None:
    """Return a common-header timestamp when present."""
    return (
        _timestamp_to_dt64(common_header.time)
        if common_header.HasField("time")
        else None
    )


def _parse_records(
    records: list[EnvelopeRecord], *, scan_mode: bool = False
) -> tuple[list[Any], ParsedMeta]:
    """Decode protobuf payloads and return messages plus selected META."""
    messages = _get_scan_proto_messages() if scan_mode else _get_proto_messages()
    parsed: list[Any] = []
    meta = ParsedMeta()
    for record in records:
        tag = record.tag
        if tag == META_TAG:
            meta = _parse_meta(record.payload)
            continue
        if tag in TS_TAGS:
            msg = messages["TimeseriesPacket"]()
        elif tag in BAND_TAGS:
            msg = messages["BandPacket"]()
        elif tag in FFT_TAGS:
            msg = messages["FFTPacket"]()
        else:
            raise InvalidFiberFileError(f"Unsupported Sintela protobuf tag {tag!r}.")
        msg.ParseFromString(record.payload)
        parsed.append((tag, msg))
    if not parsed:
        raise InvalidFiberFileError("No supported Sintela protobuf data packets found.")
    return parsed, meta


def _get_time_coord_from_samples(start: np.datetime64, sample_rate: float, size: int):
    """Build a regularly sampled time coordinate."""
    step = dc.to_timedelta64(1 / sample_rate)
    return get_coord(start=start, stop=start + step * size, step=step, units="s")


def _get_distance_coord(start_channel: int, spacing: float, count: int, step: int = 1):
    """Build the distance coordinate."""
    start = start_channel * spacing
    return get_coord(
        start=start,
        stop=start + spacing * step * count,
        step=spacing * step,
        units="m",
    )


def _get_times(times: list[np.datetime64]):
    """Build a time coordinate from packet timestamps."""
    return get_coord(data=np.asarray(times, dtype="datetime64[ns]"), units="s")


def _normalize_data_type(candidate: str) -> str:
    """Map any external string to a DASCore-valid data type."""
    clean = str(candidate or "").lower().strip()
    return clean if clean in VALID_DATA_TYPES else ""


def _assert_float_equal(name: str, values: list[float], *, rtol: float = 1e-6):
    """Ensure float values match within a small tolerance."""
    first = values[0]
    for value in values[1:]:
        if not np.isclose(first, value, rtol=rtol, atol=0.0):
            raise InvalidFiberFileError(
                f"Inconsistent {name} across Sintela protobuf packets."
            )
    return first


def _base_attrs(common_header, packet_type: str, meta: ParsedMeta, extra: dict | None = None):
    """Construct base attrs from the packet header and META metadata."""
    data_type, data_units = _TIMESERIES_DATA_TYPE_MAP.get(
        int(getattr(common_header, "timeseries_data_type", 0)),
        ("phase", "radians"),
    )
    attrs = dict(
        data_type=_normalize_data_type(data_type),
        data_category="DAS",
        data_units=data_units,
        gauge_length=float(getattr(common_header, "gauge_length", np.nan)),
        packet_type=packet_type,
        recorder_namespace=meta.recorder_namespace,
        metadata_recording_time=meta.metadata_recording_time,
        instrument_manufacturer=meta.instrument_manufacturer,
        instrument_model=meta.instrument_model,
        instrument_id=meta.serial_number,
        serial_number=meta.serial_number,
        fiber_id=meta.fiber_id,
        start_channel=int(getattr(common_header, "start_channel", 0)),
        channel_spacing=float(getattr(common_header, "channel_spacing", np.nan)),
        channel_step=None,
        sample_rate=float(getattr(common_header, "sample_rate", np.nan)),
    )
    if extra:
        attrs.update(extra)
    return SintelaProtobufAttrs(**attrs)


def _assert_equal(name: str, values: list[Any]):
    """Ensure all values in a list are equal."""
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise InvalidFiberFileError(
                f"Inconsistent {name} across Sintela protobuf packets."
            )
    return first


def _validate_single_family(parsed: list[tuple[str, Any]]) -> str:
    """Ensure a file only contains one data packet family."""
    families = {
        "timeseries" if tag in TS_TAGS else "band" if tag in BAND_TAGS else "fft"
        for tag, _ in parsed
    }
    if len(families) != 1:
        raise InvalidFiberFileError("Mixed Sintela protobuf packet families are unsupported.")
    return families.pop()


def _decode_family(parsed: list[tuple[str, Any]], meta: ParsedMeta):
    """Decode one parsed data family into data, coords, and attrs."""
    family = _validate_single_family(parsed)
    if family == "timeseries":
        return _decode_timeseries(parsed, meta)
    if family == "band":
        return _decode_band(parsed, meta)
    return _decode_fft(parsed, meta)


def _decode_timeseries(parsed: list[tuple[str, Any]], meta: ParsedMeta):
    """Decode timeseries packets into data, coords, and attrs."""
    headers = [msg.header for _tag, msg in parsed]
    common_headers = [header.common_header for header in headers]
    num_channels = _assert_equal(
        "num_channels", [int(ch.num_channels) for ch in common_headers]
    )
    sample_rate = _assert_float_equal(
        "sample_rate", [float(ch.sample_rate) for ch in common_headers]
    )
    channel_spacing = _assert_float_equal(
        "channel_spacing", [float(ch.channel_spacing) for ch in common_headers]
    )
    gauge_length = _assert_float_equal(
        "gauge_length", [float(ch.gauge_length) for ch in common_headers]
    )
    start_channel = _assert_equal(
        "start_channel", [int(ch.start_channel) for ch in common_headers]
    )
    channel_step = _assert_equal("channel_step", [int(h.channel_step) for h in headers])
    data_type = _assert_equal(
        "timeseries_data_type",
        [int(ch.timeseries_data_type) for ch in common_headers],
    )
    demod_data_type = _assert_equal(
        "demod_data_type", [int(ch.demod_data_type) for ch in common_headers]
    )
    for ch in common_headers:
        if ch.has_dropped_samples:
            raise InvalidFiberFileError("Dropped samples in Sintela protobuf stream.")
    sample_counts = [int(h.sample_count) for h in headers]
    num_samples_per_packet = [int(h.num_samples) for h in headers]
    for current, nxt, count in zip(
        sample_counts, sample_counts[1:], num_samples_per_packet[:-1], strict=False
    ):
        if current + count != nxt:
            raise InvalidFiberFileError("Non-contiguous Sintela protobuf sample counts.")
    total_samples = sum(num_samples_per_packet)
    first_time = _common_header_time(common_headers[0])
    if first_time is None:
        raise InvalidFiberFileError("Missing Sintela protobuf start time.")
    data = np.empty((total_samples, num_channels), dtype=np.float32)
    index = 0
    for _tag, msg in parsed:
        packet = np.asarray(msg.samples, dtype=np.float32)
        rows = int(msg.header.num_samples)
        expected = rows * num_channels
        if packet.size != expected:
            raise InvalidFiberFileError("Unexpected Sintela protobuf TS sample payload size.")
        data[index : index + rows] = packet.reshape(rows, num_channels)
        index += rows
    time = _get_time_coord_from_samples(first_time, sample_rate, total_samples)
    distance = _get_distance_coord(start_channel, channel_spacing, num_channels, channel_step)
    coords = get_coord_manager({"time": time, "distance": distance}, dims=DIMS_TS)
    attrs = _base_attrs(
        common_headers[0],
        packet_type=parsed[0][0],
        meta=meta,
        extra=dict(
            gauge_length=gauge_length,
            channel_step=channel_step,
            sample_rate=sample_rate,
            data_type=_TIMESERIES_DATA_TYPE_MAP.get(data_type, ("phase", "radians"))[0],
            data_units=_TIMESERIES_DATA_TYPE_MAP.get(data_type, ("phase", "radians"))[1],
            demod_data_type=str(demod_data_type),
        ),
    )
    return data, coords, attrs


def _decode_band(parsed: list[tuple[str, Any]], meta: ParsedMeta):
    """Decode band packets into data, coords, and attrs."""
    headers = [msg.header for _tag, msg in parsed]
    common_headers = [header.common_header for header in headers]
    num_channels = _assert_equal(
        "num_channels", [int(ch.num_channels) for ch in common_headers]
    )
    channel_spacing = _assert_float_equal(
        "channel_spacing", [float(ch.channel_spacing) for ch in common_headers]
    )
    gauge_length = _assert_float_equal(
        "gauge_length", [float(ch.gauge_length) for ch in common_headers]
    )
    start_channel = _assert_equal(
        "start_channel", [int(ch.start_channel) for ch in common_headers]
    )
    band_defs = []
    for header in headers:
        band_defs.append(
            tuple(
                (
                    int(info.band_data_type),
                    float(info.start),
                    float(info.end),
                    str(info.description),
                    str(info.source),
                )
                for info in header.band_data_info
            )
        )
    band_def = _assert_equal("band_data_info", band_defs)
    num_bands = len(band_def)
    if not num_bands:
        raise InvalidFiberFileError("Band packets missing band definitions.")
    times = [_common_header_time(ch) for ch in common_headers]
    if any(x is None for x in times):
        raise InvalidFiberFileError("Missing time in Sintela BAND packet.")
    data = np.empty((len(parsed), num_channels, num_bands), dtype=np.float32)
    for ind, (_tag, msg) in enumerate(parsed):
        packet = np.asarray(msg.samples, dtype=np.float32)
        expected = num_channels * num_bands
        if packet.size != expected:
            raise InvalidFiberFileError("Unexpected Sintela protobuf BAND payload size.")
        data[ind] = packet.reshape(num_channels, num_bands)
    # BAND packets do not expose channel_step, so distance comes from the
    # recorded start channel and spacing only.
    distance = _get_distance_coord(start_channel, channel_spacing, num_channels)
    band = get_coord(start=0, stop=num_bands, step=1)
    coords = get_coord_manager(
        {
            "time": _get_times(times),
            "distance": distance,
            "band": band,
            "band_start_frequency": ("band", np.asarray([x[1] for x in band_def])),
            "band_end_frequency": ("band", np.asarray([x[2] for x in band_def])),
            "band_description": ("band", np.asarray([x[3] for x in band_def], dtype=object)),
            "band_source": ("band", np.asarray([x[4] for x in band_def], dtype=object)),
        },
        dims=DIMS_BAND,
    )
    first_type = int(band_def[0][0])
    data_type, data_units = _BAND_DATA_TYPE_MAP.get(first_type, ("", ""))
    attrs = _base_attrs(
        common_headers[0],
        packet_type=parsed[0][0],
        meta=meta,
        extra=dict(
            gauge_length=gauge_length,
            data_type=data_type,
            data_units=data_units,
        ),
    )
    return data, coords, attrs


def _decode_fft(parsed: list[tuple[str, Any]], meta: ParsedMeta):
    """Decode FFT packets into data, coords, and attrs."""
    headers = [msg.header for _tag, msg in parsed]
    common_headers = [header.common_header for header in headers]
    num_channels = _assert_equal(
        "num_channels", [int(ch.num_channels) for ch in common_headers]
    )
    channel_spacing = _assert_float_equal(
        "channel_spacing", [float(ch.channel_spacing) for ch in common_headers]
    )
    gauge_length = _assert_float_equal(
        "gauge_length", [float(ch.gauge_length) for ch in common_headers]
    )
    start_channel = _assert_equal(
        "start_channel", [int(ch.start_channel) for ch in common_headers]
    )
    num_bins = _assert_equal("num_bins", [int(h.num_bins) for h in headers])
    bin_res = _assert_float_equal("bin_res", [float(h.bin_res) for h in headers])
    has_complex = _assert_equal("has_complex_data", [bool(h.has_complex_data) for h in headers])
    channel_step = _assert_equal("channel_step", [int(h.channel_step) for h in headers])
    times = [_common_header_time(ch) for ch in common_headers]
    if any(x is None for x in times):
        raise InvalidFiberFileError("Missing time in Sintela FFT packet.")
    dtype = np.complex64 if has_complex else np.float32
    data = np.empty((len(parsed), num_channels, num_bins), dtype=dtype)
    for ind, (_tag, msg) in enumerate(parsed):
        packet = np.asarray(msg.samples, dtype=np.float32)
        if has_complex:
            expected = num_channels * num_bins * 2
            if packet.size != expected:
                raise InvalidFiberFileError("Unexpected Sintela protobuf FFT payload size.")
            packet = packet.reshape(num_channels, num_bins, 2)
            packet = packet[..., 0] + 1j * packet[..., 1]
        else:
            expected = num_channels * num_bins
            if packet.size != expected:
                raise InvalidFiberFileError("Unexpected Sintela protobuf FFT payload size.")
            packet = packet.reshape(num_channels, num_bins)
        data[ind] = packet
    distance = _get_distance_coord(start_channel, channel_spacing, num_channels, channel_step)
    frequency = get_coord(start=0.0, stop=bin_res * num_bins, step=bin_res, units="Hz")
    coords = get_coord_manager(
        {"time": _get_times(times), "distance": distance, "frequency": frequency},
        dims=DIMS_FFT,
    )
    attrs = _base_attrs(
        common_headers[0],
        packet_type=parsed[0][0],
        meta=meta,
        extra=dict(
            gauge_length=gauge_length,
            channel_step=channel_step,
        ),
    )
    return data, coords, attrs


def _scan_timeseries(parsed: list[tuple[str, Any]], meta: ParsedMeta):
    """Summarize timeseries packets without allocating sample arrays."""
    headers = [msg.header for _tag, msg in parsed]
    common_headers = [header.common_header for header in headers]
    num_channels = _assert_equal(
        "num_channels", [int(ch.num_channels) for ch in common_headers]
    )
    sample_rate = _assert_float_equal(
        "sample_rate", [float(ch.sample_rate) for ch in common_headers]
    )
    channel_spacing = _assert_float_equal(
        "channel_spacing", [float(ch.channel_spacing) for ch in common_headers]
    )
    gauge_length = _assert_float_equal(
        "gauge_length", [float(ch.gauge_length) for ch in common_headers]
    )
    start_channel = _assert_equal(
        "start_channel", [int(ch.start_channel) for ch in common_headers]
    )
    channel_step = _assert_equal("channel_step", [int(h.channel_step) for h in headers])
    data_type = _assert_equal(
        "timeseries_data_type",
        [int(ch.timeseries_data_type) for ch in common_headers],
    )
    demod_data_type = _assert_equal(
        "demod_data_type", [int(ch.demod_data_type) for ch in common_headers]
    )
    for ch in common_headers:
        if ch.has_dropped_samples:
            raise InvalidFiberFileError("Dropped samples in Sintela protobuf stream.")
    sample_counts = [int(h.sample_count) for h in headers]
    num_samples_per_packet = [int(h.num_samples) for h in headers]
    for current, nxt, count in zip(
        sample_counts, sample_counts[1:], num_samples_per_packet[:-1], strict=False
    ):
        if current + count != nxt:
            raise InvalidFiberFileError("Non-contiguous Sintela protobuf sample counts.")
    total_samples = sum(num_samples_per_packet)
    first_time = _common_header_time(common_headers[0])
    if first_time is None:
        raise InvalidFiberFileError("Missing Sintela protobuf start time.")
    time = _get_time_coord_from_samples(first_time, sample_rate, total_samples)
    distance = _get_distance_coord(start_channel, channel_spacing, num_channels, channel_step)
    coords = get_coord_manager({"time": time, "distance": distance}, dims=DIMS_TS)
    attrs = _base_attrs(
        common_headers[0],
        packet_type=parsed[0][0],
        meta=meta,
        extra=dict(
            gauge_length=gauge_length,
            channel_step=channel_step,
            sample_rate=sample_rate,
            data_type=_TIMESERIES_DATA_TYPE_MAP.get(data_type, ("phase", "radians"))[0],
            data_units=_TIMESERIES_DATA_TYPE_MAP.get(data_type, ("phase", "radians"))[1],
            demod_data_type=str(demod_data_type),
        ),
    )
    return (total_samples, num_channels), coords, attrs, str(np.dtype(np.float32))


def _scan_band(parsed: list[tuple[str, Any]], meta: ParsedMeta):
    """Summarize band packets without allocating sample arrays."""
    headers = [msg.header for _tag, msg in parsed]
    common_headers = [header.common_header for header in headers]
    num_channels = _assert_equal(
        "num_channels", [int(ch.num_channels) for ch in common_headers]
    )
    channel_spacing = _assert_float_equal(
        "channel_spacing", [float(ch.channel_spacing) for ch in common_headers]
    )
    gauge_length = _assert_float_equal(
        "gauge_length", [float(ch.gauge_length) for ch in common_headers]
    )
    start_channel = _assert_equal(
        "start_channel", [int(ch.start_channel) for ch in common_headers]
    )
    band_defs = []
    for header in headers:
        band_defs.append(
            tuple(
                (
                    int(info.band_data_type),
                    float(info.start),
                    float(info.end),
                    str(info.description),
                    str(info.source),
                )
                for info in header.band_data_info
            )
        )
    band_def = _assert_equal("band_data_info", band_defs)
    num_bands = len(band_def)
    if not num_bands:
        raise InvalidFiberFileError("Band packets missing band definitions.")
    times = [_common_header_time(ch) for ch in common_headers]
    if any(x is None for x in times):
        raise InvalidFiberFileError("Missing time in Sintela BAND packet.")
    distance = _get_distance_coord(start_channel, channel_spacing, num_channels)
    band = get_coord(start=0, stop=num_bands, step=1)
    coords = get_coord_manager(
        {
            "time": _get_times(times),
            "distance": distance,
            "band": band,
            "band_start_frequency": ("band", np.asarray([x[1] for x in band_def])),
            "band_end_frequency": ("band", np.asarray([x[2] for x in band_def])),
            "band_description": ("band", np.asarray([x[3] for x in band_def], dtype=object)),
            "band_source": ("band", np.asarray([x[4] for x in band_def], dtype=object)),
        },
        dims=DIMS_BAND,
    )
    first_type = int(band_def[0][0])
    data_type, data_units = _BAND_DATA_TYPE_MAP.get(first_type, ("", ""))
    attrs = _base_attrs(
        common_headers[0],
        packet_type=parsed[0][0],
        meta=meta,
        extra=dict(
            gauge_length=gauge_length,
            data_type=data_type,
            data_units=data_units,
        ),
    )
    return (len(parsed), num_channels, num_bands), coords, attrs, str(np.dtype(np.float32))


def _scan_fft(parsed: list[tuple[str, Any]], meta: ParsedMeta):
    """Summarize FFT packets without allocating sample arrays."""
    headers = [msg.header for _tag, msg in parsed]
    common_headers = [header.common_header for header in headers]
    num_channels = _assert_equal(
        "num_channels", [int(ch.num_channels) for ch in common_headers]
    )
    channel_spacing = _assert_float_equal(
        "channel_spacing", [float(ch.channel_spacing) for ch in common_headers]
    )
    gauge_length = _assert_float_equal(
        "gauge_length", [float(ch.gauge_length) for ch in common_headers]
    )
    start_channel = _assert_equal(
        "start_channel", [int(ch.start_channel) for ch in common_headers]
    )
    num_bins = _assert_equal("num_bins", [int(h.num_bins) for h in headers])
    bin_res = _assert_float_equal("bin_res", [float(h.bin_res) for h in headers])
    has_complex = _assert_equal("has_complex_data", [bool(h.has_complex_data) for h in headers])
    channel_step = _assert_equal("channel_step", [int(h.channel_step) for h in headers])
    times = [_common_header_time(ch) for ch in common_headers]
    if any(x is None for x in times):
        raise InvalidFiberFileError("Missing time in Sintela FFT packet.")
    distance = _get_distance_coord(start_channel, channel_spacing, num_channels, channel_step)
    frequency = get_coord(start=0.0, stop=bin_res * num_bins, step=bin_res, units="Hz")
    coords = get_coord_manager(
        {"time": _get_times(times), "distance": distance, "frequency": frequency},
        dims=DIMS_FFT,
    )
    attrs = _base_attrs(
        common_headers[0],
        packet_type=parsed[0][0],
        meta=meta,
        extra=dict(
            gauge_length=gauge_length,
            channel_step=channel_step,
        ),
    )
    dtype = np.complex64 if has_complex else np.float32
    return (len(parsed), num_channels, num_bins), coords, attrs, str(np.dtype(dtype))


def read_payload(resource):
    """Decode a Sintela protobuf file into data, coords, and attrs."""
    records = _iter_envelope_records(resource, strict=True)
    parsed, meta = _parse_records(records, scan_mode=False)
    return _decode_family(parsed, meta)


def scan_payload(resource) -> list[PatchSummary]:
    """Decode a Sintela protobuf file and return PatchSummary objects."""
    records = _iter_envelope_records(resource, strict=True)
    parsed, meta = _parse_records(records, scan_mode=True)
    family = _validate_single_family(parsed)
    if family == "timeseries":
        shape, coords, attrs, dtype = _scan_timeseries(parsed, meta)
    elif family == "band":
        shape, coords, attrs, dtype = _scan_band(parsed, meta)
    else:
        shape, coords, attrs, dtype = _scan_fft(parsed, meta)
    return [
        PatchSummary.model_construct(
            attrs=attrs,
            coords=coords.to_summary_dict(),
            dims=coords.dims,
            shape=shape,
            dtype=dtype,
        )
    ]
