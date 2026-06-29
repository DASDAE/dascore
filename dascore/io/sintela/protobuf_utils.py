"""
Utilities for reading Sintela protobuf MTLV recordings.

File format
-----------
A recording is a flat sequence of MTLV (magic-tag-length-value) envelope
records.  Each record is laid out as:

    magic    4 bytes   little-endian uint32, always ``PBUF_MAGIC``
    tag      4 bytes   ASCII, null-padded (e.g. ``META``, ``TS05``, ``FFT``)
    size     4 bytes   little-endian uint32, payload length in bytes
    payload  size      a serialized protobuf message

The tag identifies the payload's packet family (see ``TS_TAGS`` etc.); the
payload is decoded with the matching message class built below.

Protobuf schema strategy
------------------------
Rather than vendoring Sintela's generated ``*_pb2.py`` modules, we build the
small subset of their schema that DASCore needs at runtime via
``descriptor_pb2`` (see ``_build_proto_messages``).  This keeps protobuf an
optional dependency, avoids committing generated code, and lets us skip the
sample payloads entirely when scanning.  The field numbers below must match
Sintela's real wire schema.

Why not just ship a ``.proto`` file?  Neither obvious file-based option works
well here:

- A checked-in ``.proto`` cannot be loaded at runtime by the ``protobuf``
  package alone -- the pure-Python runtime has no ``.proto`` text parser.
  Compiling one requires the ``protoc`` compiler (or ``grpcio-tools``), a
  heavier, non-pure-Python build/runtime dependency we don't want to add for an
  optional format.
- A committed generated ``*_pb2.py`` is tightly coupled to the installed
  protobuf runtime version (generated code has broken across protobuf major
  releases).  Since protobuf is optional and unpinned, a user could have any
  version installed.

Building descriptors at runtime through the lower-level, more stable
``descriptor_pb2`` reflection API sidesteps both problems.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from functools import cache
from typing import Any

import numpy as np
from pydantic import ValidationError

import dascore as dc
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import InvalidFiberFileError
from dascore.io.core import _make_scan_payload
from dascore.utils.misc import optional_import, suppress_warnings
from dascore.utils.models import DascoreBaseModel, PositiveFiniteFloat, PositiveInt

PBUF_MAGIC = 0x46554250
META_TAG = "META"
TS_TAGS = frozenset({"TS05", "RF01"})
FFT_TAGS = frozenset({"FFT", "FFT-"})
BAND_TAGS = frozenset({"BAND"})
# Maps each data tag to the packet message class (built by
# ``_get_proto_messages``) that decodes it.
_TAG_TO_PACKET = {
    **dict.fromkeys(TS_TAGS, "TimeseriesPacket"),
    **dict.fromkeys(BAND_TAGS, "BandPacket"),
    **dict.fromkeys(FFT_TAGS, "FFTPacket"),
}
DIMS_TS = ("time", "distance")
DIMS_BAND = ("time", "distance", "band")
DIMS_FFT = ("time", "distance", "frequency")

_TIMESERIES_DATA_TYPE_MAP = {
    # Sintela currently reports both enum codes as phase-like samples.
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
_FFT_ATTR_DEFAULTS = {
    "data_type": "power_spectral_density",
    "data_units": "",
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
    channel_step: int | None = None
    demod_data_type: str = ""


class EnvelopeRecord(DascoreBaseModel):
    """The envelope information for one MTLV record."""

    tag: str
    payload: bytes


class ParsedMeta(DascoreBaseModel):
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


def _iter_envelope_records(resource, *, strict: bool) -> Iterator[EnvelopeRecord]:
    """Read all MTLV envelope records from a binary stream."""

    def _stop(message):
        """Raise in strict mode; otherwise signal a clean stop to the caller."""
        if strict:
            raise InvalidFiberFileError(message)
        return True

    resource.seek(0)
    while True:
        # Each record opens with a 4-byte magic word; an empty read here is a
        # clean end-of-file, while a short or wrong magic is a malformed record.
        magic = resource.read(4)
        if not magic:
            break
        if len(magic) < 4 and _stop("Truncated Sintela protobuf magic header."):
            return
        if struct.unpack("<I", magic)[0] != PBUF_MAGIC and _stop(
            "Invalid Sintela protobuf magic header."
        ):
            return
        # The magic is followed by an 8-byte header: a 4-byte null-padded tag
        # and a 4-byte little-endian payload size.
        header = resource.read(8)
        if len(header) < 8 and _stop("Truncated Sintela protobuf record header."):
            return
        tag = header[:4].rstrip(b"\x00").decode("utf-8", errors="ignore")
        size = struct.unpack("<I", header[4:8])[0]
        # The payload is `size` bytes of serialized protobuf, decoded later by
        # the message class matching the tag.
        payload = resource.read(size)
        if len(payload) < size and _stop("Truncated Sintela protobuf payload."):
            return
        yield EnvelopeRecord(tag=tag, payload=payload)


def get_supported_family_tag(resource) -> str | None:
    """Return the first supported data tag in a file without using protobuf."""
    for record in _iter_envelope_records(resource, strict=False):
        if record.tag == META_TAG:
            continue
        if record.tag in _TAG_TO_PACKET:
            return record.tag
        # Detection is intentionally tolerant of unknown non-data records so a
        # valid family tag later in the file can still identify the format.
        continue
    return None


def _get_protobuf_decode_error():
    """Return protobuf's decode error type, or Exception as a fallback."""
    message_mod = optional_import("google.protobuf.message", on_missing="ignore")
    return getattr(message_mod, "DecodeError", Exception)


def _import_protobuf():
    """Import the protobuf submodules, raising if protobuf is not installed."""
    txt = "Sintela protobuf scan/read operations"
    return (
        optional_import("google.protobuf.descriptor_pb2", required_for=txt),
        optional_import("google.protobuf.descriptor_pool", required_for=txt),
        optional_import("google.protobuf.message_factory", required_for=txt),
        optional_import("google.protobuf.timestamp_pb2", required_for=txt),
    )


class _ProtoSchemaBuilder:
    """
    Assemble a protobuf ``FileDescriptorProto`` and produce message classes.

    Wraps the verbose ``descriptor_pb2`` API so each message and field can be
    declared in a single line. Field numbers must match Sintela's wire schema
    (see the module docstring); names are arbitrary local labels.
    """

    def __init__(self, package_name: str, file_name: str):
        descriptor_pb2, pool, factory, timestamp = _import_protobuf()
        self._descriptor_pool = pool
        self._message_factory = factory
        self._timestamp_pb2 = timestamp
        # Short aliases for the field types/labels Sintela messages use.
        fd = descriptor_pb2.FieldDescriptorProto
        self.INT32 = fd.TYPE_INT32
        self.UINT32 = fd.TYPE_UINT32
        self.FLOAT = fd.TYPE_FLOAT
        self.BOOL = fd.TYPE_BOOL
        self.STRING = fd.TYPE_STRING
        self.BYTES = fd.TYPE_BYTES
        self.MESSAGE = fd.TYPE_MESSAGE
        self.REPEATED = fd.LABEL_REPEATED
        self._optional = fd.LABEL_OPTIONAL
        self.package_name = package_name
        self._file_proto = descriptor_pb2.FileDescriptorProto()
        self._file_proto.name = file_name
        self._file_proto.package = package_name
        self._file_proto.dependency.append("google/protobuf/timestamp.proto")

    def message(self, name: str) -> _ProtoMessageBuilder:
        """Declare a new message type and return a helper to add its fields."""
        message_proto = self._file_proto.message_type.add()
        message_proto.name = name
        return _ProtoMessageBuilder(self, message_proto)

    def build(self, *names: str) -> dict:
        """Register the schema and return the requested message classes."""
        pool = self._descriptor_pool.DescriptorPool()
        pool.AddSerializedFile(self._timestamp_pb2.DESCRIPTOR.serialized_pb)
        pool.Add(self._file_proto)
        out = {}
        for name in names:
            descriptor = pool.FindMessageTypeByName(f"{self.package_name}.{name}")
            out[name] = self._message_factory.GetMessageClass(descriptor)
        return out


class _ProtoMessageBuilder:
    """Add fields to a single protobuf message declared by a schema builder."""

    def __init__(self, schema: _ProtoSchemaBuilder, message_proto):
        self._schema = schema
        self._message_proto = message_proto

    def add(self, name, number, type_, *, label=None, type_name=""):
        """Add one field; a ``type_name`` without a leading dot is local."""
        field = self._message_proto.field.add()
        field.name = name
        field.number = number
        field.label = self._schema._optional if label is None else label
        field.type = type_
        if type_name:
            if not type_name.startswith("."):
                type_name = f".{self._schema.package_name}.{type_name}"
            field.type_name = type_name
        return self


@cache
def _get_proto_messages(include_sample_fields: bool = True):
    """
    Build lightweight protobuf messages for supported Sintela packet types.

    When ``include_sample_fields`` is False the (potentially large) sample
    payload fields are omitted, which is all that is needed for scanning.  The
    two variants live in separate descriptor pools (distinct package/file
    names) so they can coexist; ``@cache`` keys on the flag.
    """
    suffix = "" if include_sample_fields else "_scan"
    return _build_proto_messages(
        include_sample_fields=include_sample_fields,
        package_name=f"sintela_common{suffix}",
        file_name=f"sintela_common{suffix or '_lite'}.proto",
    )


def _build_proto_messages(
    *,
    include_sample_fields: bool,
    package_name: str,
    file_name: str,
):
    """
    Build lightweight protobuf message classes for data packets.

    Descriptors are assembled by hand (rather than from generated ``*_pb2.py``)
    so protobuf stays optional and only the fields DASCore reads are declared.
    Field numbers must match Sintela's wire schema; see the module docstring.
    """
    schema = _ProtoSchemaBuilder(package_name, file_name)

    common = schema.message("CommonHeader")
    common.add("time", 1, schema.MESSAGE, type_name=".google.protobuf.Timestamp")
    common.add("num_channels", 2, schema.INT32)
    common.add("sample_rate", 3, schema.FLOAT)
    common.add("channel_spacing", 4, schema.FLOAT)
    common.add("gauge_length", 5, schema.FLOAT)
    common.add("start_channel", 6, schema.INT32)
    common.add("end_of_replay", 7, schema.BOOL)
    common.add("fiber_flipped", 8, schema.BOOL)
    common.add("loop_removed", 9, schema.BOOL)
    common.add("has_dropped_samples", 10, schema.BOOL)
    common.add("timeseries_data_type", 11, schema.INT32)
    common.add("demod_data_type", 12, schema.INT32)

    ts_header = schema.message("TimeseriesHeader")
    ts_header.add("common_header", 1, schema.MESSAGE, type_name="CommonHeader")
    ts_header.add("sample_count", 2, schema.UINT32)
    ts_header.add("num_samples", 3, schema.INT32)
    ts_header.add("channel_step", 4, schema.INT32)

    ts_packet = schema.message("TimeseriesPacket")
    ts_packet.add("header", 1, schema.MESSAGE, type_name="TimeseriesHeader")
    if include_sample_fields:
        ts_packet.add("samples", 3, schema.FLOAT, label=schema.REPEATED)
        ts_packet.add("raw_frames", 4, schema.BYTES)

    band_info = schema.message("BandDataInfo")
    band_info.add("band_data_type", 1, schema.INT32)
    band_info.add("start", 2, schema.FLOAT)
    band_info.add("end", 3, schema.FLOAT)
    band_info.add("averaging_type", 4, schema.INT32)
    band_info.add("description", 5, schema.STRING)
    band_info.add("source", 6, schema.STRING)

    band_header = schema.message("BandHeader")
    band_header.add("common_header", 1, schema.MESSAGE, type_name="CommonHeader")
    band_header.add(
        "band_data_info",
        2,
        schema.MESSAGE,
        label=schema.REPEATED,
        type_name="BandDataInfo",
    )

    band_packet = schema.message("BandPacket")
    band_packet.add("header", 1, schema.MESSAGE, type_name="BandHeader")
    if include_sample_fields:
        band_packet.add("samples", 2, schema.FLOAT, label=schema.REPEATED)

    fft_header = schema.message("FFTHeader")
    fft_header.add("common_header", 1, schema.MESSAGE, type_name="CommonHeader")
    fft_header.add("num_bins", 2, schema.INT32)
    fft_header.add("bin_res", 3, schema.FLOAT)
    fft_header.add("averaging_type", 4, schema.INT32)
    fft_header.add("channel_step", 5, schema.INT32)
    fft_header.add("normalised", 6, schema.BOOL)
    fft_header.add("has_power_data", 7, schema.BOOL)
    fft_header.add("has_complex_data", 8, schema.BOOL)

    fft_packet = schema.message("FFTPacket")
    fft_packet.add("header", 1, schema.MESSAGE, type_name="FFTHeader")
    if include_sample_fields:
        fft_packet.add("samples", 2, schema.FLOAT, label=schema.REPEATED)

    return schema.build("TimeseriesPacket", "BandPacket", "FFTPacket")


@cache
def _get_meta_message_class():
    """Build a lightweight RecordingMetadata parser for selected fields."""
    schema = _ProtoSchemaBuilder("sintela_meta", "sintela_meta_lite.proto")

    identification = schema.message("IdentificationResponse")
    identification.add("manufacturer", 1, schema.STRING)
    identification.add("system_type", 2, schema.STRING)
    identification.add("model", 3, schema.STRING)
    identification.add("serial_number", 4, schema.STRING)

    acquisition = schema.message("AcquisitionStatsResponse")
    acquisition.add("fiber_id", 8, schema.INT32)

    recording = schema.message("RecordingMetadata")
    recording.add("recorder_namespace", 1, schema.STRING)
    recording.add(
        "metadata_recording_time",
        2,
        schema.MESSAGE,
        type_name=".google.protobuf.Timestamp",
    )
    recording.add(
        "identification", 3, schema.MESSAGE, type_name="IdentificationResponse"
    )
    recording.add(
        "acquisition_stats", 7, schema.MESSAGE, type_name="AcquisitionStatsResponse"
    )

    return schema.build("RecordingMetadata")["RecordingMetadata"]


def _parse_meta(payload: bytes) -> ParsedMeta:
    """Parse selected fields from a META payload."""
    message_cls = _get_meta_message_class()
    msg = message_cls()
    decode_error = _get_protobuf_decode_error()
    with suppress_warnings():
        try:
            msg.ParseFromString(payload)
        except decode_error as exc:
            msg = f"Failed to parse Sintela protobuf META payload: {exc}"
            raise InvalidFiberFileError(msg) from exc
    identification = msg.identification if msg.HasField("identification") else None
    acquisition = msg.acquisition_stats if msg.HasField("acquisition_stats") else None
    return ParsedMeta(
        recorder_namespace=str(getattr(msg, "recorder_namespace", "") or ""),
        metadata_recording_time=(
            _timestamp_to_dt64(msg.metadata_recording_time)
            if msg.HasField("metadata_recording_time")
            else None
        ),
        instrument_manufacturer=str(getattr(identification, "manufacturer", "") or ""),
        instrument_model=str(getattr(identification, "model", "") or ""),
        serial_number=str(getattr(identification, "serial_number", "") or ""),
        fiber_id=(
            int(acquisition.fiber_id)
            if acquisition is not None and acquisition.HasField("fiber_id")
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
    messages = _get_proto_messages(include_sample_fields=not scan_mode)
    decode_error = _get_protobuf_decode_error()
    parsed: list[Any] = []
    meta = ParsedMeta()
    first_unsupported_tag = None
    for record in records:
        tag = record.tag
        if tag == META_TAG:
            meta = _parse_meta(record.payload)
            continue
        packet_name = _TAG_TO_PACKET.get(tag)
        if packet_name is None:
            first_unsupported_tag = first_unsupported_tag or tag
            continue
        msg = messages[packet_name]()
        try:
            msg.ParseFromString(record.payload)
        except decode_error as exc:
            out = f"Failed to parse Sintela protobuf {tag} payload: {exc}"
            raise InvalidFiberFileError(out) from exc
        parsed.append((tag, msg))
    if not parsed:
        if first_unsupported_tag is not None:
            raise InvalidFiberFileError(
                f"Unsupported Sintela protobuf tag {first_unsupported_tag!r}."
            )
        raise InvalidFiberFileError("No supported Sintela protobuf data packets found.")
    return parsed, meta


def _get_time_coord_from_samples(start: np.datetime64, sample_rate: float, size: int):
    """Build a regularly sampled time coordinate."""
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        msg = f"Invalid Sintela protobuf sample_rate: {sample_rate!r}."
        raise InvalidFiberFileError(msg)
    step = dc.to_timedelta64(1 / sample_rate)
    return get_coord(start=start, stop=start + step * size, step=step)


def _get_distance_coord(start_channel: int, spacing: float, count: int, step: int = 1):
    """Build the distance coordinate."""
    if not np.isfinite(spacing) or spacing <= 0:
        msg = f"Invalid Sintela protobuf channel_spacing: {spacing!r}."
        raise InvalidFiberFileError(msg)
    if isinstance(step, bool) or not isinstance(step, int | np.integer) or step <= 0:
        msg = f"Invalid Sintela protobuf channel_step: {step!r}."
        raise InvalidFiberFileError(msg)
    start = start_channel * spacing
    return get_coord(
        start=start,
        stop=start + spacing * step * count,
        step=spacing * step,
        units="m",
    )


def _get_times(times: list[np.datetime64]):
    """Build a time coordinate from packet timestamps."""
    return get_coord(data=np.asarray(times, dtype="datetime64[ns]"))


def _assert_float_equal(name: str, values: list[float], *, rtol: float = 1e-6):
    """Ensure float values match within a small tolerance."""
    if not values:
        msg = f"Cannot validate {name} for an empty Sintela protobuf payload."
        raise InvalidFiberFileError(msg)
    first = values[0]
    for value in values[1:]:
        if not np.isclose(first, value, rtol=rtol, atol=0.0):
            raise InvalidFiberFileError(
                f"Inconsistent {name} across Sintela protobuf packets."
            )
    return first


def _base_attrs(
    common_header, packet_type: str, meta: ParsedMeta, extra: dict | None = None
):
    """Construct base attrs from the packet header and META metadata.

    Each packet family supplies its own ``data_type``/``data_units`` via
    ``extra``; the fields below are shared across all families.
    """
    attrs = dict(
        data_category="DAS",
        packet_type=packet_type,
        recorder_namespace=meta.recorder_namespace,
        metadata_recording_time=meta.metadata_recording_time,
        instrument_manufacturer=meta.instrument_manufacturer,
        instrument_model=meta.instrument_model,
        # Mirror the recorder serial into the canonical PatchAttrs field while
        # preserving the raw vendor-specific name for round-tripping/debugging.
        instrument_id=meta.serial_number,
        serial_number=meta.serial_number,
        fiber_id=meta.fiber_id,
        start_channel=int(getattr(common_header, "start_channel", 0)),
        channel_step=None,
    )
    if extra:
        attrs.update(extra)
    return SintelaProtobufAttrs(**attrs)


def _get_band_attr_data_type(band_def: tuple[tuple[Any, ...], ...]) -> tuple[str, str]:
    """Return patch-level BAND data type/units."""
    mapped = [_BAND_DATA_TYPE_MAP.get(int(item[0])) for item in band_def]
    if any(item is None for item in mapped):
        return "frequency_band_energy", ""
    first = mapped[0]
    if all(item == first for item in mapped):
        return "frequency_band_energy", first[1]
    return "frequency_band_energy", ""


def _assert_equal(name: str, values: list[Any]):
    """Ensure all values in a list are equal."""
    if not values:
        msg = f"Cannot validate {name} for an empty Sintela protobuf payload."
        raise InvalidFiberFileError(msg)
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
        raise InvalidFiberFileError(
            "Mixed Sintela protobuf packet families are unsupported."
        )
    return families.pop()


def _decode_family(parsed: list[tuple[str, Any]], meta: ParsedMeta):
    """Decode one parsed data family into data, coords, and attrs."""
    family_cls = _FAMILY_CLASSES[_validate_single_family(parsed)]
    return family_cls.from_parsed(parsed, meta).decode(parsed)


class _PacketHeaderFields(DascoreBaseModel):
    """
    Validated per-packet header fields shared by all families.

    Constructing the model enforces the per-field constraints; cross-packet
    consistency is enforced separately by the callers via ``_reduce_common``.
    """

    num_channels: PositiveInt
    channel_spacing: PositiveFiniteFloat
    gauge_length: float
    start_channel: int


class _TimeseriesHeaderFields(_PacketHeaderFields):
    """Adds the timeseries-only constrained sample rate."""

    sample_rate: PositiveFiniteFloat
    num_samples: PositiveInt


class _FFTHeaderFields(_PacketHeaderFields):
    """Adds the FFT-only constrained bin fields."""

    num_bins: PositiveInt
    bin_res: PositiveFiniteFloat


def _validate_header(model_cls, **fields):
    """Build a header model, mapping pydantic errors to InvalidFiberFileError."""
    try:
        return model_cls(**fields)
    except ValidationError as exc:
        name = exc.errors()[0]["loc"][0]
        msg = f"Invalid Sintela protobuf {name}: {fields.get(name)!r}."
        raise InvalidFiberFileError(msg) from exc


def _reduce_common(fields: list[_PacketHeaderFields]):
    """Collapse validated per-packet header fields to single agreed values."""
    return (
        _assert_equal("num_channels", [f.num_channels for f in fields]),
        _assert_float_equal("channel_spacing", [f.channel_spacing for f in fields]),
        _assert_float_equal("gauge_length", [f.gauge_length for f in fields]),
        _assert_equal("start_channel", [f.start_channel for f in fields]),
    )


class _PacketMetadata(DascoreBaseModel):
    """
    Base for a validated, decodable Sintela packet family.

    Subclasses validate their headers in ``from_parsed`` and fill a
    preallocated array in ``decode``; ``shape`` and ``scan`` are shared.
    """

    common_header: Any
    packet_type: str
    num_channels: int
    gauge_length: float
    coords: Any
    attrs: SintelaProtobufAttrs

    @property
    def shape(self) -> tuple[int, ...]:
        """Patch shape, identical to the coordinate-manager shape."""
        return self.coords.shape

    @property
    def dtype(self):
        """Sample dtype for this family."""
        return np.float32

    def scan(self):
        """Summarize the family for a scan without allocating samples."""
        return (self.shape, self.coords, self.attrs, str(np.dtype(self.dtype)))


class TimeseriesMetadata(_PacketMetadata):
    """Validated timeseries packets laid out as (time, distance)."""

    total_samples: int
    sample_rate: float
    channel_spacing: float
    start_channel: int
    channel_step: int

    @classmethod
    def from_parsed(cls, parsed: list[tuple[str, Any]], meta: ParsedMeta):
        """Validate timeseries headers and build shared attrs/coords."""
        headers = [msg.header for _tag, msg in parsed]
        common_headers = [h.common_header for h in headers]
        fields = [
            _validate_header(
                _TimeseriesHeaderFields,
                num_channels=int(ch.num_channels),
                channel_spacing=float(ch.channel_spacing),
                gauge_length=float(ch.gauge_length),
                start_channel=int(ch.start_channel),
                sample_rate=float(ch.sample_rate),
                num_samples=int(h.num_samples),
            )
            for h, ch in zip(headers, common_headers, strict=False)
        ]
        num_channels, channel_spacing, gauge_length, start_channel = _reduce_common(
            fields
        )
        sample_rate = _assert_float_equal(
            "sample_rate", [f.sample_rate for f in fields]
        )
        channel_step = _assert_equal(
            "channel_step", [int(h.channel_step) for h in headers]
        )
        data_type = _assert_equal(
            "timeseries_data_type",
            [int(ch.timeseries_data_type) for ch in common_headers],
        )
        demod_data_type = _assert_equal(
            "demod_data_type", [int(ch.demod_data_type) for ch in common_headers]
        )
        for ch in common_headers:
            if ch.has_dropped_samples:
                raise InvalidFiberFileError(
                    "Dropped samples in Sintela protobuf stream."
                )
        sample_counts = [int(h.sample_count) for h in headers]
        num_samples_per_packet = [f.num_samples for f in fields]
        for current, nxt, count in zip(
            sample_counts,
            sample_counts[1:],
            num_samples_per_packet[:-1],
            strict=False,
        ):
            if current + count != nxt:
                raise InvalidFiberFileError(
                    "Non-contiguous Sintela protobuf sample counts."
                )
        total_samples = sum(num_samples_per_packet)
        first_time = _common_header_time(common_headers[0])
        if first_time is None:
            raise InvalidFiberFileError("Missing Sintela protobuf start time.")
        time = _get_time_coord_from_samples(first_time, sample_rate, total_samples)
        distance = _get_distance_coord(
            start_channel, channel_spacing, num_channels, channel_step
        )
        coords = get_coord_manager({"time": time, "distance": distance}, dims=DIMS_TS)
        mapping = _TIMESERIES_DATA_TYPE_MAP.get(data_type, ("phase", "radians"))
        attrs = _base_attrs(
            common_headers[0],
            packet_type=parsed[0][0],
            meta=meta,
            extra=dict(
                gauge_length=gauge_length,
                channel_step=channel_step,
                data_type=mapping[0],
                data_units=mapping[1],
                demod_data_type=str(demod_data_type),
            ),
        )
        return cls(
            common_header=common_headers[0],
            packet_type=parsed[0][0],
            num_channels=num_channels,
            total_samples=total_samples,
            sample_rate=sample_rate,
            channel_spacing=channel_spacing,
            gauge_length=gauge_length,
            start_channel=start_channel,
            channel_step=channel_step,
            coords=coords,
            attrs=attrs,
        )

    def decode(self, parsed: list[tuple[str, Any]]):
        """Decode timeseries packets into data, coords, and attrs."""
        data = np.empty(self.shape, dtype=self.dtype)
        index = 0
        for _tag, msg in parsed:
            packet = np.asarray(msg.samples, dtype=np.float32)
            rows = int(msg.header.num_samples)
            expected = rows * self.num_channels
            if packet.size != expected:
                raise InvalidFiberFileError(
                    "Unexpected Sintela protobuf TS sample payload size."
                )
            data[index : index + rows] = packet.reshape(rows, self.num_channels)
            index += rows
        return data, self.coords, self.attrs


class BandMetadata(_PacketMetadata):
    """Validated band packets laid out as (time, distance, band)."""

    num_bands: int
    band_def: tuple[tuple[Any, ...], ...]

    @classmethod
    def from_parsed(cls, parsed: list[tuple[str, Any]], meta: ParsedMeta):
        """Validate band headers and build shared attrs/coords."""
        headers = [msg.header for _tag, msg in parsed]
        common_headers = [h.common_header for h in headers]
        fields = [
            _validate_header(
                _PacketHeaderFields,
                num_channels=int(ch.num_channels),
                channel_spacing=float(ch.channel_spacing),
                gauge_length=float(ch.gauge_length),
                start_channel=int(ch.start_channel),
            )
            for ch in common_headers
        ]
        num_channels, channel_spacing, gauge_length, start_channel = _reduce_common(
            fields
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
                "band_start_frequency": (
                    "band",
                    np.asarray([x[1] for x in band_def]),
                ),
                "band_end_frequency": (
                    "band",
                    np.asarray([x[2] for x in band_def]),
                ),
                "band_description": (
                    "band",
                    np.asarray([x[3] for x in band_def], dtype=object),
                ),
                "band_source": (
                    "band",
                    np.asarray([x[4] for x in band_def], dtype=object),
                ),
            },
            dims=DIMS_BAND,
        )
        data_type, data_units = _get_band_attr_data_type(band_def)
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
        return cls(
            common_header=common_headers[0],
            packet_type=parsed[0][0],
            num_channels=num_channels,
            num_bands=num_bands,
            gauge_length=gauge_length,
            band_def=band_def,
            coords=coords,
            attrs=attrs,
        )

    def decode(self, parsed: list[tuple[str, Any]]):
        """Decode band packets into data, coords, and attrs."""
        data = np.empty(self.shape, dtype=self.dtype)
        for ind, (_tag, msg) in enumerate(parsed):
            packet = np.asarray(msg.samples, dtype=np.float32)
            expected = self.num_channels * self.num_bands
            if packet.size != expected:
                raise InvalidFiberFileError(
                    "Unexpected Sintela protobuf BAND payload size."
                )
            data[ind] = packet.reshape(self.num_channels, self.num_bands)
        return data, self.coords, self.attrs


class FFTMetadata(_PacketMetadata):
    """Validated FFT packets laid out as (time, distance, frequency)."""

    num_bins: int
    channel_step: int
    has_complex: bool

    @property
    def dtype(self):
        """Complex when the packets carry complex spectra, else float32."""
        return np.complex64 if self.has_complex else np.float32

    @classmethod
    def from_parsed(cls, parsed: list[tuple[str, Any]], meta: ParsedMeta):
        """Validate FFT headers and build shared attrs/coords."""
        headers = [msg.header for _tag, msg in parsed]
        common_headers = [h.common_header for h in headers]
        fields = [
            _validate_header(
                _FFTHeaderFields,
                num_channels=int(ch.num_channels),
                channel_spacing=float(ch.channel_spacing),
                gauge_length=float(ch.gauge_length),
                start_channel=int(ch.start_channel),
                num_bins=int(h.num_bins),
                bin_res=float(h.bin_res),
            )
            for h, ch in zip(headers, common_headers, strict=False)
        ]
        num_channels, channel_spacing, gauge_length, start_channel = _reduce_common(
            fields
        )
        num_bins = _assert_equal("num_bins", [f.num_bins for f in fields])
        bin_res = _assert_float_equal("bin_res", [f.bin_res for f in fields])
        has_complex = _assert_equal(
            "has_complex_data", [bool(h.has_complex_data) for h in headers]
        )
        channel_step = _assert_equal(
            "channel_step", [int(h.channel_step) for h in headers]
        )
        times = [_common_header_time(ch) for ch in common_headers]
        if any(x is None for x in times):
            raise InvalidFiberFileError("Missing time in Sintela FFT packet.")
        distance = _get_distance_coord(
            start_channel, channel_spacing, num_channels, channel_step
        )
        frequency = get_coord(
            start=0.0, stop=bin_res * num_bins, step=bin_res, units="Hz"
        )
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
                **_FFT_ATTR_DEFAULTS,
            ),
        )
        return cls(
            common_header=common_headers[0],
            packet_type=parsed[0][0],
            num_channels=num_channels,
            num_bins=num_bins,
            gauge_length=gauge_length,
            channel_step=channel_step,
            has_complex=has_complex,
            coords=coords,
            attrs=attrs,
        )

    def decode(self, parsed: list[tuple[str, Any]]):
        """Decode FFT packets into data, coords, and attrs."""
        data = np.empty(self.shape, dtype=self.dtype)
        for ind, (_tag, msg) in enumerate(parsed):
            packet = np.asarray(msg.samples, dtype=np.float32)
            if self.has_complex:
                expected = self.num_channels * self.num_bins * 2
                if packet.size != expected:
                    raise InvalidFiberFileError(
                        "Unexpected Sintela protobuf FFT payload size."
                    )
                packet = packet.reshape(self.num_channels, self.num_bins, 2)
                packet = packet[..., 0] + 1j * packet[..., 1]
            else:
                expected = self.num_channels * self.num_bins
                if packet.size != expected:
                    raise InvalidFiberFileError(
                        "Unexpected Sintela protobuf FFT payload size."
                    )
                packet = packet.reshape(self.num_channels, self.num_bins)
            data[ind] = packet
        return data, self.coords, self.attrs


# Family name (from `_validate_single_family`) -> packet class.
_FAMILY_CLASSES = {
    "timeseries": TimeseriesMetadata,
    "band": BandMetadata,
    "fft": FFTMetadata,
}


def read_payload(resource):
    """Decode a Sintela protobuf file into data, coords, and attrs."""
    records = _iter_envelope_records(resource, strict=True)
    parsed, meta = _parse_records(records, scan_mode=False)
    return _decode_family(parsed, meta)


def scan_payload(resource) -> list[dict[str, Any]]:
    """Decode a Sintela protobuf file and return FiberIO scan payloads."""
    records = _iter_envelope_records(resource, strict=True)
    parsed, meta = _parse_records(records, scan_mode=True)
    family_cls = _FAMILY_CLASSES[_validate_single_family(parsed)]
    shape, coords, attrs, dtype = family_cls.from_parsed(parsed, meta).scan()
    return [
        _make_scan_payload(
            attrs=attrs,
            coords=coords,
            dims=coords.dims,
            shape=shape,
            dtype=dtype,
        )
    ]
