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
from collections.abc import Iterable, Iterator
from functools import cache
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

import dascore as dc
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import InvalidFiberFileError
from dascore.io.core import ScanPayload, make_scan_payload
from dascore.models import OptionalFiniteFloat, PositiveFiniteFloat, PositiveInt
from dascore.utils.misc import optional_import, suppress_warnings

PBUF_MAGIC = 0x46554250
META_TAG = "META"
# "RF01" is deliberately absent: those packets are understood to carry their
# samples in the packed `raw_frames` blob, which has no decoder here. Since a
# file's shape and coords come from the headers, advertising the tag would let
# such a file scan cleanly and only fail once a patch was actually read, so it
# is left undetected until raw_frames can be decoded.
TS_TAGS = frozenset({"TS05"})
FFT_TAGS = frozenset({"FFT", "FFT-"})
BAND_TAGS = frozenset({"BAND"})
# Maps each data tag to the packet message class (built by
# ``_get_proto_messages``) that decodes it.
_TAG_TO_PACKET = {
    **dict.fromkeys(TS_TAGS, "TimeseriesPacket"),
    **dict.fromkeys(BAND_TAGS, "BandPacket"),
    **dict.fromkeys(FFT_TAGS, "FFTPacket"),
}
# sample_count is a uint32 field; contiguity checks wrap at this bound.
_SAMPLE_COUNT_MODULUS = 2**32
# Bytes of a payload pulled by metadata-only paths, where protobuf puts the
# `header` submessage; ample for any header, a fraction of a typical packet.
_HEADER_PREFIX_SIZE = 1 << 16
# Protobuf wire key for field 1, wire type 2 (length-delimited): `header`.
_HEADER_FIELD_KEY = 0x0A
# A seek skips a payload's transfer but costs a platter rotation (~18 ms
# measured on a USB HDD), more than sequentially reading the ~1.6 MB it saves.
# Smaller remainders are read and dropped, keeping readahead engaged.
_SEEK_SKIP_THRESHOLD = 1 << 21
# Window for the backwards search from EOF for the final record's framing:
# the first packet's size plus slack for jitter, and a wider second attempt
# used only when that finds nothing (a recording opening with a short warm-up
# packet does not predict its own final packet's size).
_TAIL_SEARCH_SLACK = 1 << 12
_MAX_TAIL_SEARCH = 1 << 23
# A base-128 varint encodes at most a 64-bit value, so it never exceeds this.
_MAX_VARINT_BYTES = 10
# How far endpoint timestamps may drift from the span their sample counts
# imply. Recorders resynchronize their clock against the counter, so valid
# files do NOT agree exactly: across a 10,308 file archive a quarter drifted,
# worst 56 ms over a 60 s span (~0.09%). Set an order of magnitude above that,
# since a false rejection silently costs a full read while what is worth
# catching is wrong by seconds to hours. The per-packet term is one packet,
# not several: a declared length is corruption-controlled, so scaling by it
# would let a malformed file widen its own budget.
_ENDPOINT_TIME_TOLERANCE = 1 / 100
_ENDPOINT_TIME_PACKETS = 1
_ENDPOINT_TIME_FLOOR_NS = 10_000_000
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
# Complex packets carry raw Fourier coefficients, not real power per
# frequency, so they must not inherit the power-spectral-density label.
_FFT_COMPLEX_ATTRS = {
    "data_type": "fourier_transform",
    "data_units": "",
}


def _get_fft_data_type(has_complex: bool) -> dict[str, str]:
    """Return the data_type/data_units attrs matching the FFT representation."""
    return dict(_FFT_COMPLEX_ATTRS if has_complex else _FFT_ATTR_DEFAULTS)


class SintelaProtobufAttrs(PatchAttrs):
    """Patch attributes for Sintela protobuf recordings."""

    gauge_length: OptionalFiniteFloat = None
    packet_type: str = ""
    recorder_namespace: str = ""
    metadata_recording_time: np.datetime64 | None = None
    fiber_id: int | None = None
    start_channel: int | None = None
    channel_step: int | None = None
    demod_data_type: str = ""


class _ProtobufModel(BaseModel):
    """
    Base for this module's parsing models.

    Plain pydantic: these validate values on the way out of a protobuf
    payload and are never serialized. Subclassing DascoreBaseModel would
    only claim each a tag in the model registry, naming them in documents
    they never appear in.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class EnvelopeRecord(_ProtobufModel):
    """The envelope information for one MTLV record."""

    tag: str
    payload: bytes


class ParsedMeta(_ProtobufModel):
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


def _read_varint(buf: bytes, pos: int) -> tuple[int | None, int]:
    """
    Return the varint starting at ``pos`` and the position after it.

    Gives up after ``_MAX_VARINT_BYTES``: a valid varint never runs longer, and
    an unterminated run of continuation bytes would otherwise build an integer
    as wide as the buffer, one 7-bit shift at a time.
    """
    value = shift = 0
    end = min(len(buf), pos + _MAX_VARINT_BYTES)
    while pos < end:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    return None, pos


def _leading_header_bytes(payload_prefix: bytes) -> bytes | None:
    """
    Return the serialized ``header`` submessage from the front of a payload.

    Every Sintela data packet carries ``header`` in field 1. The wire format
    permits any field order, but implementations emit ascending field numbers,
    so in practice the header sits at the front of the payload and can be
    recovered from a short prefix -- which is what lets the metadata paths
    ignore the samples behind it, ~99.99% of a recording.

    Returns None when the prefix does not open with field 1 or does not hold
    the whole submessage, so a differently ordered payload is still handled
    correctly, just read in full.
    """
    if not payload_prefix or payload_prefix[0] != _HEADER_FIELD_KEY:
        return None
    length, pos = _read_varint(payload_prefix, 1)
    if length is None or pos + length > len(payload_prefix):
        return None
    return payload_prefix[: pos + length]


def _stream_size(resource) -> int:
    """Return the total length of a seekable binary stream."""
    resource.seek(0, 2)
    return resource.tell()


def _iter_envelope_records(
    resource, *, strict: bool, headers_only: bool = False
) -> Iterator[EnvelopeRecord]:
    """
    Read all MTLV envelope records from a binary stream.

    With ``headers_only`` a data packet's payload is truncated to its leading
    ``header`` submessage and the samples behind it skipped, so a metadata-only
    pass never holds them. A payload whose header cannot be recovered from the
    prefix is re-read whole, so a yielded payload always parses on its own.
    META and unrecognized records are never truncated: field 1 of a META
    payload is an ordinary string, not a header.
    """

    def _stop(message):
        """Raise in strict mode; otherwise signal a clean stop to the caller."""
        if strict:
            raise InvalidFiberFileError(message)
        return True

    # Bytes of the current payload still to be stepped over. Deferred to the
    # top of the next iteration so a consumer that stops early -- format
    # detection returns on the first data tag -- never pays for a record it
    # did not ask for.
    pending = 0

    def _advance():
        """Step over the previous payload's tail, seeking only when it pays."""
        nonlocal pending
        if pending >= _SEEK_SKIP_THRESHOLD:
            resource.seek(pending, 1)
        elif pending > 0:
            # Streaming a small remainder beats a seek on spinning media and
            # keeps the kernel's sequential readahead engaged.
            resource.read(pending)
        pending = 0

    def _read_header_payload(offset, size):
        """Return just the header submessage of the payload at `offset`."""
        nonlocal pending
        prefix = resource.read(min(size, _HEADER_PREFIX_SIZE))
        header = _leading_header_bytes(prefix)
        if header is None:
            resource.seek(offset)
            return resource.read(size)
        pending = size - len(prefix)
        return header

    # Payload truncation is judged against the stream's length rather than a
    # short read, because a skipped tail never produces one.
    file_size = _stream_size(resource)
    resource.seek(0)
    while True:
        _advance()
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
        payload_start = resource.tell()
        if file_size - payload_start < size and _stop(
            "Truncated Sintela protobuf payload."
        ):
            return
        if headers_only and tag in _TAG_TO_PACKET:
            payload = _read_header_payload(payload_start, size)
        else:
            payload = resource.read(size)
        yield EnvelopeRecord(tag=tag, payload=payload)


def get_supported_family_tag(resource) -> str | None:
    """Return the first supported data tag in a file without using protobuf."""
    for record in _iter_envelope_records(resource, strict=False, headers_only=True):
        if record.tag == META_TAG:
            continue
        if record.tag in _TAG_TO_PACKET:
            return record.tag
        # Detection is intentionally tolerant of unknown non-data records so a
        # valid family tag later in the file can still identify the format.
        continue
    return None


def _parse_frame(buf: bytes, index: int = 0) -> tuple[str, int] | None:
    """Return the (tag, payload size) of the MTLV framing at ``buf[index:]``."""
    if index + 12 > len(buf):
        return None
    if struct.unpack("<I", buf[index : index + 4])[0] != PBUF_MAGIC:
        return None
    tag = buf[index + 4 : index + 8].rstrip(b"\x00").decode("utf-8", errors="ignore")
    size = struct.unpack("<I", buf[index + 8 : index + 12])[0]
    return tag, size


def _read_record_head(resource, offset: int, file_size: int):
    """
    Return (tag, payload size, payload prefix) for the record at ``offset``.

    None when the framing is invalid or declares a payload the file is too
    short to hold. That length check matters: later steps size reads and
    buffers from it, so one corrupt field could pull a whole file into memory.
    """
    resource.seek(offset)
    buf = resource.read(12 + _HEADER_PREFIX_SIZE)
    frame = _parse_frame(buf)
    if frame is None or offset + 12 + frame[1] > file_size:
        return None
    return (*frame, buf[12:])


def _find_first_data_record(resource, file_size: int):
    """
    Walk from the start of the file to its first data record.

    Returns the META metadata picked up on the way plus that record as
    (tag, size, payload prefix), or None when the framing is unreadable or the
    file holds no data records at all.
    """
    meta = ParsedMeta()
    offset = 0
    while offset < file_size:
        record = _read_record_head(resource, offset, file_size)
        if record is None:
            return None
        tag, size, _prefix = record
        if tag in _TAG_TO_PACKET:
            return meta, record
        if tag == META_TAG:
            resource.seek(offset + 12)
            meta = _parse_meta(resource.read(size))
        offset += 12 + size
    return None


def _find_last_record(resource, file_size: int, window: int):
    """
    Locate the file's final record by searching backwards from its end.

    The real last record is the one whose framing lands exactly on EOF. A
    stray magic word in sample data would need a length hitting that same
    byte, so the first match walking backwards is the record itself.
    """
    start = max(0, file_size - window)
    resource.seek(start)
    buf = resource.read(file_size - start)
    index = len(buf)
    while (index := buf.rfind(struct.pack("<I", PBUF_MAGIC), 0, index)) != -1:
        frame = _parse_frame(buf, index)
        if frame is not None and start + index + 12 + frame[1] == file_size:
            payload = buf[index + 12 : index + 12 + _HEADER_PREFIX_SIZE]
            return (*frame, payload)
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


def _parse_packet(tag: str, payload: bytes, messages, decode_error):
    """Decode one data record's payload into its packet message."""
    msg = messages[_TAG_TO_PACKET[tag]]()
    try:
        msg.ParseFromString(payload)
    except decode_error as exc:
        out = f"Failed to parse Sintela protobuf {tag} payload: {exc}"
        raise InvalidFiberFileError(out) from exc
    return msg


def _parse_records(
    records: Iterable[EnvelopeRecord], *, scan_mode: bool = False
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
        if tag not in _TAG_TO_PACKET:
            first_unsupported_tag = first_unsupported_tag or tag
            continue
        msg = _parse_packet(tag, record.payload, messages, decode_error)
        if scan_mode:
            # Omitting the sample fields from the scan descriptor stops them
            # being *decoded*, but protobuf still retains their raw bytes as
            # unknown fields, so a metadata-only scan would otherwise hold the
            # whole recording in memory (~99% of the payload for a typical
            # file). Drop them now that the header fields are parsed.
            msg.DiscardUnknownFields()
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
    return get_coord(start=start, step=step, shape=(size,))


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


def _get_times(times: list[np.datetime64 | None]):
    """
    Build a time coordinate from packet timestamps.

    Callers reject a packet with no header time before getting here; the
    None in the signature is what the list comprehension produces, not a
    supported input.
    """
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
    # Validated from a dict because the interrogator facts are nested names,
    # which no keyword argument can spell.
    attrs = SintelaProtobufAttrs.model_validate(
        {
            "data_category": "DAS",
            "packet_type": packet_type,
            "recorder_namespace": meta.recorder_namespace,
            "metadata_recording_time": meta.metadata_recording_time,
            "fiber_id": meta.fiber_id,
            "start_channel": int(getattr(common_header, "start_channel", 0)),
            "channel_step": None,
            "interrogator.manufacturer": meta.instrument_manufacturer,
            "interrogator.model": meta.instrument_model,
            "interrogator.serial_number": meta.serial_number,
        }
    )
    return attrs.new(**extra) if extra else attrs


def _get_band_attr_data_type(band_def: tuple[tuple[Any, ...], ...]) -> tuple[str, str]:
    """Return patch-level BAND data type/units."""
    mapped = [_BAND_DATA_TYPE_MAP.get(int(item[0])) for item in band_def]
    # Only bands which all map to the same known data type carry its units;
    # an unmapped band is None, which no mapped band compares equal to.
    first = mapped[0]
    if first is None or any(item != first for item in mapped):
        return "frequency_band_energy", ""
    return "frequency_band_energy", first[1]


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


class _PacketHeaderFields(_ProtobufModel):
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


class _PacketMetadata(_ProtobufModel):
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
    def from_parsed(
        cls,
        parsed: list[tuple[str, Any]],
        meta: ParsedMeta,
        *,
        total_samples: int | None = None,
    ):
        """
        Validate timeseries headers and build shared attrs/coords.

        ``total_samples`` lets a caller holding only the file's first and last
        packets supply the length it derived from their sample counts. Without
        it ``parsed`` is taken to be every packet in the file and their
        contiguity is checked here.
        """
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
        if total_samples is None:
            sample_counts = [int(h.sample_count) for h in headers]
            num_samples_per_packet = [f.num_samples for f in fields]
            for current, nxt, count in zip(
                sample_counts,
                sample_counts[1:],
                num_samples_per_packet[:-1],
                strict=False,
            ):
                # sample_count is a uint32 on the wire, so a long acquisition
                # (or a recorder-wide counter) can wrap to zero mid-file.
                # Compare modulo 2**32 so a wrapped-but-contiguous packet is
                # not read as a gap.
                if (current + count) % _SAMPLE_COUNT_MODULUS != nxt:
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

    def _fill_packet(self, data, index: int, tag: str, msg) -> int:
        """Copy one packet's samples into ``data`` at ``index``, return its rows."""
        packet = np.asarray(msg.samples, dtype=np.float32)
        rows = int(msg.header.num_samples)
        expected = rows * self.num_channels
        if not packet.size and msg.raw_frames:
            # Timeseries packets may carry samples in the packed `raw_frames`
            # blob instead of the repeated `samples` field. That encoding is
            # undocumented here, so fail with a specific message rather than a
            # confusing payload-size mismatch.
            msg_ = (
                f"Sintela protobuf {tag} packets store samples in "
                "raw_frames, which DASCore cannot yet decode."
            )
            raise InvalidFiberFileError(msg_)
        if packet.size != expected:
            raise InvalidFiberFileError(
                "Unexpected Sintela protobuf TS sample payload size."
            )
        data[index : index + rows] = packet.reshape(rows, self.num_channels)
        return rows

    def decode(self, parsed: list[tuple[str, Any]]):
        """Decode timeseries packets into data, coords, and attrs."""
        data = np.empty(self.shape, dtype=self.dtype)
        index = 0
        for tag, msg in parsed:
            index += self._fill_packet(data, index, tag, msg)
        return data, self.coords, self.attrs

    def decode_stream(self, resource, meta: ParsedMeta):
        """
        Fill the patch array packet by packet straight from the stream.

        The endpoint shortcut already established the shape, so samples go
        straight to their final home and each decoded packet is released at
        once; holding every packet *and* the output array, as the generic path
        must, costs roughly twice the patch.

        Headers are copied into fresh messages before the packet is dropped.
        Clearing the samples in place would not do: protobuf releases an arena
        only when the whole message dies, so a cleared packet still owns the
        space its samples occupied. The copies go to ``from_parsed`` at the
        end, so full cross-packet validation still runs and supplies the coords
        and attrs returned.
        """
        messages = _get_proto_messages(include_sample_fields=True)
        header_messages = _get_proto_messages(include_sample_fields=False)
        decode_error = _get_protobuf_decode_error()
        data = np.empty(self.shape, dtype=self.dtype)
        parsed: list[tuple[str, Any]] = []
        index = 0
        for record in _iter_envelope_records(resource, strict=True):
            if record.tag == META_TAG:
                # Every record is visited here, so META is picked up wherever
                # it sits, matching the read-everything path. The endpoint
                # scan only sees the ones before the first data packet.
                meta = _parse_meta(record.payload)
                continue
            if record.tag not in _TAG_TO_PACKET:
                continue
            if record.tag not in TS_TAGS:
                raise InvalidFiberFileError(
                    "Mixed Sintela protobuf packet families are unsupported."
                )
            msg = _parse_packet(record.tag, record.payload, messages, decode_error)
            if index + int(msg.header.num_samples) > self.shape[0]:
                # More samples than the endpoints implied: the packets in
                # between are not the contiguous run this path assumed.
                raise InvalidFiberFileError(
                    "Non-contiguous Sintela protobuf sample counts."
                )
            index += self._fill_packet(data, index, record.tag, msg)
            light = header_messages[_TAG_TO_PACKET[record.tag]]()
            # Round-tripped rather than copied: the sample-bearing and
            # header-only classes come from separate descriptor pools, so
            # CopyFrom rejects them as different types. A header is ~100 bytes.
            light.header.ParseFromString(msg.header.SerializeToString())
            parsed.append((record.tag, light))
            del msg
        metadata = type(self).from_parsed(parsed, meta)
        if index != self.shape[0]:
            # Contiguity, validated just above, should force the packet lengths
            # to sum to the endpoint-derived total. Checked unconditionally
            # anyway: the alternative to raising is handing back the
            # uninitialized tail of the output array.
            raise InvalidFiberFileError(
                "Sintela protobuf packets do not fill the expected sample count."
            )
        return data, metadata.coords, metadata.attrs


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
                **_get_fft_data_type(has_complex),
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


def _parse_packet_header(tag: str, payload_prefix: bytes):
    """Parse just the header submessage of a data packet from a payload prefix."""
    header_bytes = _leading_header_bytes(payload_prefix)
    if header_bytes is None:
        return None
    messages = _get_proto_messages(include_sample_fields=False)
    try:
        return _parse_packet(tag, header_bytes, messages, _get_protobuf_decode_error())
    except InvalidFiberFileError:
        # Leave the diagnostic to the read-everything path.
        return None


def _fits_in_file(total_samples: int, num_channels: int, file_size: int) -> bool:
    """
    Return whether a file this size could hold that many samples.

    Guards the *over*-estimate direction only: a float32 sample costs at least
    four bytes on the wire, so a length implying more bytes than exist cannot
    be real -- the shape of a counter that ran backwards, which the modular
    difference turns into billions of samples. Under-estimates pass trivially;
    ``_endpoint_time_agrees`` catches those.
    """
    if total_samples <= 0 or num_channels <= 0:
        return False
    return total_samples * num_channels * 4 <= file_size


def _endpoint_time_agrees(first_header, last_header, total_samples: int) -> bool:
    """
    Check the endpoint timestamps against the span their sample counts imply.

    The counters cannot tell one contiguous recording from two concatenated
    ones, or from a counter that reset partway; the timestamps are an
    independent witness, since for a contiguous run the elapsed time between
    the stamps should equal the samples between them over the sample rate.

    Coarse by design: recorder clocks resynchronize as a recording runs, so a
    valid file's stamps drift by milliseconds and the tolerance must absorb
    that, which leaves room for a small mid-file gap to hide. It catches the
    gross disagreement -- seconds to hours -- left by concatenation, a reset
    counter, a reconfiguration, or a misidentified final record.

    Returns False when either stamp is missing: an unverifiable shortcut is not
    worth taking when the full read is always available.
    """
    first_time = _common_header_time(first_header.common_header)
    last_time = _common_header_time(last_header.common_header)
    if first_time is None or last_time is None:
        return False
    rate = float(first_header.common_header.sample_rate)
    if not np.isfinite(rate) or rate <= 0:
        return False
    # Samples strictly between the two stamps, i.e. excluding the last packet.
    leading = total_samples - int(last_header.num_samples)
    expected_ns = round(leading / rate * 1e9)
    elapsed_ns = int((last_time - first_time).astype("timedelta64[ns]").astype(int))
    packet_ns = round(int(last_header.num_samples) / rate * 1e9)
    tolerance = max(
        _ENDPOINT_TIME_PACKETS * packet_ns,
        expected_ns * _ENDPOINT_TIME_TOLERANCE,
        _ENDPOINT_TIME_FLOOR_NS,
    )
    return abs(elapsed_ns - expected_ns) <= tolerance


def _get_endpoint_metadata(resource):
    """
    Summarize a timeseries recording from its first and last packets alone.

    A summary needs only header fields, and ``sample_count`` numbers the
    samples preceding each packet, so the last packet's count plus its length
    is the total. Scanning a half-gigabyte recording becomes two small reads.

    This assumes the packets in between are one contiguous, homogeneous run.
    Three checks test that assumption: each endpoint's declared length must fit
    its own record, the total must fit the file, and the endpoint timestamps
    must match the span those samples imply. A concatenated file, a reset
    counter, or a bogus tail match fails one of them.

    Returns ``(metadata, meta)``, or None when any check fails or the layout
    does not suit the shortcut, leaving the caller to read the whole file.
    Only META preceding the first data packet is seen here; ``read`` picks up
    any that appear later.
    """
    file_size = _stream_size(resource)
    head = _find_first_data_record(resource, file_size)
    if head is None:
        return None
    meta, (tag, size, prefix) = head
    # Only the timeseries family has an evenly sampled time coord derivable
    # from the endpoints; BAND and FFT time coords list every packet's stamp.
    if tag not in TS_TAGS:
        return None
    # Capped as well as floored: the first packet's declared size comes off
    # disk, and an implausible one would otherwise make the window swallow the
    # whole file in a single buffer.
    tail = _find_last_record(
        resource, file_size, min(size + _TAIL_SEARCH_SLACK, _MAX_TAIL_SEARCH)
    )
    if tail is None:
        # Sizing the window from the first packet assumes the last one is
        # about as big. A recording that opens with a short warm-up packet
        # breaks that, so widen once rather than give up and read everything
        # -- the retry costs one read, and only for files that need it.
        tail = _find_last_record(resource, file_size, min(file_size, _MAX_TAIL_SEARCH))
    if tail is None or tail[0] != tag:
        return None
    first = _parse_packet_header(tag, prefix)
    last = _parse_packet_header(tag, tail[2])
    if first is None or last is None:
        return None
    channels = int(first.header.common_header.num_channels)
    # Each endpoint's declared length has to fit the record carrying it. The
    # time check below cannot see the last packet's own length -- it cancels
    # out of the elapsed-time comparison -- so without this a packet could
    # claim millions of samples it has no room for and still be believed.
    if not (
        _fits_in_file(int(first.header.num_samples), channels, size)
        and _fits_in_file(int(last.header.num_samples), channels, tail[1])
    ):
        return None
    total_samples = (
        int(last.header.sample_count) - int(first.header.sample_count)
    ) % _SAMPLE_COUNT_MODULUS + int(last.header.num_samples)
    if not _fits_in_file(total_samples, channels, file_size):
        return None
    if not _endpoint_time_agrees(first.header, last.header, total_samples):
        return None
    try:
        metadata = TimeseriesMetadata.from_parsed(
            [(tag, first), (tag, last)], meta, total_samples=total_samples
        )
    except InvalidFiberFileError:
        # Endpoints that fail validation may be a genuinely bad file or a
        # false tail match; either way the full path decides, reporting the
        # same error for the former rather than trusting a doubtful header.
        return None
    return metadata, meta


def read_payload(resource):
    """Decode a Sintela protobuf file into data, coords, and attrs."""
    endpoints = _get_endpoint_metadata(resource)
    if endpoints is not None:
        metadata, meta = endpoints
        return metadata.decode_stream(resource, meta)
    records = _iter_envelope_records(resource, strict=True)
    parsed, meta = _parse_records(records, scan_mode=False)
    return _decode_family(parsed, meta)


def scan_payload(resource) -> list[ScanPayload]:
    """Decode a Sintela protobuf file and return FiberIO scan payloads."""
    endpoints = _get_endpoint_metadata(resource)
    if endpoints is not None:
        metadata = endpoints[0]
    else:
        records = _iter_envelope_records(resource, strict=True, headers_only=True)
        parsed, meta = _parse_records(records, scan_mode=True)
        family_cls = _FAMILY_CLASSES[_validate_single_family(parsed)]
        metadata = family_cls.from_parsed(parsed, meta)
    shape, coords, attrs, dtype = metadata.scan()
    return [make_scan_payload(attrs=attrs, coords=coords, shape=shape, dtype=dtype)]
