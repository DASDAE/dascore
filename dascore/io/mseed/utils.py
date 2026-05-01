"""Utilities for MiniSEED IO."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import groupby
from math import ceil, floor
from struct import unpack

import numpy as np

import dascore as dc
from dascore.constants import ONE_BILLION
from dascore.core import get_coord
from dascore.io import ScanPayload
from dascore.io.core import _patch_to_scan_payload
from dascore.utils.io import LocalPath, _normalize_source_patch_ids, _read_file_header
from dascore.utils.time import to_datetime64, to_int

_TimeLimits = tuple[int | None, int | None]


@dataclass(frozen=True)
class _TraceSegment:
    """A decoded contiguous MiniSEED segment from one source ID."""

    source_id: str
    network: str
    station: str
    location: str
    seed_channel: str
    format_version: str
    start_ns: int
    sample_rate: float
    data: np.ndarray
    sample_type: str
    encoding: str
    publication_version: int
    record_length: int

    @property
    def sample_count(self) -> int:
        """Return the number of samples in the segment."""
        return len(self.data)

    @property
    def sample_step_ns(self) -> int:
        """Return the sample spacing in nanoseconds."""
        return _sample_step_ns(self.sample_rate)

    @property
    def next_start_ns(self) -> int:
        """Return the expected start time of the next contiguous segment."""
        return self.start_ns + self.sample_count * self.sample_step_ns


@dataclass(frozen=True, order=True)
class _TraceGroupKey:
    """Compatibility fields that define one MiniSEED output patch."""

    format_version: str
    network: str
    location: str
    seed_channel: str
    start_ns: int
    sample_rate: float
    sample_count: int


def _sample_step_ns(sample_rate: float) -> int:
    """Convert a MiniSEED sample rate to nanosecond spacing."""
    if sample_rate == 0:
        msg = "MiniSEED sample rate cannot be zero."
        raise ValueError(msg)
    seconds = 1 / sample_rate if sample_rate > 0 else abs(sample_rate)
    return round(seconds * ONE_BILLION)


def _time_to_ns(value) -> int | None:
    """Convert a time-like value to epoch nanoseconds."""
    if value is None or value is ...:
        return None
    return int(to_int(to_datetime64(value)))


def _get_time_limits(time=None) -> _TimeLimits:
    """Return optional time limits as epoch nanoseconds."""
    if time is None or time is ...:
        return None, None
    return _time_to_ns(time[0]), _time_to_ns(time[1])


def _source_id_to_nslc(pymseed, source_id: str) -> tuple[str, str, str, str]:
    """Return NSLC codes from a MiniSEED source ID."""
    try:
        nslc = pymseed.sourceid2nslc(source_id)
    except Exception:
        return "", source_id, "", ""
    return tuple("" if x is None else str(x) for x in nslc)


def _iter_records(path: LocalPath, pymseed, unpack_data: bool = True):
    """Yield records from a MiniSEED file."""
    yield from pymseed.MS3Record.from_file(str(path), unpack_data=unpack_data)


def _record_overlaps_time(record, time_limits: _TimeLimits) -> bool:
    """Return True if a record overlaps a requested time range."""
    start, stop = time_limits
    record_start = int(record.starttime)
    record_stop = int(record.endtime)
    after_start = stop is None or record_start <= stop
    before_stop = start is None or record_stop >= start
    return after_start and before_stop


def _record_to_segment(
    record, pymseed, time_limits: _TimeLimits
) -> _TraceSegment | None:
    """Convert a PyMseed record to a trace segment."""
    if not _record_overlaps_time(record, time_limits):
        return None
    record.unpack_data()
    data = np.asarray(record.np_datasamples)
    network, station, location, seed_channel = _source_id_to_nslc(
        pymseed, str(record.sourceid)
    )
    segment = _TraceSegment(
        source_id=str(record.sourceid),
        network=network,
        station=station,
        location=location,
        seed_channel=seed_channel,
        format_version=str(record.formatversion),
        start_ns=int(record.starttime),
        sample_rate=float(record.samprate),
        data=data.copy(),
        sample_type=str(record.sampletype or ""),
        encoding=str(record.encoding),
        publication_version=int(getattr(record, "pubversion", 0) or 0),
        record_length=int(getattr(record, "reclen", 0) or 0),
    )
    return _trim_segment_time(segment, time_limits)


def _trim_segment_time(
    segment: _TraceSegment, time_limits: _TimeLimits
) -> _TraceSegment | None:
    """Trim a segment to a requested time range."""
    start, stop = time_limits
    if start is None and stop is None:
        return segment
    step = segment.sample_step_ns
    start_index = (
        0 if start is None else max(0, ceil((start - segment.start_ns) / step))
    )
    stop_index = (
        segment.sample_count
        if stop is None
        else min(segment.sample_count, floor((stop - segment.start_ns) / step) + 1)
    )
    if stop_index <= start_index:
        return None
    return replace(
        segment,
        start_ns=segment.start_ns + start_index * step,
        data=segment.data[start_index:stop_index].copy(),
    )


def _coalesce_source_segments(segments: list[_TraceSegment]) -> list[_TraceSegment]:
    """Merge contiguous records for the same source ID."""
    segments = sorted(segments, key=lambda x: (x.source_id, x.start_ns))
    out = []
    for source_id, source_group in groupby(segments, key=lambda x: x.source_id):
        pending = None
        for seg in source_group:
            if pending is None:
                pending = seg
                continue
            can_merge = (
                pending.source_id == source_id
                and pending.sample_rate == seg.sample_rate
                and pending.format_version == seg.format_version
                and pending.next_start_ns == seg.start_ns
                and pending.data.dtype == seg.data.dtype
            )
            if can_merge:
                pending = replace(
                    pending,
                    data=np.concatenate([pending.data, seg.data]),
                    record_length=max(pending.record_length, seg.record_length),
                )
            else:
                out.append(pending)
                pending = seg
        if pending is not None:
            out.append(pending)
    return out


def _read_segments(path: LocalPath, pymseed, time=None) -> list[_TraceSegment]:
    """Read and coalesce decoded MiniSEED records."""
    segments = []
    time_limits = _get_time_limits(time)
    for record in _iter_records(path, pymseed, unpack_data=False):
        segment = _record_to_segment(record, pymseed, time_limits)
        if segment is not None and segment.sample_count:
            segments.append(segment)
    return _coalesce_source_segments(segments)


def _get_group_key(segment: _TraceSegment) -> _TraceGroupKey:
    """Return the compatibility key used to merge traces into a patch."""
    return _TraceGroupKey(
        format_version=segment.format_version,
        network=segment.network,
        location=segment.location,
        seed_channel=segment.seed_channel,
        start_ns=segment.start_ns,
        sample_rate=segment.sample_rate,
        sample_count=segment.sample_count,
    )


def _group_segments(segments: list[_TraceSegment]):
    """Yield compatible segment groups."""
    segments = sorted(segments, key=lambda x: (_get_group_key(x), x.source_id))
    for key, group in groupby(segments, key=_get_group_key):
        yield key, list(group)


def _get_channel_map(segments: list[_TraceSegment]) -> dict[str, int]:
    """Return stable channel indices for MiniSEED sources."""
    source_ids = {
        x.source_id: (x.station, x.source_id)
        for x in sorted(segments, key=lambda y: (y.station, y.source_id))
    }
    return {source_id: ind for ind, source_id in enumerate(source_ids)}


def _source_patch_id(group_key: _TraceGroupKey) -> str:
    """Return a stable source patch ID for a MiniSEED group."""
    source_key = ".".join(
        (group_key.network, group_key.location, group_key.seed_channel)
    )
    rate_key = format(group_key.sample_rate, ".17g")
    return (
        f"v{group_key.format_version}:{source_key}:"
        f"{group_key.start_ns}:{rate_key}:{group_key.sample_count}"
    )


def _patch_from_segments(
    group_key, segments: list[_TraceSegment], channel_map: dict[str, int] | None = None
) -> dc.Patch:
    """Create a DASCore Patch from compatible MiniSEED trace segments."""
    segments = sorted(segments, key=lambda x: (x.station, x.source_id))
    first = segments[0]
    sample_count = first.sample_count
    source_ids = tuple(x.source_id for x in segments)
    channel_values = (
        tuple(channel_map[x.source_id] for x in segments)
        if channel_map is not None
        else tuple(range(len(segments)))
    )
    data = np.stack([x.data for x in segments])
    step = np.timedelta64(first.sample_step_ns, "ns")
    start = np.datetime64(first.start_ns, "ns")
    stop = start + step * sample_count
    coords = {
        "channel": get_coord(data=np.asarray(channel_values)),
        "time": get_coord(start=start, stop=stop, step=step),
        "source_id": (("channel",), source_ids),
        "network": (("channel",), tuple(x.network for x in segments)),
        "station": (("channel",), tuple(x.station for x in segments)),
        "location": (("channel",), tuple(x.location for x in segments)),
        "seed_channel": (("channel",), tuple(x.seed_channel for x in segments)),
    }
    attrs = {
        "data_type": "",
        "tag": ".".join((first.network, first.location, first.seed_channel)),
        "sample_rate": first.sample_rate,
        "sample_type": first.sample_type,
        "mseed_encoding": first.encoding,
        "mseed_publication_version": first.publication_version,
        "mseed_record_length": first.record_length,
        "_source_patch_id": _source_patch_id(group_key),
    }
    return dc.Patch(data=data, dims=("channel", "time"), coords=coords, attrs=attrs)


def _get_patches(
    path, pymseed, time=None, channel=None, source_patch_id=()
) -> list[dc.Patch]:
    """Read MiniSEED patches from a path."""
    wanted_ids = _normalize_source_patch_ids(source_patch_id)
    read_time = None if wanted_ids else time
    trim_limits = _get_time_limits(time) if wanted_ids else (None, None)
    patches = []
    all_segments = _read_segments(path, pymseed, read_time)
    channel_map = _get_channel_map(all_segments)
    segment_groups = _group_segments(all_segments)
    for group_key, segments in segment_groups:
        patch_id = _source_patch_id(group_key)
        if wanted_ids and patch_id not in wanted_ids:
            continue
        if wanted_ids:
            segments = [
                trimmed
                for segment in segments
                if (trimmed := _trim_segment_time(segment, trim_limits)) is not None
            ]
        if not segments:
            continue
        patch = _patch_from_segments(group_key, segments, channel_map=channel_map)
        if channel is not None:
            patch = patch.select(channel=channel)
        if patch.size:
            patches.append(patch)
    return sorted(patches, key=lambda x: x.get_coord("channel").min())


def _scan_patches(path, pymseed) -> list[ScanPayload]:
    """Return scan payloads for MiniSEED patches."""
    patches = []
    all_segments = _read_segments(path, pymseed)
    channel_map = _get_channel_map(all_segments)
    for group_key, segments in _group_segments(all_segments):
        patch = _patch_from_segments(group_key, segments, channel_map=channel_map)
        patches.append(patch)
    patches = sorted(patches, key=lambda x: x.get_coord("channel").min())
    return [_patch_to_scan_payload(x) for x in patches]


def _is_ascii_digit(value: int) -> bool:
    """Return True if an integer byte is an ASCII digit."""
    return 48 <= value <= 57


def _is_seed_code(value: bytes, *, allow_space: bool = True) -> bool:
    """Return True if bytes are plausible SEED code bytes."""
    allowed = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if allow_space:
        allowed += b" "
    return bool(value.strip()) and all(char in allowed for char in value)


def _detect_mseed_v2_header(header: bytes) -> bool:
    """Return True if a fixed MiniSEED 2 header looks valid."""
    if len(header) < 48:
        return False
    if not all(_is_ascii_digit(char) for char in header[:6]):
        return False
    if header[6:7] not in b"DRQM":
        return False
    if header[7] not in (0, 32):
        return False
    if not _is_seed_code(header[8:13]):
        return False
    if not _is_seed_code(header[15:18], allow_space=False):
        return False
    if not _is_seed_code(header[18:20]):
        return False
    try:
        year, day = unpack(">HH", header[20:24])
        hour, minute, second = header[24:27]
        _unused, frac_seconds, sample_count = unpack(">BHH", header[27:32])
        data_offset, blockette_offset = unpack(">HH", header[44:48])
    except Exception:
        return False
    valid_time = (
        1900 <= year <= 2600
        and 1 <= day <= 366
        and hour <= 23
        and minute <= 59
        and second <= 60
        and frac_seconds <= 9_999
    )
    valid_offsets = data_offset >= 48 and (
        blockette_offset == 0 or blockette_offset >= 48
    )
    return valid_time and valid_offsets and sample_count > 0


def _detect_format(path: LocalPath) -> tuple[str, str] | bool:
    """Return the MiniSEED version for a path or False."""
    header = _read_file_header(path, 48)
    if header.startswith(b"MS\x03"):
        return "MSEED", "3"
    if _detect_mseed_v2_header(header):
        return "MSEED", "2"
    return False
