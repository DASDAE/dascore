"""Utilities for MiniSEED IO."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import groupby
from math import ceil, floor

import numpy as np

import dascore as dc
from dascore.constants import ONE_BILLION
from dascore.core import get_coord
from dascore.io import ScanPayload
from dascore.io.core import _patch_to_scan_payload
from dascore.utils.io import LocalPath, _normalize_source_patch_ids
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


def _get_group_key(segment: _TraceSegment):
    """Return the compatibility key used to merge traces into a patch."""
    return (
        segment.format_version,
        segment.network,
        segment.location,
        segment.seed_channel,
        segment.start_ns,
        segment.sample_rate,
    )


def _group_segments(segments: list[_TraceSegment]):
    """Yield compatible segment groups."""
    segments = sorted(segments, key=lambda x: (_get_group_key(x), x.source_id))
    for key, group in groupby(segments, key=_get_group_key):
        yield key, list(group)


def _source_patch_id(group_key) -> str:
    """Return a stable source patch ID for a MiniSEED group."""
    version, network, location, seed_channel, start_ns, sample_rate = group_key
    source_key = ".".join((network, location, seed_channel))
    return f"v{version}:{source_key}:{start_ns}:{sample_rate:g}"


def _patch_from_segments(group_key, segments: list[_TraceSegment]) -> dc.Patch:
    """Create a DASCore Patch from compatible MiniSEED trace segments."""
    segments = sorted(segments, key=lambda x: (x.station, x.source_id))
    first = segments[0]
    sample_count = min(x.sample_count for x in segments)
    source_ids = tuple(x.source_id for x in segments)
    data = np.stack([x.data[:sample_count] for x in segments])
    step = np.timedelta64(first.sample_step_ns, "ns")
    start = np.datetime64(first.start_ns, "ns")
    stop = start + step * sample_count
    coords = {
        "channel": np.arange(len(segments)),
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
    segment_groups = _group_segments(_read_segments(path, pymseed, read_time))
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
        patch = _patch_from_segments(group_key, segments)
        if channel is not None:
            patch = patch.select(channel=channel)
        if patch.size:
            patches.append(patch)
    return patches


def _scan_patches(path, pymseed) -> list[ScanPayload]:
    """Return scan payloads for MiniSEED patches."""
    out = []
    for group_key, segments in _group_segments(_read_segments(path, pymseed)):
        patch = _patch_from_segments(group_key, segments)
        out.append(_patch_to_scan_payload(patch))
    return out


def _detect_format(path: LocalPath, pymseed) -> tuple[str, str] | bool:
    """Return the MiniSEED version for a path or False."""
    versions = set()
    try:
        for record in _iter_records(path, pymseed, unpack_data=False):
            versions.add(str(record.formatversion))
            break
    except Exception:
        return False
    if not versions:
        return False
    version = next(iter(versions))
    if version in {"2", "3"}:
        return "MSEED", version
    return False
