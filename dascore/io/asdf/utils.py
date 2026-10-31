"""Utilities for reading seismic ASDF files."""

from __future__ import annotations

from dataclasses import dataclass

from dascore.constants import snap_type
from dascore.exceptions import InvalidFiberFileError
from dascore.io.mseed.utils import (
    _get_group_key,
    _group_segments,
    _metadata_from_summaries,
    _patch_meta,
    _source_patch_key,
    _SourceTrace,
)
from dascore.utils.misc import unbyte

VERSION = "1.0"


@dataclass(frozen=True)
class _AsdfTrace(_SourceTrace):
    """One waveform dataset's identity, timing, tag and path."""

    tag: str
    path: str


def get_asdf_version(h5) -> str | None:
    """Return "1.0" for an ASDF 1.0.x file with waveforms, else None."""
    attrs = h5.attrs
    if unbyte(attrs.get("file_format")) != "ASDF" or "Waveforms" not in h5:
        return None
    version = str(unbyte(attrs.get("file_format_version", "")))
    return VERSION if version.split(".")[:2] == ["1", "0"] else None


def _parse_trace(name: str, dataset) -> _AsdfTrace:
    """Read a waveform's identity from its name and timing from its attrs."""
    parts = name.split("__")
    codes = parts[0].split(".")
    if len(parts) != 4 or len(codes) != 4:
        msg = f"ASDF waveform name {name!r} is not NET.STA.LOC.CHA__start__end__tag"
        raise InvalidFiberFileError(msg)
    network, station, location, seed_channel = codes
    attrs = dataset.attrs
    if "starttime" not in attrs or "sampling_rate" not in attrs:
        msg = f"ASDF waveform {name!r} lacks starttime or sampling_rate"
        raise InvalidFiberFileError(msg)
    return _AsdfTrace(
        source_id=parts[0],
        network=network,
        station=station,
        location=location,
        seed_channel=seed_channel,
        format_version=VERSION,
        start_ns=int(attrs["starttime"]),
        sample_rate=float(attrs["sampling_rate"]),
        sample_count=dataset.shape[0],
        dtype=dataset.dtype.name,
        tag=parts[-1],
        path=dataset.name,
    )


def scan_traces(h5) -> list[_AsdfTrace]:
    """Read every waveform's names and attrs, never its samples."""
    try:
        return [
            _parse_trace(name, dataset)
            for station in h5["Waveforms"].values()
            for name, dataset in station.items()
            if name != "StationXML"
        ]
    except InvalidFiberFileError as exc:
        raise InvalidFiberFileError(f"{h5.filename}: {exc}") from exc


def _group_key(trace: _AsdfTrace):
    """MiniSEED's compatibility key, split further by waveform tag."""
    return trace.tag, _get_group_key(trace)


def _patch_key(group_key) -> str:
    """Return the logical key of one tag's compatible group."""
    tag, key = group_key
    return f"{tag}:{_source_patch_key(key)}"


def _describe(group_key, traces, channel_map, snap):
    """Create one patch's metadata, tagged by source codes and waveform tag."""
    first = min(traces, key=lambda item: (item.station, item.source_id))
    attrs = {
        "data_type": "",
        "tag": ".".join((first.network, first.location, first.seed_channel, first.tag)),
        "sample_rate": first.sample_rate,
    }
    return _patch_meta(traces, attrs, _patch_key(group_key), channel_map, snap)


def get_metadata(traces: list[_AsdfTrace], snap: snap_type = ()):
    """Return each compatible group's metadata, ordered by channel."""
    return _metadata_from_summaries(traces, snap, _group_key, _describe)


def group_traces(traces: list[_AsdfTrace]) -> dict[str, list[_AsdfTrace]]:
    """Map each logical key to its traces."""
    return {
        _patch_key(key): group for key, group in _group_segments(traces, _group_key)
    }
