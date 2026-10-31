"""Utilities for reading seismic ASDF files."""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary

from dascore.constants import snap_type
from dascore.exceptions import InvalidFiberFileError
from dascore.io.mseed.utils import (
    _get_group_key,
    _group_segments,
    _patch_meta,
    _source_patch_key,
    _SourceTrace,
)
from dascore.utils.misc import unbyte

VERSION = "1.0"

# Each open file's whole scan, shared by every patch read through the handle.
_SCANS = WeakKeyDictionary()


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


def _parse_name(name: str) -> tuple[str, list[str], str]:
    """Split a waveform name into its source ID, codes and tag."""
    # Tags may hold "__"; codes and times cannot.
    parts = name.split("__", 3)
    codes = parts[0].split(".")
    if len(parts) != 4 or len(codes) != 4:
        msg = f"ASDF waveform name {name!r} is not NET.STA.LOC.CHA__start__end__tag"
        raise InvalidFiberFileError(msg)
    return parts[0], codes, parts[3]


def _parse_trace(name: str, dataset) -> _AsdfTrace:
    """Read a waveform's identity from its name and timing from its attrs."""
    source_id, (network, station, location, seed_channel), tag = _parse_name(name)
    attrs = dataset.attrs
    if "starttime" not in attrs or "sampling_rate" not in attrs:
        msg = f"ASDF waveform {name!r} lacks starttime or sampling_rate"
        raise InvalidFiberFileError(msg)
    return _AsdfTrace(
        source_id=source_id,
        network=network,
        station=station,
        location=location,
        seed_channel=seed_channel,
        format_version=VERSION,
        start_ns=int(attrs["starttime"]),
        sample_rate=float(attrs["sampling_rate"]),
        sample_count=dataset.shape[0],
        dtype=dataset.dtype.name,
        tag=tag,
        path=dataset.name,
    )


def _group_key(trace: _AsdfTrace):
    """MiniSEED's compatibility key, split further by waveform tag."""
    return trace.tag, _get_group_key(trace)


def _scan(h5, keys=frozenset()):
    """
    Return each logical key's attrs and traces, and the file's channel map.

    Every name is read, so channel indices span the file, but only the
    waveforms `keys` could name are opened; their samples never are. Empty
    waveforms are skipped, as MiniSEED skips empty records. Parse errors
    name the file.
    """
    # A key is "<tag>:v<version>:<NET.LOC.CHA>:<start>:<rate>:<count>".
    wanted = {tuple(key.split(":")[0:3:2]) for key in keys}
    sources, traces, paths = set(), [], []
    waveforms = h5["Waveforms"]
    waveforms.visit(paths.append)  # names only, without opening groups
    try:
        for path in paths:
            _, inside, name = path.partition("/")
            if not inside or name == "StationXML":
                continue
            source_id, codes, tag = _parse_name(name)
            sources.add((codes[1], source_id))
            network, _, location, seed_channel = codes
            if not wanted or (tag, f"{network}.{location}.{seed_channel}") in wanted:
                traces.append(_parse_trace(name, waveforms[path]))
    except InvalidFiberFileError as exc:
        raise InvalidFiberFileError(f"{h5.filename}: {exc}") from exc
    groups = {}
    for (tag, key), group in _group_segments(
        [x for x in traces if x.sample_count], _group_key
    ):
        attrs = {
            "data_type": "",
            "tag": f"{key.network}.{key.location}.{key.seed_channel}.{tag}",
            "sample_rate": key.sample_rate,
        }
        groups[f"{tag}:{_source_patch_key(key)}"] = (attrs, group)
    channel_map = {x: i for i, (_, x) in enumerate(sorted(sources))}
    return groups, channel_map


def scan_groups(h5):
    """Return the whole file's scan, read once while the handle lives."""
    if (out := _SCANS.get(h5)) is None:
        out = _SCANS[h5] = _scan(h5)
    return out


def get_metadata(groups, channel_map, snap: snap_type = ()):
    """Return each scanned group's metadata, ordered by channel."""
    metas = [
        _patch_meta(group, attrs, key, channel_map, snap)
        for key, (attrs, group) in groups.items()
    ]
    return sorted(metas, key=lambda x: x.coords.get_coord("channel").min())
