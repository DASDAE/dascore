"""
Utilities for reading binary sintela data.

Notes
-----
- A file holds one or more contiguous runs of packets; a dropped,
  repeated, or concatenated packet starts a new run, and each run is one
  patch. Within a run, each packet header's start time must stay on the
  grid the run's first packet sets.

- We only support v3 for now, though some commented out code here has the
  start for version 4 support.
"""

from __future__ import annotations

import re
from itertools import pairwise

import numpy as np

from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import InvalidFiberFileError, PatchAttributeError
from dascore.io.utils import step_from_rate
from dascore.utils.misc import get_buffer_size, maybe_mem_map

SYNC_WORD = 0x11223344
_RUN_KEY = re.compile(r"packets_(\d+)_(\d+)")
MAX_TRIGGERS = 8
DIMS = ("time", "distance")

# The minimum required info to determine if a file is a Sintela binary file.
base_header_dtypes = np.dtype(
    [
        ("sync_word", "<u4"),
        ("header_size", "<u4"),
        ("version", "<u4"),
    ]
)
# Every packet header holds sample_count then start_time_ns right after the base.
_STAMP_OFFSET = base_header_dtypes.itemsize

# numpy dtypes version 3 of header (excluding base)
_HEADER_DTYPES = {
    "3": np.dtype(
        [
            ("sample_count", "<u8"),
            ("start_time_ns", "<u8"),
            ("num_channels", "<u4"),
            ("num_samples", "<u4"),
            ("sample_rate", "<f4"),
            ("channel_spacing", "<f4"),
            ("gauge_length", "<f4"),
            ("trigger", "<u4"),
            ("start_channel", "<u4"),
            ("channel_step", "<u4"),
            ("gps_pps_time", "<u4"),
            ("gps_pps_sample_offset", "<i4"),
            ("gps_status", "<u4"),
            ("trigger_flags", "<u1"),
            ("status_flags", "<u1"),
            ("axial_counter_states", "<u1"),
            ("trigger_type", "<u1"),
            ("data_type", "<u1"),
            ("demod_data_type", "<u1"),
            ("unused", "<u2"),
            ("trigger_offsets", "<u2", (MAX_TRIGGERS,)),
        ]
    ),
    # "4": np.dtype([
    #     ("codec_type", "|S4"),
    #     ("sample_count", "<u8"),
    #     ("start_time_ns", "<u8"),
    #     ("num_channels", "<u4"),
    #     ("num_samples", "<u4"),
    #     ("sample_rate", "<f4"),
    #     ("channel_spacing", "<f4"),
    #     ("gauge_length", "<f4"),
    #     ("trigger", "<u4"),
    #     ("start_channel", "<u4"),
    #     ("channel_step", "<u4"),
    #     ("scale_factor", "<f4"),
    #     ("align", "<i4"),
    #     ("gps_pps_time", "<i4"),
    #     ("gps_pps_offset", "<i4"),
    #     ("gps_status", "<i4"),
    #     ("flags", "u1", (8,)),
    #     ("trigger_offsets", "<u2", (MAX_TRIGGERS,)),
    #     ("spare", "<i4"),
    # ]),
}

# The size of the headers
_HEADER_SIZES = {
    "3": base_header_dtypes.itemsize + _HEADER_DTYPES["3"].itemsize,
    # '4': base_header_dtypes.itemsize + _HEADER_DTYPES['4'].itemsize,
}

# The types of data that can be contained in file and their units.
_DATA_TYPE_MAP = {
    0: ("phase", "radians"),
    1: ("phase_difference", "radians"),
    2: ("phase_rate", "radians/s"),
    3: ("strain", "microstrain"),
    4: ("strain_rate", "microstrain/s"),
}


def _read_base_header(fid):
    """Return the first 3 elements of the sintela header."""
    data = fid.read(base_header_dtypes.itemsize)
    array = np.frombuffer(data, dtype=base_header_dtypes, count=1)
    names = array.dtype.names
    assert names is not None  # structured dtypes always have field names
    out = {x: y for x, y in zip(names, array[0])}
    return out


def _get_number_of_packets(fid, header, size):
    """Get the number of packets in the file."""
    # get filesize without depending on input being a file.
    file_size = get_buffer_size(fid)
    # Python ints: uint32 arithmetic overflows past 4 GiB under NEP 50.
    samples = int(header["num_samples"])
    channels = int(header["num_channels"])
    packet_size = int(size) + (channels * samples * 4)
    num_packets, remainder = divmod(file_size, packet_size)
    if remainder:
        msg = "Sintela binary file size not divisible by packet size."
        raise InvalidFiberFileError(msg)
    return num_packets


def _read_remaining_header(fid, base):
    """Read file header using numpy structured dtype."""
    version = str(base["version"])
    header_size = base["header_size"]
    expected_size = _HEADER_SIZES.get(version, -1)
    assert header_size == expected_size
    dtype = _HEADER_DTYPES[version]
    data = fid.read(dtype.itemsize)
    buf = np.frombuffer(data, dtype=dtype, count=1)
    names = buf.dtype.names
    assert names is not None  # structured dtypes always have field names
    header = {x: y for x, y in zip(names, buf[0])}
    assert version == "3", "only 3 support for now,"
    header["num_packets"] = _get_number_of_packets(fid, header, header_size)
    header["dtype"] = "<f4"
    return header


def _packet_size(header) -> int:
    """The bytes in one packet, header included."""
    samples = int(header["num_channels"]) * int(header["num_samples"])
    return int(header["header_size"]) + samples * np.dtype(header["dtype"]).itemsize


def _read_stamps(fid, header, indices):
    """Read (sample_count, start_time_ns) from the headers of some packets."""
    stamps = np.empty((len(indices), 2), dtype="<u8")
    for row, index in enumerate(indices):
        fid.seek(index * _packet_size(header) + _STAMP_OFFSET)
        stamps[row] = np.frombuffer(fid.read(16), dtype="<u8", count=2)
    return stamps


def _off_grid(stamps, base, offsets, header):
    """
    Return whether each packet's sample count and start time are off grid.

    A packet ``offsets`` packets after ``base`` must hold the samples
    ``offsets * num_samples`` onward from it, so its header's sample count
    must match that exactly and its start time to within half a packet;
    the tolerance absorbs clock jitter while catching a dropped, repeated,
    or concatenated packet.
    """
    num_samples = int(header["num_samples"])
    # Unrounded, so the expected offsets do not drift over a long file.
    packet_ns = num_samples / float(header["sample_rate"]) * 1e9
    # uint64 differences wrap, so a backwards count reads as a huge jump;
    # the int64 view gives signed time offsets.
    count_off = (stamps[:, 0] - base[0]).astype(np.int64)
    time_off = (stamps[:, 1] - base[1]).view(np.int64)
    bad_time = np.abs(time_off - offsets * packet_ns) > packet_ns / 2
    return count_off != offsets * num_samples, bad_time


def _check_run(stamps, header, first=0, offsets=None):
    """
    Raise unless packets sit on the grid packet ``first`` starts.

    ``stamps`` belong to the packets ``offsets`` after ``first``, by
    default every packet of the run in order.
    """
    offsets = np.arange(len(stamps)) if offsets is None else np.asarray(offsets)
    bad_count, bad_time = _off_grid(stamps, stamps[0], offsets, header)
    if np.any(bad := bad_count | bad_time):
        row = int(np.flatnonzero(bad)[0])
        field = "sample count" if bad_count[row] else "start time"
        msg = (
            f"Sintela binary packet {first + int(offsets[row])} (0-based) has "
            f"a {field} off the grid set by packet {first}, so the packets "
            "between are not one contiguous recording."
        )
        raise InvalidFiberFileError(msg)


def _find_run_starts(rid, header, count) -> dict[int, np.ndarray]:
    """
    Return the stamps of each packet that starts a run, keyed by packet.

    A dropped or repeated packet breaks the sample count. Packets whose
    counts are ``num_samples`` per packet apart are taken as one run
    without reading those between, so a span is bisected only while its
    ends disagree: a few reads per break rather than one per packet,
    which matters for large files on network shares. A drop and a repeat
    that cancel within a span go unseen until read_array checks it.
    """
    stamps = dict(zip([0, count - 1], _read_stamps(rid, header, [0, count - 1])))
    starts, spans = [0], [(0, count - 1)]
    while spans:
        lo, hi = spans.pop()
        for index in {lo, hi} - stamps.keys():
            stamps[index] = _read_stamps(rid, header, [index])[0]
        bad_count, _ = _off_grid(stamps[hi][None], stamps[lo], hi - lo, header)
        if not bad_count.any():
            continue
        if hi - lo == 1:
            starts.append(hi)
            continue
        mid = (lo + hi) // 2
        spans += [(lo, mid), (mid, hi)]
    return {x: stamps[x] for x in sorted(starts)}


def _get_complete_header(rid):
    """Get the complete header information."""
    header = _read_base_header(rid)
    header.update(_read_remaining_header(rid, header))
    return header


def _get_time_coord(header, start_ns, packets):
    """Get the time coordinate of a run of packets starting at start_ns."""
    starttime = np.asarray(start_ns).astype("datetime64[ns]")[()]
    timestep = step_from_rate(header["sample_rate"])
    total_len = packets * int(header["num_samples"])
    return get_coord(start=starttime, step=timestep, shape=(total_len,))


def _get_dist_coord(header):
    """Get the distance coordinate."""
    channel_step = header["channel_step"]
    channel_count = header["num_channels"]
    start_channel = header["start_channel"]
    dx = header["channel_spacing"]
    dist_start = start_channel * dx
    coord = get_coord(
        start=dist_start,
        step=channel_step * dx,
        stop=dist_start + (channel_count * channel_step) * dx,
        units="m",
    )
    assert len(coord) == channel_count
    return coord


def _get_attr_dict(header):
    """
    Extract info from sintela headers which will be used to create patch Attrs.
    """
    data_type, units = _DATA_TYPE_MAP.get(header["data_type"], ("phase", "radians"))
    out = dict(
        data_type=data_type,
        data_units=units,
        gauge_length=header["gauge_length"],
        gps_status=header["gps_status"],
    )
    return out


def _run_key(start, stop) -> str:
    """The key naming a run of packets; a digit-only key reads as positional."""
    return f"packets_{start}_{stop}"


def _run_blocks(fid, header, key=""):
    """
    Return the packets of the run ``key`` names as (packets, bytes) rows.

    No key names the whole file, which must then be one run. The memory
    map makes the run's stamps one strided gather, so every packet in it
    is checked here; a scan whose end packets agreed looked at no others.
    """
    fid.seek(0)
    raw = maybe_mem_map(fid)
    blocks = raw.reshape(-1, _packet_size(header))
    start, stop = 0, len(blocks)
    if key:
        match = _RUN_KEY.fullmatch(key)
        start, stop = map(int, match.groups()) if match else (0, 0)
        if not start < stop <= len(blocks):
            msg = f"No patch named '{key}' in the Sintela binary file."
            raise PatchAttributeError(msg)
    run = blocks[start:stop]
    _check_run(
        run[:, _STAMP_OFFSET : _STAMP_OFFSET + 16].copy().view("<u8"), header, start
    )
    return run


def _read_sample_range(blocks, header, start=0, stop=None):
    """
    Read time samples ``start`` to ``stop`` (half-open) of a run's packets.

    Each packet carries a header the samples must skip, so stripping them
    leaves a view numpy cannot stride over and any read copies. A packet
    is therefore the smallest unit read: only those the range touches
    are.
    """
    num_channels = header["num_channels"]
    num_samples = int(header["num_samples"])
    dtype = np.dtype(header["dtype"])
    total = len(blocks) * num_samples
    start = max(start, 0)
    stop = total if stop is None else min(stop, total)
    if stop <= start:
        return np.empty((0, num_channels), dtype=dtype)
    first, last = start // num_samples, (stop - 1) // num_samples
    # Create a view that skips headers
    packets = blocks[first : last + 1, header["header_size"] :]
    data = packets.view(dtype).reshape(-1, num_channels)
    offset = start - first * num_samples
    return data[offset : offset + (stop - start)]


def _scan_runs(rid, attr_class=PatchAttrs):
    """Return the header and (key, attrs, coords) of each run of packets."""
    header = _get_complete_header(rid)
    count = int(header["num_packets"])
    starts = _find_run_starts(rid, header, count)
    runs = list(pairwise([*starts, count]))
    attrs = attr_class(**_get_attr_dict(header))
    out = []
    for start, stop in runs:
        # Only the run's ends are checked here; read_array checks every packet.
        last = _read_stamps(rid, header, [stop - 1])
        ends = np.stack([starts[start], last[0]])
        _check_run(ends, header, start, [0, stop - 1 - start])
        key = _run_key(start, stop) if len(runs) > 1 else ""
        time = _get_time_coord(header, starts[start][1], stop - start)
        coords = {"time": time, "distance": _get_dist_coord(header)}
        out.append((key, attrs, get_coord_manager(coords=coords, dims=DIMS)))
    return header, out
