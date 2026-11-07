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

from itertools import pairwise

import numpy as np

from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import InvalidFiberFileError
from dascore.io.utils import resolve_keyed_source, step_from_rate
from dascore.utils.misc import get_buffer_size, maybe_mem_map

SYNC_WORD = 0x11223344
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
    Return whether each packet is off the grid its base packet starts.

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
    count_off = (stamps[:, 0] - base[..., 0]).astype(np.int64)
    time_off = (stamps[:, 1] - base[..., 1]).view(np.int64)
    bad_time = np.abs(time_off - offsets * packet_ns) > packet_ns / 2
    return (count_off != offsets * num_samples) | bad_time


def _packet_runs(stamps, header) -> list[tuple[int, int]]:
    """
    Return the (start, stop) packets of each contiguous run.

    A packet off the grid of the one before it starts a new run. A run
    whose clock then drifts more than half a packet off its own grid
    raises, since its time coordinate would be wrong.
    """
    breaks = _off_grid(stamps[1:], stamps[:-1], 1, header)
    bounds = [0, *(np.flatnonzero(breaks) + 1).tolist(), len(stamps)]
    runs = list(pairwise(bounds))
    for start, stop in runs:
        offsets = np.arange(stop - start)
        bad = _off_grid(stamps[start:stop], stamps[start], offsets, header)
        if np.any(bad):
            msg = (
                f"Sintela binary packet {start + int(np.argmax(bad))} (0-based) "
                f"has a start time off the grid set by packet {start}."
            )
            raise InvalidFiberFileError(msg)
    return runs


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


def _run_keys(runs) -> dict[str, tuple[int, int]]:
    """A lone run needs no key; several are keyed by their first packet."""
    if len(runs) == 1:
        return {"": runs[0]}
    return {str(start): (start, stop) for start, stop in runs}


def _run_blocks(fid, header, key=""):
    """
    Return the packets of the run ``key`` names as (packets, bytes) rows.

    The memory map makes every packet's stamps one strided gather, so the
    runs are found from all of them here; a scan that found its ends on
    one grid did not look between them.
    """
    fid.seek(0)
    raw = maybe_mem_map(fid)
    blocks = raw.reshape(-1, _packet_size(header))
    stamps = blocks[:, _STAMP_OFFSET : _STAMP_OFFSET + 16].copy().view("<u8")
    runs = _packet_runs(stamps, header)
    if not key and len(runs) > 1:
        # A scan whose two end packets agreed reported one patch.
        msg = (
            f"Sintela binary packet {runs[1][0]} (0-based) breaks the run of "
            "packets its scan assumed from the file's ends."
        )
        raise InvalidFiberFileError(msg)
    keyed = _run_keys(runs)
    start, stop = resolve_keyed_source(keyed, key, where="the Sintela binary file")
    return blocks[start:stop]


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
    # Two seeks show a file with no net dropped or added packet without
    # walking it, which matters for remote scans; read_array checks every
    # packet. Otherwise every header is read to find where the runs break.
    ends = _read_stamps(rid, header, [0, count - 1])
    if not _off_grid(ends, ends[0], np.array([0, count - 1]), header).any():
        runs = {"": (0, count)}
    else:
        runs = _run_keys(_packet_runs(_read_stamps(rid, header, range(count)), header))
    stamps = _read_stamps(rid, header, [x for x, _ in runs.values()])
    attrs = attr_class(**_get_attr_dict(header))
    out = []
    for (key, (start, stop)), (_, start_ns) in zip(runs.items(), stamps, strict=True):
        time = _get_time_coord(header, start_ns, stop - start)
        coords = {"time": time, "distance": _get_dist_coord(header)}
        out.append((key, attrs, get_coord_manager(coords=coords, dims=DIMS)))
    return header, out
