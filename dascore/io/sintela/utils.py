"""
Utilities for reading binary sintela data.

Notes
-----
- Each file must hold one contiguous run of packets. Each packet header's
  sample count and start time are checked against the grid set by the
  first packet; a dropped, repeated, or concatenated packet raises.

- We only support v3 for now, though some commented out code here has the
  start for version 4 support.
"""

from __future__ import annotations

import numpy as np

from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import InvalidFiberFileError
from dascore.io.utils import step_from_rate
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
    samples = header["num_samples"]
    channels = header["num_channels"]
    packet_size = size + (channels * samples * 4)
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


def _check_packet_continuity(indices, stamps, header):
    """
    Raise unless the packets sit on the grid the first one starts.

    The time coordinate assumes packet ``i`` holds the samples
    ``i * num_samples`` onward from the first packet, so its header's
    sample count must match that exactly and its start time to within
    half a packet; the tolerance absorbs clock jitter while catching a
    dropped, repeated, or concatenated packet.
    """
    num_samples = int(header["num_samples"])
    packet_ns = round(num_samples / float(header["sample_rate"]) * 1e9)
    offsets = np.asarray(indices, dtype=np.int64) - indices[0]
    # uint64 differences wrap, so a backwards count reads as a huge jump;
    # the int64 view gives signed time offsets.
    count_off = (stamps[:, 0] - stamps[0, 0]).astype(np.int64)
    time_off = (stamps[:, 1] - stamps[0, 1]).view(np.int64)
    bad_count = count_off != offsets * num_samples
    bad_time = np.abs(time_off - offsets * packet_ns) > packet_ns / 2
    if np.any(bad := bad_count | bad_time):
        row = int(np.flatnonzero(bad)[0])
        field = "sample count" if bad_count[row] else "start time"
        msg = (
            f"Sintela binary packet {indices[row]} (0-based) has a {field} "
            "off the grid set by packet 0, so the file is not one "
            "contiguous recording."
        )
        raise InvalidFiberFileError(msg)


def _get_complete_header(rid):
    """Get the complete header information."""
    header = _read_base_header(rid)
    header.update(_read_remaining_header(rid, header))
    return header


def _get_time_coord(header):
    """Get the time coordinate."""
    starttime = np.asarray(header["start_time_ns"]).astype("datetime64[ns]")[()]
    timestep = step_from_rate(header["sample_rate"])
    total_len = header["num_packets"] * header["num_samples"]
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


def _get_data_shape(header) -> tuple[int, int]:
    """The (time, channel) shape the file's packets hold."""
    return header["num_packets"] * header["num_samples"], header["num_channels"]


def _read_sample_range(fid, header, start=0, stop=None):
    """
    Read time samples ``start`` to ``stop`` (half-open) as (samples, channels).

    Each packet carries a header the samples must skip, so stripping them
    leaves a view numpy cannot stride over and any read copies. A packet
    is therefore the smallest unit read: only those the range touches
    are.
    """
    header_size = header["header_size"]
    num_channels = header["num_channels"]
    num_samples = header["num_samples"]
    dtype = np.dtype(header["dtype"])
    packet_size = _packet_size(header)
    # Memory map entire file as raw bytes. We do this to later skip the headers
    fid.seek(0)
    raw = maybe_mem_map(fid)
    # Compute how many blocks
    block_count = raw.size // packet_size
    blocks = raw.reshape(block_count, packet_size)
    # The map makes every packet's stamps one strided gather, so check them
    # all here; the scan only afforded the endpoints.
    stamps = blocks[:, _STAMP_OFFSET : _STAMP_OFFSET + 16].copy().view("<u8")
    _check_packet_continuity(np.arange(block_count), stamps, header)
    total, _ = _get_data_shape(header)
    start = max(start, 0)
    stop = total if stop is None else min(stop, total)
    if stop <= start:
        return np.empty((0, num_channels), dtype=dtype)
    first, last = start // num_samples, (stop - 1) // num_samples
    # Create a view that skips headers
    packets = blocks[first : last + 1, header_size:]
    data = packets.view(dtype).reshape(-1, num_channels)
    offset = start - first * num_samples
    return data[offset : offset + (stop - start)]


def _get_attrs_coords_header(rid, attr_class=PatchAttrs):
    """Get Patch attributes and coordinates."""
    header = _get_complete_header(rid)
    # Two seeks catch any net dropped or added packet without walking the
    # file, which matters for remote scans; read_array checks every packet.
    ends = [0, int(header["num_packets"]) - 1]
    _check_packet_continuity(ends, _read_stamps(rid, header, ends), header)
    coords = {"time": _get_time_coord(header), "distance": _get_dist_coord(header)}
    cm = get_coord_manager(coords=coords, dims=DIMS)
    attrs = _get_attr_dict(header)
    return attr_class(**attrs), cm, header
