"""
Utilities for reading binary sintela data.

Notes
-----
- We make the strong assumption that each file contains contiguous
  blocks of DAS data.

- We only support v3 for now, though some commented out code here has the
  start for version 4 support.
"""

import numpy as np

import dascore as dc
from dascore.compat import array
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import InvalidFiberFileError
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
    array = np.fromfile(fid, dtype=base_header_dtypes, count=1)
    out = {x: y for x, y in zip(array.dtype.names, array[0])}
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
    buf = np.fromfile(fid, dtype=dtype, count=1)
    header = {x: y for x, y in zip(buf.dtype.names, buf[0])}
    assert version == "3", "only 3 support for now,"
    header["num_packets"] = _get_number_of_packets(fid, header, header_size)
    header["dtype"] = "<f4"
    return header


def _get_complete_header(rid):
    """Get the complete header information."""
    header = _read_base_header(rid)
    header.update(_read_remaining_header(rid, header))
    return header


def _get_time_coord(header):
    """Get the time coordinate."""
    starttime = np.asarray(header["start_time_ns"]).astype("datetime64[ns]")[()]
    timestep = dc.to_timedelta64(1 / header["sample_rate"])
    total_len = header["num_packets"] * header["num_samples"]
    end_time = starttime + timestep * total_len
    time = get_coord(start=starttime, step=timestep, stop=end_time)
    assert len(time) == total_len
    return time


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


def _get_attr_dict(header, extras=None):
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
    out.update(extras if extras is not None else {})
    return out


def _load_data(fid, header):
    """Use numpy's memmap to get array information."""
    header_size = header["header_size"]
    num_channels = header["num_channels"]
    num_samples = header["num_samples"]
    dtype = np.dtype(header["dtype"])
    packet_size = header_size + num_channels * num_samples * dtype.itemsize
    # Memory map entire file as raw bytes. We do this to later skip the headers
    fid.seek(0)
    raw = maybe_mem_map(fid)
    # Compute how many blocks
    block_count = raw.size // packet_size
    # Create a view that skips headers
    data = raw.reshape(block_count, packet_size)[:, header_size:]
    data = data.view(dtype).reshape(-1, num_channels)
    return data


def _get_attrs_coords_header(rid, attr_class=PatchAttrs, extras=None):
    """Get Patch attributes and coordinates."""
    header = _get_complete_header(rid)
    coords = {"time": _get_time_coord(header), "distance": _get_dist_coord(header)}
    cm = get_coord_manager(coords=coords, dims=DIMS)
    attrs = _get_attr_dict(header, extras)
    attrs["coords"] = cm
    return attr_class(**attrs), cm, header


def _get_patch(
    resource, attr_class=PatchAttrs, extras=None, time=None, distance=None, **kwargs
):
    """Get the patch from the sintela file."""
    patch_attrs, cm, header = _get_attrs_coords_header(resource, attr_class, extras)
    data = _load_data(resource, header)
    # Apply slicing if needed. This is done before patch creation so a memmap
    # can be sliced before loading all data into memory.
    if time is not None or distance is not None:
        cm, data = cm.select(data, time=time, distance=distance)
    patch = dc.Patch(data=array(data), coords=cm, attrs=patch_attrs)
    return patch
