"""Utilities for Sentek data format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dascore as dc
from dascore.core import get_coord_manager
from dascore.io.utils import stored_coord


def _read_float32(fid, count=1):
    """Read one or more float32 values from a binary stream."""
    size = np.dtype(np.float32).itemsize * count
    data = fid.read(size)
    return np.frombuffer(data, dtype=np.float32, count=count)


def _get_version(fid):
    """Determine if Sentek file."""
    name = getattr(fid, "name", None) or getattr(fid, "path", None)
    name = name or getattr(fid, "_dascore_source_path", "")
    # Sentek files cannot change the extension, or file name.
    sw_data = str(name).endswith(".das")
    fid.seek(0)
    # There isn't anything in the header particularly useful for determining
    # if it is a Sentek file, so we do what we can here.
    # First check if sensor_num and measurement_count are positive and nearly
    # ints.
    sensor_num = _read_float32(fid)[0]
    measurement_count = _read_float32(fid)[0]
    _ = _read_float32(fid)[0]  # sampling_interval
    is_positive = (sensor_num > 1) and (measurement_count > 1)
    sens_nearly_int = np.round(sensor_num, 5) == np.round(sensor_num)
    meas_nearly_int = np.round(measurement_count, 5) == np.round(measurement_count)
    nearly_ints = sens_nearly_int and meas_nearly_int
    # Then check if strain_rate value is valid.
    strain_rate = int(_read_float32(fid)[0])
    proper_strain_rate = strain_rate in {0, 1}
    # Note: We will need to modify this later for different versions of the
    # sentek data, but for now we only support 5.
    if sw_data and is_positive and proper_strain_rate and nearly_ints:
        return ("sentek", "5")
    return False


def _get_time_from_file_name(name) -> np.datetime64:
    """Extract time contained in the file name.

    example file name: DASDMSShot00_20230328155652124.das
    """
    time_str = name.split("_")[1].split(".")[0]
    year = time_str[:4]
    month = time_str[4:6]
    day = time_str[6:8]
    hour = time_str[8:10]
    minute = time_str[10:12]
    second = float(time_str[12:]) / 1_000
    iso = f"{year}-{month}-{day}T{hour}:{minute}:{second:02f}"
    return np.datetime64(iso, "ns")


def _get_patch_attrs(fid, *, snap=True):
    """Extract metadata from the header followed by distance and time arrays.

    The six float32 header fields are the channel count, sample count,
    sampling interval, strain-rate flag, trigger position, and decimation
    factor. Signal samples follow both coordinate arrays in time-major order.
    Stored time offsets retain the reader's existing seconds-from-filename
    interpretation.
    """
    header, layout = _get_layout(fid)
    _, measurement_count, sensor_num = layout
    name = getattr(fid, "name", None) or getattr(fid, "path", None)
    name = name or getattr(fid, "_dascore_source_path", "")
    distances = _read_float32(fid, sensor_num)
    dist = stored_coord(distances, snap, name="distance", where=name)
    file_time = _get_time_from_file_name(Path(str(name)).name)
    # Offsets are float32 seconds, which near 1e5 s resolve only 1/64 s, so
    # some repeat; the grid they restate steps at the header's interval.
    offsets = _read_float32(fid, measurement_count)
    time = stored_coord(
        offsets,
        snap,
        name="time",
        where=name,
        decode=lambda x: file_time + dc.to_timedelta64(x.astype(np.float64)),
    )
    data_type = "strain_rate" if header[3] else "strain"
    coord_manager = get_coord_manager(
        {"time": time, "distance": dist}, dims=("distance", "time")
    )
    attrs = dc.PatchAttrs(data_type=data_type)
    return attrs, coord_manager, layout


def _get_layout(fid):
    """
    Return the header, and the signal's offset and shape.

    Leaves the stream at the distances, which the times follow.
    """
    fid.seek(0)
    header = _read_float32(fid, 6)
    sensor_num, measurement_count = int(header[0]), int(header[1])
    data_offset = (6 + sensor_num + measurement_count) * 4
    return header, (data_offset, measurement_count, sensor_num)
