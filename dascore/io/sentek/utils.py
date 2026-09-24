"""Utilities for Sentek data format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.io.utils import should_snap


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


def _read_coord(fid, offset, count, snap, start_time=None):
    """Read stored coordinate values, or only their endpoints when snapping."""
    fid.seek(offset)
    values = _read_float32(fid, 1 if snap else count)
    if snap and count > 1:
        fid.seek(offset + (count - 1) * 4)
        values = np.concatenate((values, _read_float32(fid)))
    if start_time is not None:
        values = start_time + dc.to_timedelta64(values.astype(np.float64))
    elif snap:
        values = values.astype(np.float64)
    if snap and count > 1:
        return get_coord(
            start=values[0],
            step=(values[-1] - values[0]) / (count - 1),
            shape=(count,),
        ).change_length(count)
    return get_coord(data=values, snap=False)


def _get_patch_attrs(fid, extras=None, *, snap=True):
    """Extract metadata from the header followed by distance and time arrays.

    The six float32 header fields are the channel count, sample count,
    sampling interval, strain-rate flag, trigger position, and decimation
    factor. Signal samples follow both coordinate arrays in time-major order.
    Stored time offsets retain the reader's existing seconds-from-filename
    interpretation.
    """
    fid.seek(0)
    header = _read_float32(fid, 6)
    sensor_num, measurement_count = int(header[0]), int(header[1])
    distance_offset = 6 * 4
    time_offset = distance_offset + sensor_num * 4
    data_offset = time_offset + measurement_count * 4
    dist = _read_coord(fid, distance_offset, sensor_num, should_snap(snap, "distance"))
    name = getattr(fid, "name", None) or getattr(fid, "path", None)
    name = name or getattr(fid, "_dascore_source_path", "")
    file_time = _get_time_from_file_name(Path(str(name)).name)
    time = _read_coord(
        fid, time_offset, measurement_count, should_snap(snap, "time"), file_time
    )
    data_type = "strain_rate" if header[3] else "strain"
    coord_manager = get_coord_manager(
        {"time": time, "distance": dist}, dims=("distance", "time")
    )
    attrs = dc.PatchAttrs(data_type=data_type, **({} if extras is None else extras))
    return attrs, coord_manager, (data_offset, measurement_count, sensor_num)
