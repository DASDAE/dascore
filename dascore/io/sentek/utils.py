"""Utilities for Sentek data format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dascore as dc
from dascore.core import get_coord, get_coord_manager


def _get_version(fid):
    """Determine if Sentek file."""
    name = fid.name
    # Sentek files cannot change the extension, or file name.
    sw_data = name.endswith(".das")
    fid.seek(0)
    # There isn't anything in the header particularly useful for determining
    # if it is a Sentek file, so we do what we can here.
    # First check if sensor_num and measurement_count are positive and nearly
    # ints.
    sensor_num = np.fromfile(fid, dtype=np.float32, count=1)[0]
    measurement_count = np.fromfile(fid, dtype=np.float32, count=1)[0]
    _ = np.fromfile(fid, dtype=np.float32, count=1)[0]  # sampling_interval
    is_positive = (sensor_num > 1) and (measurement_count > 1)
    sens_nearly_int = np.round(sensor_num, 5) == np.round(sensor_num)
    meas_nearly_int = np.round(measurement_count, 5) == np.round(measurement_count)
    nearly_ints = sens_nearly_int and meas_nearly_int
    # Then check if strain_rate value is valid.
    strain_rate = int(np.fromfile(fid, dtype=np.float32, count=1)[0])
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
    return np.datetime64(iso)


def _get_patch_attrs(fid, extras=None):
    """Extracts patch metadata.

    A few important fields in the header and their meaning:

    sensor_num: number of channels in the sensing fiber
    measurement_count: number of measurements in ONE single file
    sampling_interval: sampling interval in nanosecond (delta t)
    strain_rate: flag that is set when the loaded data represents strain rate
    trigger_position: index position where the trigger occurs
    decimation_factor: decimation factor (integer)
    """
    fid.seek(0)
    sensor_num = np.fromfile(fid, dtype=np.float32, count=1)[0]
    measurement_count = np.fromfile(fid, dtype=np.float32, count=1)[0]
    _ = np.fromfile(fid, dtype=np.float32, count=1)[0]  # sampling_interval
    strain_rate = np.fromfile(fid, dtype=np.float32, count=1)[0]
    _ = np.fromfile(fid, dtype=np.float32, count=1)[0]  # trigger_position
    _ = np.fromfile(fid, dtype=np.float32, count=1)[0]  # decimation_factor
    # create distance coordinate
    distance_start = np.fromfile(fid, dtype=np.float32, count=1)[0]
    fid.seek(int(sensor_num - 1) * 4)
    distance_stop = np.fromfile(fid, dtype=np.float32, count=1)[0]
    distance_step = (distance_stop - distance_start) / sensor_num
    dist = get_coord(start=distance_start, stop=distance_stop, step=distance_step)
    # create time coord
    file_time = _get_time_from_file_name(Path(fid.name).name)
    offset_start = np.fromfile(fid, dtype=np.float32, count=1)[0]
    fid.seek(int(measurement_count - 1) * 4)
    offset_stop = np.fromfile(fid, dtype=np.float32, count=1)[0]
    time_start = file_time + dc.to_timedelta64(offset_start)
    time_stop = file_time + dc.to_timedelta64(offset_stop)
    time_step = (time_stop - time_start) / measurement_count
    time = get_coord(start=time_start, stop=time_stop, step=time_step)

    data_type = "strain_rate" if strain_rate else "strain"
    coord_manager = get_coord_manager(
        {"time": time, "distance": dist}, dims=("distance", "time")
    )
    attrs = dc.PatchAttrs(
        coords=coord_manager, data_type=data_type, **({} if extras is None else extras)
    )
    offsets = fid.tell(), int(measurement_count), int(sensor_num)
    return attrs, coord_manager, offsets
