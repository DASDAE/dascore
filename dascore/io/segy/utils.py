"""Utilities for segy."""

from __future__ import annotations

import datetime

import numpy as np
from segyio import TraceField

import dascore as dc
from dascore.core import get_coord_manager

# --- Getting format/version


def _get_filtered_data_and_coords(segy_fi, coords, time=None, channel=None):
    """
    Read data from segy_file, possibly reading only subsections.

    Return filtered data and update coord manager.
    """
    traces_inds_to_read = np.arange(len(segy_fi.header), dtype=np.int64)
    time_slice = slice(None, None)
    traces = segy_fi.trace

    # filter time
    if time is not None:
        time_coord = coords.coord_map["time"]
        new_coord, time_slice = time_coord.select(time)
        coords = coords.update(time=new_coord)

    # filter channel
    if channel:
        channel_coord = coords.coord_map["channel"]
        new_coord, channel_inds = channel_coord.select(channel)
        coords = coords.update(channel=new_coord)
        traces_inds_to_read = traces_inds_to_read[channel_inds]

    # filter channels
    data_list = [traces[x][time_slice] for x in traces_inds_to_read]
    return np.stack(data_list, axis=-1), coords


def _get_coords(fi):
    """
    Get coordinates of the segy file.

    Time comes from the SEGY format of year, julian day, hour, minute, second.

    Distance axis is channel number. If the user knows the delta_x,
    then the axis should be modified.

    If a user knows the dx, change from channel to distance using
    patch.update_coords after reading
    """
    header_0 = fi.header[0]

    # get time array from SEGY headers
    starttime = _get_time_from_header(header_0)
    dt = dc.to_timedelta64(header_0[TraceField.TRACE_SAMPLE_INTERVAL] / 1000)
    ns = header_0[TraceField.TRACE_SAMPLE_COUNT]
    time_array = starttime + dt * np.arange(ns)

    # Get distance array from SEGY header
    channel = np.arange(len(fi.header))

    coords = get_coord_manager(
        {"time": time_array, "channel": channel}, dims=("time", "channel")
    )
    return coords


def _get_attrs(fi, coords, path, file_io):
    """Create Patch Attribute from SEGY header contents."""
    attrs = dc.PatchAttrs(
        path=path,
        file_version=file_io.version,
        file_format=file_io.name,
        coords=coords,
    )
    return attrs


def _get_time_from_header(header):
    """Creates a datetime64 object from SEGY header date information."""
    year = header[TraceField.YearDataRecorded]
    julday = header[TraceField.DayOfYear]
    hour = header[TraceField.HourOfDay]
    minute = header[TraceField.MinuteOfHour]
    second = header[TraceField.SecondOfMinute]
    # make those timedate64
    fmt = "%Y.%j.%H.%M.%S"
    s = f"{year}.{julday}.{hour}.{minute}.{second}"
    time = datetime.datetime.strptime(s, fmt)
    return dc.to_datetime64(time)
