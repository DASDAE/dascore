"""Utilities for segy."""

from __future__ import annotations

import datetime

import numpy as np
from segyio import TraceField

import dascore as dc
from dascore.core import get_coord_manager

# --- Getting format/version

DATA_ARRAY_NAMES = frozenset(["raw", "data"])
TIME_ARRAY_NAMES = frozenset(("timestamp", "time", "timestamps"))
OTHER_COORD_ARRAY_NAMES = frozenset(("channels", "distance"))

FILE_FORMAT_ATTR_NAMES = frozenset(("__format__", "file_format", "format"))
DEFAULT_ATTRS = frozenset(("CLASS", "PYTABLES_FORMAT_VERSION", "TITLE", "VERSION"))


def get_coords(fi):
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
    starttime = get_time_from_header(header_0)
    dt = dc.to_timedelta64(header_0[TraceField.TRACE_SAMPLE_INTERVAL] / 1000)
    ns = header_0[TraceField.TRACE_SAMPLE_COUNT]
    time_array = starttime + dt * np.arange(ns)

    # Get distance array from SEGY header
    channel = np.arange(len(fi.header))

    coords = get_coord_manager(
        {"time": time_array, "channel": channel}, dims=("time", "channel")
    )
    return coords


def get_attrs(fi, coords, path, file_io):
    """Create Patch Attribute from SEGY header contents."""
    attrs = dc.PatchAttrs(
        file_path=path,
        file_version=file_io.version,
        file_format=file_io.name,
        coords=coords,
    )
    return attrs


def get_time_from_header(header):
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
