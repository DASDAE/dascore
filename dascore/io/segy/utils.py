"""Utilities for segy."""

from __future__ import annotations

import datetime

# --- Getting format/version
import numpy as np
from segyio import TraceField

import dascore as dc
from dascore.core import get_coord_manager

# Valid data format codes as specified in the SEGY rev1 manual.
VALID_FORMATS = [1, 2, 3, 4, 5, 8]

# This is the maximum possible interval between two samples due to the nature
# of the SEG Y format.
MAX_INTERVAL_IN_SECONDS = 0.065535

# largest number possible with int16
MAX_NUMBER_OF_SAMPLES = 32767


def twos_comp(bytes_):
    """Get twos complement of bytestring."""
    bits = len(bytes_) * 8
    val = int.from_bytes(bytes_, "big")
    if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)  # compute negative value
    return val  # return positive value as is


def _is_segy(fp):
    """
    Return True if file pointer contains segy formatted data.

    Based on ObsPy's implementation writen by Lion Krischer.
    https://github.com/obspy/obspy/blob/master/obspy/io/segy/core.py
    """
    # # Read 400byte header into byte string.
    # fp.seek(3200)
    # header = fp.read(400)
    # data_trace_count = twos_comp(header[12:14])
    # auxiliary_trace_count = twos_comp(header[14:16])
    # sample_interval = twos_comp(header[16:18])
    # samples_per_trace = twos_comp(header[20:22])
    # data_format_code = twos_comp(header[24:26])
    # format_number_major = int.from_bytes(header[300:301])
    # format_number_minor = int.from_bytes(header[301:302])
    # fixed_len_flag = twos_comp(header[302:304])
    #
    #
    # if _format_number not in (0x0000, 0x0100, 0x0010, 0x0001):
    #     return False
    #
    # _fixed_length = unpack(fmt, _fixed_length)[0]
    # _extended_number = unpack(fmt, _extended_number)[0]
    # # Make some sanity checks and return False if they fail.
    # if (
    #     _sample_interval <= 0
    #     or _samples_per_trace <= 0
    #     or _number_of_data_traces < 0
    #     or _number_of_auxiliary_traces < 0
    #     or _fixed_length < 0
    #     or _extended_number < 0
    # ):
    #     return False
    return True


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
