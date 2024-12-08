"""Utilities for segy."""

from __future__ import annotations

import datetime

import numpy as np

# --- Getting format/version
import pandas as pd

import dascore as dc
from dascore import to_float
from dascore.core import get_coord_manager
from dascore.exceptions import InvalidSpoolError, PatchError
from dascore.utils.misc import optional_import
from dascore.utils.patch import check_patch_coords

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


def _get_segy_version(fp):
    """
    Return True if file pointer contains segy formatted data.

    Based on ObsPy's implementation writen by Lion Krischer.
    https://github.com/obspy/obspy/blob/master/obspy/io/segy/core.py
    """
    # # Read 400byte header into byte string.
    fp.seek(3200)
    header = fp.read(400)
    data_trace_count = twos_comp(header[12:14])
    auxiliary_trace_count = twos_comp(header[14:16])
    sample_interval = twos_comp(header[16:18])
    samples_per_trace = twos_comp(header[20:22])
    data_format_code = twos_comp(header[24:26])
    format_number_major = int.from_bytes(header[300:301])
    format_number_minor = int.from_bytes(header[301:302])
    fixed_len_flag = twos_comp(header[302:304])

    checks = (
        # First check that some samples and a sampling rate is defined.
        samples_per_trace > 0 and sample_interval > 0,
        # Ensure the data sample format code is valid using range in 2,1 standard
        1 <= data_format_code <= 16,
        # Check version code
        format_number_major in {0, 1, 2, 3},
        # Sanity checks for other values.
        data_trace_count >= 0,
        auxiliary_trace_count >= 0,
        format_number_minor in {0, 1, 2, 3},
        fixed_len_flag in {0, 1},
    )
    if all(checks):
        return "segy", f"{format_number_major}.{format_number_minor}"
    else:
        return False


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
    segyio = optional_import("segyio")
    trace_field = segyio.TraceField
    header_0 = fi.header[0]

    # get time array from SEGY headers
    starttime = _get_time_from_header(header_0)
    dt = dc.to_timedelta64(header_0[trace_field.TRACE_SAMPLE_INTERVAL] / 1000)
    ns = header_0[trace_field.TRACE_SAMPLE_COUNT]
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
    segyio = optional_import("segyio")
    trace_field = segyio.TraceField

    year = header[trace_field.YearDataRecorded]
    julday = header[trace_field.DayOfYear]
    hour = header[trace_field.HourOfDay]
    minute = header[trace_field.MinuteOfHour]
    second = header[trace_field.SecondOfMinute]
    # make those timedate64
    fmt = "%Y.%j.%H.%M.%S"
    s = f"{year}.{julday}.{hour}.{minute}.{second}"
    time = datetime.datetime.strptime(s, fmt)
    return dc.to_datetime64(time)


def _get_segy_compatible_patch(spool):
    """
    Get a patch that will be writable as a segy file.
    Ensure coords are distance time.
    """
    # Ensure we have a single patch with coordinates time and distance.
    spool = [spool] if isinstance(spool, dc.Patch) else spool
    if len(spool) != 1:
        msg = "Can only write a spool with as single patch as segy."
        raise InvalidSpoolError(msg)
    patch = spool[0]
    check_patch_coords(patch, dims=("time", "distance"))
    # Ensure there will be no loss in the time sampling.
    # segy supports us precision
    time_step = dc.to_float(patch.get_coord("time").step)
    new_samp = np.round(time_step, 6)
    if np.abs(new_samp - time_step).max() > 1e-7:
        msg = (
            f"The segy format support us precision for temporal sampling. "
            f"The input patch has a time step of {time_step} which will result "
            "in a loss or precision. Either manually set the time step with "
            "patch.update_coords or resample the time axis with patch.resample"
        )
        raise PatchError(msg)
    return patch.transpose("distance", "time")


def _make_time_header_dict(time_coord):
    """Make the time header dict from a time coordinate."""
    header = {}
    timestamp = pd.Timestamp(dc.to_datetime64(time_coord.min()))
    time_step_us = to_float(time_coord.step) * 1_000_000

    segyio = optional_import("segyio")
    trace_field = segyio.TraceField

    header[trace_field.YearDataRecorded] = timestamp.year
    header[trace_field.DayOfYear] = timestamp.day_of_year
    header[trace_field.HourOfDay] = timestamp.hour
    header[trace_field.MinuteOfHour] = timestamp.minute
    header[trace_field.SecondOfMinute] = timestamp.second
    header[trace_field.TRACE_SAMPLE_INTERVAL] = time_step_us

    return header
