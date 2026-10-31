"""Utilities for segy."""

from __future__ import annotations

import warnings

import numpy as np

# --- Getting format/version
import pandas as pd

import dascore as dc
from dascore import to_float
from dascore.core import get_coord_manager
from dascore.exceptions import InvalidSpoolError, PatchError
from dascore.utils.misc import optional_import

# Bytes per sample of each SEG-Y (rev 2.1) data sample format code.
_SAMPLE_BYTES = {1: 4, 2: 4, 3: 2, 4: 4, 5: 4, 6: 8, 7: 3, 8: 1}
_SAMPLE_BYTES |= {9: 8, 10: 4, 11: 2, 12: 8, 15: 3, 16: 1}


def twos_comp(bytes_, byteorder="big"):
    """Decode a signed integer from bytes."""
    return int.from_bytes(bytes_, byteorder, signed=True)


def _parse_binary_header(fp):
    """
    Return (byte order, version) if the file holds SEG-Y data, else None.

    The format code (1 to 16) is plausible in only one byte order, which
    settles it. A non-standard minor revision (eg 1.80) is reported as
    "major.0", and only when whole traces fill the file. Based on ObsPy's
    implementation written by Lion Krischer.
    https://github.com/obspy/obspy/blob/master/obspy/io/segy/core.py
    """
    # A SEG-Y file opens with a textual header, never an HDF5 signature;
    # the binary header checks alone can pass on an HDF5 file's bytes.
    fp.seek(0)
    if fp.read(8) == b"\x89HDF\r\n\x1a\n":
        return None
    # Read 400byte header into byte string.
    fp.seek(3200)
    header = fp.read(400)
    if len(header) < 400:
        return None
    major, minor = header[300:302]
    for endian in ("big", "little"):
        traces, auxiliary, interval, _, samples, _, code = (
            twos_comp(header[i : i + 2], endian) for i in range(12, 26, 2)
        )
        fixed_length, extended = (
            twos_comp(header[i : i + 2], endian) for i in (302, 304)
        )
        checks = (
            # First check that some samples are defined.
            samples > 0,
            # The interval may be 0 (defined in the trace headers) but not
            # negative.
            interval >= 0,
            # Ensure the data sample format code is valid (2.1 standard range).
            1 <= code <= 16,
            major in {0, 1, 2, 3},
            # Sanity checks for other values.
            traces >= 0,
            auxiliary >= 0,
            fixed_length in {0, 1},
        )
        if all(checks):
            break
    else:
        return None
    if endian == "little" and major == 0 and minor in {1, 2, 3}:
        # segyio writes the revision as one little-endian word (00 01 = 1.0).
        major, minor = minor, major
    if minor not in {0, 1, 2, 3, 100}:
        # Some producers write a non-standard minor revision (eg 1.80); read
        # those as the major revision when whole traces fill the file exactly.
        sample_bytes = _SAMPLE_BYTES.get(code, 0)
        size = fp.seek(0, 2) - 3600 - 3200 * max(extended, 0)
        if not sample_bytes or size <= 0 or size % (240 + samples * sample_bytes):
            return None
        minor = 0
    return endian, f"{major}.{minor}"


def _open_segy(path, segyio):
    """Open a SEG-Y file with segyio in the byte order its header uses."""
    with open(path, "rb") as fp:
        parsed = _parse_binary_header(fp)
    endian = parsed[0] if parsed else "big"
    return segyio.open(str(path), ignore_geometry=True, endian=endian)


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

    # Get time array from SEGY headers
    starttime = _get_time_from_header(header_0)
    dt = np.timedelta64(int(header_0[trace_field.TRACE_SAMPLE_INTERVAL]), "us")
    ns = header_0[trace_field.TRACE_SAMPLE_COUNT]
    time_array = starttime + dt * np.arange(ns)

    # Get distance array from SEGY header
    channel = np.arange(len(fi.header))

    coords = get_coord_manager(
        {"time": time_array, "channel": channel}, dims=("time", "channel")
    )
    return coords


def _get_time_from_header(header):
    """
    Get the start time from the SEGY trace header date fields.

    The fields are added up, so a second of 60 rolls into the next minute.
    Without a usable year and day, warn and return timedelta64(0), so time
    is relative to the first sample.
    """
    segyio = optional_import("segyio")
    trace_field = segyio.TraceField

    year = header[trace_field.YearDataRecorded]
    julday = header[trace_field.DayOfYear]
    hour = header[trace_field.HourOfDay]
    minute = header[trace_field.MinuteOfHour]
    second = header[trace_field.SecondOfMinute]
    # Years outside datetime64[ns] (eg two-digit years) would wrap silently.
    if not (1678 <= year <= 2261 and 1 <= julday <= 366):
        msg = (
            f"SEG-Y trace header date (year {year}, day {julday}) is not "
            "valid, so the time coordinate is relative to the first sample. "
            "To set the start: patch.update_coords(time="
            'patch.get_coord("time") + dc.to_datetime64(start)).'
        )
        warnings.warn(msg, UserWarning)
        return np.timedelta64(0, "ns")
    start = np.datetime64(f"{year:04d}-01-01", "ns")
    seconds = (((julday - 1) * 24 + hour) * 60 + minute) * 60 + second
    return start + np.timedelta64(seconds, "s")


def _get_patch_with_channel_coord(patch):
    """Ensure the patch has a channel coordinate."""
    dims = set(patch.dims)
    non_time = next(iter(dims - {"time"}))
    msg = (
        f"Currently the segy writer only handles 'channel' as the non-time "
        f"dimension; this results in a loss of the '{non_time}' dimension."
    )
    warnings.warn(msg)
    coord = patch.get_coord(non_time)
    array = np.arange(len(coord))
    patch = patch.update_coords(**{non_time: array}).rename_coords(
        **{non_time: "channel"}
    )
    return patch


def _get_segy_compatible_patch(spool, round_error_max=3e-9):
    """
    Get a patch that will be writable as a segy file.
    Ensure coords are ("channel", "time").
    """
    # Ensure we have a single patch with coordinates time and distance.
    spool = [spool] if isinstance(spool, dc.Patch) else spool
    if len(spool) != 1:
        msg = "Can only write a spool with as single patch as segy."
        raise InvalidSpoolError(msg)
    patch = spool[0]
    dims = set(patch.dims)
    has_distance_or_channel = dims & {"distance", "channel"}
    if len(dims) != 2 or "time" not in dims or not has_distance_or_channel:
        msg = (
            "Can only save 2D patches to SEGY with a time dimension and "
            "either channel or distance dimensions."
        )
        raise PatchError(msg)
    # Currently we only support channels not distance dimension.
    if "channel" not in dims:
        patch = _get_patch_with_channel_coord(patch)
    # Ensure there will be no loss in the time sampling.
    # segy supports us precision
    time = patch.get_coord("time", require_evenly_sampled=True)
    time_step = dc.to_float(time.step)
    new_samp = np.round(time_step, 6)
    round_error = np.abs(new_samp - time_step).max()
    if round_error > round_error_max:
        msg = (
            f"The segy format support us precision for temporal sampling. "
            f"The input patch has a time step of {time_step} which will result "
            "in a loss of precision. Either manually set the time step with "
            "patch.update_coords or resample the time axis with patch.resample"
        )
        raise PatchError(msg)
    if (start := pd.Timestamp(dc.to_datetime64(time.min()))) != start.floor("s"):
        msg = (
            "The SEGY writer stores the start time in whole seconds, so "
            f"{start} will be written as {start.floor('s')}."
        )
        warnings.warn(msg)
    return patch.transpose("channel", "time")


def _make_time_header_dict(time_coord):
    """Make the time header dict from a time coordinate."""
    header = {}
    timestamp = pd.Timestamp(dc.to_datetime64(time_coord.min()))
    time_step_ms = np.round(to_float(time_coord.step) * 1_000_000)

    segyio = optional_import("segyio")
    trace_field = segyio.TraceField

    header[trace_field.YearDataRecorded] = timestamp.year
    header[trace_field.DayOfYear] = timestamp.day_of_year
    header[trace_field.HourOfDay] = timestamp.hour
    header[trace_field.MinuteOfHour] = timestamp.minute
    header[trace_field.SecondOfMinute] = timestamp.second
    header[trace_field.TRACE_SAMPLE_INTERVAL] = int(time_step_ms)
    header[trace_field.TRACE_SAMPLE_COUNT] = len(time_coord)

    return header


def _write_segy(spool, resource, version, segyio):
    """
    Private function for writing a patch/spool as SEGY.
    """
    patch = _get_segy_compatible_patch(spool)
    time, channel = patch.get_coord("time"), patch.get_coord("channel")
    channel_step = channel.step

    time_dict = _make_time_header_dict(time)
    bin_field = segyio.BinField
    spec = segyio.spec()

    spec.format = 1  # 1 means float32 TODO look into supporting more
    spec.samples = np.ones(len(time)) * len(channel)
    spec.ilines = range(len(channel))
    spec.xlines = [1]

    # For 32 bit float for now.
    data = patch.data.astype(np.float32)

    with segyio.create(resource, spec) as f:
        # Update the file header info.
        f.bin.update(tsort=segyio.TraceSortingFormat.INLINE_SORTING)
        f.bin.update(
            {
                bin_field.Samples: time_dict[segyio.TraceField.TRACE_SAMPLE_COUNT],
                bin_field.Interval: time_dict[segyio.TraceField.TRACE_SAMPLE_INTERVAL],
                bin_field.SEGYRevision: int(version.split(".")[0]),
                bin_field.SEGYRevisionMinor: int(version.split(".")[1]),
            }
        )
        # Then iterate each channel and dump to segy.
        for num, data in enumerate(data):
            header = dict(time_dict)
            header.update(
                {
                    segyio.su.offset: channel_step,
                    segyio.su.iline: num,
                    segyio.su.xline: 1,
                }
            )
            f.header[num] = header
            f.trace[num] = data
