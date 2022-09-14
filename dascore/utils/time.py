"""
Utility for working with time.
"""

from datetime import datetime
from functools import singledispatch
from typing import Optional, Union

import numpy as np
import pandas as pd

from dascore.constants import (
    LARGEDT64,
    NUMPY_TIME_UNIT_MAPPPING,
    SMALLDT64,
    timeable_types,
)
from dascore.exceptions import TimeError


@singledispatch
def to_datetime64(obj: Union[timeable_types, np.array]):
    """Convert"""
    msg = f"type {type(obj)} is not yet supported"
    raise NotImplementedError(msg)


@to_datetime64.register(str)
def str_to_datetime64(obj: str) -> np.datetime64:
    """Convert a string to a datetime64 object."""
    return np.datetime64(obj, "ns")


@to_datetime64.register(float)
@to_datetime64.register(int)
def float_to_datetime(num: Union[float, int]) -> np.datetime64:
    """Convert a float to a single datetime"""
    ar = np.array([num])
    return array_to_datetime64(ar)[0]


@to_datetime64.register(np.ndarray)
@to_datetime64.register(list)
@to_datetime64.register(tuple)
def array_to_datetime64(array: np.array) -> Union[np.datetime64, np.ndarray]:
    """
    Convert an array of floating point timestamps to an array of np.datatime64.
    """
    array = np.array(array)
    nans = pd.isnull(array)
    # dealing with an array of datetime64 or empty array
    if np.issubdtype(array.dtype, np.datetime64) or len(array) == 0:
        if not array.shape:  # dealing with 0-D array (scalar)
            out = np.datetime64(array)
        else:
            out = array
    # just check first element to determine type.  # TODO replace with dtype check
    # dealing with numerical data
    elif not isinstance(array[0], np.datetime64) and np.isreal(array[0]):
        array[nans] = 0  # temporary replace NaNs
        try:
            # separate seconds and factions, assume ns precision
            int_sec = array.astype(np.int64).astype("datetime64[s]")
        except (TypeError, ValueError):
            out = np.array([to_datetime64(x) for x in array])
        else:
            frac_sec = array % 1.0
            ns = (frac_sec * 1_000_000_000).astype(np.int64).astype("timedelta64[ns]")
            out = int_sec.astype("datetime64[ns]") + ns
        # fill NaN Back in
        out[nans] = np.datetime64("NaT")
    elif isinstance(array[0], str):
        out = array.astype("datetime64[ns]")
    else:  # No fast path, just iterate array elements
        return np.array([to_datetime64(x) for x in array])

    return out


@to_datetime64.register(np.datetime64)
def _pass_datetime(datetime):
    """simply return the datetime"""
    return np.datetime64(datetime, "ns")


@to_datetime64.register(type(None))
@to_datetime64.register(type(pd.NaT))
def _return_NaT(datetime):
    """Convert non to NaT"""
    return np.datetime64("NaT")


@to_datetime64.register(pd.Timestamp)
def _pandas_timestamp(datetime: pd.Timestamp):
    return datetime.to_datetime64()


@singledispatch
def to_timedelta64(obj: Union[float, np.array, str]):
    """Convert"""
    if pd.isnull(obj):
        return np.datetime64("NaT")
    msg = f"type {type(obj)} is not yet supported"
    raise NotImplementedError(msg)


@to_timedelta64.register(float)
@to_timedelta64.register(int)
def float_to_timedelta64(num: Union[float, int]) -> np.datetime64:
    """Convert a float to a single datetime"""
    ar = np.array([num])
    return array_to_timedelta64(ar)[0]


@to_timedelta64.register(np.ndarray)
@to_timedelta64.register(list)
@to_timedelta64.register(tuple)
def array_to_timedelta64(array: np.array) -> np.datetime64:
    """
    Convert an array of floating point timestamps to an array of np.datatime64.
    """
    array = np.array(array)
    nans = pd.isnull(array)
    array[nans] = 0
    if np.issubdtype(array.dtype, np.timedelta64) or len(array) == 0:
        if not array.shape:  # unpack degenerate array
            return np.timedelta64(array)
        else:
            return array.astype("timedelta64[ns]")
    assert np.isreal(array[0])
    # separate seconds and factions, convert fractions to ns precision
    # sub in array
    seconds = array.astype(np.int64).astype("timedelta64[s]")
    frac_sec = array % 1.0
    ns = (frac_sec * 1_000_000_000).astype(np.int64).astype("timedelta64[ns]")
    out = seconds + ns
    out[nans] = np.timedelta64("NaT")
    return out


@to_timedelta64.register(pd.Series)
def series_to_timedelta64_series(ser: pd.Series) -> pd.Series:
    """
    Convert a series to a series of timedelta64.
    """
    return pd.to_timedelta(ser)


@to_timedelta64.register(np.timedelta64)
def pass_time_delta(time_delta):
    """simply return the time delta."""
    return to_timedelta64(time_delta / np.timedelta64(1, "s"))


@to_timedelta64.register(pd.Timedelta)
def unpack_pandas_time_delta(time_delta: pd.Timedelta):
    """simply return the time delta."""
    return time_delta.to_numpy()


@to_timedelta64.register(str)
def time_delta_from_str(time_delta_str: str):
    """simply return the time delta."""
    split = time_delta_str.split(" ")
    assert len(split) == 2
    val, units = split
    if units[-1] == "s":
        units = units[:-1]
    new_unit = NUMPY_TIME_UNIT_MAPPPING[units]
    return np.timedelta64(int(val), new_unit)


def get_select_time(
    time: Union[float, int, np.datetime64, np.timedelta64, str, datetime],
    time_min: Optional[np.datetime64] = None,
    time_max: Optional[np.datetime64] = None,
) -> np.datetime64:
    """
    Applies logic for select time.

    Parameters
    ----------
    time
        The input time argument. Can either be:
            * An absolute time expressed as a datetime64 or datettime object.
            * A relative time expressed as a float, int, or time delta.
              Positive relative times reference time_min, negative reference
              time_max.

    time_min
        The reference start time (used for relative times).
    time_max
        The reference end tiem (used for relative times).

    """
    if pd.isnull(time):
        return np.datetime64("NaT")
    if isinstance(time, (str, datetime, np.datetime64)):
        return to_datetime64(time)
    else:
        d_time = to_timedelta64(time)
        relative_to = time_min if d_time > 0 else time_max
        if pd.isnull(relative_to):
            msg = "Cannot use relative times when reference times are null"
            raise TimeError(msg)
        return relative_to + d_time


@singledispatch
def to_number(obj: Union[timeable_types, np.array]) -> np.array:
    """
    Ensure a scalar or array is a number.

    If the input values represents a time or a time-delta, convert it to a
    an int representing ns.
    """
    msg = f"type {type(obj)} is not yet supported"
    raise NotImplementedError(msg)


@to_number.register(float)
@to_number.register(int)
def float_to_num(num: Union[float, int]) -> Union[float, int]:
    """Convert a float to a single datetime"""
    return num


@to_number.register(np.ndarray)
@to_number.register(list)
@to_number.register(tuple)
def array_to_number(array: np.array) -> np.array:
    """
    Convert an array of floating point timestamps to an array of np.datatime64.
    """
    array = np.array(array)
    if not len(array):
        return array
    # dealing with an array of datetime64 or empty array
    is_dt = np.issubdtype(array.dtype, np.datetime64)
    is_td = np.issubdtype(array.dtype, np.timedelta64)
    if is_td or is_dt:
        new = to_datetime64(array) if is_dt else to_timedelta64(array)
        array = new.astype(np.int64)
    return array


@to_number.register(np.datetime64)
@to_number.register(datetime)
@to_number.register(pd.Timestamp)
def _time_to_num(datetime):
    """simply return the datetime"""
    return to_number([to_datetime64(datetime)])[0]


@to_number.register(type(None))
@to_number.register(type(pd.NaT))
@to_number.register(type(pd.NA))
def _return_number_null(null):
    """Convert non to NaT"""
    return np.NaN


@to_number.register(np.timedelta64)
def _time_detal_to_number(time_delta: np.timedelta64):
    return to_number([to_timedelta64(time_delta)])[0]


@to_number.register(pd.Series)
def _pandas_timestamp_to_num(ser: pd.Series):
    return ser.view(np.int64)


def is_datetime64(obj) -> bool:
    """Return True if object is a timedelta object or array of such."""
    if isinstance(obj, np.datetime64):
        return True
    if isinstance(obj, (np.ndarray, list, tuple, pd.Series)):
        if np.issubdtype(np.array(obj).dtype, np.datetime64):
            return True
    if isinstance(obj, pd.Timestamp):
        return True
    return False


def get_max_min_times(kwarg_time=None):
    """
    Function to get min/max times from a tuple of possible time values.

    If None, return max/min times possible.
    """
    # first unpack time from tuples
    assert kwarg_time is None or len(kwarg_time) == 2
    time_min, time_max = (None, None) if kwarg_time is None else kwarg_time
    # get defaults if starttime or endtime is none
    time_min = None if pd.isnull(time_min) else time_min
    time_max = None if pd.isnull(time_max) else time_max
    time_min = to_datetime64(time_min or SMALLDT64)
    time_max = to_datetime64(time_max or LARGEDT64)
    if time_min is not None and time_max is not None:
        if time_min > time_max:
            msg = "time_min cannot be greater than time_max."
            raise ValueError(msg)
    return time_min, time_max
