"""
Utility for working with time.
"""

from datetime import datetime
from functools import singledispatch
from typing import Union, Optional


import numpy as np
import pandas as pd

from fios.exceptions import TimeError


@singledispatch
def to_datetime64(obj: Union[float, np.array, str]):
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
def array_to_datetime64(array: np.array) -> np.datetime64:
    """
    Convert an array of floating point timestamps to an array of np.datatime64.
    """
    array = np.array(array)
    # dealing with an array of datetime64 already
    if np.issubdtype(array.dtype, np.datetime64):
        out = array
    # just check first element to determine type.  # TODO replace with dtype check
    elif np.isreal(array[0]):  # dealing with numerical data
        # separate seconds and factions, assume ns precision
        int_sec = array.astype(np.int64).astype("datetime64[s]")
        frac_sec = array % 1.0
        ns = (frac_sec * 1_000_000_000).astype(np.int64).astype("timedelta64[ns]")
        out = int_sec.astype("datetime64[ns]") + ns

    elif isinstance(array[0], str):
        out = array.astype("datetime64[ns]")
    else:  # already
        msg = f"{array.dtype} not supported"
        raise NotImplementedError(msg)

    return out


@to_datetime64.register(np.datetime64)
def _pass_datetime(datetime):
    """simply return the datetime"""
    return datetime


@singledispatch
def to_timedelta64(obj: Union[float, np.array, str]):
    """Convert"""
    if pd.isnull(obj):
        return np.datetime64('NaT')
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
    # just check first element to determine type.
    array = np.array(array)
    if np.issubdtype(array.dtype, np.timedelta64):
        return array
    assert np.isreal(array[0])
    # separate seconds and factions, convert fractions to ns precision
    seconds = array.astype(np.int64).astype("timedelta64[s]")
    frac_sec = array % 1.0
    ns = (frac_sec * 1_000_000_000).astype(np.int64).astype("timedelta64[ns]")
    out = seconds + ns
    return out


@to_timedelta64.register(np.timedelta64)
def pass_time_delta(time_delta):
    """simply return the time delta."""
    return to_timedelta64(time_delta / np.timedelta64(1, "s"))


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
        return np.datetime64('NaT')
    if isinstance(time, (str, datetime, np.datetime64)):
        return to_datetime64(time)
    else:
        dt = to_timedelta64(time)
        relative_to = time_min if dt > 0 else time_max
        if pd.isnull(relative_to):
            msg = 'Cannot use relative times when reference times are null'
            raise TimeError(msg)
        return relative_to + dt




