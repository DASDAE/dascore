"""
Utility for working with time.
"""
from typing import Union
from functools import singledispatch
import numpy as np


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


@singledispatch
def to_timedelta64(obj: Union[float, np.array, str]):
    """Convert"""
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
