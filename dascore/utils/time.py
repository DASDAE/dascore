"""Utility for working with time."""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import singledispatch

import numpy as np
import pandas as pd

from dascore.constants import (
    LARGEDT64,
    NUMPY_TIME_UNIT_MAPPING,
    ONE_SECOND,
    SMALLDT64,
    timeable_types,
)
from dascore.exceptions import TimeError


@singledispatch
def to_datetime64(obj: timeable_types | np.ndarray):
    """
    Convert an object to a datetime64.

    This function accepts a wide range of inputs and returns something
    of the same shape, but converted to numpy's datetime64 representation.

    Parameters
    ----------
    obj
        An object to convert to a datetime64. If a string is passed, it
        should conform to [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
        Floats and integers are interpreted as seconds from Jan 1st, 1970.
        Arrays and Series of floats or strings are also supported.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>>
    >>> # Convert an iso 8601 string to datetime64
    >>> dt_1 = dc.to_datetime64('2017-09-17T12:11:01.23212')
    >>>
    >>> # Convert a time stamp (seconds from 1970) to datetime64
    >>> dt_2 = dc.to_datetime64(631152000.0)
    >>>
    >>> # Convert an array of time stamps to datetime64
    >>> timestamp_array = np.random.uniform(1704400000, 1704900000)
    >>> dt_array = dc.to_datetime64(timestamp_array)
    """
    if pd.isnull(obj):
        return np.datetime64("NaT")
    msg = f"type {type(obj)} is not supported"
    raise NotImplementedError(msg)


@to_datetime64.register(str)
def _str_to_datetime64(obj: str) -> np.datetime64:
    """Convert a string to a datetime64 object."""
    # strip off timezone info so numpy doesn't complain.
    if obj.endswith("Z"):
        obj = obj[:-1]
    return np.datetime64(obj, "ns")


@to_datetime64.register(float)
@to_datetime64.register(int)
@to_datetime64.register(np.number)
def _float_to_datetime(num: float | int) -> np.datetime64:
    """Convert a float to a single datetime."""
    ar = np.asarray([num])
    return _array_to_datetime64(ar)[0]


@to_datetime64.register(np.ndarray)
@to_datetime64.register(list)
@to_datetime64.register(tuple)
def _array_to_datetime64(array: np.ndarray) -> np.datetime64 | np.ndarray:
    """Convert an array of floating point timestamps to an array of np.datatime64."""
    array = np.asarray(array)
    nans = pd.isnull(array)
    # dealing with objects
    if np.issubdtype(array.dtype, np.dtype(object)):
        array = np.asarray([to_datetime64(x) for x in array]).astype("datetime64[ns]")
    # dealing with a string
    if np.issubdtype(array.dtype, np.dtype(str)):
        array = array.astype("datetime64[ns]")
    # dealing with an array of datetime64 or empty array
    if np.issubdtype(array.dtype, np.datetime64) or len(array) == 0:
        if not array.shape:  # dealing with degenerate (0-D( array
            out = np.datetime64(array)
        else:
            out = array.astype("datetime64[ns]")
    # dealing with numerical data
    elif not np.issubdtype(array.dtype, np.datetime64) and np.isreal(array[0]):
        with np.errstate(divide="ignore", invalid="ignore"):
            array = np.array(array)  # need to make copy to write
            array[nans] = 0  # temporary replace NaNs
            abs_array = np.abs(array)
            sign = np.sign(array)
            # separate seconds and factions, assume ns precision
            int_sec = abs_array.astype(np.int64)
            frac_sec = abs_array % 1.0
            ns = (frac_sec * 1_000_000_000).astype(np.int64)
            out = (sign * (int_sec * 1_000_000_000 + ns)).astype("datetime64[ns]")
        # fill NaN Back in
        out[nans] = np.datetime64("NaT")
    return out


@to_datetime64.register(pd.Series)
def _float_to_datetime(ser: pd.Series) -> pd.Series:
    """Convert a float to a single datetime."""
    ar = to_datetime64(ser.values)
    return pd.Series(ar, index=ser.index)


@to_datetime64.register(np.datetime64)
def _pass_datetime(datetime):
    """Simply return the datetime."""
    return np.datetime64(datetime, "ns")


@to_datetime64.register(datetime)
def _datetime_to_datetime64(dt: datetime):
    """Convert python datetime to datetime64."""
    # because pandas NaT has datettime in its MRO we need to check
    # if this is nullish and return NaT if so.
    if pd.isnull(dt):
        return np.datetime64("NaT")
    return to_datetime64(np.datetime64(dt))


@to_datetime64.register(pd.Timestamp)
def _pandas_timestamp(datetime: pd.Timestamp):
    return datetime.to_datetime64()


@singledispatch
def to_timedelta64(obj: float | np.ndarray | str | timedelta):
    """
    Convert an object to timedelta64.

    This function accepts a wide range of inputs and returns something
    of the same shape, but converted to numpy's timedelta64 representation.

    Parameters
    ----------
    obj
        An object to convert to timedelta64. Can be a float, str or arary of
        such. Floats are interpreted as seconds and strings must conform to
        the output style of timedeltas (e.g. str(time_delta)).

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> # Convert a float to timedelta64 representing seconds.
    >>> td_1 = dc.to_timedelta64(10.1232)
    >>>
    >>> # This also works on negative numbers.
    >>> td_2 = dc.to_datetime64(-10.5)
    >>>
    >>> # Convert a string to timedelta64
    >>> td_str = "1000000000 nanoseconds"
    >>> td_3 = dc.to_timedelta64(td_str)

    """
    if pd.isnull(obj):
        return np.timedelta64("NaT")
    msg = f"type {type(obj)} is not supported"
    raise NotImplementedError(msg)


@to_timedelta64.register(float)
@to_timedelta64.register(int)
@to_timedelta64.register(np.number)
def _float_to_timedelta64(num: float | int) -> np.datetime64:
    """Convert a float to a single datetime."""
    ar = np.asarray([num])
    return _array_to_timedelta64(ar)[0]


@to_timedelta64.register(np.ndarray)
@to_timedelta64.register(list)
@to_timedelta64.register(tuple)
def _array_to_timedelta64(array: np.ndarray) -> np.datetime64:
    """Convert an array of floating point timestamps to an array of np.datatime64."""
    array = np.asarray(array)
    # convert pure object arrays into float so sign casting works.
    if np.issubdtype(array.dtype, np.dtype(object)):
        array = array.astype(np.float64)
    if np.issubdtype(array.dtype, np.timedelta64) or len(array) == 0:
        if not array.shape:  # unpack degenerate array
            return np.timedelta64(array)
        else:
            return array.astype("timedelta64[ns]")
    # Need to just get the ns form datetime64
    elif np.issubdtype(array.dtype, np.datetime64):
        int_array = array.view(np.int64)
        return np.array(int_array).astype("timedelta64[ns]")

    assert np.isreal(array[0])
    nans = pd.isnull(array)
    array[nans] = 0
    # inf/NaN complain, salience these types of warnings for this block.
    with np.errstate(divide="ignore", invalid="ignore"):
        # separate seconds and factions, convert fractions to ns precision,
        # sub in array. Track sign and use abs to ensure sign comes out.
        sign = np.sign(array)
        abs_array = np.abs(array)
        seconds = abs_array.astype(np.int64).astype("timedelta64[s]")
        frac_sec = abs_array % 1.0
        ns = (frac_sec * 1_000_000_000).astype(np.int64).astype("timedelta64[ns]")
        out = sign * (seconds + ns)
        out[nans] = np.timedelta64("NaT")
    return out


@to_timedelta64.register(pd.Series)
def _series_to_timedelta64_series(ser: pd.Series) -> pd.Series:
    """Convert a series to a series of timedelta64."""
    out = to_timedelta64(ser.values)
    return pd.Series(out, index=ser.index)


@to_timedelta64.register(np.timedelta64)
def _pass_time_delta(time_delta):
    """Simply return the time delta as ns precision."""
    return time_delta.astype("<m8[ns]")


@to_timedelta64.register(pd.Timedelta)
def _unpack_pandas_time_delta(time_delta: pd.Timedelta):
    """Simply return the time delta."""
    return time_delta.to_numpy()


@to_timedelta64.register(timedelta)
def _timedelta_to_timedelta64(td):
    """Return timedelta64."""
    return to_timedelta64(np.timedelta64(td))


@to_timedelta64.register(str)
def _time_delta_from_str(time_delta_str: str):
    """Simply return the time delta."""
    match time_delta_str.split():
        # Can split string into (hopefully) units an values. Standard case.
        case [val, units]:
            if units[-1] == "s":
                units = units[:-1]
            new_unit = NUMPY_TIME_UNIT_MAPPING[units]
            return np.timedelta64(int(val), new_unit)
        case [val] if val.lower() == "nat" or val.lower() == "":
            return np.timedelta64("NaT")
        case _:
            msg = f"Could not convert {time_delta_str} to timedelta64"
            raise TimeError(msg)


@singledispatch
def to_int(obj: timeable_types | np.ndarray) -> np.ndarray:
    """
    Ensure a scalar or array is a number.

    If the input values represents a time or a time-delta, convert it to a
    an int representing ns.
    """
    msg = f"type {type(obj)} is not supported"
    raise NotImplementedError(msg)


@to_int.register(float)
@to_int.register(int)
@to_int.register(np.number)
def _float_to_num(num: float | int) -> float | int:
    """Convert number to int."""
    return int(num)


@to_int.register(np.ndarray)
@to_int.register(list)
@to_int.register(tuple)
def _array_to_int(array: np.ndarray) -> np.ndarray:
    """Convert an array of floating point timestamps to an array of np.datatime64."""
    array = np.asarray(array)
    if not len(array):
        return array.astype(np.int64)
    # dealing with an array of datetime64 or empty array
    is_dt = np.issubdtype(array.dtype, np.datetime64)
    is_td = np.issubdtype(array.dtype, np.timedelta64)
    if is_td or is_dt:
        new = to_datetime64(array) if is_dt else to_timedelta64(array)
        array = new.astype(np.int64)
    return array


@to_int.register(np.datetime64)
@to_int.register(datetime)
@to_int.register(pd.Timestamp)
def _time_to_int(datetime):
    """Simply return the datetime converted to ns."""
    return to_int([to_datetime64(datetime)])[0]


@to_int.register(type(None))
@to_int.register(type(pd.NaT))
@to_int.register(type(pd.NA))
def _return_number_null(null):
    """Convert non to NaT."""
    return np.nan


@to_int.register(np.timedelta64)
def _time_delta_to_number(time_delta: np.timedelta64):
    return to_int([to_timedelta64(time_delta)])[0]


@to_int.register(pd.Series)
def _pandas_timestamp_to_num(ser: pd.Series):
    return ser.astype(np.int64)


@singledispatch
def to_float(obj: timeable_types | np.ndarray) -> np.ndarray:
    """
    Convert various datetime/timedelta things to a float.

    Time offsets represent seconds, and datetimes are seconds from 1970.
    """
    return float(obj)


@to_float.register(np.ndarray)
@to_float.register(list)
@to_float.register(tuple)
def _array_to_float(array: np.ndarray) -> np.ndarray:
    """Convert an array of floating point timestamps to an array of np.datatime64."""
    array = np.asarray(array)
    if not len(array):
        return array.astype(np.float64)
    if np.issubdtype(array.dtype, np.datetime64):
        # convert to offset from 1970
        array = array - to_datetime64(0)
    if np.issubdtype(array.dtype, np.timedelta64):
        array = array / ONE_SECOND
    return array.astype(np.float64)


@to_float.register(pd.Series)
def _series_to_float(series: pd.Series) -> pd.Series:
    """Convert a series of possible dates to floats."""
    array = to_float(series.values)
    return pd.Series(array, index=series.index)


@to_float.register(np.datetime64)
@to_float.register(datetime)
@to_float.register(pd.Timestamp)
def _time_to_float(datetime):
    """Simply return the datetime."""
    td = to_datetime64(datetime) - to_datetime64(0)
    return to_float(td)


@to_float.register(type(None))
@to_float.register(type(pd.NaT))
@to_float.register(type(pd.NA))
def _return_null(null):
    """Convert non to NaT."""
    return np.nan


@to_float.register(np.timedelta64)
@to_float.register(timedelta)
@to_float.register(pd.Timedelta)
def _time_delta_to_float(time_delta: np.timedelta64):
    return to_timedelta64(time_delta) / ONE_SECOND


def _is_dtype(obj, numpy_dtype, pandas_dtype) -> bool:
    """
    Test if a variety of object types are of numpy or pandas dtype.

    Returns True if the object is a numpy type, pandas type,
    numpy dtype, pandas dtype, or an array-like of dtype values.
    """
    # Handle scalars: np.datetime64, pandas.Timestamp
    if isinstance(obj, numpy_dtype | pandas_dtype):
        return True
    # Handle numpy/pandas datetime64 dtypes directly
    if isinstance(obj, (np.dtype | pd.api.extensions.ExtensionDtype)):
        return np.issubdtype(obj, numpy_dtype)
    # Handle pandas Series
    if isinstance(obj, pd.Series):
        return np.issubdtype(obj.dtype, numpy_dtype)
    # Handle array-like objects (numpy arrays, lists, tuples)
    if isinstance(obj, np.ndarray | list | tuple):
        return np.issubdtype(np.asarray(obj).dtype, numpy_dtype)
    return False


def is_datetime64(obj):
    """Determine if an object represents a timedelta64 dtype or value(s)."""
    return _is_dtype(obj, np.datetime64, pd.Timestamp)


def is_timedelta64(obj):
    """Determine if an object represents a timedelta64 dtype or value(s)."""
    return _is_dtype(obj, np.timedelta64, pd.Timedelta)


def dtype_time_like(dtype_or_array) -> bool:
    """Return True if dtype is time related (datetime64, timedelta64)."""
    try:
        dtype_or_array = np.dtype(dtype_or_array)
    except TypeError:
        dtype_or_array = getattr(dtype_or_array, "dtype", dtype_or_array)
    is_datetime = np.issubdtype(dtype_or_array, np.datetime64)
    is_timedelta = np.issubdtype(dtype_or_array, np.timedelta64)
    if is_timedelta or is_datetime:
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
