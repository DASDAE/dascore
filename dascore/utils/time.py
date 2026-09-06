"""Utility for working with time."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from functools import singledispatch
from typing import Any, SupportsFloat, cast, overload

import numpy as np
import pandas as pd
import pint
from pint import DimensionalityError

from dascore.constants import (
    NUMPY_TIME_UNIT_MAPPING,
    ONE_SECOND,
    timeable_types,
)
from dascore.exceptions import TimeError, UnitError

_NAT_DATETIME64 = np.datetime64("NaT", "ns")
_NAT_TIMEDELTA64 = np.timedelta64("NaT", "ns")
_EPOCH_DATETIME64 = np.datetime64(0, "ns")


def _float_array_to_ns(array):
    """Convert seconds as floats to signed integer nanoseconds."""
    # Integer inputs must be widened first; the default integer is only
    # 32 bits on some platforms (e.g. wasm32) and the multiply overflows.
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64) * 1_000_000_000
    return np.rint(array * 1_000_000_000).astype(np.int64)


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
        return _NAT_DATETIME64
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
    # Scalar fast path; the array path costs ~10x more for single values.
    num = float(num)
    if not math.isfinite(num):  # matches array path: NaN/inf -> NaT
        return _NAT_DATETIME64
    return np.datetime64(round(num * 1_000_000_000), "ns")


@to_datetime64.register(np.ndarray)
@to_datetime64.register(list)
@to_datetime64.register(tuple)
def _array_to_datetime64(array: np.ndarray) -> np.datetime64 | np.ndarray:
    """Convert an array of floating point timestamps to an array of np.datetime64."""
    array = np.asarray(array)
    # 0-D arrays cannot be indexed or iterated, which the branches below do;
    # use the length-one array it stands for and unpack the scalar at the end.
    degenerate = array.ndim == 0
    if degenerate:
        array = array.reshape(1)
    nans = pd.isnull(array)
    # dealing with objects
    if np.issubdtype(array.dtype, np.dtype(object)):
        array = np.asarray([to_datetime64(x) for x in array]).astype("datetime64[ns]")
    # dealing with a string
    if np.issubdtype(array.dtype, np.dtype(str)):
        array = array.astype("datetime64[ns]")
    # dealing with an array of datetime64 or empty array
    if np.issubdtype(array.dtype, np.datetime64) or len(array) == 0:
        out = array.astype("datetime64[ns]")
    # dealing with numerical data
    elif np.issubdtype(array.dtype, np.timedelta64) or np.isreal(array[0]):
        with np.errstate(divide="ignore", invalid="ignore"):
            array = to_float(np.array(array))  # need to make copy to write
            nans = nans | ~np.isfinite(array)
            array[nans] = 0  # temporary replace NaNs
            out = _float_array_to_ns(array).astype("datetime64[ns]")
        # fill NaN Back in
        out[nans] = _NAT_DATETIME64
    return out[0] if degenerate else out


@to_datetime64.register(pd.Series)
def _float_to_datetime(ser: pd.Series) -> pd.Series:
    """Convert a float to a single datetime."""
    ar = to_datetime64(ser.values)
    return pd.Series(ar, index=ser.index)


@to_datetime64.register(pd.arrays.StringArray)
@to_datetime64.register(pd.arrays.ArrowStringArray)
def _string_array_to_datetime64(arr: pd.arrays.StringArray):
    """
    Convert a pandas string array to datetime64.

    Both backings, since which one pandas gives text is not the caller's
    choice: a `str` column is arrow-backed wherever pyarrow is installed
    and numpy-backed where it is not. Neither class is a subclass of the
    other -- they share only `BaseStringArray` -- so registering one does
    not dispatch the other.
    """
    out = pd.to_datetime(arr, errors="coerce", format="mixed")
    return out.to_numpy(dtype="datetime64[ns]")


@to_datetime64.register(np.datetime64)
def _pass_datetime(datetime):
    """Simply return the datetime."""
    return np.datetime64(datetime, "ns")


@to_datetime64.register(datetime)
def _datetime_to_datetime64(dt: datetime):
    """Convert python datetime to datetime64."""
    # because pandas NaT has datetime in its MRO we need to check
    # if this is nullish and return NaT if so.
    if pd.isnull(dt):
        return _NAT_DATETIME64
    return to_datetime64(np.datetime64(dt, "ns"))


@to_datetime64.register(date)
def _date_to_datetime64(value: date):
    """
    Convert a python date to the instant it starts.

    YAML reads an unquoted ``2024-06-01`` as a date rather than a string,
    so this is the ordinary spelling of a day in a hand-authored file, not
    an exotic input. Registered after datetime, which is a subclass of
    date and keeps its own more specific handler.
    """
    out = np.datetime64(value.isoformat(), "ns")
    # A datetime64 spans about 1678 to 2262 and wraps silently past either
    # end, so a day outside it would come back as one centuries away. Read
    # back as text rather than through datetime64[D], which is itself
    # unreliable at the boundary: 1677-09-22 is representable but converts
    # to 2262-04-11. The other spellings of a time still wrap; see #890.
    if not str(out).startswith(value.isoformat()):
        msg = (
            f"Date {value.isoformat()} is outside the range a nanosecond "
            f"timestamp can represent; it would read as {out}."
        )
        raise ValueError(msg)
    return out


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
        An object to convert to timedelta64. Can be a float, str or array of
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
    >>> td_2 = dc.to_timedelta64(-10.5)
    >>>
    >>> # Convert a string to timedelta64
    >>> td_str = "1000000000 nanoseconds"
    >>> td_3 = dc.to_timedelta64(td_str)

    """
    if pd.isnull(obj):
        return _NAT_TIMEDELTA64
    msg = f"type {type(obj)} is not supported"
    raise NotImplementedError(msg)


@to_timedelta64.register(float)
@to_timedelta64.register(int)
@to_timedelta64.register(np.number)
def _float_to_timedelta64(num: float | int) -> np.timedelta64:
    """Convert a number of seconds to a single timedelta64."""
    # Scalar fast path; the array path costs ~10x more for single values.
    num = float(num)
    if not math.isfinite(num):  # matches array path: NaN/inf -> NaT
        return _NAT_TIMEDELTA64
    return np.timedelta64(round(num * 1_000_000_000), "ns")


@to_timedelta64.register(np.timedelta64)
def _pass_time_delta(time_delta):
    """Simply return the time delta as ns precision."""
    return time_delta.astype("<m8[ns]")


@to_timedelta64.register(np.ndarray)
@to_timedelta64.register(list)
@to_timedelta64.register(tuple)
def _array_to_timedelta64(array: np.ndarray) -> np.timedelta64 | np.ndarray:
    """Convert an array of floating point durations to np.timedelta64."""
    array = np.asarray(array)
    # See the note in _array_to_datetime64.
    degenerate = array.ndim == 0
    if degenerate:
        array = array.reshape(1)
    # convert pure object arrays into float so sign casting works.
    if np.issubdtype(array.dtype, np.dtype(object)):
        array = array.astype(np.float64)
    if np.issubdtype(array.dtype, np.timedelta64) or len(array) == 0:
        out = array.astype("timedelta64[ns]")
    # A datetime becomes its offset from the epoch. The unit has to be
    # normalized first, or viewing e.g. datetime64[s] as int64 would label
    # its second count as nanoseconds.
    elif np.issubdtype(array.dtype, np.datetime64):
        out = array.astype("datetime64[ns]").view("timedelta64[ns]")
    else:
        assert np.isreal(array[0])
        invalid = pd.isnull(array) | ~np.isfinite(array)
        # Need to make copy to 1) not change original array and 2) handle
        # immutable arrays. See #575.
        if np.any(invalid):
            array = np.array(array)
            array[invalid] = 0
        # inf/NaN complain, salience these types of warnings for this block.
        with np.errstate(divide="ignore", invalid="ignore"):
            out = _float_array_to_ns(array).astype("timedelta64[ns]")
            out[invalid] = _NAT_TIMEDELTA64
    return out[0] if degenerate else out


@to_timedelta64.register(pd.Series)
def _series_to_timedelta64_series(ser: pd.Series) -> pd.Series:
    """Convert a series to a series of timedelta64."""
    out = to_timedelta64(ser.values)
    return pd.Series(out, index=ser.index)


@to_timedelta64.register(pd.arrays.StringArray)
@to_timedelta64.register(pd.arrays.ArrowStringArray)
def _string_array_to_timedelta64(arr: pd.arrays.StringArray):
    """Convert a pandas string array, of either backing, to timedelta64."""
    out = pd.to_timedelta(arr, errors="coerce")
    return out.to_numpy(dtype="timedelta64[ns]")


@to_timedelta64.register(pd.Timedelta)
def _unpack_pandas_time_delta(time_delta: pd.Timedelta):
    """Simply return the time delta."""
    return time_delta.to_numpy()


@to_timedelta64.register(timedelta)
def _timedelta_to_timedelta64(td):
    """Return timedelta64."""
    return to_timedelta64(np.timedelta64(td, "ns"))


@to_timedelta64.register(str)
def _time_delta_from_str(time_delta_str: str):
    """Simply return the time delta."""
    match time_delta_str.split():
        # Can split string into (hopefully) units and values. Standard case.
        case [val, units]:
            if units[-1] == "s":
                units = units[:-1]
            new_unit = NUMPY_TIME_UNIT_MAPPING[units]
            return np.timedelta64(int(val), new_unit)
        case [val] if val.lower() == "nat" or val.lower() == "":
            return _NAT_TIMEDELTA64
        case _:
            msg = f"Could not convert {time_delta_str} to timedelta64"
            raise TimeError(msg)


# The public to_int/to_float below are thin wrappers over these dispatchers
# so their signatures can be overloaded. Stacking @overload directly on a
# singledispatch is not an option: the name then refers to the overload set
# and loses .register.
@singledispatch
def _to_int(obj: timeable_types | np.ndarray) -> Any:
    """Dispatch implementation for to_int."""
    msg = f"type {type(obj)} is not supported"
    raise NotImplementedError(msg)


@_to_int.register(float)
@_to_int.register(int)
@_to_int.register(np.number)
def _float_to_num(num: float | int) -> float | int:
    """Convert number to int."""
    return int(num)


@_to_int.register(np.ndarray)
@_to_int.register(list)
@_to_int.register(tuple)
def _array_to_int(array: np.ndarray) -> np.ndarray:
    """Convert an array of possible dates to int64 nanoseconds."""
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


@_to_int.register(np.datetime64)
@_to_int.register(datetime)
@_to_int.register(pd.Timestamp)
def _time_to_int(datetime):
    """Simply return the datetime converted to ns."""
    return to_int([to_datetime64(datetime)])[0]


@_to_int.register(type(None))
@_to_int.register(type(pd.NaT))
@_to_int.register(type(pd.NA))
def _return_number_null(null):
    """Convert non to NaT."""
    return np.nan


@_to_int.register(np.timedelta64)
def _time_delta_to_number(time_delta: np.timedelta64):
    return to_int([to_timedelta64(time_delta)])[0]


@_to_int.register(pd.Series)
def _pandas_timestamp_to_num(ser: pd.Series):
    return ser.astype(np.int64)


@overload
def to_int(obj: pd.Series) -> pd.Series: ...


@overload
def to_int(obj: np.ndarray | list | tuple) -> np.ndarray: ...


@overload
def to_int(obj: Any) -> int | np.integer | float: ...


def to_int(obj) -> pd.Series | np.ndarray | int | np.integer | float:
    """
    Ensure a scalar or array is a number.

    If the input values represents a time or a time-delta, convert it to a
    an int representing ns.

    A Series stays a Series and any other sequence becomes an array. Other
    inputs come back as a scalar, but not always an int: time-like values
    convert through numpy and yield np.int64, and null yields NaN.
    """
    return _to_int(obj)


@singledispatch
def _to_float(obj: timeable_types | np.ndarray) -> Any:
    """Dispatch implementation for to_float."""
    # Every time type is registered below, so only things float() already
    # understands reach this fallback.
    return float(cast("SupportsFloat", obj))


@_to_float.register(pint.Quantity)
def _quantity_to_float(quant: pint.Quantity) -> float | np.ndarray:
    """
    Convert a time quantity to seconds.

    Anything else raises: this function's output is a duration in
    seconds, so there is no meaningful float for a length or a data
    size. Without this, pint's `__float__` would silently convert any
    *dimensionless* quantity to its base units — returning 2e8 for
    `25 * MB`, whose base unit is the bit — while rejecting the time
    quantities this function is actually for.
    """
    try:
        # recurse so an array-valued quantity takes the array path and a
        # scalar one is always widened to float (a magnitude may be int)
        return to_float(quant.to("s").magnitude)
    except DimensionalityError:
        msg = (
            f"Cannot convert {quant} to a float; only time quantities "
            "have a float representation here (seconds). Convert "
            "explicitly instead, eg dascore.units.convert_units or, for "
            "data sizes, dascore.units.get_byte_count."
        )
        raise UnitError(msg) from None


@_to_float.register(np.ndarray)
@_to_float.register(list)
@_to_float.register(tuple)
def _array_to_float(array: np.ndarray) -> np.ndarray:
    """Convert an array of possible dates to floats (seconds)."""
    array = np.asarray(array)
    if not len(array):
        return array.astype(np.float64)
    if np.issubdtype(array.dtype, np.datetime64):
        # convert to offset from 1970
        array = array - _EPOCH_DATETIME64
    if np.issubdtype(array.dtype, np.timedelta64):
        array = array / ONE_SECOND
    return array.astype(np.float64)


@_to_float.register(pd.Series)
def _series_to_float(series: pd.Series) -> pd.Series:
    """Convert a series of possible dates to floats."""
    array = to_float(series.values)
    return pd.Series(array, index=series.index)


@_to_float.register(np.datetime64)
@_to_float.register(datetime)
@_to_float.register(pd.Timestamp)
def _time_to_float(datetime):
    """Simply return the datetime."""
    td = to_datetime64(datetime) - _EPOCH_DATETIME64
    return to_float(td)


@_to_float.register(type(None))
@_to_float.register(type(pd.NaT))
@_to_float.register(type(pd.NA))
def _return_null(null):
    """Convert non to NaT."""
    return np.nan


@_to_float.register(np.timedelta64)
@_to_float.register(timedelta)
@_to_float.register(pd.Timedelta)
def _time_delta_to_float(time_delta: np.timedelta64):
    return to_timedelta64(time_delta) / ONE_SECOND


@overload
def to_float(obj: pd.Series) -> pd.Series: ...


@overload
def to_float(obj: np.ndarray | list | tuple) -> np.ndarray: ...


@overload
def to_float(obj: pint.Quantity) -> float | np.ndarray: ...


@overload
def to_float(obj: Any) -> float: ...


def to_float(obj) -> pd.Series | np.ndarray | float:
    """
    Convert various datetime/timedelta things to a float.

    Time offsets represent seconds, and datetimes are seconds from 1970.
    A pint quantity of time is converted to seconds as well; any other
    quantity raises [`UnitError`](`dascore.exceptions.UnitError`), since
    a length or a data size has no float representation here.

    A Series stays a Series and any other sequence becomes an array; every
    other input, null included, comes back as a scalar. A quantity follows
    its own magnitude, so an array-valued one yields an array.
    """
    return _to_float(obj)


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
