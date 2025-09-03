"""Tests for time variables."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from dascore.compat import random_state
from dascore.exceptions import TimeError
from dascore.utils.time import (
    get_max_min_times,
    is_datetime64,
    is_timedelta64,
    to_datetime64,
    to_float,
    to_int,
    to_timedelta64,
)


class Dummy:
    """A dummy class for testing dispatching."""

    pass


class TestToDateTime64:
    """Tests for converting things to datetime64."""

    date_strs = ("1970-01-01", "2020-01-03T05:22:11.123123345", "2017-09-18T01")

    def test_float_array(self):
        """Ensure basic tests work."""
        date_strs = ["2015-01-01", "2020-03-01T21:10:10", "1970-01-02"]
        input_datetime64 = np.array(date_strs, dtype="datetime64[ns]")
        # convert to float (ns) then divide by e9 to get float in seconds
        float_array = input_datetime64.astype(np.float64) / 1_000_000_000
        out = to_datetime64(float_array)
        assert np.all(input_datetime64 == out)

    def test_single_float(self):
        """Ensure a single float can be converted to datetime."""
        out = to_datetime64(1.0)
        assert isinstance(out, np.datetime64)
        assert out == np.datetime64("1970-01-01T00:00:01.00000", "ns")

    def test_string(self):
        """Test for converting a string to a datetime object."""
        for time_str in self.date_strs:
            out = to_datetime64(time_str)
            assert isinstance(out, np.datetime64)
            assert time_str in str(out)

    def test_str_array(self):
        """Tests for converting an array of strings."""
        out = to_datetime64(self.date_strs)
        assert len(out) == len(self.date_strs)
        for el, datestr in zip(out, self.date_strs):
            assert datestr in str(el)

    def test_datetime64_array(self):
        """Tests for inputting datetime64."""
        array = to_datetime64(self.date_strs)
        out = to_datetime64(array)
        for el, datestr in zip(out, self.date_strs):
            assert datestr in str(el)

    def test_datetime64(self):
        """A datetitme64 should remain thus and equal."""
        d_time = to_datetime64("2020-01-01")
        out = to_datetime64(d_time)
        assert d_time == out

    def test_none(self):
        """None should return NaT."""
        out = to_datetime64(None)
        assert pd.isnull(out)

    def test_pandas_timestamp(self):
        """Ensure a timestamp returns the datetime64."""
        ts = pd.Timestamp("2020-01-03")
        expected = ts.to_datetime64()
        out = to_datetime64(ts)
        assert out == expected

    def test_non_ns_datetime64(self):
        """Tests that a non-nano second datatime gets converted to one."""
        datetimes = [
            np.datetime64("2011-01-01", "s"),
            np.datetime64("2011-01-01", "ms"),
            np.datetime64("2011-01-01", "ns"),
        ]
        expected = np.datetime64("2011-01-01", "ns")
        for dt in datetimes:
            out = to_datetime64(dt)
            assert out == expected
            # check string rep to ensure precision matches
            assert str(out) == str(expected)

    def test_array_with_all_nan(self):
        """Tests for NaN in array."""
        array = np.array([None, None])
        out = to_datetime64(array)
        assert pd.isnull(out).all()

    def test_array_with_one_nan(self):
        """Tests for NaN in array."""
        array = np.array([None, 1.2])
        out = to_datetime64(array)
        assert pd.isnull(out[0]) and not pd.isnull(out[1])

    def test_one_nan_one_timestamp(self):
        """Ensure (None, TimeStamp) works."""
        array = (None, pd.Timestamp("2020-01-01"))
        out = to_datetime64(array)
        assert len(out) == 2
        assert pd.isnull(out[0])
        assert out[1] == array[1]

    def test_str_tuples(self):
        """Ensure tuples of datetime strings can also be converted."""
        out1 = to_datetime64((None, "2011-01-01"))
        out2 = to_datetime64(("2011-01-01", None))
        assert pd.isnull(out2[1]) and pd.isnull(out1[0])
        assert out1[1] == out2[0]

    def test_zero_dimensional_array(self):
        """Ensure a zero-dimensional array of a datetime is just unpacked."""
        ar = np.array("2011-08-12", dtype="datetime64[ns]")
        out = to_datetime64(ar)
        assert isinstance(out, np.datetime64)

    def test_negative_float(self):
        """Negative floats should be a symmetric operation."""
        floats_to_test = [0.1, 10.0, 10.001, 100.0, 0, np.inf]
        for val in floats_to_test:
            out = to_datetime64(-val)
            expected = to_datetime64(0) - to_timedelta64(val)
            assert out == expected or (pd.isnull(out) and pd.isnull(expected))

    def test_negative_int(self):
        """Negative ints should be a symmetric operation."""
        floats_to_test = [0, 10, 100]
        for val in floats_to_test:
            out = to_datetime64(-val)
            expected = to_datetime64(0) - to_timedelta64(val)
            assert out == expected or (pd.isnull(out) and pd.isnull(expected))

    def test_datetime_non_ns_array(self):
        """Non-ns datetime arrays should be converted to ns precision."""
        ar = np.atleast_1d(np.datetime64("2012-01-01"))
        out = to_datetime64(ar)
        assert out.dtype == np.dtype("<M8[ns]")

    def test_series(self):
        """Ensure a series od datatime64 works."""
        ser = pd.Series(to_datetime64(["2020-01-12", "2024-01-02"]))
        out = to_datetime64(ser)
        assert out.equals(ser)

    def test_datetime(self):
        """Ensure datatime works."""
        dt = datetime.fromisoformat("2021-01-02")
        out = to_datetime64(dt)
        assert isinstance(out, np.datetime64)
        assert out == to_datetime64("2021-01-02")

    def test_unsupported_type(self):
        """Ensure unsupported types raise."""
        with pytest.raises(NotImplementedError):
            to_datetime64(Dummy())


class TestToTimeDelta64:
    """Tests for creating timedeltas."""

    @pytest.fixture(
        params=(
            np.timedelta64(1, "ns"),
            np.timedelta64(10, "s"),
            np.timedelta64(63, "ms"),
        )
    )
    def timedelta64(self, request):
        """Return the parametrized timedeltas."""
        return request.param

    def test_single_float(self):
        """Ensure a single float is converted to timedelta."""
        out = to_timedelta64(1.0)
        assert out == np.timedelta64(1_000_000_000, "ns")

    def test_float_array(self):
        """Ensure an array of flaots can be converted to ns timedelta."""
        ar = [1.0, 0.000000001, 0.001]
        expected = np.array([1 * 10**9, 1, 1 * 10**6], "timedelta64")
        out = to_timedelta64(ar)
        assert np.all(out == expected)

    def test_timedelta64_array(self):
        """Ensure passing timedelta array works."""
        expected = np.array([1 * 10**9, 1, 1 * 10**6], "timedelta64[s]").astype(
            "timedelta64[ns]"
        )
        out = to_timedelta64(expected)
        assert np.equal(out, expected).all()

    def test_timedetla64(self):
        """Test for passing a time delta."""
        td = to_timedelta64(123)
        out = np.timedelta64(123, "s")
        out2 = to_timedelta64(out)
        assert out == td == out2

    def test_pandas_time_delta(self):
        """Ensure pandas timedelta still works."""
        expected = np.timedelta64(1, "s")
        ptd = pd.Timedelta(1, "s")
        out = to_timedelta64(ptd)
        assert expected == out

    def test_str_roundtrip(self, timedelta64):
        """Ensure the output of str(timedelta64) is valid input."""
        obj_str = str(timedelta64)
        assert timedelta64 == to_timedelta64(obj_str)

    def test_array_with_all_nan(self):
        """Tests for NaN in array."""
        array = np.array([None, None])
        out = to_timedelta64(array)
        assert pd.isnull(out).all()

    def test_array_with_one_nan(self):
        """Tests for NaN in array."""
        array = np.array([None, 1.2])
        out = to_timedelta64(array)
        assert pd.isnull(out[0]) and not pd.isnull(out[1])

    def test_zero_dimensional_array(self):
        """A degenerate array should be unpacked."""
        array1 = np.array("1", dtype="timedelta64[s]")
        assert isinstance(to_timedelta64(array1), np.timedelta64)
        array2 = np.array(1, dtype="timedelta64[s]")
        assert isinstance(to_timedelta64(array2), np.timedelta64)

    def test_negative_float(self):
        """Negative floats should be a symmetric operation."""
        floats_to_test = [0.1, 10.0, 10.001, 100.0, 0, np.inf]
        for val in floats_to_test:
            out = to_timedelta64(-val)
            expected = -to_timedelta64(abs(val))
            assert out == expected or (pd.isnull(out) and pd.isnull(expected))

    def test_negative_int(self):
        """Negative ints should be a symmetric operation."""
        floats_to_test = [1, 10, 100]
        for val in floats_to_test:
            out = to_timedelta64(-val)
            expected = -to_timedelta64(abs(val))
            assert out == expected or (pd.isnull(out) and pd.isnull(expected))

    def test_timedelta(self):
        """Ensure python timedelta can convert to numpy timedelta64."""
        td = timedelta(hours=1)
        out = to_timedelta64(td)
        assert isinstance(out, np.timedelta64)
        assert out == to_timedelta64(3600)

    def test_unsupported_type(self):
        """Ensure unsupported types raise."""
        with pytest.raises(NotImplementedError):
            to_timedelta64(Dummy())

    def test_bad_str(self):
        """Raises if an un-parsable string is passed."""
        msg = "Could not convert"
        with pytest.raises(TimeError, match=msg):
            to_timedelta64("a bad string")
        with pytest.raises(TimeError, match=msg):
            to_timedelta64("abadstring")

    def test_nat(self):
        """Ensure we can initiate NaT."""
        out1 = to_timedelta64("NaT")
        out2 = to_timedelta64("nat")
        assert pd.isnull(out1)
        assert pd.isnull(out2)

    def test_nat_array(self):
        """Ensure an array of NaT works."""
        ar = np.array([to_timedelta64("NaT")] * 4)
        out = to_timedelta64(ar)
        assert np.all(pd.isnull(out))

    def test_array_of_datetimes(self, random_patch):
        """Ensure datetime64 array can be converted to timedelta array."""
        dt_array = random_patch.get_coord("time").values
        out = to_timedelta64(dt_array)
        assert np.all(out.astype(np.int64) == dt_array.astype(np.int64))


class TestToInt:
    """Tests for converting time-like types to ints, or passing through reals."""

    def test_timedelta64(self):
        """Ensure a timedelta64 returns the ns."""
        out = to_int(np.timedelta64(1, "s"))
        assert out == 1_000_000_000

    def test_datetime64(self):
        """Ensure int ns is returned for datetime64."""
        out = to_int(to_datetime64("1970-01-01") + np.timedelta64(1, "ns"))
        assert out == 1

    def test_timedelta64_array(self):
        """Ensure int ns is returned for datetime64."""
        array = to_datetime64(["2017-01-01", "1970-01-01", "1999-01-01"])
        out = to_int(array)
        assert np.issubdtype(out.dtype, np.int64)

    def test_timedelta_array(self):
        """Ensure a timedelta array works."""
        array = to_timedelta64([1, 1_000_000, 20])
        out = to_int(array)
        assert np.issubdtype(out.dtype, np.int64)

    def test_nullish_returns_nan(self):
        """Ensure a timedelta array works."""
        assert to_int(None) is np.nan
        assert to_int(pd.NaT) is np.nan

    def test_converted_to_int(self):
        """Ensure a number is converted to int."""
        assert to_int(10) == 10
        assert to_int(10.1) == 10

    def test_numeric_array_unchanged(self):
        """Ensure numeric arrays are not changed."""
        array = np.array([10, 12, 20])
        assert np.all(to_int(array) == array)
        array = np.array([12.3, 13.2, 12.2])
        assert np.all(to_int(array) == array)

    def test_non_ns_datetime64(self):
        """Tests that a non-nano second datatime gets converted to one."""
        datetimes = [
            np.datetime64("2011-01-01", "s"),
            np.datetime64("2011-01-01", "ms"),
            np.datetime64("2011-01-01", "ns"),
        ]
        expected = np.datetime64("2011-01-01", "ns").astype(np.int64)
        for dt in datetimes:
            out = to_int(dt)
            assert out == expected

    def test_empty_array(self):
        """Ensure an empty array comes out the other end."""
        ar = np.array([])
        out = to_int(ar)
        assert len(out) == 0

    def test_unsupported_type(self):
        """Ensure unsupported types raise."""
        with pytest.raises(NotImplementedError):
            to_int(Dummy())

    def test_empy_dt_array(self):
        """Ensure an empty datatime array gets converted to int."""
        array = np.empty(0, dtype="datetime64[ns]")
        out = to_int(array)
        assert np.issubdtype(out.dtype, np.integer)


class TestIsDateTime:
    """Ensure is_datetime64 detects datetimes."""

    def test_not_datetime(self):
        """Simple tests for things that aren't datetimes."""
        assert not is_datetime64(None)
        assert not is_datetime64(float)
        assert not is_datetime64(10)
        assert not is_datetime64(42.12)
        assert not is_datetime64(np.timedelta64(10, "s"))

    def test_is_datetime(self):
        """Things that should return True."""
        assert is_datetime64(np.datetime64("1970-01-01"))
        array = to_datetime64(["1990-01-01", "2010-01-01T12:23:22"])
        assert is_datetime64(array)

    def test_datetime_series(
        self,
    ):
        """is_datettime should work with a pandas series."""
        array = to_datetime64(["1990-01-01", "2010-01-01T12:23:22"])
        ser = pd.Series(array)
        assert is_datetime64(ser)

    def test_dtype(self):
        """Giving the function a numpy datatype should also work."""
        d1 = np.array([1.0, 2.0]).dtype
        d2 = np.array([1, 2]).astype("datetime64[ms]").dtype
        assert not is_datetime64(d1)
        assert is_datetime64(d2)


class TestToFloat:
    """Tests for converting datetime(ish) things to floats."""

    def test_float(self):
        """Ensure a single float gets converted to float."""
        assert to_float(1.0) == 1.0
        assert to_float(5) == 5.0

    def test_numerical_array(self):
        """Tests for numerical arrays."""
        ar = random_state.random(10)
        assert np.allclose(to_float(ar), ar)
        assert np.issubdtype(ar.dtype, np.float64)
        ar = np.ones(10)
        assert np.allclose(ar, 1.0)
        assert np.issubdtype(ar.dtype, np.float64)

    def test_timedelta(self):
        """Ensure time delta is floated."""
        td = to_timedelta64(100.00)
        assert np.isclose(to_float(td), 100.00)

    def test_timedelta_array(self):
        """Tests for arrays of time deltas."""
        td = to_timedelta64(np.ones(10))
        out = to_float(td)
        assert np.issubdtype(out.dtype, np.float64)
        assert np.allclose(out, 1.0)

    def test_datetime(self):
        """Ensure datetimes work."""
        dt = to_datetime64("2012-01-01")
        out = to_float(dt)
        expected = (dt - to_datetime64("1970-01-01")) / to_timedelta64(1)
        assert np.isclose(out, expected)

    def test_datetime_array(self):
        """Tests for arrays of date times."""
        dt = to_datetime64(np.ones(10))
        out = to_float(dt)
        assert np.issubdtype(out.dtype, np.float64)
        assert np.allclose(out, 1.0)

    def test_none(self):
        """Ensure None returns NaN."""
        out = to_float(None)
        assert out is np.nan

    def test_empty_array(self):
        """Empty arrays should work too."""
        ar = np.array([])
        out = to_float(ar)
        assert len(out) == 0
        assert np.issubdtype(out.dtype, np.float64)

    def test_series(self):
        """Ensure a series works."""
        ser1 = pd.Series([1, 2, 3])
        ser2 = pd.Series([to_datetime64(10), to_datetime64(1_000_000.12)])
        out1 = to_float(ser1)
        out2 = to_float(ser2)
        assert isinstance(out1, pd.Series)
        assert isinstance(out2, pd.Series)


class TestIsTimeDelta:
    """Test suite for determining time deltas."""

    def test_simple_td(self):
        """A single time delta should be true."""
        td = to_timedelta64(10)
        assert is_timedelta64(td)

    def test_timedelta_array(self):
        """Time-delta arrays should also be true."""
        td = to_timedelta64(np.array([1, 2, 3]))
        assert is_timedelta64(td)

    def datetimes_false(self):
        """Datetimes are not timedeltas :)."""
        dt = to_datetime64("2020-01-02")
        assert not is_timedelta64(dt)

    def test_dtype(self):
        """Giving the function a numpy datatype should also work."""
        d1 = np.array([1.0, 2.0]).dtype
        d2 = np.array([1, 2]).astype("timedelta64[ms]").dtype
        assert not is_timedelta64(d1)
        assert is_timedelta64(d2)


class TestGetmaxMinTimes:
    """Tests for max_min fetching."""

    def test_raises_bad_value(self):
        """Simple test to make sure error is raised if unordered tuple."""
        t1 = to_datetime64("2020-01-01")
        t2 = to_datetime64("1994-01-01")
        with pytest.raises(ValueError):
            get_max_min_times((t1, t2))
