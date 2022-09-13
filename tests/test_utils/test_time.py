"""
Tests for time variables.
"""
import functools

import numpy as np
import pandas as pd
import pytest

from dascore.utils.time import (
    get_select_time,
    is_datetime64,
    to_datetime64,
    to_number,
    to_timedelta64,
)


class TestToDateTime64:
    """Tests for converting things to datetime64."""

    date_strs = ["1970-01-01", "2020-01-03T05:22:11.123123345", "2017-09-18T01"]

    def test_float_array(self):
        """Ensure basic tests work"""
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
        """Test for converting a string to a datetime object"""
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
        """
        Ensure a zero-dimensional array of a datetime is just unpacked.
        """
        ar = np.array("2011-08-12", dtype="datetime64[ns]")
        out = to_datetime64(ar)
        assert isinstance(out, np.datetime64)


class TestToTimeDelta:
    """Tests for creating timedeltas"""

    @pytest.fixture(
        params=(
            np.timedelta64(1, "ns"),
            np.timedelta64(10, "s"),
            np.timedelta64(63, "ms"),
        )
    )
    def timedelta64(self, request):
        """Return the parametrized timedeltas"""
        return request.param

    def test_single_float(self):
        """Ensure a single float is converted to timedelta"""
        out = to_timedelta64(1.0)
        assert out == np.timedelta64(1_000_000_000, "ns")

    def test_float_array(self):
        """Ensure an array of flaots can be converted to ns timedelta"""
        ar = [1.0, 0.000000001, 0.001]
        expected = np.array([1 * 10**9, 1, 1 * 10**6], "timedelta64")
        out = to_timedelta64(ar)
        assert np.all(out == expected)

    def test_timedelta64_array(self):
        """Ensure passing timedelta array works."""
        expected = np.array([1 * 10**9, 1, 1 * 10**6], "timedelta64")
        out = to_timedelta64(expected)
        assert np.equal(out, expected).all()

    def test_timedetla64(self):
        """Test for passing a time delta."""
        td = to_timedelta64(123)
        out = np.timedelta64(123, "s")
        out2 = to_timedelta64(out)
        assert out == td == out2

    def test_pandas_time_delta(self):
        """Ensure pandas timedelta still works"""
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
        array = np.array("1", dtype="timedelta64[s]")
        out = to_timedelta64(array)
        assert isinstance(out, np.timedelta64)


class TestGetSelectTime:
    """Tests for getting select time(s)."""

    t1 = np.datetime64("2020-01-03")
    t2 = np.datetime64("2020-01-04")
    func = functools.partial(get_select_time, time_min=t1, time_max=t2)

    def test_datetime64(self):
        """Test for returning a datetime 64."""
        assert get_select_time(self.t1) == self.t1
        assert get_select_time(self.t2) == self.t2
        time = self.t1 + np.timedelta64(100, "s")
        assert get_select_time(time) == time

    def test_string(self):
        """Tests for passing time as a string"""
        tstr = "2020-01-03T01"
        out = self.func(tstr)
        assert out == np.datetime64(tstr)

    def test_positive_float_int(self):
        """Test float/ints are interpreted as relative"""
        out = self.func(1)
        expected = self.t1 + to_timedelta64(1)
        assert out == expected

        out = self.func(1.0)
        expected = self.t1 + to_timedelta64(1)
        assert out == expected

    def test_negative_float_int(self):
        """Test float/ints are interpreted as relative"""
        out = self.func(-1)
        expected = self.t2 - to_timedelta64(1)
        assert out == expected

        out = self.func(-1.0)
        expected = self.t2 - to_timedelta64(1)
        assert out == expected


class TestToNumber:
    """Tests for converting time-like types to ints, or passing through reals"""

    def test_timedelta64(self):
        """Ensure a timedelta64 returns the ns"""
        out = to_number(np.timedelta64(1, "s"))
        assert out == 1_000_000_000

    def test_datetime64(self):
        """Ensure int ns is returned for datetime64."""
        out = to_number(to_datetime64("1970-01-01") + np.timedelta64(1, "ns"))
        assert out == 1

    def test_timedelta64_array(self):
        """Ensure int ns is returned for datetime64."""
        array = to_datetime64(["2017-01-01", "1970-01-01", "1999-01-01"])
        out = to_number(array)
        assert np.issubdtype(out.dtype, np.int64)

    def test_timedelta_array(self):
        """Ensure a timedelta array works."""
        array = to_timedelta64([1, 1_000_000, 20])
        out = to_number(array)
        assert np.issubdtype(out.dtype, np.int64)

    def test_nullish_returns_nan(self):
        """Ensure a timedelta array works."""
        assert to_number(None) is np.NaN
        assert to_number(pd.NaT) is np.NaN

    def test_number_unchanged(self):
        """Ensure a number is passed through unchanged."""
        assert to_number(10) == 10
        assert to_number(10.1) == 10.1

    def test_numeric_array_unchanged(self):
        """Ensure numeric arrays are not changed."""
        array = np.array([10, 12, 20])
        assert np.all(to_number(array) == array)
        array = np.array([12.3, 13.2, 12.2])
        assert np.all(to_number(array) == array)

    def test_non_ns_datetime64(self):
        """Tests that a non-nano second datatime gets converted to one."""
        datetimes = [
            np.datetime64("2011-01-01", "s"),
            np.datetime64("2011-01-01", "ms"),
            np.datetime64("2011-01-01", "ns"),
        ]
        expected = np.datetime64("2011-01-01", "ns").astype(np.int64)
        for dt in datetimes:
            out = to_number(dt)
            assert out == expected


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
        """Things that should return True"""
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
