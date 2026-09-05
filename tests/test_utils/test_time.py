"""Tests for time variables."""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.compat import random_state
from dascore.exceptions import TimeError, UnitError

try:
    import pyarrow
except ImportError:
    pyarrow = None
from dascore.utils.time import (
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

    def test_float_array_preserves_ns_offsets(self):
        """Float datetimes should preserve representable nanosecond offsets."""
        expected = np.array(["2026-06-03T15:18:13.422442752"], dtype="datetime64[ns]")
        float_array = expected.astype(np.float64) / 1_000_000_000
        out = to_datetime64(float_array)
        assert np.all(expected == out)

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
        """A datetime64 should remain thus and equal."""
        d_time = to_datetime64("2020-01-01")
        out = to_datetime64(d_time)
        assert d_time == out

    def test_none(self):
        """None should return NaT."""
        out = to_datetime64(None)
        assert pd.isnull(out)

    def test_pandas_nat(self):
        """Pandas NaT should return numpy NaT."""
        out = to_datetime64(pd.NaT)
        assert pd.isnull(out)

    def test_pandas_timestamp(self):
        """Ensure a timestamp returns the datetime64."""
        ts = pd.Timestamp("2020-01-03")
        expected = ts.to_datetime64()
        out = to_datetime64(ts)
        assert out == expected

    def test_non_ns_datetime64(self):
        """Tests that a non-nano second datetime gets converted to one."""
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
        """Ensure a series of datetime64 works."""
        ser = pd.Series(to_datetime64(["2020-01-12", "2024-01-02"]))
        out = to_datetime64(ser)
        assert out.equals(ser)

    def test_datetime(self):
        """Ensure datetime works."""
        dt = datetime.fromisoformat("2021-01-02")
        out = to_datetime64(dt)
        assert isinstance(out, np.datetime64)
        assert out == to_datetime64("2021-01-02")

    def test_date(self):
        """Ensure a date works, being the instant its day starts."""
        out = to_datetime64(date.fromisoformat("2021-01-02"))
        assert isinstance(out, np.datetime64)
        assert out == to_datetime64("2021-01-02")

    def test_date_does_not_shadow_datetime(self):
        """A datetime is a date, so it must keep its own handler."""
        stamp = datetime.fromisoformat("2021-01-02T03:04:05")
        assert to_datetime64(stamp) == to_datetime64("2021-01-02T03:04:05")

    @pytest.mark.parametrize("iso", ["2500-01-01", "1000-01-01"])
    def test_date_outside_the_representable_range(self, iso):
        """A datetime64 wraps silently, so such a date is refused."""
        with pytest.raises(ValueError, match="outside the range"):
            to_datetime64(date.fromisoformat(iso))

    @pytest.mark.parametrize("iso", ["2262-04-11", "1677-09-22"])
    def test_the_outermost_representable_days(self, iso):
        """The check refuses what wraps without refusing what does not."""
        assert to_datetime64(date.fromisoformat(iso)) == to_datetime64(iso)

    def test_unsupported_type(self):
        """Ensure unsupported types raise."""
        with pytest.raises(NotImplementedError):
            to_datetime64(Dummy())

    def test_immutable_inputs(self):
        """Ensure immutable array inputs work."""
        # See #575.
        array_no_nan = random_state.random(100)
        array_nan = random_state.random(100)
        array_nan[10] = np.nan

        for array in [array_no_nan, array_nan]:
            array.setflags(write=False)
            time = to_datetime64(array_no_nan)
            assert np.issubdtype(time.dtype, "M8")

    def test_timedelta_array(self):
        """Ensure timedelta arrays can be converted to datetime64."""
        td = to_timedelta64(random_state.random(100))
        out = to_datetime64(td)
        assert np.issubdtype(out.dtype, "M8")

    def test_pandas_string_array(self):
        """Ensure pandas StringArray converts to datetime64[ns]."""
        arr = pd.array(self.date_strs, dtype="string")
        out = to_datetime64(arr)
        expected = np.array([dc.to_datetime64(x) for x in self.date_strs]).astype(
            "datetime64[ns]"
        )
        assert np.all(out == expected)

    @pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")
    def test_arrow_backed_string_array(self):
        """Which backing pandas gives text is not the caller's choice."""
        arr = pd.array(self.date_strs, dtype="string[pyarrow]")
        out = to_datetime64(arr)
        expected = np.array([dc.to_datetime64(x) for x in self.date_strs]).astype(
            "datetime64[ns]"
        )
        assert np.all(out == expected)


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
        """Ensure an array of floats can be converted to ns timedelta."""
        ar = [1.0, 0.000000001, 0.001]
        expected = np.array([1 * 10**9, 1, 1 * 10**6], "timedelta64[ns]")
        out = to_timedelta64(ar)
        assert np.all(out == expected)

    def test_timedelta64_array(self):
        """Ensure passing timedelta array works."""
        expected = np.array([1 * 10**9, 1, 1 * 10**6], "timedelta64[s]").astype(
            "timedelta64[ns]"
        )
        out = to_timedelta64(expected)
        assert np.equal(out, expected).all()

    def test_timedelta64(self):
        """Test for passing a time delta."""
        td = to_timedelta64(123)
        out = np.timedelta64(123, "s")
        out2 = to_timedelta64(out)
        assert out == td == out2

    def test_np_timedelta64_normalizes_to_ns(self, timedelta64):
        """Ensure numpy timedelta64 input is normalized to ns precision."""
        out = to_timedelta64(timedelta64)
        assert out.dtype == np.dtype("timedelta64[ns]")

    def test_pandas_time_delta(self):
        """Ensure pandas timedelta still works."""
        expected = np.timedelta64(1, "s")
        pandas_time_delta = pd.Timedelta(1, "s")
        out = to_timedelta64(pandas_time_delta)
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

    def test_negative_example_uses_timedelta_function(self):
        """The negative-number example should demonstrate to_timedelta64."""
        assert "dc.to_timedelta64(-10.5)" in to_timedelta64.__doc__

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

    def test_series(self):
        """A Series converts to timedeltas without losing its index."""
        ser = pd.Series([1.0, 2.0], index=["first", "second"])
        out = to_timedelta64(ser)
        expected = pd.Series(to_timedelta64(ser.values), index=ser.index)
        pd.testing.assert_series_equal(out, expected)

    def test_pandas_string_array(self):
        """Ensure pandas StringArray converts to timedelta64[ns]."""
        arr = pd.array(["1s", "2s", None], dtype="string")
        out = to_timedelta64(arr)
        expected = np.array(
            [
                np.timedelta64(1, "s"),
                np.timedelta64(2, "s"),
                np.timedelta64("NaT", "ns"),
            ]
        ).astype("timedelta64[ns]")
        assert np.all(out[:2] == expected[:2])
        assert pd.isnull(out[2])

    @pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")
    def test_arrow_backed_string_array(self):
        """A string column is arrow-backed wherever pyarrow is installed."""
        arr = pd.array(["1s", "2s"], dtype="string[pyarrow]")
        out = to_timedelta64(arr)
        assert np.all(out == np.array([1, 2]).astype("timedelta64[s]"))

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

    def test_none_nat_no_warnings(self):
        """None should return a typed NaT without warnings."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = to_timedelta64(None)
        assert pd.isnull(out)
        assert not record
        assert out.dtype == np.dtype("timedelta64[ns]")

    def test_nat_array(self):
        """Ensure an array of NaT works."""
        ar = np.array([to_timedelta64("NaT")] * 4)
        out = to_timedelta64(ar)
        assert np.all(pd.isnull(out))

    @pytest.mark.filterwarnings(
        "error:The 'generic' unit for NumPy timedelta is deprecated:DeprecationWarning"
    )
    def test_null_values_use_explicit_timedelta_unit(self):
        """Null-like values should not create generic-unit timedeltas."""
        assert pd.isnull(to_timedelta64(None))
        assert pd.isnull(to_timedelta64("NaT"))

        out = to_timedelta64([None, 1.0])

        assert out.dtype == np.dtype("timedelta64[ns]")
        assert pd.isnull(out[0])
        assert out[1] == np.timedelta64(1, "s")

    def test_non_finite_values_are_nat(self):
        """Infinite and NaN values should not overflow during conversion."""
        out = to_timedelta64([np.inf, -np.inf, np.nan, 1.0])

        assert out.dtype == np.dtype("timedelta64[ns]")
        assert np.all(pd.isnull(out[:3]))
        assert out[3] == np.timedelta64(1, "s")

    def test_array_of_datetimes(self, random_patch):
        """Ensure datetime64 array can be converted to timedelta array."""
        dt_array = random_patch.get_coord("time").values
        out = to_timedelta64(dt_array)
        assert np.all(out.astype(np.int64) == dt_array.astype(np.int64))

    def test_immutable_inputs(self):
        """Ensure immutable array inputs work."""
        # See #575.
        array_no_nan = random_state.random(100)
        array_nan = random_state.random(100)
        array_nan[10] = np.nan

        for array in [array_no_nan, array_nan]:
            array.setflags(write=False)
            time = to_timedelta64(array_no_nan)
            assert np.issubdtype(time.dtype, "m8")

    @pytest.mark.parametrize("unit", ("D", "s", "ms", "ns"))
    def test_datetime_array_is_epoch_offset(self, unit):
        """A datetime becomes its offset from the epoch, whatever its unit."""
        value = np.datetime64("2020-01-01", unit)
        expected = np.timedelta64(1577836800, "s")
        assert to_timedelta64(np.array([value])) == expected
        assert to_timedelta64(np.array(value)) == expected


class TestNanosecondBounds:
    """Tests for times which the nanosecond representation cannot hold."""

    out_of_range = ("2500-01-01", "1000-01-01")

    @pytest.mark.parametrize("iso", out_of_range)
    def test_string(self, iso):
        """A string outside the range used to come back centuries away."""
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(iso)

    @pytest.mark.parametrize("iso", out_of_range)
    def test_array_of_strings(self, iso):
        """The array path wrapped as silently as the scalar one."""
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(np.array([iso]))

    @pytest.mark.parametrize("iso", out_of_range)
    def test_object_array(self, iso):
        """An object array converts element wise and must check each one."""
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(np.array([iso], dtype=object))

    def test_only_one_bad_value_is_needed(self):
        """A single unrepresentable element rejects the whole array."""
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(np.array(["2020-01-01", "2500-01-01"]))

    def test_seconds_from_epoch(self):
        """Seconds from the epoch overflowed with an opaque OverflowError."""
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(2e10)

    def test_array_of_seconds(self):
        """The array of seconds saturated at the maximum instead of raising."""
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(np.array([2e10]))

    def test_timedelta_seconds(self):
        """Durations have the same bound as times."""
        with pytest.raises(TimeError, match="outside the range"):
            to_timedelta64(1e12)

    def test_array_of_timedelta_seconds(self):
        """The duration array saturated at the maximum instead of raising."""
        with pytest.raises(TimeError, match="outside the range"):
            to_timedelta64(np.array([1e12]))

    @pytest.mark.parametrize("iso", ["2262-04-11", "1677-09-22"])
    def test_outermost_days_still_convert(self, iso):
        """The check must not refuse what the representation can hold."""
        assert to_datetime64(iso) == np.datetime64(iso, "ns")

    def test_null_values_are_untouched(self):
        """Nulls are not out of range and still become NaT."""
        out = to_datetime64(np.array([np.nan, 1.0]))
        assert np.isnat(out[0])
        assert out[1] == to_datetime64(1.0)

    def test_empty_array(self):
        """An empty array carries no value to check."""
        assert len(to_datetime64(np.array([]))) == 0
        assert len(to_timedelta64(np.array([]))) == 0

    def test_message_names_the_value(self):
        """The error has to say which time was refused."""
        with pytest.raises(TimeError, match="2500"):
            to_datetime64(np.array(["2020-01-01", "2500-01-01"]))

    def test_error_is_also_an_overflow_error(self):
        """Handlers written for the numpy error must keep catching it."""
        with pytest.raises(OverflowError):
            to_datetime64("2500-01-01")
        with pytest.raises(OverflowError):
            to_timedelta64(1e12)

    # Strings with more than six fractional digits are parsed by numpy
    # straight into nanoseconds, where an out of range value wraps silently.
    nanosecond_strings = (
        "2500-01-01T00:00:00.000000000",
        "1000-01-01T00:00:00.000000000",
        "2262-04-11T23:47:16.854775808",  # one ns past the last valid one
        "1677-09-21T00:12:43.145224192",  # lands on the NaT sentinel
        "1677-09-21T00:12:43.145224191",
    )

    @pytest.mark.parametrize("iso", nanosecond_strings)
    def test_string_with_nanoseconds(self, iso):
        """A string numpy parses straight into ns wrapped before any check."""
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(iso)

    @pytest.mark.parametrize("iso", nanosecond_strings)
    def test_array_of_strings_with_nanoseconds(self, iso):
        """The array path parses the same way and needs the same check."""
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(np.array(["2020-01-01T00:00:00.000000001", iso]))

    @pytest.mark.parametrize(
        "iso", ["2262-04-11T23:47:16.854775807", "1677-09-21T00:12:43.145224193"]
    )
    def test_outermost_nanoseconds_still_convert(self, iso):
        """The last valid nanosecond on each side is not out of range."""
        expected = np.datetime64(iso, "ns")
        assert to_datetime64(iso) == expected
        assert to_datetime64(np.array([iso]))[0] == expected

    def test_nat_string_next_to_nanoseconds(self):
        """A NaT string is still NaT, not mistaken for a wrapped value."""
        out = to_datetime64(np.array(["NaT", "2020-01-01T00:00:00.123456789"]))
        assert np.isnat(out[0])
        assert out[1] == np.datetime64("2020-01-01T00:00:00.123456789", "ns")

    @pytest.mark.parametrize("sign", [1, -1])
    def test_seconds_at_the_float_limit(self, sign):
        """The float limit itself rounds to 2**63 and does not fit."""
        limit = sign * np.iinfo(np.int64).max / 1_000_000_000
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(np.array([limit]))
        with pytest.raises(TimeError, match="outside the range"):
            to_timedelta64(np.array([limit]))

    def test_seconds_just_inside_the_float_limit(self):
        """The largest float below the limit still converts."""
        limit = np.nextafter(np.iinfo(np.int64).max / 1_000_000_000, -np.inf)
        assert not np.isnat(to_datetime64(np.array([limit]))[0])

    @pytest.mark.parametrize("seconds", [9_223_372_037, -9_223_372_037])
    def test_whole_seconds_past_the_limit(self, seconds):
        """Integer seconds are bounded on both sides, not by magnitude."""
        with pytest.raises(TimeError, match="outside the range"):
            to_timedelta64(np.array([seconds]))

    @pytest.mark.parametrize("seconds", [9_223_372_036, -9_223_372_036])
    def test_outermost_whole_seconds_still_convert(self, seconds):
        """The last whole second on each side fits."""
        out = to_timedelta64(np.array([seconds]))
        assert out[0] == np.timedelta64(seconds, "s")

    def test_most_negative_integer(self):
        """np.abs of the lowest int64 is itself, so it must not slip through."""
        with pytest.raises(TimeError, match="outside the range"):
            to_timedelta64(np.array([np.iinfo(np.int64).min]))

    def test_timedelta_array_in_coarse_unit(self):
        """A temporal array narrows through the same check as a datetime one."""
        with pytest.raises(TimeError, match="outside the range"):
            to_timedelta64(np.array([1000], dtype="timedelta64[Y]"))
        with pytest.raises(TimeError, match="outside the range"):
            to_timedelta64(np.array(["2500-01-01"], dtype="datetime64[D]"))

    def test_datetime64_scalar(self):
        """A scalar in a coarse unit wrapped when rebuilt in nanoseconds."""
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(np.datetime64("2500-01-01", "D"))
        assert to_datetime64(np.datetime64("2020-01-01", "D")) == np.datetime64(
            "2020-01-01", "ns"
        )

    def test_timedelta64_scalar(self):
        """A scalar duration in a coarse unit overflowed with a numpy error."""
        with pytest.raises(TimeError, match="outside the range"):
            to_timedelta64(np.timedelta64(1000, "Y"))

    def test_python_datetime(self):
        """A datetime past the range wrapped when built in nanoseconds."""
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(datetime(2500, 1, 1))
        dt = datetime(2020, 1, 1, 12, 0, 0, 123456)
        assert to_datetime64(dt) == np.datetime64(dt, "ns")

    def test_python_timedelta(self):
        """A timedelta spans far more than nanoseconds can count."""
        with pytest.raises(TimeError, match="outside the range"):
            to_timedelta64(timedelta(days=200_000))
        td = timedelta(days=1, microseconds=1)
        assert to_timedelta64(td) == np.timedelta64(td, "ns")

    def test_pandas_timestamp(self):
        """A timestamp in a coarser unit is still narrowed and checked."""
        with pytest.raises(TimeError, match="outside the range"):
            to_datetime64(pd.Timestamp("2500-01-01"))
        out = to_datetime64(pd.Timestamp("2020-01-01 00:00:00.123456789"))
        assert out == np.datetime64("2020-01-01T00:00:00.123456789", "ns")
        assert out.dtype == np.dtype("datetime64[ns]")


class TestDegenerateArrays:
    """Tests for 0-D array inputs to the array converters."""

    values = (
        np.datetime64("2020-01-01", "s"),
        np.timedelta64(5, "s"),
        5.0,
        5,
    )

    @pytest.mark.parametrize("func", (to_datetime64, to_timedelta64))
    @pytest.mark.parametrize("value", values)
    def test_matches_length_one_array(self, func, value):
        """A 0-D array converts like the length-one array it stands for."""
        out = func(np.array(value))
        assert np.shape(out) == ()
        assert out == func(np.array([value]))[0]

    @pytest.mark.parametrize("func", (to_datetime64, to_timedelta64))
    def test_nan_becomes_nat(self, func):
        """A 0-D NaN converts to NaT rather than an epoch value."""
        out = func(np.array(np.nan))
        assert np.shape(out) == ()
        assert pd.isnull(out)


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

    def test_series(self):
        """A datetime Series converts to integer ns and preserves its index."""
        ser = pd.Series(to_datetime64(["1970-01-01", "2000-01-01"]))
        ser.index = ["first", "second"]
        out = to_int(ser)
        pd.testing.assert_series_equal(out, ser.astype(np.int64))

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
        """Tests that a non-nano second datetime gets converted to one."""
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

    def test_empty_dt_array(self):
        """Ensure an empty datetime array gets converted to int."""
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
        """is_datetime should work with a pandas series."""
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

    def test_unregistered_float_able(self):
        """Anything float() understands still reaches the fallback."""
        assert to_float(Decimal("3")) == 3.0
        assert to_float("1.5") == 1.5

    def test_timestamp_uses_registration(self):
        """Timestamp must dispatch, not fall through to float()."""
        # float(pd.Timestamp(...)) raises, so reaching the fallback would
        # turn this into a TypeError rather than seconds from the epoch.
        stamp = pd.Timestamp("1970-01-02")
        assert to_float(stamp) == pytest.approx(86400.0)
        with pytest.raises(TypeError):
            float(stamp)

    def test_container_return_types(self):
        """A Series stays a Series; other sequences become arrays."""
        assert isinstance(to_float(pd.Series([1.0, 2.0])), pd.Series)
        for seq in ([1.0, 2.0], (1.0, 2.0), np.arange(2.0)):
            assert isinstance(to_float(seq), np.ndarray)

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

    def test_time_quantity(self):
        """A time quantity converts to its duration in seconds."""
        assert to_float(dc.get_quantity("2 s")) == 2.0
        assert to_float(dc.get_quantity("2 min")) == 120.0
        assert to_float(dc.get_quantity("500 ms")) == 0.5

    def test_time_quantity_array(self):
        """An array-valued time quantity keeps its shape."""
        quant = np.array([1.0, 2.0]) * dc.get_quantity("min")
        out = to_float(quant)
        assert np.allclose(out, [60.0, 120.0])

    def test_time_quantity_returns_float(self):
        """An integer magnitude is still widened to float."""
        assert isinstance(to_float(dc.get_quantity("2 s")), float)

    @pytest.mark.parametrize("value", ("10 m", "25 MB", "50%", "1 strain", "5", "1 Hz"))
    def test_non_time_quantity_raises(self, value):
        """Only time quantities have a float representation."""
        with pytest.raises(UnitError, match="only time quantities"):
            to_float(dc.get_quantity(value))

    def test_data_size_is_not_silently_converted(self):
        """
        Guard the bits trap.

        pint's `__float__` converts a dimensionless quantity to base
        units, and information's base unit is the bit, so bytes used to
        come back eight times too large instead of raising.
        """
        with pytest.raises(UnitError):
            to_float(dc.get_quantity("25 MB"))


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
