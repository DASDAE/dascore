"""
Tests for time variables.
"""
import numpy as np

from fios.utils.time import to_datetime64, to_timedelta64


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

    def test_datetime64(self):
        """Tests for inputting datetime64."""
        array = to_datetime64(self.date_strs)
        out = to_datetime64(array)
        for el, datestr in zip(out, self.date_strs):
            assert datestr in str(el)

    def test_datetime64(self):
        """A datetitme64 should remain thus and equal."""
        dt = to_datetime64("2020-01-01")
        out = to_datetime64(dt)
        assert dt == out


class TestToTimeDelta:
    """Tests for creating timedeltas"""

    def test_single_float(self):
        """Ensure a single float is converted to timedelta"""
        out = to_timedelta64(1.0)
        assert out == np.timedelta64(1_000_000_000, "ns")

    def test_float_array(self):
        """Ensure an array of flaots can be converted to ns timedelta"""
        ar = [1.0, 0.000000001, 0.001]
        expected = np.array([1 * 10 ** 9, 1, 1 * 10 ** 6], "timedelta64")
        out = to_timedelta64(ar)
        assert np.all(out == expected)

    def test_timedelta64_array(self):
        """Ensure passing timedelta array works."""
        expected = np.array([1 * 10 ** 9, 1, 1 * 10 ** 6], "timedelta64")
        out = to_timedelta64(expected)
        assert np.equal(out, expected).all()

    def test_timedetla64(self):
        """Test for passing a time delta."""
        td = to_timedelta64(123)
        out = np.timedelta64(123, "s")
        out2 = to_timedelta64(out)
        assert out == td == out2
