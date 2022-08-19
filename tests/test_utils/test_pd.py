"""Tests for pandas utility functions."""
import numpy as np
import pandas as pd
import pytest

from dascore.utils.pd import adjust_segments, filter_df
from dascore.utils.time import to_datetime64


@pytest.fixture
def example_df_2():
    """create a simple df for testing. Example from Chris Albon."""
    time = to_datetime64("2020-01-03")
    time_min = [time + x * np.timedelta64(1, "s") for x in range(5)]
    time_max = time_min + np.timedelta64(10, "m")
    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
        "bp_min": [100, 120, 110, 85, 125],
        "bp_max": [110, 145, 121, 99, 165],
        "time_min": time_min,
        "time_max": time_max,
        "d_time": [np.timedelta64(1, "s") for _ in range(5)],
    }
    out = pd.DataFrame(raw_data, columns=list(raw_data))
    return out


class TestFilterDfBasic:
    """Tests for filtering dataframes."""

    @pytest.fixture
    def example_df(self):
        """create a simple df for testing. Example from Chris Albon."""
        raw_data = {
            "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
            "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
            "age": [42, 52, 36, 24, 73],
            "pre_test_score": [4, 24, 31, 2, 3],
            "post_test_score": [25, 94, 57, 62, 70],
        }
        return pd.DataFrame(raw_data, columns=list(raw_data))

    def test_string_basic(self, example_df):
        """test that specifying a string with no matching works."""
        out = filter_df(example_df, first_name="Jason")
        assert out[0]
        assert not out[1:].any()

    def test_string_matching(self, example_df):
        """unix style matching should also work."""
        # test *
        out = filter_df(example_df, first_name="J*")
        assert {"Jason", "Jake"} == set(example_df[out].first_name)
        # test ???
        out = filter_df(example_df, first_name="J???")
        assert {"Jake"} == set(example_df[out].first_name)

    def test_str_sequence(self, example_df):
        """Test str sequences find values in sequence."""
        out = filter_df(example_df, last_name={"Miller", "Jacobson"})
        assert out[:2].all()
        assert not out[2:].any()

    def test_non_str_single_arg(self, example_df):
        """test that filter index can be used on Non-nslc columns."""
        # test non strings
        out = filter_df(example_df, age=42)
        assert out[0]
        assert not out[1:].any()

    def test_non_str_sequence(self, example_df):
        """ensure sequences still work for isin style comparisons."""
        out = filter_df(example_df, age={42, 52})
        assert out[:2].all()
        assert not out[2:].any()

    def test_bad_parameter_raises(self, example_df):
        """ensure passing a parameter that doesn't have a column raises."""
        with pytest.raises(ValueError):
            filter_df(example_df, bad_column=2)

    def test_bad_parameter_doesnt_raise(self, example_df):
        """A bad parameter shouldn't raise if told filter is told to ignore."""
        out = filter_df(example_df, bad_column=2, ignore_bad_kwargs=True)
        # Each row should be kept.
        assert np.all(out)

    def test_min(self, example_df):
        """Tests for using min parameter."""
        df = example_df[filter_df(example_df, age_max=40)]
        assert df.equals(example_df[example_df["age"] < 40])

    def test_max(self, example_df):
        """Tests for using max parameter."""
        df = example_df[filter_df(example_df, post_test_score_min=60)]
        df2 = example_df[example_df["post_test_score"] >= 60]
        assert df.equals(df2)

    def test_min_and_max(self, example_df):
        """Ensure min/max can be used together."""
        ser = example_df["age"]
        con = (ser >= 20) & (ser <= 30)
        out = filter_df(example_df, age_min=20, age_max=30)
        assert all(con == out)


class TestFilterDfAdvanced:
    """Tests for advanced filtering of dataframes."""

    def test_col(self, example_df_2):
        """Test column names that end with max"""
        vals = [100, 125]
        out = filter_df(example_df_2, bp_min=vals)
        assert np.all(out == example_df_2["bp_min"].isin(vals))

    def test_min_max_range_based_on_column(self, example_df_2):
        """
        Using just bp, we should be able to specify range which spans
        two columns.
        """
        val = (100, 120)
        out = filter_df(example_df_2, bp=val)
        max_too_small = example_df_2["bp_max"] < val[0]
        min_too_big = example_df_2["bp_min"] > val[1]
        in_range = ~(max_too_small | min_too_big)
        assert all(out == in_range)

    def test_time_query_all_open(self, example_df_2):
        """Test for open time interval"""
        out = filter_df(example_df_2, time=(None, None))
        assert out.all()

    def test_time_query_one_open(self, example_df_2):
        """Test for open time interval"""
        tmax = to_datetime64(example_df_2["time_max"].max() - np.timedelta64(1, "ns"))
        out = filter_df(example_df_2, time=(tmax, None))
        # just the last row should have been selected
        assert out.iloc[-1] and out.astype(int).sum() == 1

    def test_time_query_with_string(self, example_df_2):
        """Test for time query with a string."""
        tmax = to_datetime64(example_df_2["time_max"].max() - np.timedelta64(1, "ns"))
        out1 = filter_df(example_df_2, time=(str(tmax), None))
        # just the last row should have been selected
        assert out1.iloc[-1] and out1.astype(int).sum() == 1
        # also ensure the string can be second element, all rows should be selected
        out2 = filter_df(example_df_2, time=(None, str(tmax)))
        assert len(out2) == len(example_df_2)


class TestAdjustSegments:
    """Tests for adjusting segments of dataframes."""

    def test_limits_changed(self, example_df_2):
        """Ensure limits of segments are changed when intersected by query."""
        df = example_df_2
        new_min = df["time_min"].min() + np.timedelta64(1, "s")
        new_max = df["time_max"].max() - np.timedelta64(1, "s")
        new = adjust_segments(df, time=(new_min, new_max))
        assert np.all(new["time_min"] >= new_min)
        assert np.all(new["time_max"] <= new_max)
