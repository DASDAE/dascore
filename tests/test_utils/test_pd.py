"""Tests for pandas utility functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
import pytest

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.utils.pd import (
    adjust_segments,
    dataframe_to_patch,
    fill_defaults_from_pydantic,
    filter_df,
    get_interval_columns,
    patch_to_dataframe,
)
from dascore.utils.time import to_datetime64, to_timedelta64


@pytest.fixture()
def random_df_from_patch(random_patch):
    """A dataframe created from random patch."""
    return patch_to_dataframe(random_patch)


@pytest.fixture
def example_df_2():
    """Create a simple df for testing. Example from Chris Albon."""
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
        "time_step": [np.timedelta64(1, "s") for _ in range(5)],
    }
    out = pd.DataFrame(raw_data, columns=list(raw_data))
    return out


@pytest.fixture
def example_df_timedeltas(example_df_2):
    """An example dataframe with timedelta columns."""
    time = to_timedelta64(10)
    time_min = [time + x * np.timedelta64(1, "s") for x in range(5)]
    time_max = time_min + np.timedelta64(10, "m")

    out = example_df_2.assign(time_min=time_min, time_max=time_max)
    return out


class TestFilterDfBasic:
    """Tests for filtering dataframes."""

    @pytest.fixture
    def example_df(self):
        """Create a simple df for testing. Example from Chris Albon."""
        raw_data = {
            "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
            "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
            "age": [42, 52, 36, 24, 73],
            "pre_test_score": [4, 24, 31, 2, 3],
            "post_test_score": [25, 94, 57, 62, 70],
        }
        return pd.DataFrame(raw_data, columns=list(raw_data))

    def test_string_basic(self, example_df):
        """Test that specifying a string with no matching works."""
        out = filter_df(example_df, first_name="Jason")
        assert out[0]
        assert not out[1:].any()

    def test_string_matching(self, example_df):
        """Unix style matching should also work."""
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
        """Test that filter index can be used on Non-nslc columns."""
        # test non strings
        out = filter_df(example_df, age=42)
        assert out[0]
        assert not out[1:].any()

    def test_non_str_sequence(self, example_df):
        """Ensure sequences still work for isin style comparisons."""
        out = filter_df(example_df, age={42, 52})
        assert out[:2].all()
        assert not out[2:].any()

    def test_bad_parameter_raises(self, example_df):
        """Ensure passing a parameter that doesn't have a column raises."""
        with pytest.raises(ParameterError, match="the column does not"):
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
        """Test column names that end with max."""
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
        """Test for open time interval."""
        out = filter_df(example_df_2, time=(None, None))
        assert out.all()

    def test_time_query_one_open(self, example_df_2):
        """Test for open time interval."""
        tmax = to_datetime64(example_df_2["time_max"].max() - np.timedelta64(1, "ns"))
        out = filter_df(example_df_2, time=(tmax, None))
        # just the last row should have been selected
        assert out.iloc[-1] and out.astype(np.int64).sum() == 1

    def test_time_query_with_string(self, example_df_2):
        """Test for time query with a string."""
        tmax = to_datetime64(example_df_2["time_max"].max() - np.timedelta64(1, "ns"))
        out1 = filter_df(example_df_2, time=(str(tmax), None))
        # just the last row should have been selected
        assert out1.iloc[-1] and out1.astype(np.int64).sum() == 1
        # also ensure the string can be second element, all rows should be selected
        out2 = filter_df(example_df_2, time=(None, str(tmax)))
        assert len(out2) == len(example_df_2)

    def test_empty_filter(self, example_df_2):
        """Empty filter kwargs, for convenience, should be ok."""
        out = filter_df(example_df_2, time=None)
        assert np.all(out)

    def test_timedelta_columns(self, example_df_timedeltas):
        """Ensure timedelta columns work when specifying ranges of single col."""
        df = example_df_timedeltas
        out = filter_df(df, time_step_min=0.5, time_step_max=2)
        assert out.all()


class TestAdjustSegments:
    """Tests for adjusting segments of dataframes."""

    @pytest.fixture()
    def adjacent_df(self):
        """Create a ddtaframe with adjacent times."""
        time_mins = [
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-01T00:00:10.01"),
        ]
        time_maxs = [
            np.datetime64("2020-01-01T00:00:10.01"),
            np.datetime64("2020-01-01T00:00:20"),
        ]
        dtime = to_timedelta64(0.01)
        df = (
            pd.DataFrame(index=range(2))
            .assign(time_min=time_mins, time_max=time_maxs, time_step=dtime)
            .assign(distance_min=0, distance_max=100, distance_step=1, station="BOB")
        )
        return df

    def test_limits_changed(self, example_df_2):
        """Ensure limits of segments are changed when intersected by query."""
        df = example_df_2
        new_min = df["time_min"].min() + np.timedelta64(1, "s")
        new_max = df["time_max"].max() - np.timedelta64(1, "s")
        new = adjust_segments(df, time=(new_min, new_max))
        assert np.all(new["time_min"] >= new_min)
        assert np.all(new["time_max"] <= new_max)

    def test_modified_tracked(self, example_df_2):
        """Ensure The modified flag gets added to start/end."""
        df = example_df_2
        new_min = df["time_min"].min() + np.timedelta64(1, "s")
        new_max = df["time_max"].max() - np.timedelta64(1, "s")
        new = adjust_segments(df, time=(new_min, new_max))
        assert new.iloc[0]["_modified"]
        assert new.iloc[-1]["_modified"]
        assert not np.any(new.iloc[1:-1]["_modified"])

    def test_multiple_kwargs(self, adjacent_df):
        """Ensure multiple dimensions can be adjusted with one function call."""
        time = [
            adjacent_df["time_max"].min() - to_timedelta64(5),
            adjacent_df["time_max"].max() - to_timedelta64(5),
        ]
        distance = (30, 50)
        out = adjust_segments(adjacent_df, time=time, distance=distance)
        assert len(adjacent_df) == len(out)
        assert (out["time_min"] >= time[0]).all()
        assert (out["time_max"] <= time[1]).all()
        assert (out["distance_min"] >= distance[0]).all()
        assert (out["distance_max"] <= distance[1]).all()

    def test_missing_interval_col_raises_keyerro(self, adjacent_df):
        """Ensure if an interval column is missing a KeyError is raised."""
        df = adjacent_df.drop(columns=["distance_min"])
        with pytest.raises(ParameterError):
            _ = adjust_segments(df, distance=(100, 200))


class TestFillDefaultsFromPydantic:
    """Tests for initing empty columns from pydanitc models."""

    class Model(pydantic.BaseModel):
        """Example basemodel."""

        int_with_default: int = 10
        str_with_default: str = "bob"
        float_with_default: float = 10.0
        float_no_default: float

    @pytest.fixture(scope="class")
    def simple_df(self):
        """Create a simple df for testing."""
        df = pd.DataFrame(index=range(10)).assign(
            int_with_default=20,
            str_with_default="bill",
            float_with_default=22.2,
            float_no_default=42.0,
        )
        return df

    def test_no_missing_does_nothing(self, simple_df):
        """Ensure the default gets filled."""
        out = fill_defaults_from_pydantic(simple_df, self.Model)
        assert out.equals(simple_df)

    def test_missing_with_default_fills_default(self, simple_df):
        """Ensure the default values are filled in when missing."""
        df = simple_df.drop(columns="int_with_default")
        out = fill_defaults_from_pydantic(df, self.Model)
        assert "int_with_default" in out.columns
        assert (out["int_with_default"] == 10).all()

    def test_missing_with_no_default_raises(self, simple_df):
        """Ensure Value error is raised if no default is there."""
        df = simple_df.drop(columns="float_no_default")
        with pytest.raises(ValueError, match="required value"):
            fill_defaults_from_pydantic(df, self.Model)


class TestPatchToDF:
    """Test conversion of patch to pandas.DataFrame."""

    def test_simple_to_df(self, random_patch):
        """Another helpful docstring."""
        df = patch_to_dataframe(random_patch)
        # assert the data match original data, the column/index ...
        # values match coordinate value
        assert np.all(np.equal(df.values, random_patch.data))

        assert df.index.name == random_patch.dims[0]
        assert df.columns.name == random_patch.dims[1]
        # also assert attrs match patch attrs after dumping
        assert df.attrs == random_patch.attrs.model_dump()


class TestDFtoPatch:
    """Test conversion of pandas.DataFrame to patch."""

    @pytest.fixture()
    def random_df_no_dims(self, random_df_from_patch):
        """Get the dataframe from patch but remove dim names."""
        new = pd.DataFrame(random_df_from_patch)
        new.index.name = None
        new.columns.name = ""
        return new

    def test_simple_to_patch(self, random_df_from_patch):
        """Test conversion from pandas.DataFrame to patch, simplest case."""
        df = random_df_from_patch
        patch = dataframe_to_patch(df)
        # first check dimensions came through
        dims = (df.index.name, df.columns.name)
        assert patch.dims == dims
        # and the data match original data
        assert np.all(np.equal(df.values, patch.data))
        # and coordinates match index
        assert np.all(patch.get_coord(dims[0]).values == df.index.values)
        assert np.all(patch.get_coord(dims[1]).values == df.columns.values)

    def test_attrs_as_dict(self, random_df_from_patch):
        """Test passing attributes as a dictionary get included in Patch attrs."""
        attrs = {"station": "Filler"}
        patch = dataframe_to_patch(random_df_from_patch, attrs=attrs)
        assert patch.attrs.station == attrs["station"]

    def test_attrs_as_model(self, random_df_from_patch):
        """Ensure passing attributes as a model get included in Patch."""
        attrs = dc.PatchAttrs(**{"station": "Filler"})
        patch = dataframe_to_patch(random_df_from_patch, attrs=attrs)
        assert patch.attrs.station == attrs.station

    def test_no_dims_on_df_raise(self, random_df_no_dims):
        """Ensure when no dims exist an Error is raised."""
        with pytest.raises(ValueError, match="Dimension names not found"):
            dataframe_to_patch(random_df_no_dims)

    def test_dims_on_attrs(self, random_df_no_dims):
        """Ensure dims can be passed on attrs."""
        attrs = {"dims": ("time", "distance")}
        patch = dataframe_to_patch(random_df_no_dims, attrs=attrs)
        assert patch.dims == ("time", "distance")


class TestGetIntervalColumns:
    """Tests for finding interval columns in a dataframe."""

    def test_simple(self, random_spool):
        """Ensure simple case works."""
        df = random_spool.get_contents()
        outs = get_interval_columns(df, "time")
        for ser in outs:
            assert isinstance(ser, pd.Series)
        assert len(outs) == 3

    def test_raises(self, example_df_2):
        """Ensure an error is raised if columns don't exist."""
        msg = "Cannot chunk spool or dataframe"
        with pytest.raises(ParameterError, match=msg):
            get_interval_columns(example_df_2, "money")
