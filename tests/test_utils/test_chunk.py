"""Tests for chunking dataframes."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import ChunkError
from dascore.utils.chunk import get_intervals
from dascore.utils.chunk_plan import build_chunk_plan
from dascore.utils.time import to_timedelta64

STARTTIME = np.datetime64("2020-01-03")
ENDTIME = STARTTIME + np.timedelta64(60, "s")


@pytest.fixture()
def contiguous_df():
    """Create a contiguous dataframe with time and distance dimensions."""
    # get time, adjust starttime to be one time step after end time
    time = get_intervals(STARTTIME, ENDTIME, length=np.timedelta64(10, "s"))
    dt = np.timedelta64(10, "ms")
    df = pd.DataFrame(time, columns=["time_min", "time_max"])
    df["distance_min"], df["distance_max"] = 0, 10
    df["time_step"] = dt
    df["distance_step"] = 1
    return df


@pytest.fixture()
def contiguous_sr_spaced_df(contiguous_df):
    """Separate df by one sample rate."""
    sr = contiguous_df.loc[:, "time_step"]
    out = contiguous_df.copy()
    out["time_max"] = out["time_max"] - sr
    return out


@pytest.fixture()
def contiguous_df_two_stations(contiguous_df):
    """Create contiguous df with different stations."""
    # get time, adjust starttime to be one time step after end time
    df1 = contiguous_df.assign(station="sta1")
    df2 = contiguous_df.assign(station="sta2")
    return pd.concat([df1, df2], axis=0, ignore_index=True)


class TestGetIntervals:
    """Tests for generating intervals along some continuous dimension."""

    def test_numbers_no_overlap(self):
        """Ensure simple ints with no overlap work."""
        out = get_intervals(0, 10, 1)
        expected = np.stack([np.arange(10), np.arange(1, 11)]).T
        assert np.all(out == expected)

    def test_numbers_overlap(self):
        """Ensure numbers with start overlaps can also work."""
        start, stop, step = 0, 40, 10
        out = get_intervals(start, stop, step, overlap=1)
        expected = np.stack([np.array([0, 9, 18, 27]), np.array([10, 19, 28, 37])]).T

        # ensure the step size is the step specified for each interval
        assert np.allclose(out[:, 1] - out[:, 0], step)
        assert np.allclose(out, expected)

    def test_times_no_overlap(self):
        """Test using datetime64 and timedelta64."""
        start = np.datetime64("2017-01-03")
        step = np.timedelta64(1, "s")
        stop = start + 50 * step
        out = get_intervals(start, stop, step)
        assert np.all(out[:, 1] - out[:, 0] == step)
        assert out[-1, 1] == stop
        assert out[0, 0] == start

    def test_times_overlap(self):
        """Test time interval with overlap."""
        start = np.datetime64("2017-01-03")
        step = np.timedelta64(2, "s")
        overlap = np.timedelta64(1, "s")
        stop = start + 10 * step
        out = get_intervals(start, stop, step, overlap=overlap)
        assert np.all(out[:, 1] - out[:, 0] == step)
        assert out[-1, 1] == stop
        assert out[0, 0] == start

    def test_timedelta_start_numeric_length(self):
        """
        A numeric length for a timedelta64 start should be coerced to a
        duration, just like for datetime64 starts (see #553). Previously this
        raised a type error comparing a timedelta64 with a float.
        """
        start = np.timedelta64(0, "s")
        stop = start + np.timedelta64(20, "s")
        # A numeric (float) length previously raised here for timedelta starts.
        out = get_intervals(start, stop, length=2.0)
        # Output stays timedelta64 and each interval spans the requested length.
        assert np.issubdtype(out.dtype, np.timedelta64)
        assert np.all(out[:, 1] - out[:, 0] == np.timedelta64(2, "s"))
        assert out[0, 0] == start
        assert out[-1, 1] == stop


class TestChunkPlanDF:
    """Dataframe-level chunk planning (ported from the old ChunkManager tests)."""

    @pytest.fixture()
    def df_different_sample_rates(self, contiguous_df):
        """Adjacent blocks with different sampling rates."""
        df1 = contiguous_df.copy()
        df2 = contiguous_df.copy()
        time_span = df1["time_max"].max() - df1["time_min"].min()
        df2["time_min"] += time_span
        df2["time_max"] += time_span
        df2["time_step"] = df1["time_step"] * 2
        return pd.concat([df1, df2], axis=0).reset_index(drop=True)

    def test_rechunk_contiguous(self, contiguous_df):
        """Test rechunking with no gaps."""
        time_interval = (contiguous_df["time_max"] - contiguous_df["time_min"]).max()
        new_time_interval = time_interval / 2
        out = build_chunk_plan(contiguous_df, time=new_time_interval).outputs
        assert len(out) == 2 * len(contiguous_df)
        time_step = out["time_step"].iloc[0]
        new_interval = (out["time_max"] - out["time_min"] + time_step).max()
        assert new_interval == new_time_interval

    def test_rechunk_contiguous_with_sr_separation(self, contiguous_sr_spaced_df):
        """Ensure it still works on data separated by one sample."""
        df = contiguous_sr_spaced_df
        sr = df["time_step"]
        time_interval = (sr + df["time_max"] - df["time_min"]).max()
        new_time_interval = time_interval / 2
        out = build_chunk_plan(df, time=new_time_interval).outputs
        assert len(out) == 2 * len(df)
        new_interval = (out["time_max"] - out["time_min"]).max()
        assert new_interval == (new_time_interval - sr.iloc[0])

    def test_rechunk_different_sr(self, df_different_sample_rates):
        """Segments with different sample rates don't get combined."""
        df = df_different_sample_rates
        out = build_chunk_plan(df, time=23).outputs
        dt = np.sort(np.unique(out["time_step"]))
        assert len(dt) == 2, "both dt should remain"
        # the second part of the df should start at the one minute mark
        df2 = out[out["time_step"] == dt[1]]
        time_min = df2.iloc[0]["time_min"]
        assert time_min.minute == 1

    def test_chunk_uses_step_from_each_group(self):
        """Each sampling group should use its own step for interval ends."""
        df = pd.DataFrame(
            {
                "time_min": [
                    np.datetime64("2020-01-01T00:00:00"),
                    np.datetime64("2020-02-01T00:00:00"),
                ],
                "time_max": [
                    np.datetime64("2020-01-01T00:01:39"),
                    np.datetime64("2020-02-01T00:16:30"),
                ],
                "time_step": [np.timedelta64(1, "s"), np.timedelta64(10, "s")],
            }
        )
        chunked = build_chunk_plan(df, time=50).outputs
        ten_second_group = chunked[chunked["time_step"] == np.timedelta64(10, "s")]
        first = ten_second_group.iloc[0]
        assert first["time_max"] - first["time_min"] == np.timedelta64(40, "s")

    def test_keep_leftovers(self, contiguous_df):
        """Ensure leftovers show up in outputs."""
        out = build_chunk_plan(contiguous_df, keep_partial=True, time=28).outputs
        assert len(out) == 3
        assert out["time_max"].max() == contiguous_df["time_max"].max()

    def test_overlap(self, contiguous_df):
        """Ensure overlapping segments work, with timedelta or float overlap."""
        over = to_timedelta64(10)
        out = build_chunk_plan(contiguous_df, overlap=over, time=20).outputs
        expected = over - contiguous_df["time_step"].iloc[0]
        olap = out.shift()["time_max"] - out["time_min"]
        assert np.all(pd.isnull(olap) | (olap == expected))
        out2 = build_chunk_plan(contiguous_df, overlap=10, time=20).outputs
        assert out.equals(out2)

    def test_chunk_on_split(self, terra15_file_spool):
        """Ensure chunking which creates a slice at the end time works."""
        df = terra15_file_spool.get_contents()
        dur = (df["time_max"] - df["time_min"]).iloc[0]
        seg_len = dur / 3
        dt = df["time_step"].iloc[0]
        chunk_df = build_chunk_plan(df, keep_partial=True, time=seg_len).outputs
        duration = chunk_df["time_max"] - chunk_df["time_min"]
        assert duration.sum() == ((seg_len - dt) * 3)
        assert len(duration) == 3
        assert (duration > np.timedelta64(0, "s")).all()

    def test_nan_in_df(self, contiguous_df):
        """A null envelope row breaks continuity when dropped."""
        df = contiguous_df.copy()
        df.loc[3, "time_min"] = dc.to_datetime64("NaT")
        expected_start = df.loc[4, "time_min"]
        plan = build_chunk_plan(
            df, keep_partial=True, missing_dim="drop", time=dc.to_timedelta64(15)
        )
        assert expected_start in set(plan.outputs["time_min"])

    def test_all_nan(self, contiguous_df):
        """When all rows lack the dim (and are dropped) the plan is empty."""
        nat = dc.to_datetime64("NaT")
        df = contiguous_df.assign(time_min=nat, time_max=nat)
        plan = build_chunk_plan(df, missing_dim="drop", time=dc.to_timedelta64(1.2))
        assert plan.outputs.empty

    def test_nan_in_sample_ok(self, contiguous_df):
        """Ensure a NaN in the sampling rate is ok."""
        df = contiguous_df.assign(time_step=dc.to_timedelta64("NaT"))
        dur = (df["time_max"] - df["time_min"]).iloc[0]
        chunk_df = build_chunk_plan(df, time=dc.to_timedelta64(dur / 2)).outputs
        assert isinstance(chunk_df, pd.DataFrame)
        assert len(chunk_df) == 2 * len(contiguous_df)
        assert np.all(pd.isnull(chunk_df["time_step"]))

    def test_unknown_dim_raises(self, contiguous_df):
        """An unknown chunk dimension raises a clear error."""
        with pytest.raises(ChunkError, match="Time"):
            build_chunk_plan(contiguous_df, Time=10)


class TestChunkPlanToMerge:
    """Merge-mode planning on raw dataframes."""

    @pytest.fixture()
    def gapy_df(self, contiguous_df):
        """Create a dataframe with gaps."""
        df = contiguous_df.copy()
        df["time_max"] -= df["time_step"] * 15
        return df

    @pytest.fixture()
    def gapy_df_unordered(self, gapy_df):
        """Create a dataframe with gaps that is not sorted by starttime."""
        inds = np.random.RandomState(42).permutation(gapy_df.index)
        return gapy_df.loc[inds].reset_index(drop=True)

    def test_chunk_can_merge(self, contiguous_df):
        """Ensure chunk can be used to merge unspecified segment lengths."""
        out = build_chunk_plan(contiguous_df, time=None).outputs
        assert len(out) == 1
        assert out["time_min"].min() == contiguous_df["time_min"].min()

    def test_doesnt_merge_gappy_df(self, gapy_df):
        """Ensure the gappy dataframe doesn't get merged."""
        out = build_chunk_plan(gapy_df, time=None).outputs
        assert len(gapy_df) == len(out)
        expected = (gapy_df["time_max"] - gapy_df["time_min"]).sort_values()
        durations = (out["time_max"] - out["time_min"]).sort_values()
        assert np.array_equal(expected.values, durations.values)

    def test_doesnt_merge_unordered_gappy_df(self, gapy_df_unordered):
        """Row order must not affect merge results."""
        df = gapy_df_unordered
        out = build_chunk_plan(df, time=None).outputs
        assert len(df) == len(out)
        expected = (df["time_max"] - df["time_min"]).sort_values()
        durations = (out["time_max"] - out["time_min"]).sort_values()
        assert np.array_equal(expected.values, durations.values)

    def test_no_warning_when_final_groups_stay_separate(self, contiguous_df):
        """No warning if other group components prevent final forced merge."""
        df = contiguous_df.iloc[:2].copy()
        step = df["time_step"]
        df.loc[0, "time_max"] = df.loc[0, "time_min"] + 10 * step.iloc[0]
        df.loc[1, "time_min"] = df.loc[0, "time_max"] + 5 * step.iloc[0]
        df.loc[1, "time_max"] = df.loc[1, "time_min"] + 10 * step.iloc[1]
        df["station"] = ["sta1", "sta2"]
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            plan = build_chunk_plan(df, time=None, tolerance=10, group=("station",))
        assert len(plan.outputs) == 2

    def test_forced_merge_warns(self, contiguous_df):
        """A tolerance forcing a merge across a real gap warns (#662)."""
        df = contiguous_df.iloc[:2].copy()
        step = df["time_step"].iloc[0]
        df.loc[0, "time_max"] = df.loc[0, "time_min"] + 10 * step
        df.loc[1, "time_min"] = df.loc[0, "time_max"] + 5 * step
        df.loc[1, "time_max"] = df.loc[1, "time_min"] + 10 * step
        with pytest.warns(UserWarning, match="force merging"):
            plan = build_chunk_plan(df, time=None, tolerance=10)
        assert len(plan.outputs) == 1

    def test_modified_flag_after_merge(self, contiguous_df):
        """The modified flag shows False for a simple contiguous merge."""
        df = contiguous_df.assign(time_max=lambda x: x["time_max"] - x["time_step"])
        plan = build_chunk_plan(df, time=None)
        assert len(plan.outputs) == 1
        assert plan.outputs["time_min"].min() == df["time_min"].min()
        assert not plan.members["_modified"].any()


class TestPlanMembers:
    """Sanity checks on the members (instruction) table."""

    def test_ids(self, contiguous_df):
        """Members reference real sources and outputs."""
        plan = build_chunk_plan(contiguous_df, overlap=0, time=10)
        members = plan.members
        assert set(members["_patch_id"]).issubset(set(range(len(contiguous_df))))
        assert set(members["output_id"]).issubset(set(plan.outputs["output_id"]))

    def test_different_group_columns(self, contiguous_df_two_stations):
        """Ensure members honor differences in group columns."""
        df = contiguous_df_two_stations
        plan = build_chunk_plan(
            df, overlap=0, time=10, group=("station",), keep_partial=True
        )
        joined = plan.members.merge(
            df.assign(_patch_id=np.arange(len(df)))[["_patch_id", "station"]],
            on="_patch_id",
        ).merge(
            plan.outputs[["output_id", "station"]],
            on="output_id",
            suffixes=("_src", "_out"),
        )
        assert (joined["station_src"] == joined["station_out"]).all()
        assert set(plan.outputs["station"]) == set(df["station"])

    def test_modified_flag_if_chunked(self, contiguous_df):
        """Ensure the modified flag shows up for modified rows."""
        plan = build_chunk_plan(contiguous_df, overlap=0, time=5, keep_partial=True)
        assert plan.members["_modified"].all()

    def test_modified_flag_no_chunk(self, contiguous_df):
        """Rows whose limits don't change aren't modified."""
        time_diff = contiguous_df["time_max"] - contiguous_df["time_min"]
        df = contiguous_df.assign(time_max=lambda x: x["time_max"] - x["time_step"])
        plan = build_chunk_plan(
            df, overlap=0, time=time_diff.iloc[0], keep_partial=True
        )
        assert len(plan.outputs) == len(df)
        assert not plan.members["_modified"].any()
