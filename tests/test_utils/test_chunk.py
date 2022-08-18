"""
Tests for chunking dataframes.
"""
import numpy as np
import pandas as pd
import pytest

from dascore.exceptions import ParameterError
from dascore.utils.chunk import ChunkManager, get_intervals
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
    df["d_time"] = dt
    df["d_distance"] = 1
    return df


@pytest.fixture()
def contiguous_sr_spaced_df(contiguous_df):
    """separate df by one sample rate."""
    sr = contiguous_df.loc[:, "d_time"]
    out = contiguous_df.copy()
    out.loc[:, "time_max"] = out.loc[:, "time_max"] - sr
    return out


@pytest.fixture()
def contiguous_df_two_stations(contiguous_df):
    """Create contiguous df with different stations."""
    # get time, adjust starttime to be one time step after end time
    df1 = contiguous_df.assign(station="sta1")
    df2 = contiguous_df.assign(station="sta2")
    return pd.concat([df1, df2], axis=0, ignore_index=True)


class TestArrange:
    """Tests for custom arrange function."""

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
        """Test using datetime64 and timedelta64"""
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


class TestBasicChunkDF:
    """Test basic DF chunking."""

    @pytest.fixture()
    def df_different_sample_rates(self, contiguous_df):
        """Tests for a df which does have overlaps but different sampling rates."""
        df1 = contiguous_df.copy()
        df2 = contiguous_df.copy()
        time_span = df1["time_max"].max() - df1["time_min"].min()
        df2["time_min"] += time_span
        df2["time_max"] += time_span
        df2["d_time"] = df1["d_time"] * 2
        out = pd.concat([df1, df2], axis=0).reset_index(drop=True)
        return out

    def test_rechunk_contiguous(self, contiguous_df):
        """Test rechunking with no gaps."""
        time_interval = (contiguous_df["time_max"] - contiguous_df["time_min"]).max()
        new_time_interval = time_interval / 2
        chunker = ChunkManager(time=new_time_interval)
        _, out = chunker.chunk(contiguous_df)
        assert len(out) == 2 * len(contiguous_df)
        d_time = out["d_time"].iloc[0]
        new_interval = (out["time_max"] - out["time_min"] + d_time).max()
        assert new_interval == new_time_interval

    def test_rechunk_contiguous_with_sr_separation(self, contiguous_sr_spaced_df):
        """Ensure it still works on data seperated by one sample"""
        df = contiguous_sr_spaced_df
        sr = df["d_time"]
        time_interval = (sr + df["time_max"] - df["time_min"]).max()
        new_time_interval = time_interval / 2
        chunker = ChunkManager(time=new_time_interval)
        _, out = chunker.chunk(df)
        assert len(out) == 2 * len(df)
        new_interval = (out["time_max"] - out["time_min"]).max()
        assert new_interval == (new_time_interval - sr.iloc[0])

    def test_rechunk_different_sr(self, df_different_sample_rates):
        """Ensure segments with different sample rates don't get combined."""
        df = df_different_sample_rates
        chunker = ChunkManager(overlap=None, time=23)
        _, out = chunker.chunk(df)
        dt = np.sort(np.unique(out["d_time"]))
        assert len(dt) == 2, "both dt should remain"
        # the second part of the df should start at the one minute mark
        df2 = out[out["d_time"] == dt[1]]
        time_min = df2.iloc[0]["time_min"]
        assert time_min.minute == 1

    def test_keep_leftovers(self, contiguous_df):
        """Ensure leftovers show up in df."""
        chunker = ChunkManager(overlap=None, keep_partial=True, time=28)
        _, out = chunker.chunk(contiguous_df)
        assert len(out) == 3
        assert out["time_max"].max() == contiguous_df["time_max"].max()

    def test_overlap(self, contiguous_df):
        """Ensure overlapping segments work."""
        over = to_timedelta64(10)
        chunker1 = ChunkManager(overlap=over, time=20)
        _, out = chunker1.chunk(contiguous_df)
        expected = over - contiguous_df["d_time"].iloc[0]
        olap = out.shift()["time_max"] - out["time_min"]
        assert np.all(pd.isnull(olap) | (olap == expected))
        # now ensure floats work for overlap param
        chunker2 = ChunkManager(overlap=10, time=20)
        _, out2 = chunker2.chunk(contiguous_df)
        assert out.equals(out2)

    def test_raises_overlap_no_chunksize(self):
        """Specifying an overlap and no chunk size should raise."""
        with pytest.raises(ParameterError, match="used for merging"):
            ChunkManager(time=None, overlap=10)


class TestChunkToMerge:
    """Tests for using chunking to merge contiguous, or overlapping, data."""

    @pytest.fixture()
    def gapy_df(self, contiguous_df):
        """Create a dataframe with gaps."""
        df = contiguous_df.copy()
        df["time_max"] -= df["d_time"] * 15
        return df

    @pytest.fixture()
    def gapy_df_unordered(self, gapy_df):
        """Create a dataframe with gaps that is not sorted by starttime."""
        inds = np.random.RandomState(42).permutation(gapy_df.index)
        return gapy_df.loc[inds].reset_index(drop=True)

    def test_chunk_can_merge(self, contiguous_df):
        """Ensure chunk can be used to merge unspecified segment lengths."""
        cm = ChunkManager(time=None)
        _, out = cm.chunk(contiguous_df)
        assert len(out) == 1
        assert out["time_min"].min() == contiguous_df["time_min"].min()

    def test_doesnt_merge_gappy_df(self, gapy_df):
        """Ensure the gappy dataframe doesn't get merged."""
        cm = ChunkManager(time=None)
        _, out = cm.chunk(gapy_df)
        assert len(gapy_df) == len(out)
        expected_durations = gapy_df["time_max"] - gapy_df["time_min"]
        durations = out["time_max"] - out["time_min"]
        assert expected_durations.equals(durations)

    def test_doesnt_merge_unordered_gappy_df(self, gapy_df_unordered):
        """Ensure the gappy dataframe doesn't get merged."""
        df = gapy_df_unordered
        cm = ChunkManager(time=None)
        _, out = cm.chunk(df)
        assert len(df) == len(out)
        expected_durations = df["time_max"] - df["time_min"]
        durations = out["time_max"] - out["time_min"]
        assert expected_durations.equals(durations)


class TestInstructionDF:
    """Sanity checks on intermediary df"""

    def test_indices(self, contiguous_df):
        """Ensure the input/output index belong to input/output df."""
        chunker = ChunkManager(overlap=0, time=10)
        in_df, out_df = chunker.chunk(contiguous_df)
        instruction = chunker.get_instruction_df(in_df, out_df)
        # ensure the source index is set as the index of the instruction_df
        assert instruction.index.name == "source_index"
        assert set(instruction.index).issubset(set(contiguous_df.index))
        assert set(instruction["current_index"]).issubset(set(out_df.index))

    def test_different_group_columns(self, contiguous_df_two_stations):
        """Ensure instruction df honors differences in group columns."""
        df = contiguous_df_two_stations
        chunker = ChunkManager(
            overlap=0,
            time=10,
            group_columns=("station",),
            keep_partial=True,
        )
        in_df, out_df = chunker.chunk(df)
        instruction = chunker.get_instruction_df(in_df, out_df)
        # ensure each output has exactly one station.
        for current_index, sub in instruction.groupby("current_index"):
            source = df.loc[sub.index]
            # there should only be on station in the source for this group
            unique_stations = source["station"].unique()
            assert len(unique_stations) == 1
        # ensure all stations are present.
        used = in_df.loc[instruction.index]
        assert set(used["station"]) == set(in_df["station"])
        assert set(used["_group"]) == set(in_df["_group"])
