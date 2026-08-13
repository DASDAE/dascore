"""Tests for chunking dataframes."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import ChunkError, ParameterError, UnitError
from dascore.utils.chunk import get_intervals
from dascore.utils.chunk_plan import (
    _normalize_chunk_units,
    build_chunk_plan,
    build_subdivision_plan,
    subdivision_pieces,
)
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

    def test_invalid_conflict_raises(self, contiguous_df):
        """An unsupported conflict value raises at the chunk call. See #804."""
        with pytest.raises(ParameterError, match="conflict must be one of"):
            build_chunk_plan(contiguous_df, time=None, conflict="banana")


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


class TestNormalizeChunkUnits:
    """Only numeric envelopes are re-spelled to one unit."""

    def test_time_like_envelopes_pass_through(self):
        """Time envelopes are canonical ns whatever unit the coord names.

        Converting them as floats would raise, and there is nothing to
        normalize: the magnitudes do not depend on the spelling.
        """
        df = pd.DataFrame(
            {
                "time_min": [
                    np.datetime64("2020-01-01T00:00:00"),
                    np.datetime64("2020-01-01T00:00:10"),
                ],
                "time_max": [
                    np.datetime64("2020-01-01T00:00:09"),
                    np.datetime64("2020-01-01T00:00:19"),
                ],
                "time_step": [np.timedelta64(1, "s")] * 2,
                "_time_units": ["s", "ms"],
                "_patch_id": [1, 2],
            }
        )
        out = _normalize_chunk_units(df, "time")
        assert out["time_min"].tolist() == df["time_min"].tolist()
        assert out["_time_units"].tolist() == ["s", "ms"]


class TestQuantityChunkValues:
    """Chunk lengths given as quantities of the dimension's own units."""

    def test_seconds_match_bare_value(self, contiguous_df):
        """A time in seconds equals the bare (seconds) value."""
        quant = build_chunk_plan(contiguous_df, time=5 * dc.units.s)
        bare = build_chunk_plan(contiguous_df, time=5)
        assert quant.outputs.equals(bare.outputs)

    def test_other_time_unit(self, contiguous_df):
        """Any time unit converts to the coordinate's own scale."""
        quant = build_chunk_plan(contiguous_df, time=5000 * dc.units.ms)
        bare = build_chunk_plan(contiguous_df, time=5)
        assert quant.outputs.equals(bare.outputs)

    def test_numeric_dim_unit(self, contiguous_df):
        """A quantity length converts to the frame's stated coord units."""
        # one row, so chunking distance has no time envelopes to merge
        single = contiguous_df.iloc[[0]].assign(_distance_units="m")
        quant = build_chunk_plan(single, distance=2000 * dc.units.mm)
        # a quantity's magnitude is a float, so compare to the float value
        bare = build_chunk_plan(single, distance=2.0)
        assert quant.outputs.equals(bare.outputs)

    def test_quantity_without_stated_units_raises(self, contiguous_df):
        """A frame recording no units cannot interpret a quantity length."""
        single = contiguous_df.iloc[[0]]
        with pytest.raises(UnitError, match="records no units"):
            build_chunk_plan(single, distance=2000 * dc.units.mm)

    def test_overlap_quantity(self, contiguous_df):
        """Overlap accepts the same forms as the chunk length."""
        quant = build_chunk_plan(
            contiguous_df, time=5 * dc.units.s, overlap=1000 * dc.units.ms
        )
        bare = build_chunk_plan(contiguous_df, time=5, overlap=1)
        assert quant.outputs.equals(bare.outputs)

    def test_wrong_dimensionality_raises(self, contiguous_df):
        """A length cannot chunk a time coordinate."""
        with pytest.raises(UnitError, match="time-like"):
            build_chunk_plan(contiguous_df, time=100 * dc.units.ft)

    def test_percent_raises(self, contiguous_df):
        """A percent is dimensionless but is not a data size."""
        with pytest.raises(UnitError, match="dimensionless"):
            build_chunk_plan(contiguous_df, time=dc.get_quantity("50%"))

    @pytest.mark.parametrize(
        "value", (0 * dc.units.s, -1 * dc.units.s, 0 * dc.units.MB, -1 * dc.units.MB)
    )
    def test_non_positive_raises(self, contiguous_df, value):
        """The positive-value guard applies to quantities too."""
        with pytest.raises(ParameterError, match="greater than 0"):
            build_chunk_plan(contiguous_df, time=value)


class TestSizeChunkPlanDF:
    """Data-size chunk lengths on hand-built dataframes."""

    @pytest.fixture()
    def sized_df(self, contiguous_df):
        """A frame carrying the structural columns a size chunk needs."""
        return contiguous_df.assign(_dtype="float64", dims="distance,time")

    def test_missing_dtype_raises(self, contiguous_df):
        """A frame with no dtype cannot answer a size request."""
        with pytest.raises(ChunkError, match="no dtype information"):
            build_chunk_plan(contiguous_df, time=1 * dc.units.MB)

    def test_null_dtype_raises(self, sized_df):
        """Nulls are as unusable as an absent column."""
        df = sized_df.assign(_dtype=None)
        with pytest.raises(ChunkError, match="no dtype information"):
            build_chunk_plan(df, time=1 * dc.units.MB)

    def test_unknown_step_raises(self, sized_df):
        """An unknown sampling interval makes the sample count undefined."""
        df = sized_df.assign(distance_step=np.nan)
        with pytest.raises(ChunkError, match="cannot be determined"):
            build_chunk_plan(df, time=1 * dc.units.MB)

    def test_sample_count(self, sized_df):
        """The count is floor(bytes / (itemsize * slab samples))."""
        plan = build_chunk_plan(sized_df, time=dc.get_quantity("10 kB"))
        (part,) = plan.params["size"]["partitions"]
        # 11 distance samples of float64 per time sample
        assert part["slab_samples"] == 11
        assert part["bytes_per_sample"] == 88
        assert part["n_samples"] == 10_000 // 88

    def test_params_records_request(self, sized_df):
        """Params explain what the size request resolved to."""
        plan = build_chunk_plan(sized_df, time=dc.get_quantity("10 kB"))
        assert plan.params["size"]["requested_bytes"] == 10_000
        assert plan.params["size"]["partitions"][0]["dtype"] == "float64"

    def test_params_absent_without_size(self, sized_df):
        """A plain chunk records no size diagnostics."""
        assert "size" not in build_chunk_plan(sized_df, time=5).params

    def test_frame_without_dims_column(self, contiguous_df):
        """A frame with no `dims` falls back to its complete envelopes."""
        df = contiguous_df.assign(_dtype="float64")  # deliberately no dims
        plan = build_chunk_plan(df, time=dc.get_quantity("10 kB"))
        (part,) = plan.params["size"]["partitions"]
        assert part["slab_samples"] == 11  # the distance envelope

    def test_unknown_chunk_dim_step_raises(self, sized_df):
        """The chunked dimension's own step must be known to size it."""
        df = sized_df.assign(time_step=np.timedelta64("NaT"))
        with pytest.raises(ChunkError, match="sampling interval is unknown"):
            build_chunk_plan(df, time=dc.get_quantity("10 kB"))

    def test_numeric_dim_incompatible_units_raise(self, sized_df):
        """A numeric coordinate rejects a dimensionally wrong quantity."""
        df = sized_df.assign(_distance_units="m")
        with pytest.raises(UnitError, match="incompatible with"):
            build_chunk_plan(df, distance=5 * dc.units.s)

    def test_unparsable_dtype_raises(self, sized_df):
        """A dtype string numpy cannot parse is unusable, not absent."""
        df = sized_df.assign(_dtype="not-a-real-dtype")
        with pytest.raises(ChunkError, match="dtype"):
            build_chunk_plan(df, time=1 * dc.units.MB)

    def test_missing_other_dim_envelope_raises(self, sized_df):
        """A dim without a full envelope has no derivable sample count."""
        df = sized_df.drop(columns=["distance_max"])
        with pytest.raises(ChunkError, match="cannot be determined"):
            build_chunk_plan(df, time=1 * dc.units.MB)

    def test_unknown_dtype_partition_is_empty_not_nan(self, sized_df):
        """
        A partition with no dtype carries "", never NaN.

        NaN is truthy, so it would reach the derived catalog as the
        string "nan" and poison every later np.dtype() of the column.
        """
        known = sized_df.assign(tag="a")
        unknown = sized_df.assign(tag="b", _dtype="")
        both = pd.concat([known, unknown], ignore_index=True)
        outputs = build_chunk_plan(both, time=5).outputs
        assert not outputs["_dtype"].isna().any()
        assert set(outputs["_dtype"]) == {"float64", ""}

    def test_mixed_dtypes_upcast(self, sized_df):
        """A partition mixing dtypes is sized against the upcast dtype."""
        df = sized_df.copy()
        df.loc[df.index[0], "_dtype"] = "float32"
        plan = build_chunk_plan(df, time=dc.get_quantity("10 kB"))
        assert plan.params["size"]["partitions"][0]["dtype"] == "float64"


def _one_row_df(start, step, samples, name="time"):
    """A one-row relation over one evenly sampled dimension."""
    return pd.DataFrame(
        {
            f"{name}_min": [start],
            f"{name}_max": [start + step * (samples - 1)],
            f"{name}_step": [step],
            "_patch_id": [0],
        }
    )


def _cut_plan(df, cuts, name="time"):
    """The plan a relation's per-row cuts produce, through both helpers."""
    return build_subdivision_plan(df, subdivision_pieces(df, cuts, name), name)


class TestSubdivisionPieces:
    """Turning each row's cut values into pieces of its sample grid."""

    def test_uncut_rows_pass_through(self, contiguous_df):
        """A row with no cuts is one unmodified output of the same span."""
        df = contiguous_df.assign(_patch_id=np.arange(len(contiguous_df)))
        plan = _cut_plan(df, [()] * len(df))
        assert len(plan.outputs) == len(df)
        assert not plan.members["_modified"].any()
        assert (plan.outputs["time_min"].values == df["time_min"].values).all()

    def test_pieces_partition_the_samples(self):
        """The pieces meet exactly one step apart, with nothing between."""
        start, step = STARTTIME, np.timedelta64(10, "ms")
        df = _one_row_df(start, step, 100)
        cut = start + step * 40
        plan = _cut_plan(df, [(cut,)])
        assert len(plan.outputs) == 2
        first, second = plan.outputs.iloc[0], plan.outputs.iloc[1]
        assert second["time_min"] == cut
        assert first["time_max"] == cut - step
        assert plan.members["_modified"].all()

    @pytest.mark.parametrize("index", [3, 4001, 28294, 170275])
    def test_an_on_grid_cut_lands_on_its_own_sample(self, index):
        """
        A cut sitting exactly on a sample opens the piece at that sample.

        Dividing the native types rounds once — datetime64 arithmetic
        is integer nanoseconds, so only the quotient rounds — where
        converting each operand to float seconds first rounds three times
        and puts the boundary sample one piece too early at these step
        and index combinations. The grid comparison settles it either
        way, which is what these pin.
        """
        start, step = STARTTIME, np.timedelta64(17060962, "ns")
        df = _one_row_df(start, step, index + 10)
        cut = start + step * index
        plan = _cut_plan(df, [(cut,)])
        assert plan.outputs["time_min"].iloc[1] == cut

    def test_a_cut_a_hair_past_a_sample_opens_at_the_next(self):
        """
        The other direction: a cut just above a sample still snaps up.

        Far enough into a long-stepped row, one nanosecond is below the
        precision of the ratio, so the float says the cut is *on* the
        sample it is really just past. That sample belongs to the piece
        before the cut, and the next one opens the piece after it.
        """
        start, step = STARTTIME, np.timedelta64(1, "D").astype("timedelta64[ns]")
        df = _one_row_df(start, step, 1100)
        cut = start + step * 1000 + np.timedelta64(1, "ns")
        plan = _cut_plan(df, [(cut,)])
        assert plan.outputs["time_min"].iloc[1] == start + step * 1001

    def test_an_on_grid_float_cut_is_not_pushed_past_its_sample(self):
        """
        A float grid rounds where a datetime one cannot, and both ways.

        `1.0 + 0.1` is the textbook case: the difference back out is
        `0.10000000000000009`, so the ratio is a hair above 1 and `ceil`
        answers 2 — the sample the cut lands exactly on would be left in
        the piece *before* the cut instead of opening the one after it.
        Comparing against the grid is what settles it, and a distance
        axis makes this the ordinary case rather than the exotic one.
        """
        start, step = 1.0, 0.1
        df = _one_row_df(start, step, 10, name="distance")
        cut = start + step
        assert (cut - start) / step > 1  # the misleading ratio itself
        plan = _cut_plan(df, [(cut,)], name="distance")
        assert plan.outputs["distance_min"].iloc[1] == cut

    def test_a_float_cut_past_a_sample_still_opens_at_the_next(self):
        """
        The increment, on a float grid: a cut one ulp past a sample.

        The subtraction cannot represent the gap, so the ratio says the
        cut sits exactly on the sample. It does not — the sample is
        behind it, and belongs to the piece before the cut.
        """
        start, step = 0.1, 0.25
        df = _one_row_df(start, step, 10, name="distance")
        cut = np.nextafter(start + step, np.inf)
        assert (cut - start) / step == 1  # the ratio cannot see the gap
        plan = _cut_plan(df, [(cut,)], name="distance")
        assert plan.outputs["distance_min"].iloc[1] == start + 2 * step

    def test_a_cut_on_the_maximum_yields_one_sample(self):
        """A row ending exactly on a boundary keeps its last sample apart."""
        start, step = STARTTIME, np.timedelta64(10, "ms")
        df = _one_row_df(start, step, 50)
        cut = df["time_max"].iloc[0]
        plan = _cut_plan(df, [(cut,)])
        assert len(plan.outputs) == 2
        assert plan.outputs["time_min"].iloc[1] == plan.outputs["time_max"].iloc[1]

    def test_one_cut_sequence_per_row(self):
        """A short sequence would silently drop rows from the plan."""
        df = _one_row_df(STARTTIME, np.timedelta64(10, "ms"), 10)
        with pytest.raises(AssertionError):
            subdivision_pieces(pd.concat([df, df]), [()], "time")


class TestBuildSubdivisionPlan:
    """Building a plan from pieces given directly, as a selection does."""

    def test_a_row_given_no_pieces_leaves(self):
        """A selection drops a patch this way when it keeps no sample."""
        df = _one_row_df(0.0, 1.0, 10, name="distance")
        plan = build_subdivision_plan(df, [[]], "distance")
        assert not len(plan.outputs)
        assert not len(plan.members)

    def test_a_whole_row_piece_is_unmodified(self):
        """A piece which *is* its row loads without a trim."""
        df = _one_row_df(0.0, 1.0, 10, name="distance")
        whole = (df["distance_min"].iloc[0], df["distance_max"].iloc[0])
        plan = build_subdivision_plan(df, [[whole]], "distance")
        assert len(plan.outputs) == 1
        assert not plan.members["_modified"].any()

    def test_disjoint_pieces_become_separate_outputs(self):
        """Two runs of kept channels are two patches, with a hole between."""
        df = _one_row_df(0.0, 1.0, 10, name="distance")
        plan = build_subdivision_plan(df, [[(0.0, 2.0), (7.0, 9.0)]], "distance")
        assert len(plan.outputs) == 2
        assert plan.outputs["distance_max"].iloc[0] == 2.0
        assert plan.outputs["distance_min"].iloc[1] == 7.0
        assert plan.members["_modified"].all()

    def test_one_piece_sequence_per_row(self):
        """A short sequence would silently drop rows from the plan."""
        df = _one_row_df(0.0, 1.0, 10, name="distance")
        with pytest.raises(AssertionError):
            build_subdivision_plan(pd.concat([df, df]), [[]], "distance")


class TestSnappedCutExactness:
    """The property the two correction loops exist to guarantee."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_an_on_grid_cut_always_lands_on_its_own_sample(self, seed):
        """
        Over many float grids, a cut on a sample opens the piece there.

        The ratio alone gets this wrong on a few percent of inputs in
        either direction, and which few depends on the values, so a
        handful of hand-picked cases cannot stand for it. The pieces are
        read straight off `subdivision_pieces` rather than a plan so the
        grid arithmetic is what is under test.
        """
        rng = np.random.default_rng(seed)
        for _ in range(300):
            start = float(rng.uniform(-1e4, 1e4))
            step = float(rng.uniform(1e-6, 10))
            index = int(rng.integers(1, 200_000))
            cut = start + index * step
            df = _one_row_df(start, step, index + 2, name="distance")
            pieces = subdivision_pieces(df, [(cut,)], "distance")[0]
            assert len(pieces) == 2
            # The second piece opens exactly on the cut, and the first
            # ends one step short of it, whichever way the ratio erred.
            assert pieces[1][0] == cut
            assert pieces[0][1] == cut - step

    def test_a_cut_between_samples_opens_at_the_next(self):
        """The other direction, over the same spread of grids."""
        rng = np.random.default_rng(3)
        for _ in range(300):
            start = float(rng.uniform(-1e4, 1e4))
            step = float(rng.uniform(1e-6, 10))
            index = int(rng.integers(1, 200_000))
            on_grid = start + index * step
            cut = np.nextafter(on_grid, np.inf)
            df = _one_row_df(start, step, index + 2, name="distance")
            pieces = subdivision_pieces(df, [(cut,)], "distance")[0]
            # A cut past a sample leaves that sample behind, so the piece
            # it opens starts at the next one.
            assert pieces[1][0] > on_grid
            assert pieces[1][0] >= cut
