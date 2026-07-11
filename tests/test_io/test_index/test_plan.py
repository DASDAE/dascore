"""Tests for the chunk planner (chunking formalities spec)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import (
    ChunkError,
    CoordMergeError,
    InvalidSpoolQueryError,
    ParameterError,
)
from dascore.io.index.catalog import PatchCatalog
from dascore.io.index.plan import ChunkPlan, build_chunk_plan
from dascore.utils.time import to_timedelta64

ONE_S = np.timedelta64(1, "s")


def _flat(patches) -> pd.DataFrame:
    """Get the flat relation for a list of patches."""
    return PatchCatalog.from_patches(list(patches)).to_df()


@pytest.fixture(scope="module")
def random_flat() -> pd.DataFrame:
    """Flat relation of the contiguous random_das example spool."""
    return _flat(dc.get_example_spool("random_das"))


@pytest.fixture(scope="module")
def diverse_flat() -> pd.DataFrame:
    """Flat relation of the diverse example spool."""
    return _flat(dc.get_example_spool("diverse_das"))


class TestValidation:
    """Parameter validation per the spec errors table."""

    def test_no_kwargs_raises(self, random_flat):
        """Exactly one dimension kwarg is required."""
        with pytest.raises(ParameterError, match="one dimension"):
            build_chunk_plan(random_flat)

    def test_two_kwargs_raise(self, random_flat):
        """Two chunk kwargs raise."""
        with pytest.raises(ParameterError, match="one dimension"):
            build_chunk_plan(random_flat, time=10, distance=10)

    def test_non_positive_value_raises(self, random_flat):
        """Chunk lengths must be positive."""
        with pytest.raises(ParameterError, match="greater than 0"):
            build_chunk_plan(random_flat, time=0)

    def test_merge_mode_forbids_overlap(self, random_flat):
        """Merge mode does not accept overlap/keep_partial."""
        with pytest.raises(ParameterError, match="merging"):
            build_chunk_plan(random_flat, time=None, overlap=1)
        with pytest.raises(ParameterError, match="merging"):
            build_chunk_plan(random_flat, time=..., keep_partial=True)

    def test_overlap_ge_length_raises(self, random_flat):
        """D6: overlap >= length raises cleanly."""
        with pytest.raises(ParameterError, match="overlap"):
            build_chunk_plan(random_flat, time=2, overlap=2)

    def test_unknown_group_raises(self, random_flat):
        """Explicit group names must exist somewhere in the spool."""
        with pytest.raises(InvalidSpoolQueryError, match="bob"):
            build_chunk_plan(random_flat, time=None, group=("bob",))

    def test_unknown_dim_raises(self, random_flat):
        """Chunking a dimension no patch has raises."""
        with pytest.raises(ChunkError, match="quelle"):
            build_chunk_plan(random_flat, quelle=10)

    def test_bad_missing_dim_raises(self, random_flat):
        """missing_dim accepts only raise/drop."""
        with pytest.raises(ParameterError, match="missing_dim"):
            build_chunk_plan(random_flat, time=None, missing_dim="bob")


class TestMergePlan:
    """Merge-mode planning on contiguous data."""

    def test_contiguous_spool_single_output(self, random_flat):
        """A contiguous spool merges to one output."""
        plan = build_chunk_plan(random_flat, time=None)
        assert isinstance(plan, ChunkPlan)
        assert plan.merge_mode
        assert len(plan.outputs) == 1
        out = plan.outputs.iloc[0]
        assert out["time_min"] == random_flat["time_min"].min()
        assert out["time_max"] == random_flat["time_max"].max()
        # every source patch appears exactly once as a member
        assert len(plan.members) == len(random_flat)
        assert set(plan.members["_patch_id"]) == set(random_flat["_patch_id"])

    def test_members_unmodified_when_contiguous(self, random_flat):
        """Contiguous members load whole (no trims)."""
        plan = build_chunk_plan(random_flat, time=None)
        assert not plan.members["_modified"].any()

    def test_diverse_partitions(self, diverse_flat):
        """The diverse spool partitions by identity attrs, never raising."""
        plan = build_chunk_plan(diverse_flat, time=None)
        assert len(plan.outputs) > 1
        # every output's members share that output's group attr values
        merged = plan.members.merge(
            diverse_flat[["_patch_id", "network", "station", "tag"]],
            on="_patch_id",
        ).merge(
            plan.outputs[["output_id", "network", "station", "tag"]],
            on="output_id",
            suffixes=("_src", "_out"),
        )
        for col in ("network", "station", "tag"):
            src, out = merged[f"{col}_src"], merged[f"{col}_out"]
            equal = (src == out) | (src.isnull() & out.isnull())
            assert equal.all()

    def test_gap_splits_partition(self):
        """A gap larger than tolerance yields separate outputs."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0)
        time = p1.get_coord("time")
        dt = time.step
        span = time.max() - time.min() + dt
        p2 = dc.get_example_patch(time_min=time.max() + dt)  # contiguous
        p3 = dc.get_example_patch(time_min=time.max() + span + 10 * dt)
        plan = build_chunk_plan(_flat([p1, p2, p3]), time=None)
        assert len(plan.outputs) == 2

    def test_plan_records_params(self, random_flat):
        """Plans record resolved parameters, not config references."""
        with dc.set_config(sampling_group_tolerance=0.02):
            plan = build_chunk_plan(random_flat, time=None)
        assert plan.params["sampling_group_tolerance"] == 0.02
        assert isinstance(plan.params["group"], tuple)


class TestSegmentPlan:
    """Segment-mode planning."""

    def test_intervals_cover_envelope(self, random_flat):
        """Chunk segments tile the envelope with the requested length."""
        plan = build_chunk_plan(random_flat, time=3)
        out = plan.outputs
        lengths = (out["time_max"] - out["time_min"]) + out["time_step"]
        expected = to_timedelta64(3)
        assert (abs(lengths - expected) <= out["time_step"]).all()

    def test_members_reference_real_patches(self, random_flat):
        """All members point at rows of the input relation."""
        plan = build_chunk_plan(random_flat, time=3)
        assert set(plan.members["_patch_id"]) <= set(random_flat["_patch_id"])
        # member trims stay within their output's envelope
        joined = plan.members.merge(
            plan.outputs[["output_id", "time_min", "time_max"]],
            on="output_id",
            suffixes=("", "_out"),
        )
        assert (joined["time_min"] >= joined["time_min_out"]).all()
        assert (joined["time_max"] <= joined["time_max_out"]).all()

    def test_too_short_partition_skipped(self):
        """D8: partitions shorter than the length are skipped silently."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0)  # ~8s long
        time = p1.get_coord("time")
        gap = time.max() - time.min() + 100 * ONE_S
        p2 = dc.get_example_patch(time_min=time.min() + gap)
        df = _flat([p1, p2])
        plan = build_chunk_plan(df, time=5)
        assert len(plan.outputs) == 2  # one 5s chunk per 8s partition
        with pytest.raises(ChunkError, match="sufficient length"):
            build_chunk_plan(df, time=100)

    def test_overlap(self, random_flat):
        """Overlapping chunks step by length minus overlap."""
        plan = build_chunk_plan(random_flat, time=4, overlap=2)
        starts = plan.outputs["time_min"].sort_values().values
        strides = np.diff(starts)
        assert (abs(strides - to_timedelta64(2)) <= to_timedelta64(0.01)).all()

    def test_middle_value_step(self):
        """D7: the partition step is the middle value of member steps."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0)
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        plan = build_chunk_plan(_flat([p1, p2]), time=None)
        assert plan.outputs["time_step"].iloc[0] == time.step


class TestMissingDim:
    """Spec section 7 (D2): patches lacking the chunk dim."""

    @pytest.fixture()
    def flat_with_null(self, random_flat):
        """A flat relation with one null time envelope."""
        df = random_flat.copy()
        df.loc[df.index[0], ["time_min", "time_max"]] = (pd.NaT, pd.NaT)
        return df

    def test_raise_by_default(self, flat_with_null):
        """Null chunk-dim envelopes raise by default."""
        with pytest.raises(ChunkError, match="missing_dim"):
            build_chunk_plan(flat_with_null, time=None)

    def test_drop_opt_in(self, flat_with_null):
        """missing_dim='drop' excludes the offending rows."""
        plan = build_chunk_plan(flat_with_null, time=None, missing_dim="drop")
        dropped = flat_with_null["_patch_id"].iloc[0]
        assert dropped not in set(plan.members["_patch_id"])


class TestConflict:
    """Spec 2.5: attr policing within a partition."""

    @pytest.fixture(scope="class")
    def conflicted_patches(self):
        """Two contiguous patches with a differing non-group attr."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0)
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        p2 = p2.update_attrs(data_units="m/s")
        return [p1, p2]

    def test_raise(self, conflicted_patches):
        """Differing non-group attrs raise by default."""
        with pytest.raises(CoordMergeError, match="data_units"):
            build_chunk_plan(_flat(conflicted_patches), time=None)

    def test_keep_first(self, conflicted_patches):
        """keep_first carries the first member's value."""
        df = _flat(conflicted_patches)
        plan = build_chunk_plan(df, time=None, conflict="keep_first")
        first_id = df.sort_values("time_min")["_patch_id"].iloc[0]
        expected = df.loc[df["_patch_id"] == first_id, "data_units"].iloc[0]
        assert plan.outputs["data_units"].iloc[0] == expected

    def test_drop(self, conflicted_patches):
        """Drop omits the conflicting attr from outputs."""
        plan = build_chunk_plan(_flat(conflicted_patches), time=None, conflict="drop")
        assert "data_units" not in plan.outputs.columns


class TestGroupParameter:
    """Group attrs partition instead of raising."""

    @pytest.fixture(scope="class")
    def two_station_flat(self):
        """Contiguous patches from two stations."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0)
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        p3 = p1.update_attrs(station="XX2")
        p4 = p2.update_attrs(station="XX2")
        return _flat([p1, p2, p3, p4])

    def test_station_partitions(self, two_station_flat):
        """Different stations produce separate outputs, no error."""
        plan = build_chunk_plan(two_station_flat, time=None)
        assert len(plan.outputs) == 2
        assert set(plan.outputs["station"]) == set(two_station_flat["station"])

    def test_group_override(self, two_station_flat):
        """An explicit empty group means station conflicts raise."""
        with pytest.raises(CoordMergeError, match="station"):
            build_chunk_plan(two_station_flat, time=None, group=())

    def test_config_group(self, two_station_flat):
        """Config groupby_attrs drives the default partitioning."""
        with dc.set_config(groupby_attrs=("network",)):
            with pytest.raises(CoordMergeError, match="station"):
                build_chunk_plan(two_station_flat, time=None)


class TestDeterminism:
    """Spec section 8."""

    def test_repeat_identical(self, diverse_flat):
        """Identical inputs give identical plans."""
        p1 = build_chunk_plan(diverse_flat, time=None)
        p2 = build_chunk_plan(diverse_flat, time=None)
        pd.testing.assert_frame_equal(p1.outputs, p2.outputs)
        pd.testing.assert_frame_equal(p1.members, p2.members)

    def test_input_order_invariant(self, diverse_flat):
        """Row order of the flat relation does not change the plan."""
        shuffled = diverse_flat.sample(frac=1, random_state=0)
        p1 = build_chunk_plan(diverse_flat, time=None)
        p2 = build_chunk_plan(shuffled, time=None)
        cols = ["time_min", "time_max"]
        pd.testing.assert_frame_equal(
            p1.outputs[cols].reset_index(drop=True),
            p2.outputs[cols].reset_index(drop=True),
        )


class TestOracleParity:
    """Sanity against ChunkManager (the dev-time oracle) where compatible."""

    def test_merge_envelopes_match(self, random_flat):
        """Merge-mode output envelopes match spool.chunk(time=None)."""
        spool = dc.get_example_spool("random_das")
        merged = spool.chunk(time=None)
        contents = merged.get_contents()
        plan = build_chunk_plan(random_flat, time=None)
        assert len(plan.outputs) == len(contents)
        assert plan.outputs["time_min"].iloc[0] == contents["time_min"].iloc[0]
        assert plan.outputs["time_max"].iloc[0] == contents["time_max"].iloc[0]

    def test_segment_envelopes_match(self, random_flat):
        """Segment-mode envelopes match spool.chunk(time=3)."""
        spool = dc.get_example_spool("random_das")
        chunked = spool.chunk(time=3)
        contents = chunked.get_contents().sort_values("time_min")
        plan = build_chunk_plan(random_flat, time=3)
        outs = plan.outputs.sort_values("time_min")
        assert len(outs) == len(contents)
        assert np.array_equal(outs["time_min"].values, contents["time_min"].values)
        assert np.array_equal(outs["time_max"].values, contents["time_max"].values)
