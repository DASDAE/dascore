"""Tests for the chunk planner (chunking formalities spec)."""

from __future__ import annotations

import inspect as _inspect

import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.utils.chunk_plan as cp
from dascore.exceptions import (
    ChunkError,
    CoordMergeError,
    InvalidSpoolQueryError,
    ParameterError,
)
from dascore.io.index.catalog import PatchCatalog
from dascore.utils.chunk_plan import (
    ChunkPlan,
    _sampling_group,
    build_chunk_plan,
    samples_adjusted_envelopes,
)
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
            diverse_flat[["_patch_id", "acquisition_key", "tag"]],
            on="_patch_id",
        ).merge(
            plan.outputs[["output_id", "acquisition_key", "tag"]],
            on="output_id",
            suffixes=("_src", "_out"),
        )
        for col in ("acquisition_key", "tag"):
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
        with dc.config_context(sampling_group_tolerance=0.02):
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

    def test_unknown_policy_raises(self, conflicted_patches):
        """A misspelled conflict policy cannot silently behave like drop."""
        with pytest.raises(ParameterError, match="conflict must be"):
            build_chunk_plan(_flat(conflicted_patches), time=None, conflict="keep_fist")


class TestGroupParameter:
    """Group attrs partition instead of raising."""

    @pytest.fixture(scope="class")
    def two_source_flat(self):
        """Contiguous patches from two data sources."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0)
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        p3 = p1.update_attrs(acquisition_key="XX2.R2D1..RAW")
        p4 = p2.update_attrs(acquisition_key="XX2.R2D1..RAW")
        return _flat([p1, p2, p3, p4])

    def test_source_partitions(self, two_source_flat):
        """Different data sources produce separate outputs, no error."""
        plan = build_chunk_plan(two_source_flat, time=None)
        assert len(plan.outputs) == 2
        expected = set(two_source_flat["acquisition_key"])
        assert set(plan.outputs["acquisition_key"]) == expected

    def test_group_override(self, two_source_flat):
        """An explicit empty group means acquisition_key conflicts raise."""
        with pytest.raises(CoordMergeError, match="acquisition_key"):
            build_chunk_plan(two_source_flat, time=None, group=())

    def test_config_group(self, two_source_flat):
        """Config groupby_attrs drives the default partitioning."""
        with dc.config_context(groupby_attrs=("tag",)):
            with pytest.raises(CoordMergeError, match="acquisition_key"):
                build_chunk_plan(two_source_flat, time=None)


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
    """Sanity: plans describe exactly what spool.chunk produces."""

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


class TestChunkPlanAccessor:
    """Tests for the public spool.chunk_plan diagnostic."""

    def test_matches_chunk(self):
        """The plan describes exactly what chunk produces."""
        spool = dc.get_example_spool("random_das")
        plan = spool.chunk_plan(time=3)
        chunked = spool.chunk(time=3)
        assert len(plan.outputs) == len(chunked)
        contents = chunked.get_contents().sort_values("time_min")
        outs = plan.outputs.sort_values("time_min")
        assert np.array_equal(outs["time_min"].values, contents["time_min"].values)

    def test_records_params(self):
        """Plans record the resolved parameters."""
        spool = dc.get_example_spool("random_das")
        plan = spool.chunk_plan(time=None, tolerance=2.0)
        assert plan.params["tolerance"] == 2.0
        assert isinstance(plan.params["group"], tuple)
        assert plan.merge_mode

    def test_members_reference_sources(self):
        """Members bind outputs to source patches without loading data."""
        spool = dc.get_example_spool("diverse_das")
        plan = spool.chunk_plan(time=None)
        assert set(plan.members["output_id"]) == set(plan.outputs["output_id"])
        assert len(plan.members) == len(spool)

    def test_directory_spool(self, tmp_path):
        """chunk_plan works on file-backed spools."""
        patch = dc.get_example_patch()
        dc.write(patch, tmp_path / "a.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        plan = spool.chunk_plan(time=None)
        assert len(plan.outputs) == 1


class TestUnderscoreDimNames:
    """Dims with underscores must not confuse column ownership."""

    def test_chunk_event_time(self):
        """Adjacent patches along an underscore dim merge cleanly."""
        patch = dc.get_example_patch().rename_coords(time="event_time")
        coord = patch.get_coord("event_time")
        middle = coord.values[len(coord) // 2]
        p1 = patch.select(event_time=(None, middle))
        p2 = patch.select(event_time=(middle + coord.step, None))
        merged = dc.spool([p1, p2]).chunk(event_time=None)
        assert len(merged) == 1
        out = merged[0].get_coord("event_time")
        assert out.min() == coord.min()
        assert out.max() == coord.max()

    def test_segment_event_time(self):
        """Segmenting along an underscore dim works too."""
        patch = dc.get_example_patch().rename_coords(time="event_time")
        spool = dc.spool([patch])
        chunked = spool.chunk(event_time=2)
        assert len(chunked) > 1
        for sub in chunked:
            assert "event_time" in sub.dims


class TestChunkPlanCoverageEdges:
    """Remaining chunk-planner branches."""

    def test_partial_overlap_members(self):
        """Overlapping sources drop the covered span (non-overlap skip)."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0)
        step = p1.get_coord("time").step
        # p2 overlaps p1's tail by 100 samples.
        p2 = dc.get_example_patch(time_min=p1.get_coord("time").max() - 100 * step)
        merged = dc.spool([p1, p2]).chunk(time=None)
        # the overlap is removed, so the merge is shorter than the naive sum.
        naive = p1.get_coord("time").size + p2.get_coord("time").size
        assert merged[0].get_coord("time").size < naive

    def test_user_stacklevel_fallback(self, monkeypatch):
        """With no user frame in the stack, the stacklevel falls back to 1."""

        # Every frame reports a dascore path, so no "user" frame is found.
        class _Frame:
            filename = cp.__file__

        monkeypatch.setattr(_inspect, "stack", lambda: [_Frame()] * 3)
        assert cp._user_stacklevel() == 1


class TestSamplingGroups:
    """Sampling-group partitioning invariants (2026-07-18 F5)."""

    def test_close_steps_group(self):
        """Steps within tolerance of the anchor share a group."""
        labels = _sampling_group(pd.Series([1.0, 1.04]), 0.05)
        assert labels.nunique() == 1

    def test_chain_does_not_drift_past_tolerance(self):
        """Adjacent-close steps cannot chain past the group anchor."""
        labels = _sampling_group(pd.Series([1.00, 1.04, 1.08]), 0.05)
        assert list(labels) == [0, 0, 1]

    def test_negative_steps_group_by_magnitude(self):
        """Widely different negative steps never share a group."""
        labels = _sampling_group(pd.Series([-1.0, -5.0]), 0.05)
        assert labels.nunique() == 2

    def test_mixed_orientation_never_groups(self):
        """Equal magnitudes with opposite signs stay separate."""
        labels = _sampling_group(pd.Series([1.0, -1.0]), 0.05)
        assert labels.nunique() == 2

    def test_unknown_steps_share_one_group(self):
        """NaN steps keep their historical single-group behavior."""
        labels = _sampling_group(pd.Series([1.0, np.nan, np.nan]), 0.05)
        assert labels.nunique() == 2
        assert labels.iloc[1] == labels.iloc[2]

    def test_descending_contiguous_merges(self):
        """Contiguous descending patches produce a single merge output."""
        p = dc.get_example_patch()
        flipped = p.flip("time")
        t = p.get_coord("time")
        span = t.max() - t.min() + t.step
        shifted = flipped.update_coords(time=flipped.get_coord("time").data + span)
        plan = dc.spool([shifted, flipped]).chunk_plan(time=None)
        assert len(plan.outputs) == 1
        assert len(plan.members) == 2


class TestSamplesAdjustedEnvelopes:
    """Samples residual envelope adjustment (2026-07-18 F4)."""

    @staticmethod
    def _frame():
        """One ascending row: 10 samples at step 1 spanning [0, 9]."""
        return pd.DataFrame({"time_min": [0.0], "time_max": [9.0], "time_step": [1.0]})

    def test_exclusive_stop(self):
        """The stop index is exclusive: (0, 5) ends at sample 4."""
        out = samples_adjusted_envelopes(self._frame(), (({"time": (0, 5)}, True),))
        assert out["time_max"].iloc[0] == 4.0

    def test_empty_window_drops_row(self):
        """A zero-length window contributes nothing."""
        out = samples_adjusted_envelopes(self._frame(), (({"time": (3, 3)}, True),))
        assert len(out) == 0

    def test_start_past_end_drops_row(self):
        """A window beyond the patch contributes nothing."""
        out = samples_adjusted_envelopes(self._frame(), (({"time": (50, 60)}, True),))
        assert len(out) == 0

    def test_stop_clamps(self):
        """A stop past the end clamps to the envelope max."""
        out = samples_adjusted_envelopes(self._frame(), (({"time": (2, 99)}, True),))
        assert out["time_min"].iloc[0] == 2.0
        assert out["time_max"].iloc[0] == 9.0

    def test_descending_orientation(self):
        """On a descending coord, sample 0 sits at the envelope max."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [9.0], "time_step": [-1.0]})
        out = samples_adjusted_envelopes(df, (({"time": (0, 5)}, True),))
        assert out["time_max"].iloc[0] == 9.0
        assert out["time_min"].iloc[0] == 5.0

    def test_public_same_dim_envelope_matches_patch(self):
        """The chunked catalog's envelope matches the loaded patch (F4)."""
        p = dc.get_example_patch()
        out = dc.spool([p]).select(time=(0, 10), samples=True).chunk(time=None)
        patch = out[0]
        want = pd.Timestamp(patch.get_coord("time").max())
        got = out.get_contents()["time_max"].iloc[0]
        assert got == want
        assert patch.shape[patch.get_axis("time")] == 10


class TestChunkOnlyOnDims:
    """Chunking is defined on dimensions; non-dim coords are 'missing'."""

    @pytest.fixture()
    def aux_time_patch(self):
        """A patch carrying time only as a coord riding distance."""
        p = dc.get_example_patch()
        t = p.get_coord("time")
        base = p.mean("time").squeeze()
        n = base.shape[base.get_axis("distance")]
        return base.update_coords(time=("distance", t.data[:n]))

    def test_aux_only_raises_with_detail(self, aux_time_patch):
        """The default error explains the name rides as a coordinate."""
        with pytest.raises(ChunkError, match="non-dimensional coordinate"):
            dc.spool([aux_time_patch]).chunk(time=None)

    def test_mixed_population_raises_by_default(self, aux_time_patch):
        """Mixed dim/aux populations fail eagerly, not at patch access."""
        sp = dc.spool([dc.get_example_patch(), aux_time_patch])
        with pytest.raises(ChunkError, match="lack the chunk dimension"):
            sp.chunk(time=None)

    def test_drop_excludes_aux_patches(self, aux_time_patch):
        """missing_dim='drop' keeps only patches with the real dimension."""
        sp = dc.spool([dc.get_example_patch(), aux_time_patch])
        out = sp.chunk(time=None, missing_dim="drop", conflict="drop")
        assert len(out) == 1
        assert out[0].dims == ("distance", "time")

    def test_segmenting_aux_coord_raises(self):
        """chunk(<aux coord>=value) is rejected, not accidentally served."""
        p = dc.get_example_patch()
        q = p.update_coords(sensor=("distance", np.arange(p.shape[0], dtype=float)))
        with pytest.raises(ChunkError, match="non-dimensional coordinate"):
            dc.spool([q]).chunk(sensor=100)


class TestSkippedPartitionPolicing:
    """Partitions which produce no outputs are never policed (D8)."""

    @staticmethod
    def _two_partition_frame():
        """Rows forming one long and one too-short partition."""
        return pd.DataFrame(
            {
                "time_min": [0.0, 10.0, 100.0],
                "time_max": [9.0, 19.0, 101.0],
                "time_step": [1.0, 1.0, 1.0],
                "_patch_id": [0, 1, 2],
                "dims": ["time"] * 3,
            }
        )

    def test_skipped_partition_values_untouched(self):
        """A skipped partition's attrs are not policed — not even hashed."""
        df = self._two_partition_frame()
        # an unhashable value in the too-short partition must not matter
        df["note"] = ["ok", "ok", ["unhashable"]]
        plan = build_chunk_plan(df, time=5)
        assert set(plan.outputs["note"]) == {"ok"}

    def test_conflict_in_skipped_partition_ignored(self):
        """Conflicts confined to a skipped partition never raise."""
        df = self._two_partition_frame()
        df.loc[len(df)] = [102.0, 103.0, 1.0, 3, "time"]
        df["note"] = ["ok", "ok", "a", "b"]
        plan = build_chunk_plan(df, time=5)
        assert set(plan.outputs["note"]) == {"ok"}

    def test_earlier_conflict_outranks_later_resolution_error(self):
        """A policed conflict raises before a later partition's error."""
        df = self._two_partition_frame()
        df["time_max"] = [9.0, 19.0, 109.0]  # both partitions long enough
        df["note"] = ["a", "b", "c"]  # conflict in the first partition
        df["_dtype"] = ["float64", "float64", ""]  # size fails in the second
        with pytest.raises(CoordMergeError, match="note"):
            build_chunk_plan(df, time=dc.units.get_quantity("40 bytes"))

    @staticmethod
    def _one_partition_frame():
        """Two contiguous rows forming a single active partition."""
        return pd.DataFrame(
            {
                "time_min": [0.0, 10.0],
                "time_max": [9.0, 19.0],
                "time_step": [1.0, 1.0],
                "_patch_id": [0, 1],
                "dims": ["time"] * 2,
            }
        )

    def test_unhashable_active_value_raises(self):
        """An unhashable policed value in an active partition errors loudly."""
        df = self._one_partition_frame()
        df["note"] = [[1], [1]]  # attr values must be hashable
        with pytest.raises(TypeError, match="unhashable"):
            build_chunk_plan(df, time=...)


class TestDimlessFrames:
    """Plain planner frames without a dims column still plan."""

    def test_frame_without_dims_column(self):
        """Dimension membership checks are skipped when dims is absent."""
        df = pd.DataFrame(
            {
                "time_min": [0.0, 10.0],
                "time_max": [10.0, 20.0],
                "time_step": [1.0, 1.0],
            }
        )
        plan = build_chunk_plan(df, time=None)
        assert len(plan.outputs) == 1
        assert len(plan.members) == 2


class TestNegativeSamplesEnvelopes:
    """Negative samples indices resolve per patch (2026-07-18)."""

    @staticmethod
    def _frame():
        """One ascending row: 10 samples at step 1 spanning [0, 9]."""
        return pd.DataFrame({"time_min": [0.0], "time_max": [9.0], "time_step": [1.0]})

    def test_negative_start_resolves(self):
        """(-3, None) selects the last three samples."""
        out = samples_adjusted_envelopes(self._frame(), (({"time": (-3, None)}, True),))
        assert out["time_min"].iloc[0] == 7.0
        assert out["time_max"].iloc[0] == 9.0

    def test_negative_stop_resolves(self):
        """(None, -2) drops the last two samples."""
        out = samples_adjusted_envelopes(self._frame(), (({"time": (None, -2)}, True),))
        assert out["time_max"].iloc[0] == 7.0

    def test_unknown_step_keeps_envelope(self):
        """Rows whose count is unknown keep their candidacy envelope."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [9.0], "time_step": [np.nan]})
        out = samples_adjusted_envelopes(df, (({"time": (-3, None)}, True),))
        assert len(out) == 1
        assert out["time_min"].iloc[0] == 0.0
        assert out["time_max"].iloc[0] == 9.0

    def test_drop_empty_false_keeps_rows(self):
        """Equality's variant keeps presented-but-empty rows."""
        out = samples_adjusted_envelopes(
            self._frame(), (({"time": (3, 3)}, True),), drop_empty=False
        )
        assert len(out) == 1

    def test_public_negative_window_contents_honest(self):
        """Derived contents match the loaded patch for negative windows."""
        p = dc.get_example_patch()
        out = dc.spool([p]).select(time=(-10, None), samples=True).chunk(time=None)
        got = out.get_contents()["time_min"].iloc[0]
        want = pd.Timestamp(out[0].get_coord("time").min())
        assert got == want
        assert out[0].shape[out[0].get_axis("time")] == 10


class TestNonIntSamplesIndices:
    """Non-integer samples indices leave envelopes untouched."""

    def test_float_index_skipped(self):
        """A float index cannot adjust; the envelope stays candidacy."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [9.0], "time_step": [1.0]})
        out = samples_adjusted_envelopes(df, (({"time": (0.5, None)}, True),))
        assert out["time_min"].iloc[0] == 0.0
        assert out["time_max"].iloc[0] == 9.0
