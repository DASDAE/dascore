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
from dascore.units import m, percent
from dascore.utils.chunk_plan import (
    ChunkPlan,
    _directions,
    _normalize_numeric_units,
    _order_key,
    _sampling_group,
    _value_family,
    build_chunk_plan,
    build_concat_plan,
    patch_local_adjusted_envelopes,
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
            # the members of an output state exactly what it does; the
            # missing value is the one spelled two ways
            missing = (src.isnull() | (src == "")) & (out.isnull() | (out == ""))
            assert ((src == out) | missing).all()

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
        with pytest.raises(ChunkError, match="longest contiguous segment"):
            build_chunk_plan(df, time=100)

    def test_middle_value_step(self):
        """D7: the partition step is the middle value of member steps."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0)
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        plan = build_chunk_plan(_flat([p1, p2]), time=None)
        assert plan.outputs["time_step"].iloc[0] == time.step


class TestConflict:
    """Spec 2.5: attr policing within a partition."""

    @pytest.fixture(scope="class")
    def conflicted_patches(self):
        """Two contiguous patches with conflicting known values of an attr."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0).update_attrs(data_type="velocity")
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        p2 = p2.update_attrs(data_type="strain_rate")
        return [p1, p2]

    def test_raise(self, conflicted_patches):
        """Conflicting known values raise by default."""
        with pytest.raises(CoordMergeError, match="data_type"):
            build_chunk_plan(_flat(conflicted_patches), time=None)

    def test_missing_value_is_a_conflict(self, conflicted_patches):
        """A member lacking the attr conflicts with one which has it."""
        p1, p2 = conflicted_patches
        df = _flat([p1.update_attrs(data_type=""), p2])
        with pytest.raises(CoordMergeError, match="data_type"):
            build_chunk_plan(df, time=None)
        plan = build_chunk_plan(df, time=None, conflict="drop")
        assert len(plan.outputs) == 1
        assert "data_type" not in plan.outputs.columns

    def test_keep_first(self, conflicted_patches):
        """keep_first carries the first member's value."""
        df = _flat(conflicted_patches)
        plan = build_chunk_plan(df, time=None, conflict="keep_first")
        first_id = df.sort_values("time_min")["_patch_id"].iloc[0]
        expected = df.loc[df["_patch_id"] == first_id, "data_type"].iloc[0]
        assert plan.outputs["data_type"].iloc[0] == expected

    def test_keep_first_means_the_first_member(self, conflicted_patches):
        """keep_first takes the first member's value even where it states none."""
        p1, p2 = conflicted_patches
        df = _flat([p1.update_attrs(data_type=""), p2])
        plan = build_chunk_plan(df, time=None, conflict="keep_first")
        assert len(plan.outputs) == 1
        assert pd.isnull(plan.outputs["data_type"].iloc[0])

    def test_drop(self, conflicted_patches):
        """Drop omits the conflicting attr from outputs."""
        plan = build_chunk_plan(_flat(conflicted_patches), time=None, conflict="drop")
        assert "data_type" not in plan.outputs.columns


class TestGroupParameter:
    """Group attrs partition instead of raising."""

    @pytest.fixture(scope="class")
    def two_source_flat(self):
        """Contiguous patches from two data sources."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0)
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        p1, p2 = (x.update_attrs(acquisition_key="XX1.R2D1..RAW") for x in (p1, p2))
        p3 = p1.update_attrs(acquisition_key="XX2.R2D1..RAW")
        p4 = p2.update_attrs(acquisition_key="XX2.R2D1..RAW")
        return _flat([p1, p2, p3, p4])

    def test_missing_value_is_its_own_kind(self, two_source_flat):
        """A row lacking a kind value is not the kind of one which states it."""
        df = two_source_flat
        one = df[df["acquisition_key"] == "XX1.R2D1..RAW"].copy()
        one.loc[one.index[-1], "acquisition_key"] = ""
        plan = build_chunk_plan(one, time=None)
        assert len(plan.outputs) == 2
        keys = sorted(plan.outputs["acquisition_key"].fillna(""))
        assert keys == ["", "XX1.R2D1..RAW"]

    def test_un_keyed_rows_are_one_kind(self):
        """Un-keyed rows beside two acquisitions stay together, apart from both."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0)
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        keyed = [
            p.update_attrs(acquisition_key=key)
            for key in ("XX1.R2D1..RAW", "XX2.R2D1..RAW")
            for p in (p1, p2)
        ]
        plan = build_chunk_plan(_flat([*keyed, p1, p2]), time=None)
        assert len(plan.outputs) == 3
        keys = sorted(plan.outputs["acquisition_key"].fillna(""))
        assert keys == ["", "XX1.R2D1..RAW", "XX2.R2D1..RAW"]

    def test_missing_spellings_are_one_kind(self, two_source_flat):
        """Null and "" spell one missing value, so they are one kind."""
        df = two_source_flat.copy()
        df["acquisition_key"] = ["", None, "", None]
        plan = build_chunk_plan(df, time=None)
        assert len(plan.outputs) == 1

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
        """Config patch_kind_attrs drives the default partitioning."""
        with dc.config_context(patch_kind_attrs=("tag",)):
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
        # Three un-keyed patches and three keyed ones cover the same span
        # with the same tag, but a missing key is a kind of its own, so the
        # two sets partition apart and nothing is deduplicated as an overlap.
        assert len(plan.members) == len(spool)
        assert set(plan.members["_patch_id"]) <= set(spool._df["_patch_id"])
        # The plan rows and the assembled patches agree on the key.
        out = spool.chunk(time=None)
        keys = out.get_contents()["acquisition_key"]
        assert [x.attrs.acquisition_key for x in out] == list(keys)

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


class TestSamplesAdjustedEnvelopes:
    """Samples residual envelope adjustment (2026-07-18 F4)."""

    @staticmethod
    def _frame():
        """One ascending row: 10 samples at step 1 spanning [0, 9]."""
        return pd.DataFrame({"time_min": [0.0], "time_max": [9.0], "time_step": [1.0]})

    def test_exclusive_stop(self):
        """The stop index is exclusive: (0, 5) ends at sample 4."""
        out = patch_local_adjusted_envelopes(
            self._frame(), (({"time": (0, 5)}, True, False),)
        )
        assert out["time_max"].iloc[0] == 4.0

    def test_empty_window_drops_row(self):
        """A zero-length window contributes nothing."""
        out = patch_local_adjusted_envelopes(
            self._frame(), (({"time": (3, 3)}, True, False),)
        )
        assert len(out) == 0

    def test_start_past_end_drops_row(self):
        """A window beyond the patch contributes nothing."""
        out = patch_local_adjusted_envelopes(
            self._frame(), (({"time": (50, 60)}, True, False),)
        )
        assert len(out) == 0

    def test_stop_clamps(self):
        """A stop past the end clamps to the envelope max."""
        out = patch_local_adjusted_envelopes(
            self._frame(), (({"time": (2, 99)}, True, False),)
        )
        assert out["time_min"].iloc[0] == 2.0
        assert out["time_max"].iloc[0] == 9.0

    def test_descending_orientation(self):
        """On a descending coord, sample 0 sits at the envelope max."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [9.0], "time_step": [-1.0]})
        out = patch_local_adjusted_envelopes(df, (({"time": (0, 5)}, True, False),))
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


class TestRelativeAdjustedEnvelopes:
    """Relative residual envelope adjustment."""

    def test_open_percent_bound(self):
        """Open and percent bounds project in each row range."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [9.0]})
        residuals = (({"time": (None, -50 * percent)}, False, True),)
        out = patch_local_adjusted_envelopes(df, residuals)
        assert out["time_min"].iloc[0] == 0.0
        assert out["time_max"].iloc[0] == 4.5

    @pytest.mark.parametrize("bound", [1 * m, np.array([1, 2]), "bad"])
    def test_unresolved_bound_keeps_envelope(self, bound):
        """Bounds the relation cannot project defer to exact patch selection."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [9.0]})
        # Pair the unresolvable bound with a resolvable one: the control
        # below moves time_max, so an untouched envelope here means the
        # pair was deferred, not that nothing was projected at all.
        residuals = (({"time": (bound, -4.5)}, False, True),)
        assert patch_local_adjusted_envelopes(df, residuals).equals(df)
        control = (({"time": (0.0, -4.5)}, False, True),)
        out = patch_local_adjusted_envelopes(df, control)
        assert out["time_max"].iloc[0] == 4.5

    @pytest.mark.parametrize("bounds", [(-np.inf, -4.5), (4.5, np.inf)])
    def test_infinity_away_from_the_patch_is_open(self, bounds):
        """An infinity pointing away from the patch leaves that side open."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [9.0]})
        residuals = (({"time": bounds}, False, True),)
        out = patch_local_adjusted_envelopes(df, residuals)
        assert out["time_min"].iloc[0] == (0.0 if bounds[0] == -np.inf else 4.5)
        assert out["time_max"].iloc[0] == (4.5 if bounds[0] == -np.inf else 9.0)

    @pytest.mark.parametrize("bounds", [(np.inf, None), (None, -np.inf)])
    def test_infinity_past_the_patch_empties_it(self, bounds):
        """Numeric infinity is a real bound: past the patch selects nothing."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [9.0]})
        residuals = (({"time": bounds}, False, True),)
        out = patch_local_adjusted_envelopes(df, residuals, drop_empty=False)
        assert out["_patch_local_empty"].all()
        assert out["time_min"].isna().all()

    def test_non_finite_time_bound_is_open(self):
        """A datetime coordinate cannot offset by infinity, so it loads whole."""
        start = np.datetime64("2020-01-01")
        df = pd.DataFrame(
            {"time_min": [start], "time_max": [start + np.timedelta64(9, "s")]}
        )
        residuals = (({"time": (np.inf, None)}, False, True),)
        out = patch_local_adjusted_envelopes(df, residuals)
        assert out["time_min"].iloc[0] == start
        assert out["time_max"].iloc[0] == df["time_max"].iloc[0]

    def test_numeric_quantity_uses_row_units(self):
        """Quantity offsets convert into each numeric coordinate's units."""
        df = pd.DataFrame(
            {"distance_min": [10.0], "distance_max": [19.0], "_distance_units": ["m"]}
        )
        residuals = (({"distance": (100 * dc.units.cm, None)}, False, True),)
        out = patch_local_adjusted_envelopes(df, residuals)
        assert out["distance_min"].iloc[0] == 11.0
        assert out["distance_max"].iloc[0] == 19.0

    def test_missing_envelope_columns_pass_through(self):
        """A residual for an absent coordinate does not alter the relation."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [9.0]})
        residuals = (({"depth": (1, -1)}, False, True),)
        out = patch_local_adjusted_envelopes(df, residuals)
        assert out.equals(df)

    def test_each_row_uses_its_own_envelope(self):
        """Equal offsets produce different absolute ranges per patch."""
        df = pd.DataFrame(
            {
                "time_min": [0.0, 100.0],
                "time_max": [9.0, 109.0],
                "time_step": [1.0, 1.0],
            }
        )
        residuals = (({"time": (1, -1)}, False, True),)
        out = patch_local_adjusted_envelopes(df, residuals)
        assert out["time_min"].tolist() == [1.0, 101.0]
        assert out["time_max"].tolist() == [8.0, 108.0]

    def test_first_window_survives_chunk(self):
        """A derived spool describes and loads each patch-local window."""
        spool = dc.get_example_spool(
            "random_das", shape=(20, 200), time_step=dc.to_timedelta64(0.04)
        )
        out = spool.select(time=(0, 2), relative=True).chunk(time=None)
        assert len(out) == len(spool)
        for source, selected in zip(spool, out, strict=True):
            expected = source.select(time=(0, 2), relative=True)
            assert selected.equals(expected)

    @pytest.fixture()
    def short_spool(self):
        """Three patches whose patch-local windows are separated by gaps."""
        return dc.get_example_spool(
            "random_das", shape=(20, 200), time_step=dc.to_timedelta64(0.04)
        )

    def test_samples_take_precedence_over_relative(self, short_spool):
        """Combined flags project with Patch.select's sample semantics."""
        selected = short_spool.select(time=(0, 10), samples=True, relative=True).chunk(
            time=None
        )
        assert len(selected) == len(short_spool)
        assert all(x.shape[x.get_axis("time")] == 10 for x in selected)

    @pytest.mark.parametrize(
        "window",
        [
            (np.nan, 2),
            (0 * dc.units.s, 2 * dc.units.s),
            (0 * percent, 50 * percent),
        ],
        ids=("null", "quantity", "percent"),
    )
    def test_bound_forms_survive_chunk(self, short_spool, window):
        """Null, quantity, and percent bounds plan the loaded windows exactly."""
        selected = short_spool.select(time=window, relative=True).chunk(time=None)
        assert len(selected) == len(short_spool)
        for source, patch in zip(short_spool, selected, strict=True):
            assert patch.equals(source.select(time=window, relative=True))

    def test_empty_windows_produce_empty_plan(self, short_spool):
        """Selected empty patches remain members but contribute no plan outputs."""
        selected = short_spool.select(time=(100, 101), relative=True)
        assert len(selected) == len(short_spool)
        assert len(selected.chunk_plan(time=None).outputs) == 0
        assert len(selected.chunk(time=None)) == 0


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
        out = patch_local_adjusted_envelopes(
            self._frame(), (({"time": (-3, None)}, True, False),)
        )
        assert out["time_min"].iloc[0] == 7.0
        assert out["time_max"].iloc[0] == 9.0

    def test_negative_stop_resolves(self):
        """(None, -2) drops the last two samples."""
        out = patch_local_adjusted_envelopes(
            self._frame(), (({"time": (None, -2)}, True, False),)
        )
        assert out["time_max"].iloc[0] == 7.0

    def test_unknown_step_keeps_envelope(self):
        """Rows whose count is unknown keep their candidacy envelope."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [9.0], "time_step": [np.nan]})
        out = patch_local_adjusted_envelopes(df, (({"time": (-3, None)}, True, False),))
        assert len(out) == 1
        assert out["time_min"].iloc[0] == 0.0
        assert out["time_max"].iloc[0] == 9.0

    def test_drop_empty_false_keeps_rows(self):
        """Equality's variant keeps presented-but-empty rows."""
        out = patch_local_adjusted_envelopes(
            self._frame(), (({"time": (3, 3)}, True, False),), drop_empty=False
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
        out = patch_local_adjusted_envelopes(
            df, (({"time": (0.5, None)}, True, False),)
        )
        assert out["time_min"].iloc[0] == 0.0
        assert out["time_max"].iloc[0] == 9.0


class TestConcatPlan:
    """build_concat_plan: chunk's partitioning, then order-based grouping."""

    @pytest.fixture(scope="class")
    def trio(self):
        """Three same-kind patches in time order."""
        t0 = np.datetime64("2020-01-01", "ns")
        p1 = dc.get_example_patch(time_min=t0)
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        # a gap of eight samples between p2 and p3
        p3 = dc.get_example_patch(time_min=p2.get_coord("time").max() + 9 * time.step)
        return p1, p2, p3

    def test_one_output_per_partition(self, trio):
        """Gaps do not split; kind does."""
        p1, p2, p3 = trio
        plan = build_concat_plan(_flat([p1, p2, p3]), time=None)
        assert len(plan.outputs) == 1
        assert len(plan.members) == 3
        assert plan.params["mode"] == "concat"
        plan = build_concat_plan(_flat([p1, p2.update_attrs(tag="x"), p3]), time=None)
        assert len(plan.outputs) == 2

    def test_count_groups_in_order(self, trio):
        """The count splits each partition's rows in dimension order."""
        p1, p2, p3 = trio
        plan = build_concat_plan(_flat([p3, p1, p2]), time=2)
        assert len(plan.outputs) == 2
        first = plan.members[plan.members["output_id"] == 0]
        assert len(first) == 2
        # the first output spans p1 and p2, whatever order the rows came in
        assert plan.outputs["time_min"].iloc[0] == p1.get_coord("time").min()
        assert plan.outputs["time_max"].iloc[0] == p2.get_coord("time").max()

    def test_missing_kind_value_partitions(self, trio):
        """A missing kind value is a kind of its own; units are policed."""
        p1, p2, _ = trio
        keyed = p2.update_attrs(acquisition_key="XX.R2D1..RAW").set_units("m")
        plan = build_concat_plan(_flat([p1, keyed]), time=None)
        assert len(plan.outputs) == 2
        keys = sorted(plan.outputs["acquisition_key"].fillna(""))
        assert keys == ["", "XX.R2D1..RAW"]
        # different units conflict, policed as a chunk plan polices them
        same_kind = p1.update_attrs(acquisition_key="XX.R2D1..RAW")
        mixed = _flat([same_kind.set_units("km"), keyed])
        with pytest.raises(CoordMergeError, match="data_units"):
            build_concat_plan(mixed, time=None)
        plan = build_concat_plan(mixed, time=None, conflict="drop")
        assert len(plan.outputs) == 1
        # dropping the conflict still leaves the row stating the units the
        # members are converted to, which is what the patch will hold
        assert dc.get_quantity(plan.outputs["data_units"].iloc[0]) == dc.get_quantity(
            "km"
        )

    def test_new_dimension_is_described(self, trio):
        """An output along a new dimension names it, claiming no envelope."""
        p1 = trio[0]
        copies = [p1, p1.new(), p1.new()]
        plan = build_concat_plan(_flat(copies), wave_rank=None)
        assert len(plan.outputs) == 1
        row = plan.outputs.iloc[0]
        assert row["dims"].endswith(",wave_rank")
        assert "wave_rank_min" not in plan.outputs.columns
        assert len(plan.members) == 3
        # patches whose existing coordinates differ cannot share a new dimension
        assert len(build_concat_plan(_flat(trio), wave_rank=None).outputs) == 3
        # a relation of dimensionless rows spells the new dimension cleanly
        flat = _flat(copies)
        spatial = [x for x in flat.columns if "time" in x or "distance" in x]
        flat = flat.drop(columns=spatial)
        flat["dims"] = ""
        plan = build_concat_plan(flat, wave_rank=None)
        assert plan.outputs["dims"].tolist() == ["wave_rank"]

    def test_dimension_units_partition(self, trio):
        """Unitless coordinate values stay apart from unitful ones."""
        p1 = trio[0]
        distance = p1.get_coord("distance")
        shifted = p1.update_coords(distance_min=distance.max() + distance.step)
        unitless = shifted.get_coord("distance").set_units(None)
        bare = shifted.update_coords(distance=unitless)
        assert len(build_concat_plan(_flat([p1, bare]), distance=None).outputs) == 2
        assert len(build_concat_plan(_flat([p1, shifted]), distance=None).outputs) == 1
        # a patch with no values along the dimension states nothing which
        # could clash, so it follows the one spelling the others state
        collapsed = p1.mean("distance")
        assert (
            len(build_concat_plan(_flat([p1, collapsed]), distance=None).outputs) == 1
        )
        # but not when they state two, which would be a guess
        assert (
            len(build_concat_plan(_flat([p1, bare, collapsed]), distance=None).outputs)
            == 3
        )

    def test_empty_and_null_coordinate_units_are_one_spelling(self, trio):
        """A coordinate stated with no units spells that null or ""."""
        p1, p2, _ = trio
        df = _flat([p1, p2])
        col = "_time_units"
        assert col in df.columns
        df = df.copy()
        df.loc[df.index[0], col] = ""
        df.loc[df.index[1], col] = None
        plan = build_concat_plan(df, time=None)
        assert len(plan.outputs) == 1

    def test_relation_without_step_or_dtype(self, trio):
        """An envelope without a step, or rows without a dtype, still plan."""
        df = _flat(trio).drop(columns=["time_step"])
        plan = build_concat_plan(df, time=None)
        assert "time_step" not in plan.outputs.columns
        assert len(plan.outputs) == 1
        df = _flat(trio).assign(_dtype=None)
        plan = build_concat_plan(df, time=None)
        assert plan.outputs["_dtype"].iloc[0] == ""
        # a relation without a dims column still plans; the coordinates it
        # identifies are settled when the output loads, not here
        df = _flat(trio).drop(columns=["dims"]).assign(_sensor_def_key=None)
        df.loc[0, "_sensor_def_key"] = "fp:a"
        df.loc[1, "_sensor_def_key"] = "fp:b"
        plan = build_concat_plan(df, time=None, conflict="drop")
        assert len(plan.outputs) == 1
        assert "dropped_coords" not in plan.params

    def test_coordinate_only_some_members_have_is_not_dropped(self, trio):
        """A coordinate other members lack rides along, in the catalog too."""
        p1, p2, _ = trio
        n = p1.shape[p1.get_axis("distance")]
        lat = p1.update_coords(latitude=("distance", np.arange(n, dtype=float)))
        plan = build_concat_plan(_flat([lat, p2]), time=None, conflict="drop")
        assert "dropped_coords" not in plan.params or not plan.params["dropped_coords"]
        out = dc.spool([lat, p2]).concatenate(time=None, conflict="drop")
        assert "latitude" in out[0].coords.coord_map
        assert "latitude_min" in out.get_contents().columns

    def test_directions(self):
        """Steps of any kind give a sign; a missing or signless one gives NaN."""
        steps = pd.Series([2, -0.5, pd.Timedelta(-1, "s"), None, "x"], dtype=object)
        got = _directions(steps)
        assert got[0] == 1.0 and got[1] == -1.0 and got[2] == -1.0
        assert np.isnan(got[3]) and np.isnan(got[4])

    def test_value_less_dimension_is_identified_by_its_members(self, trio):
        """A dimension carried without values takes a digest of what it joins."""
        p1, p2, _ = trio
        blank = p1.mean("time")  # time survives as a value-less coordinate
        rows = _flat([blank, p2.mean("time")])
        plan = build_concat_plan(rows, time=None)
        key = plan.outputs["_time_def_key"].iloc[0]
        assert str(key).startswith("cat:")
        # an output joining other members is a different coordinate
        rows = _flat([blank, p2.mean("time"), p1.mean("time").new()])
        other = build_concat_plan(rows, time=None).outputs["_time_def_key"].iloc[0]
        assert other != key
        # a member whose identity the relation does not state leaves the
        # output without one, rather than digesting a hole
        blind = _flat([blank, p2.mean("time")])
        blind.loc[0, "_time_def_key"] = None
        plan = build_concat_plan(blind, time=None)
        assert pd.isnull(plan.outputs["_time_def_key"].iloc[0])
        # and one member alone keeps the identity it was given
        lone = build_concat_plan(_flat([blank]), time=None)
        assert lone.outputs["_time_def_key"].iloc[0] == rows["_time_def_key"].iloc[0]

    def test_directions_of_a_constant_step(self):
        """A zero step points neither way."""
        assert np.isnan(_directions(pd.Series([0.0]))[0])
        assert np.isnan(_directions(pd.Series([pd.Timedelta(0)], dtype=object))[0])

    def test_created_dimension_leaves_other_identities_alone(self, trio):
        """The new dimension's identity is its own, not another dimension's."""
        p1, p2, _ = trio
        rows = _flat([p1, p2])
        before = list(rows["_time_def_key"])
        assert len(set(before)) == 2
        plan = build_concat_plan(rows, wave_rank=None)
        # the two patches differ along time, so they cannot share an output
        assert len(plan.outputs) == 2
        assert list(plan.outputs["_time_def_key"]) == before
        assert plan.outputs["_wave_rank_def_key"].str.startswith("fp:").all()

    def test_public_units_attr_is_refused_for_a_new_dimension(self, trio):
        """An attr spelled like the new dimension's units cannot survive it."""
        p1 = trio[0]
        rows = _flat([p1, p1.new()])
        rows["batch_units"] = "m"
        with pytest.raises(ParameterError, match="batch_units"):
            build_concat_plan(rows, batch=None)

    def test_keep_first_floats_a_converted_dtype(self, trio):
        """An output which converts its members' data says so in its dtype."""
        p1, p2, _ = trio
        ints = p1.new(data=p1.data.astype("int32")).set_units("m")
        km = p2.new(data=p2.data.astype("int32")).set_units("km")
        df = _flat([ints, km])
        assert set(df["_dtype"]) == {"int32"}
        plan = build_concat_plan(df, time=None, conflict="keep_first")
        assert plan.outputs["_dtype"].iloc[0] == "float64"
        # without a conversion the dtype stands
        same = _flat([ints, p2.new(data=p2.data.astype("int32")).set_units("m")])
        plan = build_concat_plan(same, time=None, conflict="keep_first")
        assert plan.outputs["_dtype"].iloc[0] == "int32"

    def test_normalize_numeric_units_without_an_envelope(self):
        """A coordinate the relation gives no envelope needs no conversion."""
        df = pd.DataFrame({"_sensor_units": ["m", "cm"]})
        assert _normalize_numeric_units(df, "sensor") is df

    def test_value_family(self):
        """Envelope values sort into four families; a missing one into none."""
        assert _value_family(np.datetime64("2020-01-01")) == "datetime"
        assert _value_family(pd.Timestamp("2020-01-01")) == "datetime"
        assert _value_family(np.timedelta64(1, "s")) == "timedelta"
        assert _value_family("a") == "text"
        assert _value_family(3) == "number"
        assert _value_family(None) == "" and _value_family(np.nan) == ""

    def test_order_key_ranks_timedeltas_natively(self):
        """Timedelta envelopes rank by duration, not by spelling."""
        deltas = pd.Series([pd.Timedelta(10, "s"), pd.Timedelta(2, "s")], dtype=object)
        key = _order_key(deltas)
        assert key[1] < key[0]

    def test_order_key_ranks_each_kind_among_its_own(self):
        """Labels and numbers rank separately (kinds never share a partition)."""
        key = _order_key(pd.Series(["b", 10, None, "a", 2], dtype=object))
        assert np.isnan(key[2])
        assert key[3] < key[0]
        assert key[4] < key[1]

    def test_envelope_attrs_refused_per_partition(self, trio):
        """A partition elsewhere owning the name does not excuse an attr here."""
        p1, p2, _ = trio
        as_dim = p1.rename_coords(distance="batch")
        rows = _flat([p2, p2.new()])
        rows["batch_step"] = 3  # an ordinary attr, as the relation holds it
        frame = pd.concat([_flat([as_dim]), rows], ignore_index=True)
        with pytest.raises(ParameterError, match="batch_step"):
            build_concat_plan(frame, batch=None)
        # the same rows without the attr plan as two partitions
        frame = pd.concat([_flat([as_dim]), _flat([p2, p2.new()])], ignore_index=True)
        assert len(build_concat_plan(frame, batch=None).outputs) == 2

    def test_empty_and_bad_arguments(self, trio):
        """An empty relation plans nothing; bad arguments raise."""
        plan = build_concat_plan(_flat(trio).iloc[:0], time=None)
        assert plan.outputs.empty and plan.members.empty
        with pytest.raises(ParameterError):
            build_concat_plan(_flat(trio), time=None, distance=None)
        with pytest.raises(ParameterError):
            build_concat_plan(_flat(trio), time=0)
