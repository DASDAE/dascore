"""
Test file for chunk/merge.

This is separated from spool tests because these need to be quite
extensive.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import ChunkError, CoordMergeError, ParameterError
from dascore.utils.misc import get_middle_value
from dascore.utils.time import to_timedelta64


@pytest.fixture(scope="class")
def spool_dt_perturbed(random_patch) -> dc.BaseSpool:
    """Create a spool with patches that have slightly different dts."""
    dts = np.array((0.999722, 0.99985, 0.99973, 0.99986))
    current_max = random_patch.attrs.time_max
    patches = []
    for dt in dts:
        patch = random_patch.update_attrs(time_min=current_max, time_step=dt)
        patches.append(patch)
    return dc.spool(patches)


class TestChunk:
    """Tests for merging/chunking patches."""

    @pytest.fixture(scope="class")
    def random_spool_df(self, random_spool):
        """Get contents and sort the contents of random_spool."""
        df = random_spool.get_contents().sort_values("time_min").reset_index(drop=True)
        return df

    def test_merge_eq(self, adjacent_spool_no_overlap):
        """Ensure merged spools are equal."""
        sp1 = adjacent_spool_no_overlap.chunk(time=2)
        sp2 = adjacent_spool_no_overlap.chunk(time=2)
        assert sp1 == sp2

    def test_merge_chunk_adjacent_no_overlap(self, adjacent_spool_no_overlap):
        """Ensure chunking works on simple case of contiguous data w/ no overlap."""
        new = adjacent_spool_no_overlap.chunk(time=None)
        out_list = list(new)
        assert len(new) == len(out_list) == 1

    def test_adjacent_merge_no_overlap(self, adjacent_spool_no_overlap):
        """Test that the adjacent patches get merged."""
        spool = adjacent_spool_no_overlap
        st_len = len(spool)
        merged_st = spool.chunk(time=None)
        merged_len = len(merged_st)
        assert merged_len < st_len
        assert merged_len == 1

    def test_chunk_doesnt_modify_original(self, random_spool):
        """Chunking shouldn't modify original spool."""
        first = random_spool.get_contents().copy()
        _ = random_spool.chunk(time=2)
        second = random_spool.get_contents().copy()
        assert first.equals(second)

    def test_patches_match_df_contents(self, random_spool):
        """Ensure the patch content matches the dataframe."""
        new = random_spool.chunk(time=2)
        # get contents of chunked spool
        chunk_df = new.get_contents()
        new_patches = list(new)
        new_spool = dc.spool(new_patches)
        # get content of spool created from patches in chunked spool.
        new_content = new_spool.get_contents()
        # these should be (nearly) identical.
        common = set(chunk_df.columns) & set(new_content.columns)
        cols = sorted(common - {"history"})  # no need to compare history
        assert chunk_df[cols].equals(new_content[cols])

    def test_merge_empty_spool(self, tmp_path_factory):
        """Ensure merge doesn't raise on empty spools."""
        spool = dc.spool([])
        merged = spool.chunk(time=None)
        assert len(merged) == 0

    def test_chunk_across_boundary(self, random_spool, random_spool_df):
        """Ensure query across a boundary works."""
        df = random_spool_df
        dt = df["time_step"][0]
        time_1 = df.loc[0, "time_max"] - to_timedelta64(1.00000123123)
        time_2 = time_1 + to_timedelta64(1)
        spool = random_spool.select(time=(time_1, time_2)).chunk(time=None)
        assert len(spool) == 1
        patch = spool[0]
        assert np.abs(patch.attrs["time_min"] - time_1) < dt
        assert np.abs(patch.attrs["time_max"] - time_2) < dt

    def test_uneven_chunk_iteration(self, random_spool, random_spool_df):
        """Ensure uneven start/end still yield consistent slices."""
        df = random_spool_df
        dt = df["time_step"][0]
        one_sec = to_timedelta64(1)
        time_1 = df.loc[0, "time_max"] - to_timedelta64(1.00000123123)
        time_2 = time_1 + to_timedelta64(10)
        spool_2 = random_spool.select(time=(time_1, time_2)).chunk(time=1)
        assert len(spool_2) == 10
        patches = list(spool_2)
        durations = [x.attrs["time_max"] - x.attrs["time_min"] for x in patches]
        # there should ba a single duration
        assert len(set(durations)) == 1
        duration = durations[0] / one_sec
        assert np.abs(duration - 1) <= (2.2 * dt / one_sec)

    def test_merge_1_dim_patches(self, memory_spool_dim_1_patches):
        """Ensure patches with one sample in time can be merged."""
        spool = memory_spool_dim_1_patches
        # patches should have
        new = spool.chunk(time=None)
        assert len(new) == 1
        patch = new[0]
        assert patch.attrs.time_min == spool[0].attrs.time_min
        assert patch.attrs.time_max == spool[-1].attrs.time_max
        assert patch.attrs.time_step == spool[0].attrs.time_step

    def test_small_segments_no_partial(self, diverse_spool):
        """Test issue #262 with no partials."""
        spool = diverse_spool.chunk(time=10)
        contents = spool.get_contents()
        duration = contents["time_max"] - contents["time_min"]
        dt = contents["time_step"]
        assert (((duration + dt) / dc.to_timedelta64(1)) >= 10).all()

    def test_small_segments_with_partial(self, diverse_spool):
        """Test issue #262 with partials."""
        diverse_contents = diverse_spool.get_contents()
        spool = diverse_spool.chunk(time=10, keep_partial=True)
        contents = spool.get_contents()
        duration = contents["time_max"] - contents["time_min"]
        dt = contents["time_step"]
        dur_dt = duration + dt
        # First, there should be some times less than 10 seconds
        assert ((dur_dt / dc.to_timedelta64(1)) < 9).any()
        # and the far out time should still be there
        assert contents["time_min"].min() == diverse_contents["time_min"].min()

    def test_raise_increment_too_big(self, diverse_spool):
        """Ensure code raises an error if the increment is too large."""
        msg = "No segments with sufficient length"
        with pytest.raises(ChunkError, match=msg):
            diverse_spool.chunk(time=10000)

    def test_too_big_partial(self, diverse_spool):
        """When chunk is too large, all contiguous blocks should merge."""
        spool1 = diverse_spool.chunk(time=100000, keep_partial=True)
        spool2 = diverse_spool.chunk(time=...)
        assert spool1 == spool2

    def test_too_big_overlap_raises(self, diverse_spool):
        """Overlap > chunk an error should raise."""
        msg = "overlap is greater than chunk size"
        with pytest.raises(ParameterError, match=msg):
            diverse_spool.chunk(time=10, overlap=11)

    def test_issue_474(self, random_spool):
        """Ensure spools can be chunked with the duration reported by coord."""
        # See #474
        patch1 = random_spool.chunk(time=...)[0]
        duration = patch1.coords.coord_range("time")
        merged2 = random_spool.chunk(time=duration)
        patch2 = merged2[0]
        assert patch1.equals(patch2)

    def test_issue_475(self, diverse_spool):
        """Ensure the partially chunked spool can be merged."""
        # See #475
        spool = diverse_spool.chunk(time=3, overlap=1, keep_partial=True)
        merged_spool = spool.chunk(time=None)
        assert isinstance(merged_spool, dc.BaseSpool)
        assert len(merged_spool)


class TestChunkMerge:
    """Tests for merging patches together using chunk method."""

    @pytest.fixture()
    def desperate_spool_no_overlap(self, random_patch) -> dc.BaseSpool:
        """
        Create spool that do not overlap at all.
        Ensure the patches are not sorted in temporal order.
        """
        pa1 = random_patch
        t2 = random_patch.attrs["time_max"]
        time_step = random_patch.attrs["time_step"] * 1_000
        pa2 = random_patch.update_attrs(time_min=t2 + time_step)
        t3 = pa2.attrs["time_max"]
        pa3 = pa2.update_attrs(time_min=t3 + time_step)
        return dc.spool([pa2, pa1, pa3])

    @pytest.fixture()
    def spool_complete_overlap(self, random_patch) -> dc.BaseSpool:
        """Create a spool which overlaps each other completely."""
        return dc.spool([random_patch, random_patch])

    @pytest.fixture()
    def spool_slight_gap(self, random_patch) -> dc.BaseSpool:
        """Create a spool which has a 1.1 * dt gap."""
        pa1 = random_patch
        t2 = random_patch.attrs["time_max"]
        dt = random_patch.attrs["time_step"]
        pa2 = random_patch.update_attrs(time_min=t2 + dt * 1.1)
        t3 = pa2.attrs["time_max"]
        pa3 = pa2.update_attrs(time_min=t3 + dt * 1.1)
        return dc.spool([pa2, pa1, pa3])

    @pytest.fixture(scope="class")
    def adjacent_spool_overlap(self, adjacent_spool_no_overlap):
        """Create a spool with several patches that have 50% overlap."""
        patches = list(adjacent_spool_no_overlap)
        out = [patches[0]]
        for ind in range(1, len(patches)):
            previous = patches[ind - 1]
            current = patches[ind]
            new_time = previous.coords.get_array("time")[40]
            out.append(current.update_attrs(time_min=new_time))
        return dc.spool(out)

    @pytest.fixture(scope="class")
    def adjacent_spool_monotonic(self, wacky_dim_patch):
        """Create a spool with no overlap that isnt evenly sampled."""
        pa1 = wacky_dim_patch
        dt = dc.to_timedelta64(0.2)
        pa2 = pa1.update_attrs(time_min=pa1.attrs.time_max + dt)
        return dc.spool([pa1, pa2])

    @pytest.fixture(scope="class")
    def adjacent_spool_monotonic_overlap(self, wacky_dim_patch):
        """Create a spool with overlap that isnt evenly sampled."""
        pa1 = wacky_dim_patch
        dt = dc.to_timedelta64(1)
        pa2 = pa1.update_attrs(time_min=pa1.attrs.time_max - dt)
        return dc.spool([pa1, pa2])

    @pytest.fixture(scope="class")
    def distance_adjacent(self, random_patch):
        """Create a spool with two distance adjacent patches."""
        pa1 = random_patch
        new_dist = pa1.attrs.distance_min + pa1.attrs.distance_step
        pa2 = pa1.update_attrs(distance_min=new_dist)
        return dc.spool([pa1, pa2])

    @pytest.fixture(scope="class")
    def distance_adjacent_no_order(self, wacky_dim_patch):
        """Create a spool with two distance adjacent patches."""
        pa1 = wacky_dim_patch
        new_dist = pa1.attrs.distance_min + 1
        pa2 = pa1.update_attrs(distance_min=new_dist)
        return dc.spool([pa1, pa2])

    @pytest.fixture(scope="class")
    def adjacent_spool_different_attrs(self, adjacent_spool_no_overlap):
        """An adjacent spool with on attribute that is different on each patch."""
        out = []
        for num, patch in enumerate(adjacent_spool_no_overlap):
            out.append(patch.update_attrs(my_attr=num))
        # since
        return dc.spool(out)

    @pytest.fixture(scope="class")
    def patches_conflicting_private_coord(self, random_patch):
        """Create two patches that have conflicting private coords."""
        dist_ax = random_patch.get_axis("distance")
        rand = np.random.RandomState(42)
        c1 = rand.random(random_patch.shape[dist_ax])
        c2 = rand.random(c1.shape)

        time = random_patch.get_coord("time")
        p1 = random_patch.update_coords(_bad_coord=("distance", c1))
        p2 = random_patch.update_coords(
            _bad_coord=("distance", c2), time=time + time.coord_range()
        )
        return p1, p2

    def test_merge_unequal_other(self, distance_adjacent):
        """When distance values are not equal time shouldn't be merge-able."""
        with pytest.raises(CoordMergeError):
            distance_adjacent.chunk(time=...)

    def test_merge_adjacent(self, adjacent_spool_no_overlap):
        """Test simple merge of patches."""
        len_1 = len(adjacent_spool_no_overlap)
        out_spool = adjacent_spool_no_overlap.chunk(time=None)
        assert len(out_spool) < len_1
        assert len(out_spool) == 1
        out_patch = out_spool[0]
        # make sure coords are consistent with attrs
        assert out_patch.attrs["time_max"] == out_patch.coords.max("time")
        assert out_patch.attrs["time_min"] == out_patch.coords.min("time")
        # ensure the spacing is still uniform
        time = out_patch.coords.get_array("time")
        spacing = time[1:] - time[:-1]
        unique_spacing = np.unique(spacing)
        assert len(unique_spacing) == 1
        assert unique_spacing[0] == out_patch.attrs["time_step"]

    def test_no_overlap(self, desperate_spool_no_overlap):
        """Spools with no overlap should not be merged."""
        len_1 = len(desperate_spool_no_overlap)
        out = desperate_spool_no_overlap.chunk(time=None)
        assert len_1 == len(out)

    def test_complete_overlap(self, spool_complete_overlap, random_patch):
        """Ensure complete overlap results in dropped data for overlap section."""
        out = spool_complete_overlap.chunk(time=None)
        assert len(out) == 1
        pa = out[0]
        data = pa.data
        assert data.shape == random_patch.data.shape

    def test_slight_gap(self, spool_slight_gap):
        """Ensure gaps slightly more than 1 time interval still work."""
        out = spool_slight_gap.chunk(time=None)
        assert len(out) == 1

    def test_ellipsis(self, spool_slight_gap):
        """Ensure ellipsis does the same as none."""
        pa1 = spool_slight_gap.chunk(time=...)
        pa2 = spool_slight_gap.chunk(time=None)
        assert pa1 == pa2

    def test_merge_transposed_patches(self, spool_complete_overlap):
        """Ensure if one of the patches is transposed merge still works."""
        # TODO for now this won't work; its probably a silly edge case to complicate
        # the code over, but maybe revisit.

    def test_merge_monotonic_no_overlap(self, adjacent_spool_monotonic):
        """Ensure monotonic coords can merge."""
        sp = adjacent_spool_monotonic.chunk(time=...)
        assert len(sp) == 1
        pa = sp[0]
        assert isinstance(pa, dc.Patch)
        # the time coordinate should remain sorted but not evenly sampled.
        time = pa.coords.coord_map["time"]
        assert time.sorted
        assert not time.evenly_sampled

    def test_merge_monotonic_overlap(self, adjacent_spool_monotonic_overlap):
        """Ensure monotonic overlaps are eliminated."""
        old_sp = adjacent_spool_monotonic_overlap
        sp = old_sp.chunk(time=...)
        assert len(sp) == 1
        # basic patch check
        pa = sp[0]
        assert isinstance(pa, dc.Patch)
        # the time coordinate should remain sorted but not evenly sampled.
        time = pa.coords.coord_map["time"]
        assert time.sorted
        assert not time.evenly_sampled
        # times should remain the same
        old_df = old_sp.get_contents()
        new_df = sp.get_contents()
        assert old_df["time_min"].min() == new_df["time_min"].min()
        assert old_df["time_max"].max() == new_df["time_max"].max()

    def test_merge_distance(self, distance_adjacent):
        """Ensure distance dims can be merged for monotonic case."""
        sp = distance_adjacent.chunk(distance=...)
        assert len(sp) == 1
        pa = sp[0]
        assert isinstance(pa, dc.Patch)
        # distance_step should remain identical
        assert distance_adjacent[0].attrs.distance_step == sp[0].attrs.distance_step
        # ensure correct bounds are there.
        old_df = distance_adjacent.get_contents()
        new_df = sp.get_contents()
        assert old_df["distance_min"].min() == new_df["distance_min"].min()
        assert old_df["distance_max"].max() == new_df["distance_max"].max()

    def test_merge_distance_no_order(self, distance_adjacent_no_order):
        """Ensure distance can be merged with unsorted coords."""
        sp = distance_adjacent_no_order.chunk(distance=...)
        assert len(sp) == 1
        pa = sp[0]
        assert isinstance(pa, dc.Patch)
        assert not np.any(pd.isnull(pa.coords.get_array("distance")))

    def test_merge_patches_close_dt(self, memory_spool_small_dt_differences):
        """Slightly different dt values should still merge."""
        old_spool = memory_spool_small_dt_differences
        new_spool = old_spool.chunk(time=None)
        old_contents = old_spool.get_contents()
        time_step_expected = get_middle_value(old_contents["time_step"])
        assert len(new_spool) == 1
        # need to iterate to make sure patch can be loaded.
        for patch in new_spool:
            assert isinstance(patch, dc.Patch)
            assert patch.attrs.time_step == time_step_expected

    def test_merge_patches_very_different_dt(self, memory_spool_small_dt_differences):
        """Slightly different dt values should still merge."""
        spool = memory_spool_small_dt_differences
        patches_1 = [x for x in spool]
        # create new patches with higher dt, this creates overlap that should
        # be trimmed out.
        patches_2 = [x.update_attrs(time_step=x.attrs.time_step * 33) for x in spool]
        patches = patches_2 + patches_1
        random.shuffle(patches)  # mix the patches, ensure order isnt required.
        new_spool = dc.spool(patches).chunk(time=None)
        assert len(new_spool) == 2
        time_steps = new_spool.get_contents()["time_step"]
        for patch, time_step in zip(new_spool, time_steps):
            diff = np.abs(patch.attrs.time_step - time_step)
            assert diff / time_step < 0.01

    def test_overlap_merge_doesnt_change_dt(self, adjacent_spool_overlap):
        """Trimming overlap shouldn't change dt."""
        spool = adjacent_spool_overlap.chunk(time=None)
        contents = spool.get_contents()
        assert len(spool) == 1
        patch_new, patch_old = spool[0], adjacent_spool_overlap[0]
        new_at, old_at = patch_new.attrs, patch_old.attrs
        assert new_at["time_max"] == contents["time_max"].max()
        assert (
            new_at["time_step"] == old_at["time_step"] == contents["time_step"].iloc[0]
        )

    def test_perturbed_dt(self, spool_dt_perturbed):
        """Ensure patches still merge if dt is slightly off."""
        out = spool_dt_perturbed.chunk(time=...)
        assert len(out) == 1
        for patch in out:
            assert isinstance(patch, dc.Patch)

    def test_merge_select(self, adjacent_spool_no_overlap):
        """Ensure spools can be merged *then* selected."""
        # get start/endtimes to encompass the last half of the first patch.
        # and the first half of the second patch.
        df = adjacent_spool_no_overlap.get_contents().sort_values("time_min")
        time = (df["time_max"] - df["time_min"]) / 2 + df["time_min"]
        time_tup = (time.iloc[0], time.iloc[1])
        # merge spool together and select
        merged = adjacent_spool_no_overlap.chunk(time=...)
        selected = merged.select(time=time_tup)
        assert len(selected) == 1
        # get patch, double check start/endtime
        patch = selected[0]
        coord = patch.get_coord("time")
        time_min, time_max, time_step = coord.min(), coord.max(), coord.step
        assert time_min >= time_tup[0]
        assert (time_min - time_step) < time_tup[0]
        assert time_max <= time_tup[1]
        assert (time_max + time_step) > time_tup[1]

    def test_attrs_conflict(self, adjacent_spool_different_attrs):
        """Test various cases for specifying what to do when attrs conflict."""
        spool = adjacent_spool_different_attrs
        # when we don't specify to ignore or drop attrs this should raise.
        match = "all values for my_attr"
        with pytest.raises(CoordMergeError, match=match):
            spool.chunk(time=...)
        # however, when we specify drop attrs this shouldn't.
        out = spool.chunk(time=..., conflict="keep_first")
        assert isinstance(out, dc.BaseSpool)
        assert len(out) == 1
        # make sure we can read the patch
        patch = out[0]
        assert isinstance(patch, dc.Patch)

    def test_chunk_patches_with_non_coord(self, random_patch):
        """Tests for chunking when some patches have non coordinate dimensions."""
        patches = [random_patch.mean("time") for _ in range(3)]
        spool = dc.spool(patches)
        chunked = spool.chunk(time=None)
        # Since the time dims are NaN, this can't work.
        assert not len(chunked)

    def test_merge_with_conflicting_private_coords(
        self,
        patches_conflicting_private_coord,
    ):
        """
        Private coords that conflict should be dropped and not block merge
        when conflict="drop".

        Otherwise they should raise.
        """
        p1, p2 = patches_conflicting_private_coord
        merged_spool = dc.spool([p1, p2]).chunk(time=None, conflict="drop")
        merge_patch = merged_spool[0]
        assert len(merged_spool) == 1
        # Since the private coords conflicted, they should have been dropped.
        coord_names = list(merge_patch.coords.coord_map)
        assert not any([x.startswith("_") for x in coord_names])
        # Without conflict drop this should raise.
        with pytest.raises(CoordMergeError, match="conflict"):
            dc.spool([p1, p2]).chunk(time=None)[0]

    def test_chunk_merge_then_chunk_split(self, random_spool):
        """
        Test chaining chunk(time=...) followed by chunk(time=duration).
        See #533.
        """
        spool = random_spool

        # First merge all patches along time, then chunk into 2s segments
        chunk_1_spool = spool.chunk(time=...)
        result_spool = chunk_1_spool.chunk(time=2)

        # Should be able to access patches
        first_patch = result_spool[0]
        assert isinstance(first_patch, dc.Patch)

        # Should have more patches (chunking into smaller pieces)
        assert len(result_spool) > len(spool)

        # Verify NO patches have NaN values and dataframe consistency
        result_contents = result_spool.get_contents().reset_index(drop=True)
        for i, patch in enumerate(result_spool):
            # Assert no NaN values in patch attributes
            assert not pd.isna(patch.attrs["time_min"]), f"Patch {i} has NaN time_min"
            assert not pd.isna(patch.attrs["time_max"]), f"Patch {i} has NaN time_max"

            # Verify dataframe contains reasonable time values
            df_row = result_contents.iloc[i]
            df_time_min = dc.to_datetime64(df_row["time_min"])
            df_time_max = dc.to_datetime64(df_row["time_max"])

            # Dataframe times should not be NaN or invalid
            assert not pd.isna(df_time_min), f"DF row {i} has NaN time_min"
            assert not pd.isna(df_time_max), f"DF row {i} has NaN time_max"
            assert df_time_min <= df_time_max, f"DF row {i} has invalid time range"

    def test_multiple_chained_chunks(self, random_spool):
        """Test multiple chained chunk operations. See #533."""
        # Chain multiple chunk operations
        result_spool = random_spool.chunk(time=...).chunk(time=5).chunk(time=2)

        # Should be able to access all patches
        for i in range(len(result_spool)):
            patch = result_spool[i]
            assert isinstance(patch, dc.Patch)

    def test_chunk_split_then_merge(self, random_spool):
        """Test chaining chunk split followed by merge. See #533."""
        # First chunk into smaller pieces, then merge back
        result_spool = random_spool.chunk(time=1).chunk(time=...)

        # Should be able to access patches (this test the fix works)
        first_patch = result_spool[0]
        assert isinstance(first_patch, dc.Patch)

        # The merge operation should result in fewer patches than the chunked operation
        chunked_spool = random_spool.chunk(time=1)
        assert len(result_spool) <= len(chunked_spool)

        # Verify NO patches have NaN values and dataframe consistency
        result_contents = result_spool.get_contents().reset_index(drop=True)
        for i, patch in enumerate(result_spool):
            # Assert no NaN values in patch attributes
            assert not pd.isna(patch.attrs["time_min"]), f"Patch {i} has NaN time_min"
            assert not pd.isna(patch.attrs["time_max"]), f"Patch {i} has NaN time_max"

            # Verify dataframe contains reasonable time values
            df_row = result_contents.iloc[i]
            df_time_min = dc.to_datetime64(df_row["time_min"])
            df_time_max = dc.to_datetime64(df_row["time_max"])

            # Dataframe times should not be NaN or invalid
            assert not pd.isna(df_time_min), f"DF row {i} has NaN time_min"
            assert not pd.isna(df_time_max), f"DF row {i} has NaN time_max"
            assert df_time_min <= df_time_max, f"DF row {i} has invalid time range"
