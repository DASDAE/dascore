"""
Test file for chunk/merge.

This is separated from spool tests because these need to be quite
extensive.
"""

from __future__ import annotations

import random
import warnings

import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.utils.patch_assembly as assembly_module
from dascore.exceptions import ChunkError, CoordMergeError, ParameterError, UnitError
from dascore.units import get_quantity
from dascore.utils.misc import get_middle_value
from dascore.utils.patch import _get_merged_coord
from dascore.utils.patch_assembly import PatchAssembler, _match_merge_units
from dascore.utils.time import to_timedelta64


@pytest.fixture(scope="class")
def spool_dt_perturbed(random_patch) -> dc.BaseSpool:
    """Create a spool with patches that have slightly different dts."""
    dts = np.array((0.999722, 0.99985, 0.99973, 0.99986))
    current_max = random_patch.get_coord("time").max()
    patches = []
    for dt in dts:
        coords = random_patch.coords.update(time_min=current_max, time_step=dt)
        patch = random_patch.new(coords=coords)
        patches.append(patch)
    return dc.spool(patches)


class TestChunk:
    """Tests for merging/chunking patches."""

    @pytest.fixture(scope="class")
    def random_spool_df(self, random_spool):
        """Get contents and sort the contents of random_spool."""
        df = random_spool.get_contents().sort_values("time_min").reset_index(drop=True)
        return df

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
        # len fields may differ by ±1 between summary-based and data-based
        # counts; identity/provenance columns legitimately differ between
        # plan rows and re-scanned live patches
        skip = {
            "history",
            "source_path",
            "source_format",
            "source_version",
            "source_patch_key",
        }
        skip |= {c for c in common if c.endswith("_len")}
        cols = sorted(common - skip)
        comp1, comp2 = chunk_df[cols], new_content[cols]
        equal_cols = (comp1 == comp2) | (pd.isnull(comp1) & pd.isnull(comp2))
        assert equal_cols.all().all()

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
        time_coord = patch.get_coord("time")
        assert np.abs(time_coord.min() - time_1) < dt
        assert np.abs(time_coord.max() - time_2) < dt

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
        durations = [
            x.get_coord("time").max() - x.get_coord("time").min() for x in patches
        ]
        # there should be a single duration
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
        assert patch.coords["time"].min() == spool[0].coords["time"].min()
        assert patch.coords["time"].max() == spool[-1].coords["time"].max()
        assert patch.coords["time"].step == spool[0].coords["time"].step

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
        """Overlap >= chunk size should raise a clear error."""
        msg = "overlap is greater than or equal to chunk size"
        with pytest.raises(ParameterError, match=msg):
            diverse_spool.chunk(time=10, overlap=11)
        # Equal overlap would mean zero-stride segments; also rejected.
        with pytest.raises(ParameterError, match=msg):
            diverse_spool.chunk(time=10, overlap=10)

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
        time_coord = random_patch.coords["time"]
        t2 = time_coord.max()
        time_step = time_coord.step * 1_000
        pa2 = random_patch.new(
            coords=random_patch.coords.update(time_min=t2 + time_step)
        )
        t3 = pa2.coords["time"].max()
        pa3 = pa2.new(coords=pa2.coords.update(time_min=t3 + time_step))
        return dc.spool([pa2, pa1, pa3])

    @pytest.fixture()
    def spool_complete_overlap(self, random_patch) -> dc.BaseSpool:
        """Create a spool which overlaps each other completely."""
        return dc.spool([random_patch, random_patch])

    @pytest.fixture()
    def spool_slight_gap(self, random_patch) -> dc.BaseSpool:
        """Create a spool which has a 1.1 * dt gap."""
        pa1 = random_patch
        time_coord = random_patch.coords["time"]
        t2 = time_coord.max()
        dt = time_coord.step
        pa2 = random_patch.new(
            coords=random_patch.coords.update(time_min=t2 + dt * 1.1)
        )
        t3 = pa2.coords["time"].max()
        pa3 = pa2.new(coords=pa2.coords.update(time_min=t3 + dt * 1.1))
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
            out.append(current.new(coords=current.coords.update(time_min=new_time)))
        return dc.spool(out)

    @pytest.fixture(scope="class")
    def adjacent_spool_monotonic(self, wacky_dim_patch):
        """Create a spool with no overlap that isnt evenly sampled."""
        pa1 = wacky_dim_patch
        dt = dc.to_timedelta64(0.2)
        pa2 = pa1.new(
            coords=pa1.coords.update(time_min=pa1.get_coord("time").max() + dt)
        )
        return dc.spool([pa1, pa2])

    @pytest.fixture(scope="class")
    def adjacent_spool_monotonic_overlap(self, wacky_dim_patch):
        """Create a spool with overlap that isnt evenly sampled."""
        pa1 = wacky_dim_patch
        dt = dc.to_timedelta64(1)
        pa2 = pa1.new(coords=pa1.coords.update(time_min=pa1.coords["time"].max() - dt))
        return dc.spool([pa1, pa2])

    @pytest.fixture(scope="class")
    def distance_adjacent(self, random_patch):
        """Create a spool with two distance adjacent patches."""
        pa1 = random_patch
        distance = pa1.coords["distance"]
        new_dist = distance.min() + distance.step
        pa2 = pa1.new(coords=pa1.coords.update(distance_min=new_dist))
        return dc.spool([pa1, pa2])

    @pytest.fixture(scope="class")
    def distance_adjacent_no_order(self, wacky_dim_patch):
        """Create a spool with two distance adjacent patches."""
        pa1 = wacky_dim_patch
        new_dist = pa1.coords["distance"].min() + 1
        pa2 = pa1.new(coords=pa1.coords.update(distance_min=new_dist))
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
        """Unequal distance coords partition rather than raise (0.2 change).

        Patches whose non-chunked dimension coordinates differ are never
        combined; they simply land in separate output patches.
        """
        out = distance_adjacent.chunk(time=...)
        assert len(out) == len(distance_adjacent)

        # the differing distance envelopes are preserved, not merged/duplicated
        def _distance_envelopes(spool):
            df = spool.get_contents()
            return sorted(zip(df["distance_min"], df["distance_max"], strict=True))

        assert _distance_envelopes(out) == _distance_envelopes(distance_adjacent)

    def test_merge_adjacent(self, adjacent_spool_no_overlap):
        """Test simple merge of patches."""
        len_1 = len(adjacent_spool_no_overlap)
        out_spool = adjacent_spool_no_overlap.chunk(time=None)
        assert len(out_spool) < len_1
        assert len(out_spool) == 1
        out_patch = out_spool[0]
        # make sure coords are consistent with attrs
        assert out_patch.coords["time"].max() == out_patch.coords.max("time")
        assert out_patch.coords["time"].min() == out_patch.coords.min("time")
        # ensure the spacing is still uniform
        time = out_patch.coords.get_array("time")
        spacing = time[1:] - time[:-1]
        unique_spacing = np.unique(spacing)
        assert len(unique_spacing) == 1
        assert unique_spacing[0] == out_patch.coords["time"].step

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
        assert (
            distance_adjacent[0].coords["distance"].step
            == sp[0].coords["distance"].step
        )
        # ensure correct bounds are there.
        old_df = distance_adjacent.get_contents()
        new_df = sp.get_contents()
        assert old_df["distance_min"].min() == new_df["distance_min"].min()
        assert old_df["distance_max"].max() == new_df["distance_max"].max()

    def test_non_si_merge_tolerance_uses_coord_units(self, random_patch):
        """Canonical index steps are not interpreted in native coord units."""
        size = len(random_patch.get_coord("distance"))
        first = dc.get_coord(data=np.arange(size, dtype=float), units="km")
        second = dc.get_coord(
            data=np.arange(size, dtype=float) + size + 8,
            units="km",
        )
        patches = [
            random_patch.update_coords(distance=first),
            random_patch.update_coords(distance=second),
        ]
        # Index summaries store numeric dimension steps in canonical SI.
        summaries = pd.DataFrame({"distance_step": [1000.0, 1000.0]})
        manager = _get_merged_coord(
            summaries,
            "distance",
            [patch.coords for patch in patches],
            tolerance=1.5,
        )
        merged = manager.coord_map["distance"]
        assert not merged.evenly_sampled
        assert np.array_equal(
            merged.values, np.concatenate([first.values, second.values])
        )

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
            assert patch.coords["time"].step == time_step_expected

    def test_merge_patches_very_different_dt(self, memory_spool_small_dt_differences):
        """Slightly different dt values should still merge."""
        spool = memory_spool_small_dt_differences
        patches_1 = [x for x in spool]
        # create new patches with higher dt, this creates overlap that should
        # be trimmed out.
        patches_2 = [
            x.new(coords=x.coords.update(time_step=x.coords["time"].step * 33))
            for x in spool
        ]
        patches = patches_2 + patches_1
        random.shuffle(patches)  # mix the patches, ensure order isnt required.
        new_spool = dc.spool(patches).chunk(time=None)
        assert len(new_spool) == 2
        time_steps = new_spool.get_contents()["time_step"]
        for patch, time_step in zip(new_spool, time_steps):
            diff = np.abs(patch.coords["time"].step - time_step)
            assert diff / time_step < 0.01

    def test_overlap_merge_doesnt_change_dt(self, adjacent_spool_overlap):
        """Trimming overlap shouldn't change dt."""
        spool = adjacent_spool_overlap.chunk(time=None)
        contents = spool.get_contents()
        assert len(spool) == 1
        patch_new, patch_old = spool[0], adjacent_spool_overlap[0]
        new_time_coord = patch_new.get_coord("time")
        old_time_coord = patch_old.get_coord("time")
        assert new_time_coord.max() == contents["time_max"].max()
        assert (
            new_time_coord.step == old_time_coord.step == contents["time_step"].iloc[0]
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

    def test_missing_attr_is_a_conflict(self):
        """A member lacking an attr conflicts with one which states it."""
        p1 = dc.get_example_patch().update_attrs(data_type="velocity")
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        assert p2.attrs.data_type == ""
        with pytest.raises(CoordMergeError, match="data_type"):
            dc.spool([p1, p2]).chunk(time=None)
        out = dc.spool([p1, p2]).chunk(time=None, conflict="drop")
        assert len(out) == 1
        assert out[0].attrs.data_type == ""

    def test_surviving_member_takes_the_rows_attrs(self):
        """Whichever duplicate survives overlap removal, patch and row agree."""
        p1 = dc.get_example_patch().update_attrs(foo="a", data_type="velocity")
        p2 = dc.get_example_patch()  # same span, knows neither
        with pytest.raises(CoordMergeError, match=r"foo|data_type"):
            dc.spool([p1, p2]).chunk(time=None)
        # keep_first takes the first member's values, stated or not, and
        # the assembled patch says exactly what its row does.
        out = dc.spool([p1, p2]).chunk(time=None, conflict="keep_first")
        assert len(out) == 1
        assert out.get_contents().iloc[0]["foo"] == "a"
        assert out[0].attrs.foo == "a"
        # the other order keeps the first member's silence
        out = dc.spool([p2, p1]).chunk(time=None, conflict="keep_first")
        assert len(out) == 1
        assert pd.isnull(out.get_contents().iloc[0].get("foo"))
        assert out[0].attrs.get("foo") is None

    def test_attrs_named_like_coordinates_survive_a_chunk(self):
        """Attrs which look like coordinate metadata are still attrs."""
        extra = {
            "latitude": "north",  # a coordinate some other patch has
            "foo_min": "a",  # an envelope pair with no foo coordinate
            "foo_max": "b",
            "time_zone": "UTC",  # a name prefixed by a real dimension
            "gauge": 10 * dc.get_quantity("m"),  # a quantity
            "shots": 7,  # a plain number
        }
        p1 = dc.get_example_patch().update_attrs(**extra)
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        p2 = p2.update_attrs(**extra)
        n = p1.shape[p1.get_axis("distance")]
        # a patch which holds latitude as a coordinate rather than an attr
        elsewhere = dc.get_example_patch(
            time_min=time.max() + 10 * time.step, tag="other"
        ).update_coords(latitude=("distance", np.arange(n, dtype=float)))
        out = dc.spool([p2, p1, elsewhere]).chunk(time=None)
        merged = out.select(tag="random")[0]
        row = out.get_contents().set_index("tag").loc["random"]
        for name, value in extra.items():
            assert merged.attrs.get(name) == value
        assert row["time_zone"] == "UTC"

    def test_dropped_coordinate_envelope_is_not_an_attr(self):
        """A coordinate only one member has leaves no stray attrs behind."""
        p1 = dc.get_example_patch()
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        n = p1.shape[p1.get_axis("distance")]
        p1 = p1.update_coords(latitude=("distance", np.arange(n, dtype=float)))
        out = dc.spool([p1, p2]).chunk(time=None, conflict="drop")
        patch = out[0]
        assert patch.attrs.get("latitude_min") is None
        assert patch.attrs.get("latitude_max") is None

    def test_history_warns_not_raises(self):
        """Differing histories merge with a warning, carrying the first's."""
        p1 = dc.get_example_patch()
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        p2 = p2.pass_filter(time=(None, 10))
        with pytest.warns(UserWarning, match="histories differ"):
            patch = dc.spool([p1, p2]).chunk(time=None)[0]
        assert patch.shape[1] == 2 * p1.shape[1]
        assert patch.attrs.history == p1.attrs.history

    def test_attrs_conflict(self, adjacent_spool_different_attrs):
        """Test various cases for specifying what to do when attrs conflict."""
        spool = adjacent_spool_different_attrs
        # when we don't specify to ignore or drop attrs this should raise.
        match = "my_attr holds conflicting values"
        with pytest.raises(CoordMergeError, match=match):
            spool.chunk(time=...)
        # however, when we specify drop attrs this shouldn't.
        out = spool.chunk(time=..., conflict="keep_first")
        assert isinstance(out, dc.BaseSpool)
        assert len(out) == 1
        # make sure we can read the patch
        patch = out[0]
        assert isinstance(patch, dc.Patch)

    def test_invalid_conflict_raises(self, adjacent_spool_different_attrs):
        """
        An unrecognized conflict value should raise rather than silently
        selecting undocumented behavior. See #804.
        """
        spool = adjacent_spool_different_attrs
        for bad_value in ("banana", "", None):
            with pytest.raises(ParameterError, match="conflict must be one of"):
                spool.chunk(time=..., conflict=bad_value)

    def test_chunk_patches_with_non_coord(self, random_patch):
        """Tests for chunking when some patches have non coordinate dimensions."""
        patches = [random_patch.mean("time") for _ in range(3)]
        spool = dc.spool(patches)
        # Losing patches silently would be data loss; this raises by default
        # (0.2 change) with missing_dim="drop" restoring the old behavior.
        with pytest.raises(ChunkError, match="missing_dim"):
            spool.chunk(time=None)
        chunked = spool.chunk(time=None, missing_dim="drop")
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
            time_coord = patch.get_coord("time")
            assert not pd.isna(time_coord.min()), f"Patch {i} has NaN time_min"
            assert not pd.isna(time_coord.max()), f"Patch {i} has NaN time_max"

            # Verify dataframe contains reasonable time values
            df_row = result_contents.iloc[i]
            df_time_min = dc.to_datetime64(df_row["time_min"])
            df_time_max = dc.to_datetime64(df_row["time_max"])

            # Dataframe times should not be NaN or invalid
            assert not pd.isna(df_time_min), f"DF row {i} has NaN time_min"
            assert not pd.isna(df_time_max), f"DF row {i} has NaN time_max"
            assert df_time_min <= df_time_max, f"DF row {i} has invalid time range"

    def test_chunk_non_adjacent_within_tolerance_warns(self, random_patch):
        """
        Non-adjacent patches can still merge, but the coordinate type may change.

        In this case a warning should be issued. See #662.
        """
        base = random_patch.update_attrs(history="")
        time = random_patch.get_coord("time")

        # Case 1: the patches should in fact merge with no warning.
        patch_no_gap = base.update_coords(time_min=time.max() + time.step).update_attrs(
            history=""
        )
        # No warning should be raised.
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            out = dc.spool((base, patch_no_gap)).chunk(time=None)
        assert len(out) == 1
        assert out[0].get_coord("time").step is not None

        # Case 2: The patches should not merge.
        patch_w_gap = base.update_coords(
            time_min=time.max() + time.step * 5,
        ).update_attrs(history="")
        # No warning, no merge.
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            out = dc.spool((base, patch_w_gap)).chunk(time=None)
        assert len(out) == 2

        # Case 3: The patches should merge and a warning issued.
        match = "There is a gap in the patch along dimension time"
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            out = dc.spool((base, patch_w_gap)).chunk(time=None, tolerance=10)
        assert len(caught_warnings) == 1
        assert match in str(caught_warnings[0].message)
        assert caught_warnings[0].filename == __file__
        assert len(out) == 1


def _bare_assembler():
    """An assembler with no frames, for direct streaming-merge tests."""
    return PatchAssembler(load_patch=None, merge_kwargs={})


class TestStreamingMerge:
    """
    Tests for the streaming merge path, which copies each patch into a
    pre-allocated array rather than concatenating all loaded patches.
    """

    def test_streaming_path_used(self, adjacent_spool_no_overlap, monkeypatch):
        """Ensure simple merges take the streaming path."""
        called = []
        original = PatchAssembler._merge_patches_streaming

        def wrapper(self, *args, **kwargs):
            called.append(True)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(PatchAssembler, "_merge_patches_streaming", wrapper)
        merged = adjacent_spool_no_overlap.chunk(time=None)
        assert isinstance(merged[0], dc.Patch)
        assert called

    def test_matches_materialized_merge(self, adjacent_spool_no_overlap, monkeypatch):
        """Streaming and concatenating merges must produce identical patches."""
        streamed = adjacent_spool_no_overlap.chunk(time=None)[0]
        # Disabling the sample estimate forces the materialized path.
        monkeypatch.setattr(
            assembly_module, "_estimate_merge_samples", lambda df, dim: None
        )
        materialized = adjacent_spool_no_overlap.chunk(time=None)[0]
        assert np.array_equal(streamed.data, materialized.data)
        assert streamed.coords == materialized.coords

    @pytest.mark.parametrize("small_first", [True, False])
    def test_mixed_dtype_upcast(self, random_patch, small_first):
        """Merging mixed dtypes must upcast like np.concatenate."""
        p1 = random_patch.update_attrs(history=[])
        time = p1.get_coord("time")
        p2 = p1.update_coords(time_min=time.max() + time.step).update_attrs(history=[])
        if small_first:
            p1 = p1.update(data=p1.data.astype(np.float32))
        else:
            p2 = p2.update(data=p2.data.astype(np.float32))
        merged = dc.spool([p1, p2]).chunk(time=None)[0]
        assert merged.data.dtype == np.float64

    def test_transposes_patch_to_first_patch_dims(self, random_patch, monkeypatch):
        """Streaming merge should tolerate patches with the same dims reordered."""
        assembler = _bare_assembler()
        p1 = random_patch.update_attrs(history=[])
        time = p1.get_coord("time")
        p2 = p1.update_coords(time_min=time.max() + time.step).update_attrs(history=[])
        p2 = p2.transpose(*reversed(p2.dims))
        patches = iter([p1, p2])
        monkeypatch.setattr(
            assembler, "_load_trimmed_patch", lambda patch_kwargs, joined: next(patches)
        )
        time_axis = p1.get_axis("time")
        samples = p1.data.shape[time_axis] * 2
        out = assembler._merge_patches_streaming(None, [{}, {}], "time", samples)
        assert out.dims == p1.dims
        assert out.data.shape[time_axis] == samples

    def test_incompatible_shapes_raise_merge_error(self, random_patch, monkeypatch):
        """Streaming merge should wrap non-merge-dimension shape mismatches."""
        assembler = _bare_assembler()
        p1 = random_patch.update_attrs(history=[])
        time = p1.get_coord("time")
        p2 = p1.update_coords(time_min=time.max() + time.step).update_attrs(history=[])
        distance = p2.get_coord("distance")
        p2 = p2.select(distance=(None, distance.max() - distance.step))
        patches = iter([p1, p2])
        monkeypatch.setattr(
            assembler, "_load_trimmed_patch", lambda patch_kwargs, joined: next(patches)
        )
        msg = "their shapes are incompatible"
        with pytest.raises(CoordMergeError, match=msg):
            samples = p1.data.shape[p1.get_axis("time")] * 2
            assembler._merge_patches_streaming(None, [{}, {}], "time", samples)

    def test_unexpected_merge_dimension_raises(self, random_patch, monkeypatch):
        """Streaming merge should validate the actual varying dimension."""
        assembler = _bare_assembler()
        p1 = random_patch.update_attrs(history=[])
        dist = p1.get_coord("distance")
        p2 = p1.update_coords(distance_min=dist.max() + dist.step).update_attrs(
            history=[]
        )
        patches = iter([p1, p2])
        monkeypatch.setattr(
            assembler, "_load_trimmed_patch", lambda patch_kwargs, joined: next(patches)
        )
        msg = "expected them to vary along time"
        with pytest.raises(CoordMergeError, match=msg):
            samples = p1.data.shape[p1.get_axis("time")] * 2
            assembler._merge_patches_streaming(None, [{}, {}], "time", samples)


class TestDescendingChunk:
    """Public chunk behavior for descending coordinates (2026-07-18 F5)."""

    def test_contiguous_descending_patches_merge(self):
        """Two contiguous descending patches chunk into one patch."""
        p = dc.get_example_patch()
        flipped = p.flip("time")
        t = p.get_coord("time")
        span = t.max() - t.min() + t.step
        shifted = flipped.update_coords(time=flipped.get_coord("time").data + span)
        merged = dc.spool([shifted, flipped]).chunk(time=None, conflict="drop")
        assert len(merged) == 1
        patch = merged[0]
        time = patch.get_coord("time")
        assert time.reverse_sorted
        n_time = p.shape[p.get_axis("time")]
        assert patch.shape[patch.get_axis("time")] == 2 * n_time
        assert time.min() == t.min()


class TestMixedUnitChunk:
    """Chunk partitioning and merging across unit differences."""

    @staticmethod
    def _shifted(patch, units=None):
        """The example patch shifted to be distance-contiguous, in units."""
        d = patch.get_coord("distance")
        span = d.max() - d.min() + d.step
        values = d.data + span
        if units == "ft":
            values = values / 0.3048
        out = patch.update_coords(distance=values)
        return out.set_units(distance=units) if units else out

    def test_incompatible_dimensionality_splits(self):
        """Metre and second patches with contiguous SI magnitudes stay apart."""
        p = dc.get_example_patch()
        pm = p.set_units(distance="m")
        ps = self._shifted(p, "s")
        sp = dc.spool([pm, ps])
        plan = sp.chunk_plan(distance=None)
        assert len(plan.outputs) == 2
        out = sp.chunk(distance=None, conflict="drop")
        assert {str(x.get_coord("distance").units) for x in out} == {"1 m", "1 s"}

    def test_unitless_and_unitful_split(self):
        """A unitless patch never merges with a unitful one."""
        p = dc.get_example_patch()
        sp = dc.spool([p.set_units(distance="m"), self._shifted(p)])
        assert len(sp.chunk(distance=None, conflict="drop")) == 2

    def test_compatible_units_convert_and_merge(self):
        """Metres and feet (one dimensionality) merge, converted, unit-true."""
        p = dc.get_example_patch()
        pm = p.set_units(distance="m")
        pf = self._shifted(p, "ft")
        out = dc.spool([pm, pf]).chunk(distance=None, conflict="drop")
        assert len(out) == 1
        patch = out[0]
        coord = patch.get_coord("distance")
        assert str(coord.units) == "1 m"
        n = p.shape[p.get_axis("distance")]
        assert patch.shape[patch.get_axis("distance")] == 2 * n
        assert float(coord.max()) == pytest.approx(2 * n - 1)

    def test_mixed_spelling_members_trim_in_plan_units(self):
        """Length-chunking across m and ft members loses no samples.

        Plan trims are magnitudes in the partition's normalized unit; a
        member stored under another spelling must convert them, not read
        them natively (adversarial round, D1).
        """
        pm = dc.get_example_patch().set_units(distance="m")
        d = pm.get_coord("distance")
        span = float(d.max() - d.min() + d.step)
        values = (d.data + span) / 0.3048
        pf = pm.update_coords(distance=values).set_units(distance="ft")
        sp = dc.spool([pm, pf])
        out = sp.chunk(distance=200, conflict="keep_first", keep_partial=True)
        n = pm.shape[pm.get_axis("distance")]
        assert sum(p.shape[p.get_axis("distance")] for p in out) == 2 * n
        assert all(p.shape[p.get_axis("distance")] for p in out)  # none empty

    @staticmethod
    def _continuous_mixed_spool():
        """Metre and feet patches covering one continuous 600 m span."""
        pm = dc.get_example_patch().set_units(distance="m")
        d = pm.get_coord("distance")
        span = float(d.max() - d.min() + d.step)
        pf = pm.update_coords(distance=(d.data + span) / 0.3048)
        return dc.spool([pm, pf.set_units(distance="ft")])

    def test_single_member_output_speaks_plan_units(self):
        """An output the merge never visits still matches its plan row.

        Merging converts its members to one unit, but an output drawing
        on a single member skips that path — so a feet member could be
        published under a row claiming metres, making `get_contents`
        describe an envelope no patch it yields actually has.
        """
        spool = self._continuous_mixed_spool().chunk(
            distance=200, keep_partial=True, conflict="keep_first"
        )
        df = spool.get_contents()
        assert set(df["distance_units"]) == {"m"}
        for patch, (_, row) in zip(spool, df.iterrows(), strict=True):
            coord = patch.get_coord("distance")
            assert str(coord.units) == "1 m"
            assert float(coord.min()) == pytest.approx(row["distance_min"])
            assert float(coord.max()) == pytest.approx(row["distance_max"])
        # the envelope the last row advertises is selectable
        assert len(spool.select(distance=(400, 599))) == 1

    def test_provenance_column_survives_a_like_named_coord(self):
        """A coordinate named ``{dim}_source`` is not mistaken for units.

        Its own unit column is ``_distance_source_units``, which the
        provenance column must not collide with.
        """
        spool = self._continuous_mixed_spool()
        patches = []
        for patch in spool:
            size = patch.coord_shapes["distance"][0]
            values = np.arange(size, dtype=float)
            new = patch.update_coords(distance_source=("distance", values))
            patches.append(new.set_units(distance_source="s"))
        chunked = dc.spool(patches).chunk(
            distance=200, keep_partial=True, conflict="keep_first"
        )
        rows = chunked._catalog.resolver.member_rows
        assert set(rows["_distance_units_source"]) == {"m", "ft"}
        assert set(rows["_distance_units"]) == {"m"}
        n = sum(p.shape[p.get_axis("distance")] for p in chunked)
        assert n == 600

    def test_rechunk_keeps_one_source_units_column(self):
        """Re-chunking a derived view must not duplicate the unit column.

        The plan records the file's own spelling beside its normalized
        unit; renaming again on the second pass produced two columns of
        one name, and one of them silently vanished when the rows became
        load kwargs — so a member could be trimmed in the wrong unit.
        """
        pm = dc.get_example_patch().set_units(distance="m")
        d = pm.get_coord("distance")
        span = float(d.max() - d.min() + d.step)
        pf = pm.update_coords(distance=(d.data + span) / 0.3048)
        pf = pf.set_units(distance="ft")
        first = dc.spool([pm, pf]).chunk(distance=200, keep_partial=True)
        second = first.chunk(distance=None, conflict="keep_first")
        rows = second._catalog.resolver.member_rows
        assert not rows.columns.duplicated().any()
        # the file's own spelling survives, not the plan's normalized unit
        assert set(rows["_distance_units_source"]) == {"m", "ft"}

    def test_affine_quantity_length_is_a_delta(self):
        """20 degC of extent is 36 degF, never 68 (adversarial round, D2)."""
        degf = dc.get_example_patch().set_units(distance="degF")
        length = 20 * dc.get_quantity("degC")
        out = dc.spool([degf]).chunk(distance=length, keep_partial=True)
        coord = out[0].get_coord("distance")
        assert float(coord.max() - coord.min()) <= 36.1

    def test_scaled_unit_quantity_length(self):
        """A scaled unit spelling converts for chunk lengths too."""
        patch = dc.get_example_patch().set_units(distance="1e-9 strain/s")
        length = 50 * dc.get_quantity("1e-9 strain/s")
        out = dc.spool([patch]).chunk(distance=length, keep_partial=True)
        assert len(out) == 6

    def test_same_units_unchanged(self):
        """The ordinary same-unit merge keeps its behavior and units."""
        p = dc.get_example_patch()
        sp = dc.spool([p.set_units(distance="m"), self._shifted(p, "m")])
        out = sp.chunk(distance=None, conflict="drop")
        assert len(out) == 1
        assert str(out[0].get_coord("distance").units) == "1 m"


class TestNonSIUnitTrim:
    """Plan trims speak the coordinate's own units."""

    @pytest.fixture()
    def foot_spool(self):
        """A one patch spool whose distance coordinate is in feet."""
        return dc.spool([dc.get_example_patch().convert_units(distance="ft")])

    def test_chunk_keeps_every_sample(self, foot_spool):
        """Chunking a non-SI coordinate must not drop data."""
        patch = foot_spool[0]
        axis = patch.get_axis("distance")
        expected = patch.shape[axis]
        # A quantity length of 100 m is an exact multiple of the 1 m
        # sample step, so every sample belongs to some chunk; a length
        # that is not a step multiple loses boundary samples on any
        # coordinate, units aside (#870). keep_partial so the chunks
        # cover the coordinate exactly.
        length = 100 * dc.get_quantity("m")
        out = foot_spool.chunk(distance=length, keep_partial=True)
        assert sum(x.shape[x.get_axis("distance")] for x in out) == expected

    def test_chunk_pieces_match_the_plan(self, foot_spool):
        """Each assembled piece spans the interval the plan advertised."""
        length = 100 * dc.get_quantity("m")
        plan = foot_spool.chunk_plan(distance=length)
        out = foot_spool.chunk(distance=length)
        assert len(out) == len(plan.outputs)
        for patch, (_, row) in zip(out, plan.outputs.iterrows(), strict=True):
            coord = patch.get_coord("distance")
            # the plan speaks the coordinate's own feet
            assert float(coord.min()) == pytest.approx(row["distance_min"])
            assert float(coord.max()) == pytest.approx(row["distance_max"])

    def test_trim_is_the_same_physical_interval(self, foot_spool):
        """A 100 m chunk covers 100 m of a coordinate stored in feet."""
        out = foot_spool.chunk(distance=100 * dc.get_quantity("m"))
        span = out[0].get_coord("distance").convert_units("m")
        assert float(span.max() - span.min()) == pytest.approx(99, abs=1.0)

    def test_bare_length_means_native_units(self, foot_spool):
        """A bare 100 means 100 of the coordinate's own feet."""
        out = foot_spool.chunk(distance=100, keep_partial=True)
        coords = [x.get_coord("distance") for x in out]
        assert all(float(c.max() - c.min()) <= 100 for c in coords)
        # ~981 ft of coordinate cut into 100 ft pieces
        assert len(out) == 10

    def test_si_coord_unchanged(self):
        """The ordinary SI case keeps its behavior."""
        spool = dc.spool([dc.get_example_patch()])
        patch = spool[0]
        axis = patch.get_axis("distance")
        out = spool.chunk(distance=None)
        assert sum(x.shape[x.get_axis("distance")] for x in out) == patch.shape[axis]

    def test_unitless_coord_unchanged(self):
        """A coordinate with no units is trimmed by bare magnitudes."""
        patch = dc.get_example_patch().set_units(distance=None)
        spool = dc.spool([patch])
        axis = patch.get_axis("distance")
        out = spool.chunk(distance=None)
        assert sum(x.shape[x.get_axis("distance")] for x in out) == patch.shape[axis]


class TestChainedChunk:
    """Chunking a derived spool along another dimension (round-4 F1)."""

    def test_other_dim_keeps_prior_boundaries(self):
        """Re-chunking distance must not undo a time concatenation."""
        p1 = dc.get_example_patch()
        t = p1.get_coord("time")
        p2 = p1.update_coords(time_min=t.max() + t.step)
        merged = dc.spool([p1, p2]).chunk(time=None, conflict="drop")
        current = merged[0]
        d = current.get_coord("distance")
        size = (d.max() - d.min()) / 2
        actual = merged.chunk(distance=size, keep_partial=True, conflict="drop")
        expected = dc.spool([current]).chunk(
            distance=size, keep_partial=True, conflict="drop"
        )
        assert sorted(x.shape for x in actual) == sorted(x.shape for x in expected)
        got = {
            (str(x.get_coord("time").min()), str(x.get_coord("time").max()))
            for x in actual
        }
        want = {
            (str(x.get_coord("time").min()), str(x.get_coord("time").max()))
            for x in expected
        }
        assert got == want

    def test_segment_then_segment(self):
        """chunk(time=...) then chunk(distance=...) partitions both dims."""
        p = dc.get_example_patch()  # (300, 2000), 8 s
        out = dc.spool([p]).chunk(time=2).chunk(distance=100)
        assert len(out) == 12
        assert {x.shape for x in out} == {(100, 500)}

    def test_same_dim_rechunk_still_collapses(self):
        """Re-chunking the same dim re-plans from members (no nesting)."""
        p1 = dc.get_example_patch()
        t = p1.get_coord("time")
        p2 = p1.update_coords(time_min=t.max() + t.step)
        merged = dc.spool([p1, p2]).chunk(time=None, conflict="drop")
        rechunk = merged.chunk(time=2)
        assert len(rechunk) == 8
        assert {x.shape for x in rechunk} == {(300, 500)}

    def test_size_after_merge(self, random_spool):
        """A merged spool still knows its dtype."""
        target = dc.get_quantity("1 MB")
        out = random_spool.chunk(time=None).chunk(time=target)
        assert max(x.data.nbytes for x in out) <= target.to("byte").magnitude

    def test_size_after_other_dim_chunk(self, random_spool):
        """A derived catalog carries the dtype to the next chunk."""
        target = dc.get_quantity("1 MB")
        out = random_spool.chunk(distance=100).chunk(time=target)
        assert max(x.data.nbytes for x in out) <= target.to("byte").magnitude

    def test_size_then_size(self, random_spool):
        """Chunking by size twice narrows the plan monotonically.

        This asserts on the plan rather than the assembled patches: a
        same-dim rechunk currently mis-assembles one output regardless of
        how the length is expressed (a plain `chunk(time=3.332).chunk(
        time=1.664)` hits it too). See #832; tighten this to assert
        `nbytes` once that is fixed.
        """
        first = random_spool.chunk(time=2 * dc.units.MB)
        plan = first.chunk_plan(time=1 * dc.units.MB)
        (part,) = plan.params["size"]["partitions"]
        step = plan.outputs["time_step"].iloc[0]
        spans = plan.outputs["time_max"] - plan.outputs["time_min"] + step
        samples = (spans / step).round().astype(int)
        assert (samples <= part["n_samples"]).all()
        assert len(plan.outputs) > len(first)


class TestMatchMergeUnits:
    """The member unit normalizer's defensive paths.

    Plan-driven assembly re-expresses each member in the plan's unit
    before this runs, so these branches are exercised directly rather
    than through a chunk.
    """

    def test_incompatible_units_pass_through(self):
        """Dimensionality mismatches pass through for the merge to police."""
        patch = dc.get_example_patch().set_units(distance="m")
        target = get_quantity("s").units
        out, kept = _match_merge_units(patch, "distance", target)
        assert out is patch  # unconverted
        assert kept == target

    def test_compatible_units_convert(self):
        """A member spelled differently converts to the target."""
        patch = dc.get_example_patch().set_units(distance="ft")
        target = get_quantity("m").units
        out, kept = _match_merge_units(patch, "distance", target)
        assert kept == target
        coord = out.get_coord("distance")
        assert get_quantity(coord.units) == get_quantity("m")
        original = patch.get_coord("distance")
        assert float(coord.max()) == pytest.approx(float(original.max()) * 0.3048)


class TestUnitChunkValue:
    """Chunk lengths carrying the coordinate's own units."""

    def test_distance_unit_converts(self, random_spool):
        """A distance in feet converts to the coordinate's metres."""
        out = random_spool.chunk(distance=100 * dc.units.ft)
        assert len(out) == len(random_spool.chunk(distance=30.48))

    def test_unitless_coord_raises(self, random_spool):
        """A unit-bearing length needs a coordinate with units."""
        patches = [x.set_units(distance=None) for x in random_spool]
        with pytest.raises(UnitError, match="no units"):
            dc.spool(patches).chunk(distance=100 * dc.units.ft)


class TestSizeChunk:
    """Chunk lengths expressed as a data size."""

    @staticmethod
    def _make(dtype, start, distance=50, samples=400):
        """A patch of a given dtype starting at a given time."""
        rng = np.random.default_rng(42)
        data = rng.random((distance, samples)).astype(dtype)
        coords = {
            "distance": np.arange(distance) * 1.0,
            "time": start + np.arange(samples) * np.timedelta64(4, "ms"),
        }
        return dc.Patch(data=data, coords=coords, dims=("distance", "time"))

    @pytest.mark.parametrize(
        # One decimal unit and one binary one; the four sizes all split the
        # spool, so what the other two added was the arithmetic.
        "size",
        ("1 MiB", "500 kB"),
        ids=lambda x: x.replace(" ", ""),
    )
    def test_never_exceeds_request(self, random_spool, size):
        """Every output patch fits inside the requested size."""
        quant = dc.get_quantity(size)
        limit = quant.to("byte").magnitude
        sizes = [x.data.nbytes for x in random_spool.chunk(time=quant)]
        assert max(sizes) <= limit
        # and the request is not trivially under-delivered
        assert max(sizes) > 0.8 * limit

    def test_binary_and_decimal_prefixes_differ(self, random_spool):
        """MiB is 2**20 bytes while MB is 10**6, as pint defines them."""
        decimal = random_spool.chunk(time=1 * dc.units.MB)[0]
        binary = random_spool.chunk(time=1 * dc.units.MiB)[0]
        assert binary.data.nbytes > decimal.data.nbytes
        assert binary.data.nbytes <= 1024**2

    def test_smaller_dtype_gives_more_samples(self):
        """Half the itemsize fits twice the samples in the same bytes."""
        start = np.datetime64("2020-01-01T00:00:00")
        target = dc.get_quantity("40 kB")
        wide = dc.spool([self._make("float64", start)]).chunk(time=target)
        narrow = dc.spool([self._make("float32", start)]).chunk(time=target)
        wide_samples = wide[0].shape[wide[0].get_axis("time")]
        narrow_samples = narrow[0].shape[narrow[0].get_axis("time")]
        assert narrow_samples == 2 * wide_samples

    def test_mixed_steps_in_one_partition_stay_bounded(self):
        """
        Sizing must use the partition's smallest step, not its median.

        Steps within `sampling_group_tolerance` share a partition, so a
        member sampled faster than the median fits more samples into the
        same length and would overshoot the byte target.
        """
        t0 = np.datetime64("2020-01-01T00:00:00")
        patches, start = [], t0
        for step_ns in (4_000_000, 4_000_000, 3_850_000):  # 3.75% apart
            step = np.timedelta64(step_ns, "ns")
            times = start + np.arange(500) * step
            patches.append(
                dc.Patch(
                    data=np.zeros((25, 500)),
                    dims=("distance", "time"),
                    coords={"distance": np.arange(25) * 1.0, "time": times},
                )
            )
            start = times[-1] + step
        spool = dc.spool(patches).sort("time")
        target = dc.get_quantity("100 kB")
        # the steps must actually share one partition or this proves nothing
        assert len(spool.chunk_plan(time=target).params["size"]["partitions"]) == 1
        out = spool.chunk(time=target)
        assert max(x.data.nbytes for x in out) <= target.to("byte").magnitude

    def test_int_float_partition_promotes_above_both(self):
        """
        A mixed partition is sized by `np.result_type`, not max itemsize.

        int32 and float32 are both 4 bytes but promote to 8-byte
        float64, so a max-itemsize estimate would under-count and
        produce patches at twice the requested size.
        """
        t0 = np.datetime64("2020-01-01T00:00:00")

        def make(dtype, start):
            return dc.Patch(
                data=np.zeros((50, 250), dtype=dtype),
                dims=("distance", "time"),
                coords={
                    "distance": np.arange(50) * 1.0,
                    "time": start + np.arange(250) * np.timedelta64(4, "ms"),
                },
            )

        second = t0 + np.timedelta64(1000, "ms")
        spool = dc.spool([make("int32", t0), make("float32", second)]).sort("time")
        target = dc.get_quantity("100 kB")
        (part,) = spool.chunk_plan(time=target).params["size"]["partitions"]
        assert part["dtype"] == "float64"
        assert part["itemsize"] == 8
        out = spool.chunk(time=target)
        assert max(x.data.nbytes for x in out) <= target.to("byte").magnitude

    def test_slab_larger_than_target_warns(self):
        """One sample is the floor; the request cannot be honored below it."""
        # 200 channels of float64 is a 1,600 byte slab, so one sample already
        # exceeds the request. The default patch's 400 byte slab would not.
        start = np.datetime64("2020-01-01T00:00:00")
        spool = dc.spool([self._make("float64", start, distance=200, samples=10)])
        with pytest.warns(UserWarning, match="larger than the requested size"):
            out = spool.chunk(time=1 * dc.units.kB)
        patch = out[0]
        assert patch.shape[patch.get_axis("time")] == 1

    def test_keep_partial_stays_bounded(self, random_spool):
        """A partial segment is smaller than the request, never larger."""
        target = dc.get_quantity("2 MB")
        out = random_spool.chunk(time=target, keep_partial=True)
        assert max(x.data.nbytes for x in out) <= target.to("byte").magnitude

    def test_overlap_in_coord_units(self, random_spool):
        """Overlap may use the coordinate's units while the length is a size."""
        target = dc.get_quantity("2 MB")
        plain = random_spool.chunk(time=target)
        out = random_spool.chunk(time=target, overlap=1 * dc.units.s)
        assert max(x.data.nbytes for x in out) <= target.to("byte").magnitude
        assert len(out) > len(plain)  # the overlap is honored, not dropped

    def test_overlap_as_size(self, random_spool):
        """Overlap may also be expressed as a size."""
        target = dc.get_quantity("2 MB")
        plain = random_spool.chunk(time=target)
        out = random_spool.chunk(time=target, overlap=1 * dc.units.MB)
        assert max(x.data.nbytes for x in out) <= target.to("byte").magnitude
        assert len(out) > len(plain)  # overlap yields more segments

    def test_overlap_larger_than_length_raises(self, random_spool):
        """The existing overlap guard applies to sizes too."""
        with pytest.raises(ParameterError):
            random_spool.chunk(time=1 * dc.units.MB, overlap=2 * dc.units.MB)

    def test_descending_coord(self, random_spool):
        """A reverse-sorted coordinate chunks by size like any other."""
        patches = [
            x.snap_coords("time").sort_coords("time", reverse=True)
            for x in random_spool
        ]
        target = dc.get_quantity("1 MB")
        out = dc.spool(patches).chunk(time=target)
        assert max(x.data.nbytes for x in out) <= target.to("byte").magnitude

    def test_one_dimensional_patch(self):
        """With no other dims a slab is one element."""
        start = np.datetime64("2020-01-01T00:00:00")
        data = np.arange(1000, dtype="float64")
        coords = {"time": start + np.arange(1000) * np.timedelta64(1, "s")}
        patch = dc.Patch(data=data, coords=coords, dims=("time",))
        plan = dc.spool([patch]).chunk_plan(time=dc.get_quantity("1 kB"))
        (part,) = plan.params["size"]["partitions"]
        assert part["slab_samples"] == 1
        assert part["n_samples"] == 1000 // 8

    def test_three_dimensional_patch(self):
        """A slab is the product of every other dimension."""
        start = np.datetime64("2020-01-01T00:00:00")
        data = np.zeros((3, 5, 100), dtype="float64")
        coords = {
            "distance": np.arange(3) * 1.0,
            "depth": np.arange(5) * 1.0,
            "time": start + np.arange(100) * np.timedelta64(1, "s"),
        }
        patch = dc.Patch(data=data, coords=coords, dims=("distance", "depth", "time"))
        plan = dc.spool([patch]).chunk_plan(time=dc.get_quantity("1 kB"))
        (part,) = plan.params["size"]["partitions"]
        assert part["slab_samples"] == 15

    def test_sub_sample_gaps_stay_bounded(self):
        """
        Members packed closer than one sample must not overshoot.

        Near-contiguous files whose boundaries miss the grid fit more
        samples into a span than span/step + 1, because each member
        contributes its own trailing sample.
        """
        t0 = np.datetime64("2020-01-01T00:00:00")
        second = np.timedelta64(1_000_000_000, "ns")
        patches, offset = [], 0
        for _ in range(30):
            start = t0 + np.timedelta64(offset, "ns")
            patches.append(
                dc.Patch(
                    data=np.zeros((50, 20)),
                    dims=("distance", "time"),
                    coords={
                        "distance": np.arange(50) * 1.0,
                        "time": start + np.arange(20) * second,
                    },
                )
            )
            offset += 19 * 1_000_000_000 + 10_000_000  # 10 ms short of the grid
        target = dc.get_quantity("100 kB")
        out = dc.spool(patches).chunk(time=target)
        assert max(x.data.nbytes for x in out) <= target.to("byte").magnitude

    def test_gridded_partition_has_unit_packing(self, random_spool):
        """A partition that tiles its grid is sized without correction."""
        plan = random_spool.chunk_plan(time=1 * dc.units.MB)
        (part,) = plan.params["size"]["partitions"]
        assert part["packing"] == 1.0

    def test_output_dtype_matches_assembled_patch(self):
        """
        A plan row must claim the dtype its patch actually assembles to.

        Claiming the partition-wide upcast would both over-size a later
        size chunk and make a chunked spool compare unequal to its own
        materialized twin.
        """
        t0 = np.datetime64("2020-01-01T00:00:00")

        def make(dtype, start):
            rng = np.random.default_rng(1)
            return dc.Patch(
                data=rng.random((40, 400)).astype(dtype),
                dims=("distance", "time"),
                coords={
                    "distance": np.arange(40) * 1.0,
                    "time": start + np.arange(400) * np.timedelta64(4, "ms"),
                },
            )

        spool = dc.spool(
            [make("float32", t0), make("float64", t0 + np.timedelta64(1600, "ms"))]
        ).sort("time")
        out = spool.chunk(time=1.0)
        claimed = out._df["_dtype"].tolist()
        assert claimed == [str(x.data.dtype) for x in out]
        assert out == dc.spool(list(out))

    def test_merge_mode_rejects_size_overlap(self, random_spool):
        """A size overlap is still an overlap, which merging forbids."""
        with pytest.raises(ParameterError, match="keep_partial and overlap"):
            random_spool.chunk(time=None, overlap=1 * dc.units.MB)

    @pytest.mark.parametrize(
        "value", (float("inf") * dc.units.MB, float("nan") * dc.units.MB)
    )
    def test_non_finite_size_raises(self, random_spool, value):
        """
        A non-finite size is a bad request, not a merge.

        NaN is null, so without an explicit guard a nan-valued size
        would silently merge the whole spool into one patch when the
        user asked for a size cap.
        """
        with pytest.raises(ParameterError, match="finite"):
            random_spool.chunk(time=value)

    def test_array_size_raises(self, random_spool):
        """A chunk length is one value, not an array of them."""
        with pytest.raises(ParameterError, match="single quantity"):
            random_spool.chunk(time=np.array([1.0, 2.0]) * dc.units.MB)
