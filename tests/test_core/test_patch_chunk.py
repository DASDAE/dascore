"""
Test file for chunk/merge.

This is separated from spool tests because these need to be quite
extensive.
"""

from __future__ import annotations

import os
import random
import warnings
from datetime import timedelta
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.examples as ex
import dascore.utils.patch_assembly as assembly_module
from dascore.core.coords import CoordRange, CoordSegmented
from dascore.core.source import ArraySource
from dascore.exceptions import (
    ChunkError,
    CoordMergeError,
    InvalidFiberIOError,
    ParameterError,
    UnitError,
)
from dascore.io.febus.core import FebusPatchAttrs
from dascore.io.index import planned
from dascore.units import get_quantity
from dascore.utils.gaps import GapTolerance
from dascore.utils.misc import get_middle_value, suppress_warnings
from dascore.utils.patch import _get_merged_coord
from dascore.utils.patch_assembly import PatchAssembler, _match_merge_units
from dascore.utils.time import to_int, to_timedelta64


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
        msg = "longest contiguous segment along 'time'"
        with pytest.raises(ChunkError, match=msg):
            diverse_spool.chunk(time=10000)

    def test_increment_too_big_names_knobs(self, diverse_spool):
        """The error should point at tolerance and keep_partial (#1046)."""
        with pytest.raises(ChunkError) as info:
            diverse_spool.chunk(time=10000)
        msg = str(info.value)
        assert "tolerance" in msg and "keep_partial" in msg

    def test_increment_too_big_names_units(self):
        """The segment length carries the unit its coordinate states."""
        p1 = dc.get_example_patch().set_units(distance="mm")
        gap = p1.get_coord("time").max() + to_timedelta64(1000)
        p2 = dc.get_example_patch(time_min=gap).set_units(distance="mm")
        with pytest.raises(ChunkError, match=r"is 300\.0 mm,"):
            dc.spool([p1, p2]).chunk(distance=2 * get_quantity("m"))

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
        with pytest.raises(CoordMergeError, match=r"\bfoo\b|\bdata_type\b"):
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

    def test_attrs_named_like_coordinates_are_policed_as_attrs(self):
        """Attrs which look like coordinate metadata are policed by `conflict`.

        A column a coordinate owns is refused whatever `conflict` says,
        so keeping the first value is what proves these are read as
        ordinary attrs rather than as the envelope of some coordinate.
        """
        # values differ, so only the conflict policy can decide them
        first = {
            "latitude": "north",  # a coordinate another patch has
            "foo_min": "a",  # an envelope pair with no foo coordinate
            "foo_max": "b",
            "time_zone": "UTC",  # a name prefixed by a real dimension
            "gauge": 10 * dc.get_quantity("m"),  # a quantity
            "shots": 7,  # a plain number
        }
        second = {
            "latitude": "south",
            "foo_min": "c",
            "foo_max": "d",
            "time_zone": "MST",
            "gauge": 20 * dc.get_quantity("m"),
            "shots": 9,
        }
        p1 = dc.get_example_patch().update_attrs(**first)
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + time.step)
        p2 = p2.update_attrs(**second)
        n = p1.shape[p1.get_axis("distance")]
        # a patch which holds latitude as a coordinate rather than an attr
        elsewhere = dc.get_example_patch(
            time_min=time.max() + 10 * time.step, tag="other"
        ).update_coords(latitude=("distance", np.arange(n, dtype=float)))
        spool = dc.spool([p1, p2, elsewhere])
        # they are attrs, so they conflict rather than being coordinate metadata
        with pytest.raises(CoordMergeError, match=r"latitude|foo_min|time_zone"):
            spool.chunk(time=None)
        out = spool.chunk(time=None, conflict="keep_first")
        merged = out.select(tag="random")[0]
        row = out.get_contents().set_index("tag").loc["random"]
        for name, value in first.items():
            assert merged.attrs.get(name) == value
            assert name in row or name == "gauge"
        assert row["time_zone"] == "UTC" and row["latitude"] == "north"

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
    return PatchAssembler(load_patch=None, merge_kwargs={}, plan_dim="time")


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
        """
        Re-chunking preserves one source-units column so load kwargs trim in the correct
        units.
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


class TestMixedUnitSpellings:
    """One length spelled two ways is two coordinates to the planner."""

    @pytest.fixture(scope="class")
    def mixed_patches(self, random_patch):
        """Two time-contiguous patches, distance in metres and in cm."""
        coord = random_patch.get_coord("time")
        later = random_patch.update_coords(time_min=coord.max() + coord.step)
        distance = random_patch.get_coord("distance")
        metres = random_patch.update_coords(distance=distance.set_units("m"))
        centimetres = later.update_coords(
            distance=dc.get_coord(
                start=distance.min() * 100,
                step=distance.step * 100,
                shape=distance.shape,
                units="cm",
            )
        )
        return metres, centimetres

    def test_memory_spool_does_not_merge(self, mixed_patches):
        """Merging them would give one of the two spellings for both."""
        out = dc.spool(list(mixed_patches)).chunk(time=None)
        assert len(out) == 2

    def test_indexed_spool_does_not_merge(self, mixed_patches, tmp_path_factory):
        """The stored def keys keep the spellings apart, as the envelopes are."""
        path = tmp_path_factory.mktemp("mixed_units")
        for num, patch in enumerate(mixed_patches):
            dc.write(patch, path / f"{num}.h5", "dasdae")
        spool = dc.spool(path).update()
        assert spool._catalog.to_df()["_distance_def_key"].nunique() == 2
        assert len(spool.chunk(time=None)) == 2


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


class TestQuantityTolerance:
    """Continuity tolerances stated in the coordinate's own units."""

    @staticmethod
    def _gapped(patch, samples):
        """Two patches whose boundary spans `samples` steps; `samples - 1` missing."""
        base = patch.update_attrs(history="")
        time = patch.get_coord("time")
        after = base.update_coords(
            time_min=time.max() + time.step * samples
        ).update_attrs(history="")
        return dc.spool((base, after))

    @staticmethod
    def _shifted(patch, dim, steps):
        """Two patches whose boundary along `dim` spans `steps` steps."""
        base = patch.update_attrs(history="")
        coord = patch.get_coord(dim)
        after = base.update_coords(
            **{f"{dim}_min": coord.max() + coord.step * steps}
        ).update_attrs(history="")
        return dc.spool((base, after))

    def test_merges_gap_it_spans(self, random_patch):
        """A tolerance wider than the hole merges over it; a tighter one does not."""
        step = dc.to_float(random_patch.get_coord("time").step)
        spool = self._gapped(random_patch, 5)
        with pytest.warns(UserWarning, match="gap in the patch"):
            merged = spool.chunk(time=None, tolerance=get_quantity(f"{5 * step} s"))
        assert len(merged) == 1
        assert len(spool.chunk(time=None, tolerance=get_quantity(f"{step} s"))) == 2

    def test_merged_coord_keeps_its_step(self, random_patch):
        """A merge under an absolute tolerance keeps the patches' own step."""
        coord = random_patch.get_coord("time")
        step = dc.to_float(coord.step)
        spool = self._gapped(random_patch, 3)
        with pytest.warns(UserWarning, match="gap in the patch"):
            merged = spool.chunk(time=None, tolerance=get_quantity(f"{4 * step} s"))
        assert merged[0].get_coord("time").step == coord.step

    def test_distance_unit_converts(self, random_spool):
        """A tolerance in feet is read in the coordinate's metres, not as metres."""
        patch = random_spool[0]
        step = float(patch.get_coord("distance").step)
        spool = self._shifted(patch, "distance", 9)
        hole = step * 9
        # 20 ft is 6.1 m, under the 9 m hole; 40 ft is 12.2 m, over it.
        # Read as metres instead, both would clear it and both legs would
        # merge, so the pair pins the conversion in both directions.
        assert step == 1.0 and hole == 9.0
        with suppress_warnings(UserWarning):
            wide = spool.chunk(distance=None, tolerance=40 * dc.units.ft)
        assert len(wide) == 1
        assert len(spool.chunk(distance=None, tolerance=20 * dc.units.ft)) == 2

    def test_non_time_merge_assembles(self, random_spool):
        """A distance merge under a converted tolerance assembles whole."""
        spool = self._shifted(random_spool[0], "distance", 3)
        with pytest.warns(UserWarning, match="gap in the patch"):
            merged = spool.chunk(distance=None, tolerance=4 / 0.3048 * dc.units.ft)[0]
        coord = merged.get_coord("distance")
        assert coord.step is not None
        assert len(coord) == sum(len(x.get_coord("distance")) for x in spool)

    def test_dimensionless_is_a_sample_count(self, random_patch):
        """A dimensionless quantity is samples, not the coordinate's units."""
        # A hole of 6 steps is 0.024 s: six samples merge it, and six
        # seconds would too, so the gap has to be wide in seconds and
        # narrow in samples to tell the two readings apart.
        step = dc.to_float(random_patch.get_coord("time").step)
        spool = self._gapped(random_patch, 3)
        seconds_would_merge = 3 * step < 6
        assert seconds_would_merge
        assert (
            len(spool.chunk(time=None, tolerance=get_quantity("2 dimensionless"))) == 2
        )
        with suppress_warnings(UserWarning):
            quantity = spool.chunk(time=None, tolerance=get_quantity("6 dimensionless"))
            number = spool.chunk(time=None, tolerance=6)
        assert len(quantity) == len(number) == 1

    def test_timedelta_is_absolute(self, random_patch):
        """A timedelta says the same thing as a time quantity."""
        step = random_patch.get_coord("time").step
        spool = self._gapped(random_patch, 5)
        with suppress_warnings(UserWarning):
            delta = spool.chunk(time=None, tolerance=6 * step)
            quantity = spool.chunk(
                time=None, tolerance=get_quantity(f"{6 * dc.to_float(step)} s")
            )
        assert len(delta) == len(quantity) == 1
        assert delta[0].equals(quantity[0])
        assert len(spool.chunk(time=None, tolerance=step)) == 2

    def test_datetime_timedelta_accepted(self, random_patch):
        """The stdlib timedelta is a timedelta too."""
        spool = self._gapped(random_patch, 5)
        with suppress_warnings(UserWarning):
            out = spool.chunk(time=None, tolerance=timedelta(seconds=1))
        assert len(out) == 1

    def test_timedelta_reads_a_numeric_time_coord(self, random_patch):
        """A numeric coordinate measured in seconds takes a timedelta."""
        coord = random_patch.get_coord("time")
        numeric = dc.core.get_coord(
            start=0.0, stop=float(len(coord)), step=1.0, units="s"
        )
        base = random_patch.rename_coords(time="shot").update_coords(shot=numeric)
        spool = self._shifted(base, "shot", 3)
        with suppress_warnings(UserWarning):
            delta = spool.chunk(shot=None, tolerance=to_timedelta64(4))
            quantity = spool.chunk(shot=None, tolerance=get_quantity("4 s"))
        assert len(delta) == len(quantity) == 1
        # materialized, since the merge converts the tolerance a second
        # time to bound the snap, and a raw timedelta cannot bound a
        # numeric coordinate's deviations
        assert delta[0].get_coord("shot") == quantity[0].get_coord("shot")
        assert len(spool.chunk(shot=None, tolerance=to_timedelta64(1))) == 2

    def test_sub_step_tolerance_keeps_contiguity(self, random_spool):
        """A margin narrower than the step never splits adjacent patches."""
        # The boundary between adjacent patches is one full step, so a
        # tolerance under it must still read as "nothing missing".
        tiny = get_quantity("1 ns")
        assert len(random_spool.chunk(time=None, tolerance=tiny)) == 1
        assert random_spool.get_gaps(tolerance=tiny).empty
        assert (random_spool.get_coverage(tolerance=tiny)["coverage"] == 1).all()

    def test_unknown_step_gap_is_found(self, random_patch):
        """An absolute tolerance needs no sampling interval to measure a gap."""

        def _jitter(patch, offset):
            time = patch.get_coord("time").values + offset
            rng = np.random.default_rng(13)
            jittered = time + (rng.random(len(time)) * 1e6).astype("timedelta64[ns]")
            return patch.update_coords(time=np.sort(jittered)).update_attrs(history="")

        spool = dc.spool(
            [
                _jitter(random_patch, np.timedelta64(0, "s")),
                _jitter(random_patch, np.timedelta64(20, "s")),
            ]
        )
        assert pd.isnull(spool.get_contents()["time_step"]).all()
        # the sample count has no step to scale, so it merges blindly
        assert len(spool.chunk(time=None)) == 1
        assert len(spool.chunk(time=None, tolerance=get_quantity("0.5 s"))) == 2
        # and the merging side of the same branch, which needs no step
        with suppress_warnings(UserWarning):
            merged = spool.chunk(time=None, tolerance=get_quantity("30 s"))
        assert len(merged) == 1
        assert len(merged[0].get_coord("time")) == sum(
            len(x.get_coord("time")) for x in spool
        )

    def test_affine_units_convert_as_a_delta(self, random_patch):
        """A tolerance is a difference, so an affine unit's offset cancels."""
        coord = random_patch.get_coord("distance")
        celsius = dc.core.get_coord(
            start=0.0, stop=float(len(coord)), step=1.0, units="degC"
        )
        base = random_patch.rename_coords(distance="temp").update_coords(temp=celsius)
        spool = self._shifted(base, "temp", 3)
        # 4 K of extent is 4 degC of extent; read as a point it would be
        # -269.15 degC, which simplify refuses as negative.
        with suppress_warnings(UserWarning):
            merged = spool.chunk(temp=None, tolerance=4 * dc.units.kelvin)[0]
        assert merged.get_coord("temp").step is not None

    def test_wrong_dimensionality_raises(self, random_spool):
        """A tolerance must measure the dimension it is applied to."""
        with pytest.raises(UnitError, match="must have units of time"):
            random_spool.chunk(time=None, tolerance=10 * dc.units.m)

    def test_timedelta_on_other_dim_raises(self, random_spool):
        """A time tolerance cannot measure a coordinate of metres."""
        with pytest.raises(UnitError, match="incompatible with the coordinate"):
            random_spool.chunk(distance=None, tolerance=to_timedelta64(1))

    def test_unitless_coord_raises(self, random_spool):
        """A unit-bearing tolerance needs a coordinate with units."""
        patches = [x.set_units(distance=None) for x in random_spool]
        with pytest.raises(UnitError, match="no units"):
            dc.spool(patches).chunk(distance=None, tolerance=10 * dc.units.ft)

    def test_data_size_raises(self, random_spool):
        """A data size does not describe a hole along a coordinate."""
        with pytest.raises(UnitError, match="data size"):
            random_spool.chunk(time=None, tolerance=get_quantity("25 MB"))

    def test_percent_raises(self, random_spool):
        """A percentage is neither a count nor a length."""
        with pytest.raises(UnitError, match="percentage"):
            random_spool.chunk(time=None, tolerance=get_quantity("50%"))

    def test_unrepresentable_time_raises(self, random_spool):
        """A time too large for a timedelta64 says so, rather than overflowing."""
        with pytest.raises(ParameterError, match="too large"):
            random_spool.chunk(time=None, tolerance=get_quantity("1e11 s"))

    @pytest.mark.parametrize(
        "tolerance,match",
        [
            (np.nan, "finite"),
            (np.timedelta64("NaT", "s"), "finite"),
            (np.inf * dc.units.s, "finite"),
            (-1, "not be negative"),
            (get_quantity("-1 dimensionless"), "not be negative"),
            (get_quantity("-1 s"), "not be negative"),
            (-to_timedelta64(1), "not be negative"),
            (np.array([2]), "single value"),
            (np.array([1.0, 2.0]), "single value"),
            (np.array([1.0, 2.0]) * dc.units.s, "single value"),
            ("2 s", "get_quantity"),
        ],
    )
    def test_unmeasurable_tolerance_raises(self, random_spool, tolerance, match):
        """A tolerance no gap could be measured against is refused."""
        with pytest.raises(ParameterError, match=match):
            random_spool.chunk(time=None, tolerance=tolerance)

    def test_infinite_sample_count_merges_everything(self, random_patch):
        """An infinite count is a coherent request: no boundary is a gap."""
        spool = self._gapped(random_patch, 500)
        with pytest.warns(UserWarning, match="gap in the patch"):
            merged = spool.chunk(time=None, tolerance=np.inf)
            assert len(merged) == 1
            # and the patch it advertises actually loads: an infinite
            # count is no bound on the snap, not a bound of NaT
            assert isinstance(merged[0].get_coord("time"), CoordSegmented)

    def test_exchanged_boundary_warns(self):
        """A forced merge warns even when the partition count is unchanged."""
        t0 = np.datetime64("2020-01-01T00:00:00", "ns")
        rng = np.random.default_rng(42)

        def _patch(start, step, samples=20):
            step = to_timedelta64(step)
            coord = dc.core.get_coord(
                start=start, stop=start + step * samples, step=step
            )
            data = rng.random((5, samples))
            coords = {"distance": np.arange(5) * 1.0, "time": coord}
            return dc.Patch(data=data, coords=coords, dims=("distance", "time"))

        # Steps within the sampling group tolerance, so all three patches
        # share a cell; the holes are chosen so the default (1.5 steps) and
        # an excess of 0.51 s over the step split at *different* boundaries.
        first = _patch(t0, 1.0)
        second = _patch(first.get_coord("time").max() + to_timedelta64(1.505), 1.0)
        third = _patch(second.get_coord("time").max() + to_timedelta64(1.555), 1.04)
        spool = dc.spool([first, second, third])
        default = spool.chunk(time=None)
        with pytest.warns(UserWarning, match="force merging"):
            absolute = spool.chunk(time=None, tolerance=get_quantity("0.51 s"))
        assert len(default) == len(absolute) == 2
        # the same count, but not the same split
        assert default[0].get_coord("time").max() != absolute[0].get_coord("time").max()

    def test_merge_uses_the_normalized_tolerance(self, random_spool):
        """The merge gets the tolerance the plan resolved, not the raw one.

        Asserted on what the merge is handed rather than on a merged
        coordinate: a dimensionless quantity reaching `simplify` is read
        as *seconds*, which only shows up in a merged coordinate for
        gap geometries where the snap bound is the binding constraint.
        """
        out = random_spool.chunk(time=None, tolerance=get_quantity("2"))
        handed = out._catalog.resolver.merge_kwargs["tolerance"]
        # a dimensionless quantity is the sample count it spells out
        assert handed == GapTolerance.samples(2.0)

    def test_snapping_never_relabels_across_a_hole(self, random_patch):
        """However wide the tolerance, a hole is missing data, not a slower rate."""
        step = random_patch.get_coord("time").step
        spool = self._gapped(random_patch, 40)
        with pytest.warns(UserWarning, match="gap in the patch"):
            snapped = spool.chunk(time=None, tolerance=41 * step)[0]
            exact = spool.chunk(time=None, tolerance=41 * step, snap_coords=False)[0]
        snapped_coord, exact_coord = (x.get_coord("time") for x in (snapped, exact))
        # snapping the merge is now a no-op: the hole stays a seam and
        # every label keeps the value the source patch gave it
        assert isinstance(snapped_coord, CoordSegmented)
        assert snapped_coord.step == step
        assert np.array_equal(snapped_coord.values, exact_coord.values)

    def test_snapping_still_absorbs_sub_sample_jitter(self, random_patch):
        """Labels a fraction of a step off the grid do collapse to a range."""
        coord = random_patch.get_coord("time")
        offset = np.timedelta64(int(to_int(coord.step) // 3), "ns")
        base = random_patch.update_attrs(history="")
        after = base.update_coords(
            time_min=coord.max() + coord.step + offset
        ).update_attrs(history="")
        spool = dc.spool((base, after))
        snapped = spool.chunk(time=None)[0].get_coord("time")
        exact = spool.chunk(time=None, snap_coords=False)[0].get_coord("time")
        assert isinstance(snapped, CoordRange)
        assert isinstance(exact, CoordSegmented)
        # no sample moved far enough to land on another grid position,
        # and none moved past the tolerance the merge was given either
        deviation = abs(snapped.values - exact.values).max()
        assert deviation < coord.step
        assert deviation <= 1.5 * coord.step

    def test_plan_records_normalized_tolerance(self, random_spool):
        """The plan records the tolerance it actually used."""
        plan = random_spool.chunk_plan(time=None, tolerance=get_quantity("2 s"))
        assert plan.params["tolerance"] == GapTolerance.absolute(get_quantity("2 s"))
        plan = random_spool.chunk_plan(time=None, tolerance=get_quantity("2"))
        assert plan.params["tolerance"] == GapTolerance.samples(2.0)


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


class TestChunkEdgeBetweenPatches:
    """
    Chunk lengths that are not a whole number of samples (#1008, #893).

    An output edge then lands between the last sample of one patch and
    the first of the next; chunking must give the same result as chunking
    the patches once merged.
    """

    @pytest.fixture(scope="class")
    def contiguous_spool(self):
        """Three exactly contiguous patches of five one-second samples."""
        step = np.timedelta64(1, "s")
        t0 = np.datetime64("2026-01-01T00:00:00", "ns")
        patches = []
        for _ in range(3):
            time = t0 + np.arange(5) * step
            coords = {"time": time, "distance": np.arange(3.0)}
            data = np.random.default_rng(len(patches)).random((5, 3))
            patches.append(
                dc.Patch(data=data, coords=coords, dims=("time", "distance"))
            )
            t0 = time[-1] + step
        return dc.spool(patches)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(time=4.5),
            dict(time=5.5),
            dict(time=5.5, keep_partial=True),
            dict(time=6, overlap=1.5),
            dict(time=2.3, overlap=0.4),
        ],
    )
    def test_matches_chunking_merged_patch(self, contiguous_spool, kwargs):
        """Patch boundaries must not change what a chunk contains."""
        merged = contiguous_spool.chunk(time=None)
        assert len(merged) == 1
        out, expected = contiguous_spool.chunk(**kwargs), merged.chunk(**kwargs)
        assert len(out) == len(expected)
        assert all(a.equals(b) for a, b in zip(out, expected))

    @pytest.mark.parametrize("length", [25, 35, 50])
    def test_nested_patches_chunk_as_the_outer_one(self, length):
        """Patches inside a longer one must not add or remove samples."""

        def _patch(first, samples):
            step = np.timedelta64(1, "s")
            t0 = np.datetime64("2026-01-01", "ns") + first * step
            data = np.arange(samples * 2, dtype=float).reshape(samples, 2)
            coords = {"time": t0 + np.arange(samples) * step, "distance": [0.0, 1.0]}
            return dc.Patch(data=data, coords=coords, dims=("time", "distance"))

        outer = _patch(0, 101)
        nested = dc.spool([outer, _patch(10, 11), _patch(30, 11)])
        out = nested.chunk(time=length, keep_partial=True)
        expected = dc.spool([outer]).chunk(time=length, keep_partial=True)
        assert len(out) == len(expected)
        assert all(a.equals(b) for a, b in zip(out, expected))

    def test_boundary_in_sub_tolerance_gap(self):
        """A boundary inside a gap the tolerance merges over chunks cleanly."""
        p1 = dc.get_example_patch(time_min="2020-01-01")
        time = p1.get_coord("time")
        p2 = dc.get_example_patch(time_min=time.max() + 1.4 * time.step)
        span = (time.max() - time.min()) + 0.7 * time.step
        length = span / np.timedelta64(1, "s") / 2
        out = dc.spool([p1, p2]).chunk(time=length)
        assert len(out) == 4
        coords = [patch.get_coord("time") for patch in out]
        assert all(len(coord) == len(coords[0]) for coord in coords)
        assert all(a.max() < b.min() for a, b in pairwise(coords))


class TestChunkWithAssociatedCoords:
    """Chunking must not trim on coordinates riding another dimension."""

    @pytest.fixture(scope="class")
    def base_patch(self):
        """A patch whose distance coordinate carries the associated ones."""
        return dc.get_example_patch()

    @pytest.fixture(scope="class")
    def string_patch(self, base_patch):
        """A patch with a three-valued string coordinate on distance."""
        dist = base_patch.get_array("distance")
        third = len(dist) // 3
        label = np.full(len(dist), "middle", dtype="<U6")
        label[:third], label[-third:] = "start", "end"
        return base_patch.update_coords(label=("distance", label))

    @pytest.fixture(scope="class")
    def nan_patch(self, base_patch):
        """A patch with a numeric coordinate missing on some channels."""
        dist = base_patch.get_array("distance")
        values = np.where(dist > dist.mean(), dist * 2.0, np.nan)
        return base_patch.update_coords(hole_depth=("distance", values))

    @pytest.fixture(scope="class")
    def numeric_patch(self, base_patch):
        """A patch with a plain numeric coordinate on distance."""
        dist = base_patch.get_array("distance")
        return base_patch.update_coords(depth=("distance", dist * 2.0))

    def test_string_coord_keeps_all_channels(self, string_patch):
        """A string coordinate is not a range the trim may select on."""
        out = list(dc.spool(string_patch).chunk(time=2))
        assert len(out) > 1
        expected = set(np.unique(string_patch.get_array("label")))
        for patch in out:
            assert patch.shape[0] == string_patch.shape[0]
            assert set(np.unique(patch.get_array("label"))) == expected

    def test_nan_coord_keeps_all_channels(self, nan_patch):
        """Channels the coordinate says nothing about must survive."""
        expected = np.isnan(nan_patch.get_array("hole_depth")).sum()
        assert expected  # the fixture must actually hold missing values
        out = list(dc.spool(nan_patch).chunk(time=2))
        assert len(out) > 1
        for patch in out:
            assert patch.shape[0] == nan_patch.shape[0]
            assert np.isnan(patch.get_array("hole_depth")).sum() == expected

    def test_numeric_coord_keeps_all_channels(self, numeric_patch):
        """The well-behaved case must keep working."""
        out = list(dc.spool(numeric_patch).chunk(time=2))
        assert len(out) > 1
        for patch in out:
            assert patch.shape[0] == numeric_patch.shape[0]

    def test_chunk_along_shared_dim(self, nan_patch, string_patch):
        """Chunking the dimension the coordinate rides must not raise."""
        for patch in (nan_patch, string_patch):
            out = list(dc.spool(patch).chunk(distance=100))
            assert len(out) > 1
            total = sum(x.shape[0] for x in out)
            assert total == patch.shape[0]
            assert all(x.shape[1] == patch.shape[1] for x in out)

    @pytest.mark.parametrize("kwargs", [{"time": 2}, {"distance": 100}])
    def test_matches_file_backed_spool(self, string_patch, tmp_path_factory, kwargs):
        """An in-memory spool must chunk like a spool of the same file."""
        path = tmp_path_factory.mktemp("associated_coord_chunk")
        ex.spool_to_directory(dc.spool(string_patch), path=path)
        memory = list(dc.spool(string_patch).chunk(**kwargs))
        file_backed = list(dc.spool(path).update().chunk(**kwargs))
        assert len(memory) == len(file_backed) > 1
        labels = set(np.unique(string_patch.get_array("label")))
        for patch, other in zip(memory, file_backed):
            assert patch.shape == other.shape
            assert np.all(patch.get_array("label") == other.get_array("label"))
            # pin the file-backed result outright: matching a broken
            # in-memory result would otherwise read as agreement
            if "time" in kwargs:
                assert other.shape[0] == string_patch.shape[0]
                assert set(np.unique(other.get_array("label"))) == labels


@pytest.fixture
def calls(monkeypatch):
    """Count patch loads and named member arrays through the plan resolver."""
    counts = {"patch": 0, "array": 0}
    load_patch = planned.PlanResolver._load_member
    array_source = planned.PlanResolver._member_array_source

    def count_patch(self, kwargs):
        counts["patch"] += 1
        return load_patch(self, kwargs)

    def count_array(self, row, shape):
        counts["array"] += 1
        return array_source(self, row, shape)

    monkeypatch.setattr(planned.PlanResolver, "_load_member", count_patch)
    monkeypatch.setattr(planned.PlanResolver, "_member_array_source", count_array)
    return counts


def _force_patch_path(monkeypatch):
    """Make every member load as a patch, as a format without read_array would."""
    monkeypatch.setattr(
        planned.PlanResolver, "_member_array_source", lambda self, row, shape: None
    )


class TestChunkFromIndex:
    """
    Untrimmed members use index metadata and read_array; incomplete rows fall back to
    the equivalent patch-loading path.
    """

    @pytest.fixture(scope="class")
    def dasdae_directory_spool(self, tmp_path_factory):
        """Adjacent DASDAE files sharing their attrs, so they merge."""
        path = tmp_path_factory.mktemp("chunk_from_index")
        spool = ex.get_example_spool(
            "random_das", length=6, time_gap=np.timedelta64(0, "s")
        )
        for num, patch in enumerate(spool):
            patch = patch.update_attrs(tag="x", vendor_thing=5, data_type="strain")
            patch.io.write(path / f"p{num}.h5", "dasdae")
        return dc.spool(path).update()

    def test_matches_patch_path(self, dasdae_directory_spool, calls, monkeypatch):
        """The index-built merge equals the patch-built one in every part."""
        fast = dasdae_directory_spool.chunk(time=None)[0]
        assert calls == {"patch": 0, "array": len(dasdae_directory_spool)}
        _force_patch_path(monkeypatch)
        slow = dasdae_directory_spool.chunk(time=None)[0]
        assert fast.dims == slow.dims
        assert np.array_equal(fast.data, slow.data)
        assert fast.coords == slow.coords
        assert dict(fast.attrs) == dict(slow.attrs)
        # coord equality compares values, so the two things the fast path
        # rebuilds by hand -- the stored dtype and the stated units --
        # have to be compared for themselves
        for name, coord in fast.coords.coord_map.items():
            other = slow.coords.coord_map[name]
            assert coord.dtype == other.dtype, name
            assert coord.units == other.units, name
        # the lineage ids are folded from the members, so they have to
        # come back from the row rather than at their defaults
        assert fast.attrs.origin_id

    @pytest.mark.parametrize(
        "value", [np.datetime64("2021-01-01"), np.timedelta64(5, "s")]
    )
    def test_temporal_attrs_remain_writable(self, tmp_path, calls, monkeypatch, value):
        """Index reconstruction keeps temporal and integer attrs usable by IO."""
        directory = tmp_path / "sources"
        patches = ex.get_example_spool("random_das", length=2, time_gap=0)
        for index, patch in enumerate(patches):
            patch.update_attrs(event_value=value, n_ch=5).io.write(
                directory / f"{index}.h5", "dasdae"
            )
        spool = dc.spool(directory).update()
        # Export/import the catalog too: dtype metadata must survive unions.
        spool = spool + dc.spool([])
        fast = spool.chunk(time=None)[0]
        assert calls == {"patch": 0, "array": 2}
        _force_patch_path(monkeypatch)
        slow = spool.chunk(time=None)[0]
        for name in ("event_value", "n_ch"):
            assert type(fast.attrs[name]) is type(slow.attrs[name])
            assert fast.attrs[name] == slow.attrs[name]
        path = tmp_path / "merged.h5"
        fast.io.write(path, "dasdae")
        restored = dc.read(path)[0]
        assert restored.attrs.event_value == fast.attrs.event_value
        assert restored.attrs.n_ch == fast.attrs.n_ch

    def test_large_integer_attr_loads_patch(self, tmp_path, calls):
        """An integer SQL cannot store exactly must never be rebuilt from it."""
        value = 2**53 + 1
        for index, patch in enumerate(ex.get_example_spool("random_das", length=2)):
            patch.update_attrs(counter=value).io.write(
                tmp_path / f"{index}.h5", "dasdae"
            )
        out = dc.spool(tmp_path).update().chunk(time=None)[0]
        assert out.attrs.counter == value
        assert calls == {"patch": 2, "array": 0}

    def test_trailing_trim_reads_no_arrays(self, dasdae_directory_spool, calls):
        """A trim on the last member prevents speculative reads of earlier ones."""
        stop = dasdae_directory_spool.get_contents()["time_max"].max()
        stop -= np.timedelta64(1, "s")
        view = dasdae_directory_spool.select(time=(None, stop))
        out = view.chunk(time=None)[0]
        assert out.get_coord("time").max() <= stop
        assert calls == {"patch": len(dasdae_directory_spool), "array": 0}

    def test_history_is_not_rebuilt(self, tmp_path_factory, calls, monkeypatch):
        """A member built from the index states no history, by design.

        The index holds none, being a list rather than a column, so a
        merge which takes the array path carries none; only a format
        which stores a history has one to lose.
        """
        path = tmp_path_factory.mktemp("chunk_history")
        spool = ex.get_example_spool(
            "random_das", length=3, time_gap=np.timedelta64(0, "s")
        )
        for num, patch in enumerate(spool):
            patch.pass_filter(time=(None, 100)).io.write(path / f"p{num}.h5", "dasdae")
        file_spool = dc.spool(path).update()
        assert file_spool[0].attrs.history  # the file kept it
        fast = file_spool.chunk(time=None)[0]
        assert calls["array"] == 3
        assert fast.attrs.history == ()
        _force_patch_path(monkeypatch)
        slow = file_spool.chunk(time=None)[0]
        assert slow.attrs.history
        # and history is the only thing the two paths disagree on
        differs = {
            k
            for k in set(dict(fast.attrs)) | set(dict(slow.attrs))
            if dict(fast.attrs).get(k) != dict(slow.attrs).get(k)
        }
        assert differs == {"history"}
        # data_id is indexed, so the fold sees what the members
        # carried rather than a default
        assert fast.attrs.data_id == slow.attrs.data_id
        assert fast.attrs.data_id

    def test_unstateable_coord_loads_patch(self, tmp_path_factory, calls):
        """A coordinate the index cannot describe is still known to exist.

        Its values cannot be stated, so the member must be loaded; the
        catalog records the name so the fast path knows to stand aside.
        """
        path = tmp_path_factory.mktemp("chunk_bool_coord")
        spool = ex.get_example_spool(
            "random_das", length=3, time_gap=np.timedelta64(0, "s")
        )
        for num, patch in enumerate(spool):
            quality = np.arange(len(patch.get_coord("distance"))) % 2 == 0
            patch.update_coords(quality=("distance", quality)).io.write(
                path / f"p{num}.h5", "dasdae"
            )
        file_spool = dc.spool(path).update()
        assert "quality" in file_spool._catalog.backend.coord_dims_map()
        out = file_spool.chunk(time=None)[0]
        assert "quality" in out.coords.coord_map
        assert calls["array"] == 0

    def test_trimmed_member_loads_patch(
        self, dasdae_directory_spool, calls, monkeypatch
    ):
        """One trimmed member sends its whole merge down the patch path.

        A loaded patch can carry what the index cannot hold, so mixing
        the two sources within one merge could make it refuse attrs or
        coordinates it accepts when every member is loaded alike.
        """
        contents = dasdae_directory_spool.get_contents().sort_values("time_min")
        # a range starting inside the second file trims it, keeps the rest
        # whole, and drops the first
        start = dc.to_datetime64(contents["time_min"].iloc[1]) + np.timedelta64(1, "s")
        narrowed = dasdae_directory_spool.select(time=(start, None))
        out = narrowed.chunk(time=None)[0]
        assert out.get_coord("time").min() >= start
        assert calls["array"] == 0
        assert calls["patch"] == len(dasdae_directory_spool) - 1
        # and the mix of paths assembles what the patch path alone would
        _force_patch_path(monkeypatch)
        slow = narrowed.chunk(time=None)[0]
        assert np.array_equal(out.data, slow.data)
        assert out.coords == slow.coords
        assert dict(out.attrs) == dict(slow.attrs)

    def test_associated_coords_load_patch(self, tmp_path_factory, calls):
        """A catalog holding an associated coordinate never takes the array path."""
        path = tmp_path_factory.mktemp("chunk_assoc_coords")
        spool = ex.get_example_spool(
            "random_das", length=3, time_gap=np.timedelta64(0, "s")
        )
        for num, patch in enumerate(spool):
            dist = patch.get_coord("distance")
            lat = np.linspace(40, 41, len(dist))
            patch = patch.update_coords(latitude=("distance", lat))
            patch.io.write(path / f"p{num}.h5", "dasdae")
        out = dc.spool(path).update().chunk(time=None)[0]
        assert "latitude" in out.coords.coord_map
        assert calls["array"] == 0
        assert calls["patch"] == 3

    def test_a_coord_associated_anywhere_loads_patch(self, tmp_path_factory, calls):
        """
        A name which is a dimension on one patch may ride another on the next.

        The catalog keeps one dims spelling per name, the first seen, so
        the first patch's dimensional `distance` would make the later
        patches' associated one look dimensional, and the array path
        would drop it.
        """
        path = tmp_path_factory.mktemp("chunk_assoc_anywhere")
        spool = ex.get_example_spool(
            "random_das", length=3, time_gap=np.timedelta64(0, "s")
        )
        spool[0].io.write(path / "a0.h5", "dasdae")
        # indexed first, so its dimensional `distance` is the spelling seen first
        dc.spool(path).update()
        for num, patch in enumerate(spool[1:]):
            patch = patch.rename_coords(distance="sensor")
            values = patch.get_coord("sensor").values * 2.0
            patch = patch.update_coords(distance=("sensor", values))
            patch.io.write(path / f"b{num}.h5", "dasdae")
        merged = dc.spool(path).update().chunk(time=None)
        riding = [p for p in merged if "sensor" in p.dims]
        assert len(riding) == 1
        assert "distance" in riding[0].coords.coord_map
        assert calls["array"] == 0

    def test_a_moved_source_loads_patch(self, tmp_path_factory, calls):
        """
        A renamed file's row states no id until the file is read again.

        Folding no id is not folding the one the patch carries, so a
        merge built from such rows would not carry the id the patch
        path's merge does.
        """
        path = tmp_path_factory.mktemp("chunk_moved")
        spool = ex.get_example_spool(
            "random_das", length=2, time_gap=np.timedelta64(0, "s")
        )
        for num, patch in enumerate(spool):
            patch.io.write(path / f"p{num}.h5", "dasdae")
        dc.spool(path).update()
        (path / "p0.h5").rename(path / "q0.h5")
        moved = dc.spool(path).update()
        assert (moved.get_contents()["origin_id"] == "").any()
        fast = moved.chunk(time=None)[0]
        assert calls == {"patch": 2, "array": 0}
        read = dc.spool([dc.read(p)[0] for p in sorted(path.glob("*.h5"))])
        assert fast.attrs.origin_id == read.chunk(time=None)[0].attrs.origin_id

    def test_extended_float_coordinate_survives_merge(self, tmp_path, permanent_config):
        """File coordinate precision survives merging, including extended floats."""
        with permanent_config(allow_dasdae_format_unpickle=False):
            start = np.nextafter(np.longdouble(1), np.longdouble(2))
            coord = dc.get_coord(
                start=start, step=np.longdouble("0.1"), shape=10, units="m"
            )
            spool = ex.get_example_spool("random_das", length=2, time_gap=0)
            for num, patch in enumerate(spool):
                patch = patch.select(distance=(0, 10), samples=True)
                patch.update_coords(distance=coord).io.write(
                    tmp_path / f"p{num}.h5", "dasdae"
                )
            indexed = dc.spool(tmp_path).update()
            source = indexed[0].get_coord("distance")

            def read(row):
                return dc.read(tmp_path / row["source_path"])[0]

            # Exercise the two-file assembly independently of coordinate-id grouping.
            assembler = PatchAssembler(
                load_patch=read,
                array_source=lambda row, shape: read(row)._source,
                merge_kwargs={},
                plan_dim="time",
            )
            rows = indexed._catalog.to_df().assign(current_index=0)
            merged = assembler._patch_from_instruction_df(rows)
            assert len(merged) == 1
            out = merged[0].get_coord("distance")
            assert out.dtype == source.dtype
            assert np.array_equal(out.values, source.values)

    def test_extended_float_attribute_survives_merge(self, tmp_path, permanent_config):
        """A stored calibration scalar must retain all precision on every platform."""
        with permanent_config(allow_dasdae_format_unpickle=False):
            value = np.nextafter(np.longdouble(1), np.longdouble(2))
            spool = ex.get_example_spool("random_das", length=2, time_gap=0)
            for num, patch in enumerate(spool):
                patch.update_attrs(calibration=value).io.write(
                    tmp_path / f"p{num}.h5", "dasdae"
                )
            indexed = dc.spool(tmp_path).update()
            assert indexed[0].attrs["calibration"] == value
            out = indexed.chunk(time=None)[0]
            assert out.attrs["calibration"] == value
            assert np.asarray(out.attrs["calibration"]).dtype == np.asarray(value).dtype

    @pytest.mark.parametrize("value", ["unknown", "10"])
    def test_directory_override_keeps_its_string_type(self, tmp_path, calls, value):
        """A path override replaces a source number before reconstruction."""
        directory = tmp_path / f"channel_count={value}"
        directory.mkdir()
        spool = ex.get_example_spool("random_das", length=2, time_gap=0)
        for num, patch in enumerate(spool):
            patch.update_attrs(channel_count=5).io.write(
                directory / f"p{num}.h5", "dasdae"
            )
        with pytest.warns(UserWarning, match="override attrs"):
            indexed = dc.spool(tmp_path).update()
        out = indexed.chunk(time=None)[0]
        assert out.attrs["channel_count"] == value
        assert calls == {"patch": 0, "array": 2}
        directory.rename(tmp_path / "channel_count=renamed")
        moved = dc.spool(tmp_path).update().chunk(time=None)[0]
        assert moved.attrs["channel_count"] == "renamed"

    @pytest.mark.parametrize("case", ["empty", "envelope", "vendor", "skipped"])
    def test_unreconstructable_attributes_keep_the_patch_path(
        self, tmp_path, calls, monkeypatch, case
    ):
        """Missing custom attrs, flat collisions, and vendor models survive merging."""
        spool = ex.get_example_spool("random_das", length=2, time_gap=0)
        for num, patch in enumerate(spool):
            if case == "empty":
                patch = patch.update_attrs(empty_extra="")
            elif case == "envelope":
                patch = patch.update_attrs(distance_units="custom")
            elif case == "skipped":
                patch = patch.update_attrs(coords="user metadata")
            else:
                patch = patch.update(attrs=FebusPatchAttrs())
            patch.io.write(tmp_path / f"p{num}.h5", "dasdae")
        with suppress_warnings(UserWarning):
            indexed = dc.spool(tmp_path).update()
            out = indexed.chunk(time=None)[0]
        assert calls == {"patch": 2, "array": 0}
        _force_patch_path(monkeypatch)
        with suppress_warnings(UserWarning):
            expected = indexed.chunk(time=None)[0]
        assert type(out.attrs) is type(expected.attrs)
        assert dict(out.attrs) == dict(expected.attrs)
        assert np.array_equal(out.data, expected.data)
        assert out.coords == expected.coords

    def test_an_attr_the_index_cannot_hold_loads_patch(self, tmp_path_factory, calls):
        """An array attr is on the patch and in no column, so no row stands in."""
        path = tmp_path_factory.mktemp("chunk_array_attr")
        spool = ex.get_example_spool(
            "random_das", length=3, time_gap=np.timedelta64(0, "s")
        )
        for num, patch in enumerate(spool):
            patch = patch.update_attrs(gauge=np.array([1.0, 2.0]))
            patch.io.write(path / f"p{num}.h5", "dasdae")
        merged = dc.spool(path).update().chunk(time=None)
        out = merged[0]
        assert np.array_equal(out.attrs["gauge"], [1.0, 2.0])
        assert calls == {"patch": 3, "array": 0}
        # A plan on another dimension consumes the derived rows.
        nested = merged.chunk(distance=100)
        assert all(np.array_equal(p.attrs["gauge"], [1.0, 2.0]) for p in nested)

    def test_row_the_index_could_not_fully_describe_loads_patch(self):
        """A cleared id or an attr the index could not hold means the patch path."""
        assembler = PatchAssembler(
            load_patch=lambda kwargs: None,
            merge_kwargs={},
            plan_dim="time",
            array_source=lambda row, shape: ArraySource.full(shape, 0.0),
        )
        row = {
            "dims": "distance,time",
            "distance_min": 0,
            "distance_max": 2,
            "distance_step": 1,
            "time_min": np.datetime64("2020-01-01"),
            "time_max": np.datetime64("2020-01-01T00:00:03"),
            "time_step": np.timedelta64(1, "s"),
            "origin_id": "abc",
            "_attrs_complete": 1,
        }
        assert assembler._meta_from_index(row) is not None
        assert assembler._meta_from_index(row | {"origin_id": ""}) is None
        assert assembler._meta_from_index(row | {"origin_id": None}) is None
        assert assembler._meta_from_index(row | {"_attrs_complete": 0}) is None

    def test_row_without_range_loads_patch(self):
        """A dimension the row cannot state as a range means the patch path."""
        assembler = PatchAssembler(
            load_patch=lambda kwargs: None,
            merge_kwargs={},
            plan_dim="time",
            array_source=lambda row, shape: ArraySource.full(shape, 0.0),
        )
        row = {
            "dims": "distance,time",
            "distance_min": 0,
            "distance_max": 2,
            "distance_step": 1,
        }
        assert assembler._meta_from_index(row) is None

    def test_unnameable_member_loads_patch(self):
        """A member no source can name sends the whole merge to the patches."""
        row = {
            "dims": "distance,time",
            "distance_min": 0,
            "distance_max": 2,
            "distance_step": 1,
            "time_min": np.datetime64("2020-01-01"),
            "time_max": np.datetime64("2020-01-01T00:00:03"),
            "time_step": np.timedelta64(1, "s"),
        }
        good = PatchAssembler(
            load_patch=lambda kwargs: None,
            merge_kwargs={},
            plan_dim="time",
            array_source=lambda row, shape: ArraySource.full(shape, 0.0),
        )
        meta = good._meta_from_index(row)
        assert meta is not None
        assert meta.dims == ("distance", "time")
        assert meta.coords.shape == (3, 4)
        recipe = good._recipe([row], [meta], meta.dims, 1, "time")
        assert recipe is not None
        assert recipe.shape == (3, 4)
        silent = PatchAssembler(
            load_patch=lambda kwargs: None,
            merge_kwargs={},
            plan_dim="time",
            array_source=lambda row, shape: None,
        )
        assert silent._recipe([row], [meta], meta.dims, 1, "time") is None

    def test_file_which_changed_shape_loads_patch(
        self, dasdae_directory_spool, calls, monkeypatch
    ):
        """A file which no longer holds the array its row states is not trusted."""
        from dascore.io import core as io_core  # noqa: PLC0415

        expected = dasdae_directory_spool.chunk(time=None)[0]

        def changed(source):
            msg = f"{source.path} gave a different array than it declared."
            raise InvalidFiberIOError(msg)

        monkeypatch.setattr(io_core, "_load_array_source", changed)
        calls.update(patch=0, array=0)
        out = dasdae_directory_spool.chunk(time=None)[0]
        assert calls["patch"] == len(dasdae_directory_spool)
        assert out.equals(expected)

    def test_whole_member_skips_a_residual_it_lies_inside(
        self, dasdae_directory_spool, calls, monkeypatch
    ):
        """A value selection which cuts no member is a no-op on every one.

        An unmodified row lies wholly inside such a selection, so the
        exactness re-application has nothing to do and the member can be
        built from the index anyway.
        """
        contents = dasdae_directory_spool.get_contents()
        span = (contents["time_min"].min(), contents["time_max"].max())
        selected = dasdae_directory_spool.select(time=span)
        fast = selected.chunk(time=None)[0]
        assert calls == {"patch": 0, "array": len(dasdae_directory_spool)}
        _force_patch_path(monkeypatch)
        slow = selected.chunk(time=None)[0]
        assert fast == slow

    def test_unitless_dim_stays_unitless(self, tmp_path):
        """A coordinate with no units does not gain one from the row.

        Row values come out of a frame, so an unstated unit is NaN, and
        a NaN unit builds a dimensionless quantity nothing can convert.
        """
        # a coordinate which states no units is what puts the NaN in the row
        patch = dc.get_example_patch().set_units(distance=None)
        assert patch.get_coord("distance").units is None
        time = patch.get_coord("time")
        for num in range(3):
            start = time.min() + num * 100 * time.step
            sub = patch.select(time=(start, start + 99 * time.step))
            sub.io.write(tmp_path / f"m{num}.h5", "dasdae")
        merged = dc.spool(tmp_path).update().chunk(time=None)[0]
        units = {n: c.units for n, c in merged.coords.coord_map.items()}
        assert units == {
            n: c.units for n, c in patch.coords.coord_map.items() if n in units
        }
        # a unit which was never stated does not block a conversion
        merged.convert_units(distance="ft")

    def test_falling_back_reads_no_arrays(self):
        """A merge the index cannot describe does not read arrays first.

        The rows say whether the index can stand in for every member, so
        a merge which must load patches pays for no discarded reads.
        """
        reads = []
        assembler = PatchAssembler(
            load_patch=lambda kwargs: None,
            merge_kwargs={},
            plan_dim="time",
            array_source=lambda row, shape: (
                reads.append(row) or ArraySource.full(shape, 0.0)
            ),
        )
        stateable = {
            "dims": "distance,time",
            "distance_min": 0,
            "distance_max": 2,
            "distance_step": 1,
            "time_min": np.datetime64("2020-01-01"),
            "time_max": np.datetime64("2020-01-01T00:00:03"),
            "time_step": np.timedelta64(1, "s"),
        }
        silent = dict(stateable, time_step=np.timedelta64("NaT", "s"))
        assert assembler._member_meta_from_index([stateable, silent]) is None
        assert not reads

    def test_row_range_refuses_what_it_cannot_state(self):
        """A range the row states differently than the file had is refused."""
        row_range = assembly_module._row_range
        base = {"x_min": 0.0, "x_max": 4.0, "x_step": 1.0}
        # no stored dtype: the envelope is taken as it stands
        assert row_range(base, "x") == (0.0, 4.0, 1.0)
        # an integer coordinate comes back an integer one
        typed = base | {"_x_coord_dtype": "int64"}
        assert [type(x).__name__ for x in row_range(typed, "x")] == ["int64"] * 3
        # a descending coordinate: the row states the value order, not
        # the sample order, so its start is unknown
        assert row_range(base | {"x_step": -1.0}, "x") is None
        # a converted envelope is float in truth, whatever the file held
        converted = typed | {"_x_units_source": "cm", "_x_units": "m"}
        assert row_range(converted, "x") is None
        # and past 2**53 a float cannot have held the integer exactly
        big = typed | {"x_min": 0.0, "x_max": float(2**53 + 8)}
        assert row_range(big, "x") is None

    def test_attrs_from_row(self):
        """The row's attrs are the file's; bookkeeping and envelopes are not."""
        row = {
            "dims": "distance,time",
            "output_id": 3,
            "_modified": False,
            "_time_units": "s",
            "time_min": 0,
            "time_max": 1,
            "time_step": 1,
            "distance_min": 0,
            "distance_max": 1,
            "distance_step": 1,
            "source_path": "/a.h5",
            "source_format": "DASDAE",
            "source_version": "1",
            "source_patch_key": "DAS__x",
            "origin_id": "abc",
            "tag": "raw",
            "vendor_thing": 5,
            "blank": np.nan,
        }
        attrs = assembly_module._attrs_from_row(row, ("distance", "time"))
        assert attrs.tag == "raw"
        assert attrs["vendor_thing"] == 5
        assert attrs.origin_id == "abc"
        assert "_source_patch_key" not in attrs
        assert row["source_patch_key"] == "DAS__x"
        for name in ("output_id", "source_path", "time_min", "blank", "_modified"):
            assert name not in dict(attrs)


class TestRecipeMerge:
    """A merge the rows describe is read as one recipe over its members."""

    step = np.timedelta64(10_000_000, "ns")

    def _write(self, directory, patches):
        """Write one file per patch and return the indexed spool."""
        for num, patch in enumerate(patches):
            patch.io.write(directory / f"m{num}.h5", "dasdae")
        return dc.spool(directory).update()

    def _patch(self, start, samples, channels=4, dtype="float32", dims=None, seed=0):
        """A patch on the shared grid, with its own data."""
        rng = np.random.default_rng(seed)
        data = rng.random((samples, channels))
        time = dc.core.get_coord(start=start, step=self.step, shape=(samples,))
        distance = dc.core.get_coord(start=0.0, step=1.0, shape=(channels,))
        patch = dc.Patch(
            data=data.astype(dtype),
            coords={"time": time, "distance": distance},
            dims=("time", "distance"),
        )
        return patch if dims is None else patch.transpose(*dims)

    def _grid(self, count, samples):
        """The start of each of `count` patches laid end to end."""
        origin = np.datetime64("2020-01-01")
        return [origin + self.step * samples * num for num in range(count)]

    def test_even_files_match_numpy(self, tmp_path, calls):
        """Files of different lengths merge into the array numpy makes."""
        starts, patches, lengths = [np.datetime64("2020-01-01")], [], (5, 9, 4)
        for num, samples in enumerate(lengths):
            patches.append(self._patch(starts[-1], samples, seed=num))
            starts.append(starts[-1] + self.step * samples)
        spool = self._write(tmp_path, patches)
        out = spool.chunk(time=None)[0]
        assert calls == {"patch": 0, "array": len(lengths)}
        expected = np.concatenate([x.data for x in patches], axis=0)
        assert np.array_equal(out.data, expected)
        assert out.data.dtype == expected.dtype
        assert out.shape == expected.shape
        assert out.dims == ("time", "distance")
        assert out.get_coord("time").step == self.step
        assert len(out.get_coord("time")) == sum(lengths)

    def test_mixed_dtypes_promote(self, tmp_path, calls):
        """Members stored at different dtypes promote as numpy does."""
        starts = self._grid(3, 6)
        dtypes = ("float32", "int16", "float64")
        patches = [
            self._patch(start, 6, dtype=dtype, seed=num)
            for num, (start, dtype) in enumerate(zip(starts, dtypes))
        ]
        spool = self._write(tmp_path, patches)
        out = spool.chunk(time=None)[0]
        assert calls == {"patch": 0, "array": 3}
        expected = np.concatenate([x.data for x in patches], axis=0)
        assert out.data.dtype == np.result_type(*[np.dtype(x) for x in dtypes])
        assert out.data.dtype == expected.dtype
        assert np.array_equal(out.data, expected)

    def test_members_stored_the_other_way_round(self, tmp_path, calls):
        """Dimension order partitions the plan, so each order merges alone."""
        starts = self._grid(4, 7)
        orders = (None, None, ("distance", "time"), ("distance", "time"))
        patches = [
            self._patch(start, 7, dims=dims, seed=num)
            for num, (start, dims) in enumerate(zip(starts, orders))
        ]
        spool = self._write(tmp_path, patches)
        merged = spool.chunk(time=None)
        assert len(merged) == 2
        for out, pair in zip(merged, (patches[:2], patches[2:])):
            axis = out.dims.index("time")
            assert out.dims == pair[0].dims
            expected = np.concatenate([x.data for x in pair], axis=axis)
            assert np.array_equal(out.data, expected)
        assert calls == {"patch": 0, "array": 4}

    def test_a_member_in_another_order_takes_the_patch_path(self):
        """A recipe places whole arrays, so it refuses one stored transposed."""
        rows = [{"dims": "time,distance"}, {"dims": "distance,time"}]
        metas = [
            assembly_module._MemberMeta(
                dims=dims,
                coords=dc.core.coordmanager.get_coord_manager(
                    {
                        "time": dc.core.get_coord(start=0.0, step=1.0, shape=(3,)),
                        "distance": dc.core.get_coord(start=0.0, step=1.0, shape=(2,)),
                    },
                    dims=dims,
                ),
                attrs=dc.PatchAttrs(),
                extent=(3, 2) if dims[0] == "time" else (2, 3),
                window=(slice(0, 3), slice(0, 2))
                if dims[0] == "time"
                else (slice(0, 2), slice(0, 3)),
            )
            for dims in (("time", "distance"), ("distance", "time"))
        ]
        assembler = PatchAssembler(
            load_patch=lambda kwargs: None,
            merge_kwargs={},
            plan_dim="time",
            array_source=lambda row, shape: ArraySource.full(shape, 1.0),
        )
        assert assembler._recipe(rows[:1], metas[:1], metas[0].dims, 0, "time")
        assert assembler._recipe(rows, metas, metas[0].dims, 0, "time") is None

    def test_incompatible_shapes_refuse_the_merge(self):
        """Members which disagree off the merged axis cannot be laid together."""
        metas = []
        for channels in (2, 3):
            coords = {
                "time": dc.core.get_coord(start=0.0, step=1.0, shape=(3,)),
                "distance": dc.core.get_coord(start=0.0, step=1.0, shape=(channels,)),
            }
            metas.append(
                assembly_module._MemberMeta(
                    dims=("time", "distance"),
                    coords=dc.core.coordmanager.get_coord_manager(
                        coords, dims=("time", "distance")
                    ),
                    attrs=dc.PatchAttrs(),
                    extent=(3, channels),
                    window=(slice(0, 3), slice(0, channels)),
                )
            )
        assembler = PatchAssembler(
            load_patch=lambda kwargs: None,
            merge_kwargs={},
            plan_dim="time",
            array_source=lambda row, shape: ArraySource.full(shape, 1.0),
        )
        rows = [{"dims": "time,distance"}] * 2
        with pytest.raises(CoordMergeError, match="not being merged"):
            assembler._recipe(rows, metas, metas[0].dims, 0, "time")

    def test_three_dimensional_members(self, tmp_path, calls):
        """A cube merges along its planned dimension and no other."""
        rng = np.random.default_rng(3)
        patches, starts = [], self._grid(3, 5)
        for num, start in enumerate(starts):
            data = rng.random((5, 4, 2)).astype(np.float32)
            coords = {
                "time": dc.core.get_coord(start=start, step=self.step, shape=(5,)),
                "distance": dc.core.get_coord(start=0.0, step=1.0, shape=(4,)),
                "depth": dc.core.get_coord(start=0.0, step=2.0, shape=(2,)),
            }
            patches.append(
                dc.Patch(data=data, coords=coords, dims=("time", "distance", "depth"))
            )
        spool = self._write(tmp_path, patches)
        out = spool.chunk(time=None)[0]
        assert calls == {"patch": 0, "array": 3}
        assert out.shape == (15, 4, 2)
        assert np.array_equal(out.data, np.concatenate([x.data for x in patches], 0))

    def test_multi_patch_file(self, tmp_path, calls):
        """Several members of one file each read their own array."""
        starts = self._grid(6, 5)
        patches = [self._patch(x, 5, seed=n) for n, x in enumerate(starts)]
        dc.write(dc.spool(patches[:3]), tmp_path / "a.h5", "dasdae")
        dc.write(dc.spool(patches[3:]), tmp_path / "b.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        out = spool.chunk(time=None)[0]
        assert calls == {"patch": 0, "array": 6}
        assert np.array_equal(out.data, np.concatenate([x.data for x in patches], 0))

    def test_trimmed_overlap_takes_the_recipe_path(self, tmp_path, calls):
        """An overlap trims the second member to a window of its source."""
        first = self._patch(np.datetime64("2020-01-01"), 10, seed=1)
        second = self._patch(np.datetime64("2020-01-01") + self.step * 6, 10, seed=2)
        spool = self._write(tmp_path, [first, second])
        out = spool.chunk(time=None)[0]
        assert calls == {"patch": 0, "array": 2}
        # the second patch starts 6 samples in, so its first 4 are the overlap
        kept = second.data[4:]
        assert out.shape[0] == 16
        assert np.array_equal(out.data, np.concatenate([first.data, kept], axis=0))

    @pytest.mark.parametrize("conflict", ["drop", "keep_first"])
    def test_conflicting_attrs(self, tmp_path, calls, conflict):
        """Every conflict policy reaches the same patch either way."""
        starts = self._grid(3, 6)
        patches = [
            self._patch(start, 6, seed=num).update_attrs(instrument_id=f"i{num}")
            for num, start in enumerate(starts)
        ]
        spool = self._write(tmp_path, patches)
        out = spool.chunk(time=None, conflict=conflict)[0]
        assert calls == {"patch": 0, "array": 3}
        expected = np.concatenate([x.data for x in patches], axis=0)
        assert np.array_equal(out.data, expected)
        if conflict == "keep_first":
            assert out.attrs.instrument_id == "i0"
        else:
            assert "instrument_id" not in dict(out.attrs)

    @pytest.mark.parametrize("fill", [None, 0.0])
    def test_gap_between_members(self, tmp_path, calls, fill):
        """A bridged hole holds the fill value and nothing else moves."""
        gap = 3
        first = self._patch(np.datetime64("2020-01-01"), 8, seed=1)
        start = np.datetime64("2020-01-01") + self.step * (8 + gap)
        second = self._patch(start, 8, seed=2)
        spool = self._write(tmp_path, [first, second])
        kwargs = {} if fill is None else {"fill_value": fill}
        merged = spool.chunk(time=None, tolerance=10, **kwargs)
        assert len(merged) == 1
        out = merged[0]
        assert calls == {"patch": 0, "array": 2}
        if fill is None:
            joined = np.concatenate([first.data, second.data], axis=0)
            assert np.array_equal(out.data, joined)
        else:
            hole = np.full((gap, first.shape[1]), fill, dtype=first.data.dtype)
            joined = np.concatenate([first.data, hole, second.data], axis=0)
            assert np.array_equal(out.data, joined)

    def test_every_case_matches_the_patch_path(self, tmp_path_factory, monkeypatch):
        """The recipe merge and the patch merge agree in every part."""
        cases = {
            "even": [(0, 5, 4, "float32", None), (5, 9, 4, "float32", None)],
            "dtypes": [(0, 6, 4, "float32", None), (6, 6, 4, "int16", None)],
            "transposed": [
                (0, 7, 4, "float32", None),
                (7, 7, 4, "float32", ("distance", "time")),
            ],
        }
        for name, spec in cases.items():
            path = tmp_path_factory.mktemp(f"recipe_{name}")
            origin = np.datetime64("2020-01-01")
            patches = [
                self._patch(
                    origin + self.step * offset, samples, channels, dtype, dims, seed
                )
                for seed, (offset, samples, channels, dtype, dims) in enumerate(spec)
            ]
            spool = self._write(path, patches)
            fast = spool.chunk(time=None)[0]
            with monkeypatch.context() as patcher:
                _force_patch_path(patcher)
                slow = spool.chunk(time=None)[0]
            assert fast.dims == slow.dims
            assert fast.data.dtype == slow.data.dtype
            assert np.array_equal(fast.data, slow.data)
            assert fast.coords == slow.coords
            assert dict(fast.attrs) == dict(slow.attrs)
            assert fast.attrs.history == slow.attrs.history


class TestChunkFillValue:
    """Filling the samples a bridged hole is missing."""

    hole = 9  # samples missing between the two patches

    @pytest.fixture()
    def gapped_spool(self, random_patch):
        """Two patches of one grid with `hole` samples missing between them."""
        first = random_patch.select(time=(0, 100), samples=True)
        second = random_patch.select(time=(100 + self.hole, ...), samples=True)
        return dc.spool([first, second])

    @staticmethod
    def _fill_positions(patch, dim="time"):
        """Which positions along dim hold no data."""
        axis = patch.get_axis(dim)
        other = tuple(x for x in range(patch.ndim) if x != axis)
        return np.flatnonzero(np.isnan(patch.data).all(axis=other))

    def test_merge_fills_the_hole(self, gapped_spool):
        """A merge over the hole comes back evenly sampled at the true step."""
        first, second = gapped_spool
        step = first.get_coord("time").step
        merged = gapped_spool.chunk(time=None, tolerance=10, fill_value=np.nan)[0]
        coord = merged.get_coord("time")
        assert isinstance(coord, CoordRange)
        assert coord.step == step
        # the fill belongs where the samples are missing, not at an end
        start = len(first.get_coord("time"))
        expected = np.arange(start, start + self.hole)
        assert np.array_equal(self._fill_positions(merged), expected)
        # and the real samples still carry the labels they arrived with
        kept = np.setdiff1d(np.arange(len(coord)), expected)
        source = np.concatenate([first.data, second.data], axis=1)
        assert np.array_equal(merged.data[:, kept], source)

    def test_merge_without_fill_value_keeps_the_hole(self, gapped_spool):
        """Without one the merge is segmented, as it is with no tolerance to span."""
        with pytest.warns(UserWarning, match="fill_value"):
            merged = gapped_spool.chunk(time=None, tolerance=10)[0]
        assert isinstance(merged.get_coord("time"), CoordSegmented)

    def test_fill_value_does_not_widen_the_tolerance(self, gapped_spool):
        """A hole the tolerance does not span is still a boundary."""
        assert len(gapped_spool.chunk(time=None, fill_value=np.nan)) == 2

    def test_filling_merge_does_not_warn(self, gapped_spool):
        """The forced-merge warning is about uneven sampling, which filling ends."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gapped_spool.chunk(time=None, tolerance=10, fill_value=np.nan)[0]

    def test_nothing_to_fill_is_untouched(self, random_spool):
        """A contiguous spool merges as it would without a fill value."""
        merged = random_spool.chunk(time=None, fill_value=np.nan)[0]
        assert isinstance(merged.get_coord("time"), CoordRange)
        assert not np.isnan(merged.data).any()

    def test_single_member_hole_fills_in_place(self, gapped_spool):
        """A lone member carrying its own hole is filled at the hole, not the end."""
        with pytest.warns(UserWarning, match="fill_value"):
            gapped = gapped_spool.chunk(time=None, tolerance=20)[0]
        assert isinstance(gapped.get_coord("time"), CoordSegmented)
        # one source patch, so nothing is merged and only the fill reshapes it
        filled = dc.spool([gapped]).chunk(time=None, tolerance=20, fill_value=np.nan)[0]
        where = np.flatnonzero(np.isnan(filled.data).all(axis=0))
        first = len(gapped_spool[0].get_coord("time"))
        assert np.array_equal(where, np.arange(first, first + self.hole))
        # and the samples either side keep the values their labels had
        kept = np.setdiff1d(np.arange(filled.shape[1]), where)
        assert np.array_equal(filled.data[:, kept], gapped.data)

    def test_filled_spool_rechunks_without_losing_fill(self, gapped_spool):
        """Re-planning a filled spool must not collapse back to its sources."""
        filled = gapped_spool.chunk(time=None, tolerance=10, fill_value=np.nan)
        again = filled.chunk(time=None)
        assert len(again) == 1
        before, after = filled[0], again[0]
        assert after.shape == before.shape
        assert np.array_equal(after.data, before.data, equal_nan=True)

    def test_hole_is_kept_when_steps_differ_slightly(self, random_patch):
        """Steps too close to tell apart are still not licence to spread a hole."""
        coord = random_patch.get_coord("time")
        first = random_patch.select(time=(0, 100), samples=True)
        second = random_patch.select(time=(110, ...), samples=True)
        # a step one nanosecond longer: well inside the sampling group
        nudged = second.new(
            coords=second.coords.update(time_step=coord.step + to_timedelta64(1e-9))
        )
        spool = dc.spool([first, nudged])
        with pytest.warns(UserWarning, match="fill_value"):
            merged = spool.chunk(time=None, tolerance=20)[0]
        merged_coord = merged.get_coord("time")
        assert isinstance(merged_coord, CoordSegmented)
        # the pathology: one range whose step was stretched to cover the hole
        assert abs(merged_coord.segments[0].step - coord.step) < to_timedelta64(1e-8)

    def test_tolerance_bounds_a_hole_the_planner_never_saw(self, random_patch):
        """A pending selection hides a hole from the plan; tolerance still rules."""
        first = random_patch.select(time=(0, 100), samples=True)
        second = random_patch.select(time=(600, 1000), samples=True)
        with pytest.warns(UserWarning, match="fill_value"):
            gapped = dc.spool([first, second]).chunk(time=None, tolerance=600)[0]
        # the selection resolves against the patch, so the plan describes
        # the whole of it rather than its runs, holes and all
        spool = dc.spool([gapped]).select(time=(0.1, -0.1), relative=True)
        with suppress_warnings(UserWarning):
            narrow = spool.chunk(time=None, tolerance=1, fill_value=np.nan)[0]
            wide = spool.chunk(time=None, tolerance=600, fill_value=np.nan)[0]
        assert not np.isnan(narrow.data).any()
        assert int(np.isnan(wide.data).all(axis=0).sum()) == 500
        assert len(wide.get_coord("time")) == len(narrow.get_coord("time")) + 500

    def test_infinite_tolerance_fills_every_hole(self, gapped_spool):
        """No boundary is a gap, so no hole is too wide to fill."""
        merged = gapped_spool.chunk(time=None, tolerance=np.inf, fill_value=np.nan)[0]
        assert isinstance(merged.get_coord("time"), CoordRange)
        assert len(self._fill_positions(merged)) == self.hole

    def test_absolute_tolerance_fills(self, gapped_spool, random_patch):
        """A tolerance stated in the coordinate's own units bounds the fill."""
        step = random_patch.get_coord("time").step
        wide = gapped_spool.chunk(
            time=None, tolerance=self.hole * step, fill_value=np.nan
        )[0]
        assert len(self._fill_positions(wide)) == self.hole
        # a hole wider than the tolerance is left as a boundary
        narrow = gapped_spool.chunk(time=None, tolerance=2 * step, fill_value=np.nan)
        assert len(narrow) == 2

    def test_pending_sample_selection_still_chunks(self, random_patch):
        """A selection the plan cannot describe leaves nothing to lend or fill."""
        first = random_patch.select(time=(0, 100), samples=True)
        second = random_patch.select(time=(600, 1000), samples=True)
        with pytest.warns(UserWarning, match="fill_value"):
            gapped = dc.spool([first, second]).chunk(time=None, tolerance=600)[0]
        spool = dc.spool([gapped]).select(time=(0, 499), samples=True)
        with suppress_warnings(UserWarning):
            assert len(spool.chunk(time=None, tolerance=600, fill_value=np.nan)) == 0

    def test_integer_data_raises(self, gapped_spool):
        """Integers have no null, so NaN cannot be what a hole holds."""
        spool = dc.spool([x.new(data=x.data.astype(np.int32)) for x in gapped_spool])
        with pytest.raises(ParameterError, match="Cannot fill data of dtype"):
            spool.chunk(time=None, tolerance=10, fill_value=np.nan)[0]

    def test_integer_fill_value(self, gapped_spool):
        """A value of the data's own dtype fills it."""
        spool = dc.spool(
            [x.new(data=x.data.astype(np.int32) + 1) for x in gapped_spool]
        )
        merged = spool.chunk(time=None, tolerance=10, fill_value=0)[0]
        assert merged.data.dtype == np.int32
        axis = merged.get_axis("time")
        other = tuple(x for x in range(merged.ndim) if x != axis)
        assert int((merged.data == 0).all(axis=other).sum()) == self.hole


class TestChunkFillWindows:
    """Windows laid over a bridged hole, which no source patch feeds."""

    @pytest.fixture()
    def spool_and_tolerance(self, random_patch):
        """Two 2 s patches of one grid with a 4 s hole between them."""
        start = random_patch.get_coord("time").min()
        first = random_patch.select(time=(start, start + to_timedelta64(2)))
        second = random_patch.select(time=(start + to_timedelta64(6), ...))
        return dc.spool([first, second]), to_timedelta64(5)

    def test_windows_cover_the_hole(self, spool_and_tolerance):
        """Every second of the bridged span comes back, filled where it must be."""
        spool, tolerance = spool_and_tolerance
        chunked = spool.chunk(time=1, tolerance=tolerance, fill_value=np.nan)
        patches = list(chunked)
        assert len(patches) == 8
        # one window per second, none of them short
        lengths = {len(x.get_coord("time")) for x in patches}
        assert lengths == {250}
        # the three windows wholly inside the hole hold nothing else
        filled = [bool(np.isnan(x.data).all()) for x in patches]
        assert filled == [False, False, False, True, True, True, False, False]

    def test_partial_window_is_padded(self, spool_and_tolerance):
        """A window the sources only partly feed is filled out, not left short."""
        spool, tolerance = spool_and_tolerance
        chunked = spool.chunk(time=1, tolerance=tolerance, fill_value=np.nan)
        patch = chunked[2]
        assert len(patch.get_coord("time")) == 250
        assert int(np.isnan(patch.data).all(axis=0).sum()) == 249

    def test_patches_match_the_rows(self, spool_and_tolerance):
        """Every assembled patch spans the window its contents row advertises."""
        spool, tolerance = spool_and_tolerance
        chunked = spool.chunk(time=1, tolerance=tolerance, fill_value=np.nan)
        contents = chunked.get_contents()
        for num, row in enumerate(contents.itertuples()):
            coord = chunked[num].get_coord("time")
            assert coord.min() == row.time_min
            assert coord.max() == row.time_max

    def test_windows_are_dropped_without_a_fill_value(self, spool_and_tolerance):
        """The plan only publishes an output no source feeds when it can fill it."""
        spool, tolerance = spool_and_tolerance
        with suppress_warnings(UserWarning):
            plain = spool.chunk(time=1, tolerance=tolerance)
        assert len(plain) == 5

    def test_streaming_merge_fills(self, random_patch, tmp_path_factory):
        """The index-backed merge streams into a buffer; it fills too."""
        path = tmp_path_factory.mktemp("fill_stream")
        first = random_patch.select(time=(0, 100), samples=True)
        second = random_patch.select(time=(109, ...), samples=True)
        for name, patch in (("a.h5", first), ("b.h5", second)):
            dc.write(patch, path / name, "dasdae")
        spool = dc.spool(path).update()
        merged = spool.chunk(time=None, tolerance=10, fill_value=np.nan)[0]
        coord = merged.get_coord("time")
        assert isinstance(coord, CoordRange)
        assert coord.step == random_patch.get_coord("time").step
        # placed at the hole the sources left, not appended at an end
        where = np.flatnonzero(np.isnan(merged.data).all(axis=0))
        start = len(first.get_coord("time"))
        assert np.array_equal(where, np.arange(start, start + 9))

    def test_window_edges_off_the_grid_do_not_move_labels(self):
        """A chunk length of a fraction of a sample must not re-anchor the data."""
        values = np.arange(8.0)
        patch = dc.Patch(
            data=values[None, :],
            coords={"distance": np.array([0.0]), "time": values},
            dims=("distance", "time"),
        )
        spool = dc.spool([patch])
        # 2.5 samples per window, so no window edge lands on a position
        plain = list(spool.chunk(time=2.5))
        filled = list(spool.chunk(time=2.5, fill_value=np.nan))
        assert len(filled) == len(plain)
        for fill, expected in zip(filled, plain):
            # nothing was missing, so filling changes neither labels nor data
            assert np.array_equal(
                fill.get_coord("time").values, expected.get_coord("time").values
            )
            assert np.array_equal(fill.data, expected.data)

    def test_all_fill_window_takes_the_partition_dtype(self, spool_and_tolerance):
        """With no member to ask, an all-fill window is the partition's dtype."""
        spool, tolerance = spool_and_tolerance
        first, second = spool
        # float16 with float32 combines to float32, which is neither
        # member's dtype nor the no-dtype fallback of float64
        mixed = dc.spool(
            [
                first.new(data=first.data.astype(np.float16)),
                second.new(data=second.data.astype(np.float32)),
            ]
        )
        chunked = mixed.chunk(time=1, tolerance=tolerance, fill_value=np.nan)
        # a fed window keeps its own member's dtype; the all-fill one cannot
        assert chunked[0].data.dtype == np.float16
        assert chunked[4].data.dtype == np.float32

    def test_fill_window_copies_a_coord_no_envelope_could_state(self, random_patch):
        """An unevenly sampled coordinate is copied, not rebuilt from a row."""
        values = np.sort(np.random.default_rng(0).uniform(0, 300, 300))
        patch = random_patch.update_coords(distance=values)
        start = patch.get_coord("time").min()
        first = patch.select(time=(start, start + to_timedelta64(2)))
        second = patch.select(time=(start + to_timedelta64(6), ...))
        spool = dc.spool([first, second])
        with suppress_warnings(UserWarning):
            chunked = spool.chunk(
                time=1, tolerance=to_timedelta64(5), fill_value=np.nan
            )
        filled = chunked[4]
        assert np.isnan(filled.data).all()
        assert np.array_equal(filled.get_coord("distance").values, values)

    def test_descending_dimension_fills(self, random_patch):
        """A coordinate running the other way fills at the same positions."""
        size = random_patch.shape[0]
        patch = random_patch.update_coords(distance=np.arange(size)[::-1] * 1.0)
        spool = dc.spool(
            [patch.select(distance=(200, 299)), patch.select(distance=(0, 100))]
        )
        with suppress_warnings(UserWarning):
            chunked = spool.chunk(distance=20, tolerance=200, fill_value=np.nan)
        assert {len(x.get_coord("distance")) for x in chunked} == {20}
        # every window runs high to low, as its sources do
        for windowed in chunked:
            coord = windowed.get_coord("distance")
            assert coord.values[0] > coord.values[-1]
        # the hole spans 101 to 199, so those windows hold nothing else
        all_fill = [bool(np.isnan(x.data).all()) for x in chunked]
        assert all_fill == [False] * 6 + [True] * 4 + [False] * 5

    def test_windows_off_a_hidden_grid_are_refused(self, random_patch):
        """A window holding no sample position is no output at all."""
        first = random_patch.select(time=(0, 100), samples=True)
        second = random_patch.select(time=(600, 1000), samples=True)
        with pytest.warns(UserWarning, match="fill_value"):
            gapped = dc.spool([first, second]).chunk(time=None, tolerance=600)[0]
        # the selection resolves against the patch, so the partition
        # envelope the windows are laid over is not stated on the grid
        spool = dc.spool([gapped]).select(time=(0.1, -0.1), relative=True)
        with suppress_warnings(UserWarning), pytest.raises(ChunkError, match="chunk"):
            spool.chunk(time=0.5, tolerance=600, fill_value=np.nan)

    def test_fill_window_needs_a_span_and_a_step(self, random_patch):
        """Neither the row's span nor the sibling's step can be missing."""
        coords = random_patch.coords
        row = {
            "dims": "distance,time",
            "time_min": random_patch.get_coord("time").min(),
            "time_max": random_patch.get_coord("time").max(),
            "time_step": random_patch.get_coord("time").step,
        }
        # the span the window covers is what says how many samples it holds
        assembly_module.patch_from_fill(coords, row, "time", np.nan)
        with pytest.raises(ChunkError, match="Cannot fill a hole along 'time'"):
            assembly_module.patch_from_fill(
                coords, {**row, "time_step": None}, "time", np.nan
            )

    def test_all_fill_windows_share_one_read(self, spool_and_tolerance):
        """The sibling a fill window copies its structure from is read once."""
        spool, tolerance = spool_and_tolerance
        chunked = spool.chunk(time=1, tolerance=tolerance, fill_value=np.nan)
        resolver = chunked._catalog.resolver
        reads = []
        original = resolver._load_member
        resolver._load_member = lambda *a, **kw: (
            reads.append(1),
            original(*a, **kw),
        )[1]
        # three windows lie wholly inside the hole, between two fed ones
        for index in (3, 4, 5):
            assert np.isnan(chunked[index].data).all()
        first_pass = len(reads)
        assert 0 < first_pass <= 2  # one sibling on each side of the hole
        for index in (3, 4, 5):
            assert np.isnan(chunked[index].data).all()
        assert len(reads) == first_pass  # the coordinates are kept

    def test_fill_window_keeps_associated_coords(self, random_patch_with_lat_lon):
        """A window with no data still carries what its siblings carry."""
        patch = random_patch_with_lat_lon
        start = patch.get_coord("time").min()
        first = patch.select(time=(start, start + to_timedelta64(2)))
        second = patch.select(time=(start + to_timedelta64(6), ...))
        spool = dc.spool([first, second])
        with suppress_warnings(UserWarning):
            chunked = spool.chunk(
                time=1, tolerance=to_timedelta64(5), fill_value=np.nan
            )
        fed, filled = chunked[0], chunked[4]
        assert set(fed.coords.coord_map) == set(filled.coords.coord_map)
        # riding distance, which the chunk did not touch, so the values
        # are the ones every patch here states
        for name in ("latitude", "longitude"):
            assert np.array_equal(
                fed.get_coord(name).values, filled.get_coord(name).values
            )
        # and the rows agree well enough to be merged back together
        with suppress_warnings(UserWarning):
            merged = chunked.chunk(time=None)[0]
        assert len(merged.get_coord("time")) == 2000
        assert len(chunked.select(latitude=...)) == len(chunked)


def _rewrite_dasdae(path, patch, group=None):
    """Rewrite a single-patch DASDAE file, keeping its waveform group name."""
    import h5py  # noqa: PLC0415

    from dascore.io.dasdae.utils import _save_patch  # noqa: PLC0415

    with h5py.File(path, "a") as handle:
        waveforms = handle["waveforms"]
        name = group or next(iter(waveforms))
        _save_patch(patch, waveforms, name)
    return name


class TestStaleSourceCheck:
    """A recipe reads a promised window, so a changed file leaves the route."""

    step = np.timedelta64(10_000_000, "ns")

    def _patch(self, start, samples, channels=4, seed=0):
        """A patch on the shared grid with its own data."""
        rng = np.random.default_rng(seed)
        time = dc.core.get_coord(start=start, step=self.step, shape=(samples,))
        distance = dc.core.get_coord(start=0.0, step=1.0, shape=(channels,))
        return dc.Patch(
            data=rng.random((samples, channels)).astype("float32"),
            coords={"time": time, "distance": distance},
            dims=("time", "distance"),
        )

    @pytest.fixture
    def spool_and_paths(self, tmp_path):
        """Two adjacent single-patch files and the spool indexing them."""
        start = np.datetime64("2020-01-01")
        paths = []
        for num in range(2):
            patch = self._patch(start + self.step * 8 * num, 8, seed=num)
            path = tmp_path / f"m{num}.h5"
            patch.io.write(path, "dasdae")
            paths.append(path)
        return dc.spool(tmp_path).update(), paths

    def test_unchanged_sources_stay_on_the_recipe(self, spool_and_paths, calls):
        """Nothing changed, so nothing is loaded as a patch."""
        spool, _ = spool_and_paths
        out = spool.chunk(time=None)[0]
        assert calls == {"patch": 0, "array": 2}
        assert out.shape[0] == 16

    def test_longer_rewrite_takes_the_patch_path(self, spool_and_paths, calls):
        """A file grown under the same key must not give its first n samples."""
        spool, paths = spool_and_paths
        start = np.datetime64("2020-01-01")
        grown = self._patch(start, 12, seed=7)
        _rewrite_dasdae(paths[0], grown)
        out = spool.chunk(time=None)[0]
        assert calls["patch"] == 2, "the changed file abandons the recipe"
        assert np.array_equal(out.data[:12], grown.data)

    def test_same_shape_rewrite_takes_the_patch_path(self, spool_and_paths, calls):
        """Identical shape, different values: the shape check cannot see it."""
        spool, paths = spool_and_paths
        start = np.datetime64("2020-01-01")
        replaced = self._patch(start, 8, seed=99)
        _rewrite_dasdae(paths[0], replaced)
        # a filesystem which does not move mtime on its own is still a
        # filesystem whose file changed, so it is moved here
        moved = paths[0].stat().st_mtime_ns + 10**9
        os.utime(paths[0], ns=(moved, moved))
        out = spool.chunk(time=None)[0]
        assert calls["patch"] == 2
        assert np.array_equal(out.data[:8], replaced.data)

    def test_a_stat_which_will_not_answer_is_ignored(
        self, spool_and_paths, calls, monkeypatch
    ):
        """A source with no size and no mtime says nothing about itself."""
        spool, _ = spool_and_paths
        monkeypatch.setattr(planned, "_source_stats", lambda path: (None, None))
        assert spool.chunk(time=None)[0].shape[0] == 16
        assert calls == {"patch": 0, "array": 2}

    def test_one_stat_per_file(self, tmp_path, monkeypatch):
        """A multi-patch file is stat-ed once, not once per member."""
        start = np.datetime64("2020-01-01")
        patches = [self._patch(start + self.step * 8 * n, 8, seed=n) for n in range(3)]
        dc.write(dc.spool(patches[:2]), tmp_path / "a.h5", "dasdae")
        patches[2].io.write(tmp_path / "b.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        stats = []
        original = planned._source_stats
        monkeypatch.setattr(
            planned,
            "_source_stats",
            lambda path: (stats.append(str(path)), original(path))[1],
        )
        assert spool.chunk(time=None)[0].shape[0] == 24
        assert len(stats) == len(set(stats)) == 2, "three members, two files"

    def test_a_remote_source_is_never_stat_ed(self, spool_and_paths, monkeypatch):
        """A store which charges for metadata is left alone."""
        spool, _ = spool_and_paths
        stats = []
        monkeypatch.setattr(
            planned,
            "_source_stats",
            lambda path: (stats.append(str(path)), (None, None))[1],
        )
        monkeypatch.setattr(planned, "is_local_path", lambda path: False)
        chunked = spool.chunk(time=None)
        resolver = chunked._catalog.resolver
        rows = resolver.member_rows.to_dict("records")
        assert resolver._sources_unchanged(rows)
        assert not stats

    @pytest.mark.parametrize(
        ("size_delta", "mtime_delta", "unchanged"),
        [(0, 0, True), (1, 0, False), (0, 1, False), (1, 1, False)],
    )
    def test_both_size_and_mtime_decide(
        self, spool_and_paths, monkeypatch, size_delta, mtime_delta, unchanged
    ):
        """Either field moving on its own says the file is not what it was."""
        spool, _ = spool_and_paths
        resolver = spool.chunk(time=None)._catalog.resolver
        rows = resolver.member_rows.to_dict("records")
        mtime, size = resolver.source_stats[rows[0]["source_path"]]
        monkeypatch.setattr(
            planned,
            "_source_stats",
            lambda path: (size + size_delta, mtime + mtime_delta),
        )
        assert resolver._sources_unchanged(rows[:1]) is unchanged

    def test_a_source_the_index_never_stat_ed_is_ignored(self, spool_and_paths):
        """An index which recorded neither size nor mtime refuses nothing."""
        spool, _ = spool_and_paths
        resolver = spool.chunk(time=None)._catalog.resolver
        rows = resolver.member_rows.to_dict("records")
        assert resolver.source_stats
        resolver.source_stats = {}
        assert resolver._sources_unchanged(rows)

    def test_a_rechunk_keeps_what_the_index_recorded(self, spool_and_paths, calls):
        """The stats come from the index, which a derived catalog is not."""
        spool, paths = spool_and_paths
        chunked = spool.chunk(time=None)
        assert chunked.chunk(time=None)._catalog.resolver.source_stats
        grown = self._patch(np.datetime64("2020-01-01"), 12, seed=7)
        _rewrite_dasdae(paths[0], grown)
        assert list(chunked.chunk(time=None))
        assert calls["patch"] > 0


class TestTrimmedRecipeMerge:
    """A member the plan trims is a window of its source in the recipe."""

    step = np.timedelta64(10_000_000, "ns")

    def _spool(self, directory, count=5, samples=8, channels=3):
        """`count` adjacent single-patch files and their indexed spool."""
        start = np.datetime64("2020-01-01")
        for num in range(count):
            rng = np.random.default_rng(num)
            time = dc.core.get_coord(
                start=start + self.step * samples * num,
                step=self.step,
                shape=(samples,),
            )
            distance = dc.core.get_coord(start=0.0, step=1.0, shape=(channels,))
            patch = dc.Patch(
                data=rng.random((samples, channels)).astype("float32"),
                coords={"time": time, "distance": distance},
                dims=("time", "distance"),
            )
            patch.io.write(directory / f"m{num}.h5", "dasdae")
        return dc.spool(directory).update()

    def test_a_trim_reads_only_its_window(self, tmp_path, calls, monkeypatch):
        """Chunk boundaries inside files stay on the recipe and stay exact."""
        spool = self._spool(tmp_path)
        chunked = spool.chunk(time=to_timedelta64(0.12))
        fast = list(chunked)
        assert calls["patch"] == 0
        _force_patch_path(monkeypatch)
        slow = list(spool.chunk(time=to_timedelta64(0.12)))
        assert len(fast) == len(slow) > 1
        for one, other in zip(fast, slow, strict=True):
            assert np.array_equal(one.data, other.data)
            assert one.coords == other.coords
            assert dict(one.attrs) == dict(other.attrs)

    def test_a_residual_keeps_a_trim_on_the_patch_path(self, tmp_path):
        """A selection left to the patch re-trims what a window already read."""
        spool = self._spool(tmp_path)
        coord = spool.chunk(time=None)[0].get_coord("time")
        selected = spool.select(time=(coord.values[3], coord.values[-4]))
        chunked = selected.chunk(time=to_timedelta64(0.12))
        assert list(chunked)
        resolver = chunked._catalog.resolver
        assert resolver.parent_residuals
        rows = resolver.member_rows
        trimmed = rows[rows["_modified"]].to_dict("records")
        assert trimmed
        assert not any(resolver._can_load_member_from_index(x) for x in trimmed)

    def test_a_member_in_another_unit_keeps_the_patch_path(self, tmp_path, calls):
        """A trim in the plan's unit is not a window on the file's grid."""
        first = dc.get_example_patch().set_units(distance="m")
        coord = first.get_coord("distance")
        span = float(coord.max() - coord.min() + coord.step)
        second = first.update_coords(distance=(coord.data + span) / 0.3048)
        second = second.set_units(distance="ft")
        for num, patch in enumerate((first, second)):
            patch.io.write(tmp_path / f"u{num}.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        kwargs = {"distance": 200, "keep_partial": True, "conflict": "keep_first"}
        out = list(spool.chunk(**kwargs))
        assert calls["patch"] > 0, "a feet member is not windowed in metres"
        assert sum(x.shape[x.get_axis("distance")] for x in out) == 2 * len(coord)

    def test_an_uneven_source_keeps_the_patch_path(self, tmp_path, calls):
        """A source the index states no step for has no grid to window."""
        start = np.datetime64("2020-01-01")
        values = np.concatenate(
            [start + self.step * np.arange(4), start + self.step * np.arange(10, 14)]
        )
        distance = dc.core.get_coord(start=0.0, step=1.0, shape=(3,))
        gappy = dc.Patch(
            data=np.random.default_rng(0).random((8, 3)).astype("float32"),
            coords={"time": dc.core.get_coord(values=values), "distance": distance},
            dims=("time", "distance"),
        )
        gappy.io.write(tmp_path / "g.h5", "dasdae")
        even = dc.core.get_coord(
            start=start + self.step * 14, step=self.step, shape=(8,)
        )
        dc.Patch(
            data=np.random.default_rng(1).random((8, 3)).astype("float32"),
            coords={"time": even, "distance": distance},
            dims=("time", "distance"),
        ).io.write(tmp_path / "e.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        with suppress_warnings(UserWarning):
            out = spool.chunk(time=None, tolerance=np.inf)
        assert list(out)
        assert calls["patch"] > 0

    def test_a_trimmed_member_states_its_window_id(self, tmp_path):
        """A trim carries the id of the window, not of the whole array."""
        spool = self._spool(tmp_path)
        chunked = spool.chunk(time=to_timedelta64(0.12))
        ids = [patch.attrs.data_id for patch in chunked]
        assert len(set(ids)) == len(ids)

    def test_a_grid_which_does_not_restate_its_envelope_is_refused(self):
        """A trimmed row's grid counts the source's samples, not its own."""
        source = dc.core.get_coord(
            start=np.datetime64("2020-01-01T00:00:00.000000000"),
            step=(1, 3),
            shape=(9,),
        )
        grid = (source.step_numerator, source.step_denominator, 0, len(source))
        row = {
            "time_min": source.min(),
            "time_max": source.max(),
            "time_step": source.step,
            "_time_coord_dtype": "datetime64",
            "_time_grid": grid,
        }
        assert assembly_module.coord_from_row(row, "time") == source
        # the window the plan trimmed to, beside the source's own grid
        trim = source.select((source.values[2], source.values[5]))[0]
        row = {**row, "time_min": trim.min(), "time_max": trim.max()}
        rebuilt = assembly_module.coord_from_row(row, "time")
        assert len(rebuilt) == len(trim), "the envelope, not the source's length"

    @pytest.fixture(scope="class")
    def grids(self, tmp_path_factory):
        """A time-chunked and a distance-chunked spool of adjacent files."""
        out = {}
        for name, kind in (("time", "datetime"), ("distance", "number")):
            directory = tmp_path_factory.mktemp(f"trim_{name}")
            for num in range(5):
                rng = np.random.default_rng(num)
                if kind == "datetime":
                    plan = dc.core.get_coord(
                        start=np.datetime64("2020-01-01") + self.step * 7 * num,
                        step=self.step,
                        shape=(7,),
                    )
                    other = dc.core.get_coord(start=0.0, step=1.0, shape=(4,))
                else:
                    plan = dc.core.get_coord(start=14.0 * num, step=2.0, shape=(7,))
                    other = dc.core.get_coord(
                        start=np.datetime64("2020-01-01"), step=self.step, shape=(4,)
                    )
                coords = {name: plan, "other": other}
                dims = (name, "other")
                patch = dc.Patch(
                    data=(rng.normal(size=(7, 4)) * 100).astype("int16"),
                    coords=coords,
                    dims=dims,
                )
                patch.io.write(directory / f"m{num}.h5", "dasdae")
            out[name] = dc.spool(directory).update()
        return out

    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_every_chunking_matches_the_patch_path(self, grids, dim, monkeypatch):
        """Random chunk lengths, overlaps and edges read the same samples.

        The recipe places a window of each source; the patch path loads
        each source and selects it. Whatever the boundaries do -- land on
        a sample, half a sample past one, on a file edge, or outside the
        spool -- the two must agree in every part of the patch.
        """
        spool = grids[dim]
        step = 2.0 if dim == "distance" else float(to_int(self.step))
        span = 7 * step
        rng = random.Random(20260921)
        lengths = [
            span * 0.5,
            span,
            span * 2 + step / 2,
            span - step / 2,
            *[rng.uniform(step * 2, span * 3) for _ in range(4)],
        ]
        routes = 0
        for index, length in enumerate(lengths):
            for keep_partial in (True, False):
                for overlap in (0.0, (step * 2, length / 4)[index % 2]):
                    kwargs = {dim: length, "keep_partial": keep_partial}
                    if overlap:
                        kwargs["overlap"] = overlap
                    if dim == "time":
                        kwargs[dim] = to_timedelta64(length / 1e9)
                        if overlap:
                            kwargs["overlap"] = to_timedelta64(overlap / 1e9)
                    fast = list(spool.chunk(**kwargs))
                    with monkeypatch.context() as context:
                        _force_patch_path(context)
                        slow = list(spool.chunk(**kwargs))
                    assert len(fast) == len(slow), kwargs
                    routes += sum(x.shape[0] > 7 for x in fast)
                    for one, other in zip(fast, slow, strict=True):
                        assert one.dims == other.dims, kwargs
                        assert one.data.dtype == other.data.dtype, kwargs
                        assert np.array_equal(one.data, other.data), kwargs
                        assert one.coords == other.coords, kwargs
                        assert dict(one.attrs) == dict(other.attrs), kwargs
                        for cname, coord in one.coords.coord_map.items():
                            mate = other.coords.coord_map[cname]
                            assert coord.dtype == mate.dtype, (kwargs, cname)
                            assert coord.units == mate.units, (kwargs, cname)
                            assert coord.step == mate.step, (kwargs, cname)
        assert routes, "some outputs merged more than one file"

    @pytest.fixture
    def trimmed_row(self, tmp_path):
        """One member row a plan trimmed, and the assembler which reads it."""
        spool = self._spool(tmp_path)
        chunked = spool.chunk(time=to_timedelta64(0.12))
        resolver = chunked._catalog.resolver
        rows = resolver.member_rows
        trimmed = rows[rows["_modified"]].iloc[0].to_dict()
        return resolver._assembler(), trimmed

    def test_a_row_missing_one_source_bound_is_refused(self, trimmed_row):
        """All three of the source's low, high and step, or none of them."""
        assembler, row = trimmed_row
        assert assembler._trim_window(row, "time", None) is not None
        for end in assembly_module.SOURCE_RANGE_ENDS:
            column = assembly_module.source_range_column("time", end)
            assert assembler._trim_window({**row, column: None}, "time", None) is None

    def test_a_converted_envelope_is_refused(self, trimmed_row):
        """A trim in the plan's unit is not a window on the file's grid."""
        assembler, row = trimmed_row
        other = {**row, "_time_units_source": "ms"}
        assert assembler._trim_window(other, "time", None) is None

    def test_a_window_naming_no_sample_is_refused(self, trimmed_row):
        """A trim outside its source names nothing to read."""
        assembler, row = trimmed_row
        past = row["_time_src_high"] + row["_time_src_step"] * 100
        row = {**row, "time_min": past, "time_max": past + row["_time_src_step"]}
        assert assembler._trim_window(row, "time", None) is None

    def test_a_trim_which_does_not_name_its_array_is_refused(self, trimmed_row):
        """A window's id builds on the whole array's, which the row states."""
        assembler, row = trimmed_row
        assert assembler._meta_from_index(row) is not None
        assert assembler._meta_from_index({**row, "data_id": None}) is None

    def test_a_row_without_its_source_range_falls_back(self, tmp_path, calls):
        """Without the source's own range a trim cannot be placed."""
        spool = self._spool(tmp_path)
        chunked = spool.chunk(time=to_timedelta64(0.12))
        resolver = chunked._catalog.resolver
        columns = [
            assembly_module.source_range_column("time", end)
            for end in assembly_module.SOURCE_RANGE_ENDS
        ]
        resolver.member_rows = resolver.member_rows.drop(columns=columns)
        assert list(chunked)
        assert calls["patch"] > 0

    def test_a_resource_of_another_rank_abandons_the_recipe(
        self, tmp_path, monkeypatch
    ):
        """Windows name one range per axis, so fewer axes leave the route."""
        spool = self._spool(tmp_path, count=2)
        path = tmp_path / "m0.h5"
        time = dc.core.get_coord(
            start=np.datetime64("2020-01-01"), step=self.step, shape=(8,)
        )
        flat = dc.Patch(
            data=np.arange(8, dtype="float32"), coords={"time": time}, dims=("time",)
        )
        _rewrite_dasdae(path, flat)
        # the stat check would catch this first; the fallback is what is tested
        monkeypatch.setattr(
            planned.PlanResolver, "_sources_unchanged", lambda self, rows: True
        )
        with pytest.raises(ValueError, match="axes"):
            spool.chunk(time=None)[0]
