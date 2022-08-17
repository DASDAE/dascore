"""
Test for stream functions.
"""
import numpy as np
import pandas as pd
import pytest

import dascore
from dascore.utils.time import to_timedelta64


class TestSpoolBasics:
    """Tests for the basics of the spool."""

    def test_not_default_str(self, random_spool):
        """Ensure the default str is not used on the spool."""
        out = str(random_spool)
        assert "object at" not in out


class TestSpoolIterablity:
    """Tests for indexing/iterating Spools"""

    def test_len(self, random_spool):
        """Ensure the stream has a length"""
        assert len(random_spool) == len(list(random_spool))

    def test_index(self, random_spool):
        """Ensure the stream can be indexed."""
        assert isinstance(random_spool[0], dascore.Patch)

    def test_list_o_patches(self, random_spool):
        """Ensure random_string can be iterated"""
        for pa in random_spool:
            assert isinstance(pa, dascore.Patch)
        patch_list = list(random_spool)
        for pa in patch_list:
            assert isinstance(pa, dascore.Patch)

    def test_index_error(self, random_spool):
        """Ensure an IndexError is raised when indexing beyond spool."""
        spool_len = len(random_spool)
        with pytest.raises(IndexError, match="out of bounds"):
            _ = random_spool[spool_len]

    def test_index_returns_corresponding_patch(self, random_spool):
        """Ensure the index returns the correct patch"""
        spool_list = list(random_spool)
        for num, (patch1, patch2) in enumerate(zip(spool_list, random_spool)):
            patch3 = random_spool[num]
            assert patch1 == patch2 == patch3


class TestGetContents:
    """Ensure the contents of the spool can be returned via dataframe."""

    def test_no_filter(self, random_spool):
        """Ensure the entirety of the contents are returned."""
        df = random_spool.get_contents()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(random_spool)

    def test_filter(self, random_spool):
        """Ensure the dataframe can be filtered."""
        full_df = random_spool.get_contents()
        new_max = full_df["time_min"].max() - to_timedelta64(1)
        sub = random_spool.select(time=(None, new_max)).get_contents()
        assert len(sub) == (len(full_df) - 1)
        assert (sub["time_min"] < new_max).all()


class TestSelect:
    """Tests for selecting/trimming spools."""

    def test_select_network(self, diverse_spool):
        """Ensure a tuple can be used to select spools within network."""
        network_set = {"das2", "das3"}
        out = diverse_spool.select(network=network_set)
        for patch in out:
            assert patch.attrs["network"] in network_set

    def test_select_tag_wildcard(self, diverse_spool):
        """Ensure wildcards can be used on str columns."""
        out = diverse_spool.select(tag="some*")
        for patch in out:
            assert patch.attrs["tag"].startswith("some")


class TestChunk:
    """
    Tests for merging/chunking patches.
    """

    def test_merge_chunk_adjacent_no_overlap(self, adjacent_spool_no_overlap):
        """
        Ensure chunking works on simple case of contiguous data w/ no overlap.
        """
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
        """Chunking shouldn't modify original spool"""
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
        new_patch = dascore.MemorySpool(new_patches)
        # get content of spool created from patches in chunked spool.
        new_content = new_patch.get_contents()
        # these should be (nearly) identical.
        common = set(chunk_df.columns) & set(new_content.columns)
        cols = sorted(common - {"history"})  # no need to compare history
        assert chunk_df[cols].equals(new_content[cols])


class TestMergePatches:
    """Tests for merging patches together."""

    @pytest.fixture()
    def desperate_spool_no_overlap(self, random_patch) -> dascore.MemorySpool:
        """
        Create streams that do not overlap at all.
        Ensure the patches are not sorted in temporal order.
        """
        pa1 = random_patch
        t2 = random_patch.attrs["time_max"]
        d_time = random_patch.attrs["d_time"] * 1_000
        pa2 = random_patch.update_attrs(time_min=t2 + d_time)
        t3 = pa2.attrs["time_max"]
        pa3 = pa2.update_attrs(time_min=t3 + d_time)
        return dascore.MemorySpool([pa2, pa1, pa3])

    @pytest.fixture()
    def spool_complete_overlap(self, random_patch) -> dascore.MemorySpool:
        """
        Create a stream which overlaps each other completely.
        """
        return dascore.MemorySpool([random_patch, random_patch])

    @pytest.fixture()
    def spool_slight_gap(self, random_patch) -> dascore.MemorySpool:
        """
        Create a stream which has a 1.1 * dt gap.
        """
        pa1 = random_patch
        t2 = random_patch.attrs["time_max"]
        dt = random_patch.attrs["d_time"]
        pa2 = random_patch.update_attrs(time_min=t2 + dt * 1.1)
        t3 = pa2.attrs["time_max"]
        pa3 = pa2.update_attrs(time_min=t3 + dt * 1.1)
        return dascore.MemorySpool([pa2, pa1, pa3])

    def test_merge_adjacent(self, adjacent_spool_no_overlap):
        """Test simple merge of patches."""
        len_1 = len(adjacent_spool_no_overlap)
        out_stream = adjacent_spool_no_overlap.chunk(time=None)
        assert len(out_stream) < len_1
        assert len(out_stream) == 1
        out_patch = out_stream[0]
        # make sure coords are consistent with attrs
        assert out_patch.attrs["time_max"] == out_patch.coords["time"].max()
        assert out_patch.attrs["time_min"] == out_patch.coords["time"].min()
        # ensure the spacing is still uniform
        time = out_patch.coords["time"]
        spacing = time[1:] - time[:-1]
        unique_spacing = np.unique(spacing)
        assert len(unique_spacing) == 1
        assert unique_spacing[0] == out_patch.attrs["d_time"]

    def test_no_overlap(self, desperate_spool_no_overlap):
        """streams with no overlap should not be merged."""
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
