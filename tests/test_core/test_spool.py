"""
Test for stream functions.
"""
import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.clients.filespool import FileSpool
from dascore.core.spool import BaseSpool, MemorySpool
from dascore.utils.time import to_datetime64, to_timedelta64


class TestSpoolBasics:
    """Tests for the basics of the spool."""

    def test_not_default_str(self, random_spool):
        """Ensure the default str is not used on the spool."""
        out = str(random_spool)
        assert "object at" not in out

    def test_spool_from_emtpy_sequence(self):
        """Ensure a spool can be created from empty list."""
        out = dc.spool([])
        assert isinstance(out, BaseSpool)
        assert len(out) == 0


class TestSlicing:
    """Tests for slicing spools to get sub-spools."""

    def test_simple_slice(self, random_spool):
        """Ensure a slice works with get_item, should return spool."""
        new_spool = random_spool[1:]
        assert isinstance(new_spool, type(random_spool))
        assert len(new_spool) == (len(random_spool) - 1)

    def test_skip_slice(self, random_spool):
        """Skipping values should also work."""
        new_spool = random_spool[::2]
        assert new_spool[0].equals(random_spool[0])
        assert new_spool[1].equals(random_spool[2])


class TestSpoolIterablity:
    """Tests for indexing/iterating Spools"""

    def test_len(self, random_spool):
        """Ensure the stream has a length"""
        assert len(random_spool) == len(list(random_spool))

    def test_index(self, random_spool):
        """Ensure the stream can be indexed."""
        assert isinstance(random_spool[0], dc.Patch)

    def test_list_o_patches(self, random_spool):
        """Ensure random_string can be iterated"""
        for pa in random_spool:
            assert isinstance(pa, dc.Patch)
        patch_list = list(random_spool)
        for pa in patch_list:
            assert isinstance(pa, dc.Patch)

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

    def test_multiple_selects(self, diverse_spool):
        """Ensure selects can be stacked."""
        contents = diverse_spool.get_contents()
        duration = contents["time_max"] - contents["time_min"]
        new_max = (contents["time_min"] + duration / 2).max()
        out = (
            diverse_spool.select(network="das2")
            .select(tag="ran*")
            .select(time=(None, new_max))
        )
        assert len(out)
        for patch in out:
            attrs = patch.attrs
            assert attrs["network"] == "das2"
            assert attrs["tag"].startswith("ran")
            assert attrs["time_max"] <= new_max

    def test_multiple_range_selects(self, adjacent_spool_no_overlap):
        """
        Ensure multiple range slects can be used in one call (eg time and distance).
        """
        spool = adjacent_spool_no_overlap
        contents = spool.get_contents()
        # get new time/distance ranges and select them
        time_min = to_datetime64(contents["time_min"].min() + to_timedelta64(4))
        time_max = to_datetime64(contents["time_max"].max() - to_timedelta64(4))
        distance_min = contents["distance_min"].min() + 50
        distance_max = contents["distance_min"].max() - 50
        new_spool = spool.select(
            time=(time_min, time_max), distance=(distance_min, distance_max)
        )
        # First check content df honors new ranges
        new_contents = new_spool.get_contents()
        assert (new_contents["time_min"] >= time_min).all()
        assert (new_contents["time_max"] <= time_max).all()
        assert (new_contents["distance_min"] >= distance_min).all()
        assert (new_contents["distance_max"] <= distance_max).all()
        # then check patches
        for patch in new_spool:
            assert patch.attrs["time_min"] >= time_min
            assert patch.attrs["time_max"] <= time_max
            assert patch.attrs["distance_min"] >= distance_min
            assert patch.attrs["distance_max"] <= distance_max


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
        new_patch = dc.MemorySpool(new_patches)
        # get content of spool created from patches in chunked spool.
        new_content = new_patch.get_contents()
        # these should be (nearly) identical.
        common = set(chunk_df.columns) & set(new_content.columns)
        cols = sorted(common - {"history"})  # no need to compare history
        assert chunk_df[cols].equals(new_content[cols])

    def test_merge_empty_spool(self, tmp_path_factory):
        """Ensure merge doesn't raise on empty spools"""
        spool = dc.spool([])
        merged = spool.chunk(time=None)
        assert len(merged) == 0


class TestMergePatchesWithChunk:
    """Tests for merging patches together using chunk method."""

    @pytest.fixture()
    def desperate_spool_no_overlap(self, random_patch) -> dc.MemorySpool:
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
        return dc.MemorySpool([pa2, pa1, pa3])

    @pytest.fixture()
    def spool_complete_overlap(self, random_patch) -> dc.MemorySpool:
        """
        Create a stream which overlaps each other completely.
        """
        return dc.MemorySpool([random_patch, random_patch])

    @pytest.fixture()
    def spool_slight_gap(self, random_patch) -> dc.MemorySpool:
        """
        Create a stream which has a 1.1 * dt gap.
        """
        pa1 = random_patch
        t2 = random_patch.attrs["time_max"]
        dt = random_patch.attrs["d_time"]
        pa2 = random_patch.update_attrs(time_min=t2 + dt * 1.1)
        t3 = pa2.attrs["time_max"]
        pa3 = pa2.update_attrs(time_min=t3 + dt * 1.1)
        return dc.MemorySpool([pa2, pa1, pa3])

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

    def test_merge_transposed_patches(self, spool_complete_overlap):
        """Ensure if one of the patches is transposed merge still works"""
        spoo = spool_complete_overlap
        # transpose patch and remove transpose from history so patch will
        # merge with other patch
        new_patch = spoo[0].transpose().update_attrs(history=spoo[1].attrs["history"])
        new = dc.spool(
            [new_patch, spoo[1]],
        )
        new_merged = new.chunk(time=None)
        old_merged = spool_complete_overlap.chunk(time=None)
        assert len(new_merged) == len(old_merged) == 1
        assert new_merged[0].equals(old_merged[0])


class TestGetSpool:
    """Test getting spool from various sources."""

    def test_spool_from_spool(self, random_spool):
        """Ensure a spool is valid input to get spool."""
        out = dc.spool(random_spool)
        for p1, p2 in zip(out, random_spool):
            assert p1.equals(p2)

    def test_spool_from_patch_sequence(self, random_spool):
        """Ensure a list of patches returns a spool"""
        spool_list = dc.spool(list(random_spool))
        spool_tuple = dc.spool(tuple(random_spool))
        for p1, p2, p3 in zip(spool_tuple, spool_list, random_spool):
            assert p1.equals(p2)
            assert p2.equals(p3)

    def test_spool_from_single_file(self, terra15_das_example_path):
        """Ensure a single file path returns a spool."""
        out1 = dc.spool(terra15_das_example_path)
        assert isinstance(out1, BaseSpool)
        # ensure format works.
        out2 = dc.spool(terra15_das_example_path, file_format="terra15")
        assert isinstance(out2, BaseSpool)
        assert len(out1) == len(out2)

    def test_non_existent_file_raises(self):
        """A path that doesn't exist should raise."""
        with pytest.raises(Exception, match="get spool from"):
            dc.spool("here_or_there?")

    def test_non_supported_type_raises(self):
        """A type that can't contain patches should raise."""
        with pytest.raises(Exception, match="not get spool from"):
            dc.spool(1.2)

    def test_file_spool(self, random_spool, tmp_path_factory):
        """
        Tests for getting a file spool vs in-memory spool. Basically,
        if a format supports scanning a FileSpool is returned. If it doesn't,
        all the file contents have to be loaded into memory to scan so a
        MemorySpool is just returned.
        """
        path = tmp_path_factory.mktemp("file_spoolin")
        dasdae_path = path / "patch.h5"
        pickle_path = path / "patch.pkl"
        dc.write(random_spool, dasdae_path, "dasdae")
        dc.write(random_spool, pickle_path, "pickle")

        dasdae_spool = dc.spool(dasdae_path)
        assert isinstance(dasdae_spool, FileSpool)

        pickle_spool = dc.spool(pickle_path)
        assert isinstance(pickle_spool, MemorySpool)
