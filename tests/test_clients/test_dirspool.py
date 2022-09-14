"""
Tests for FileSpool.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.constants import ONE_SECOND
from dascore.core.schema import PatchFileSummary
from dascore.utils.hdf5 import HDFPatchIndexManager
from dascore.utils.misc import register_func

FILE_SPOOLS = []


@pytest.fixture(scope="class")
@register_func(FILE_SPOOLS)
def one_file_spool(one_file_dir):
    """Create a directory with a single DAS file."""
    spool = dc.DirectorySpool(one_file_dir)
    return spool.update()


@pytest.fixture(scope="class", params=FILE_SPOOLS)
def file_spool(request):
    """Meta fixture for getting all file spools."""
    return request.getfixturevalue(request.param)


class TestFileSpool:
    """Test that the file spool works."""

    def test_isinstance(self, file_spool):
        """Simply ensure expected type was returned."""
        assert isinstance(file_spool, dc.DirectorySpool)


class TestFileIndex:
    """Tests for returning summaries of all files in managed directory."""

    @pytest.fixture(scope="class")
    def basic_index_df(self, basic_file_spool):
        """Return the index file of the basic bank."""
        spool = basic_file_spool
        return spool.get_contents()

    def test_index_exists(self, basic_file_spool):
        """An index should be returned."""
        assert basic_file_spool.indexer.index_path.exists()

    def test_index_len(self, basic_index_df, two_patch_directory):
        """An index should be returned."""
        bank_paths = list(Path(two_patch_directory).rglob("*hdf5"))
        assert isinstance(basic_index_df, pd.DataFrame)
        assert len(bank_paths) == len(basic_index_df)

    def test_index_columns(self, basic_index_df):
        """Ensure expected columns show up in the index."""
        schema_fields = PatchFileSummary.__fields__
        assert set(basic_index_df).issuperset(schema_fields)

    def test_patches_extracted(self, basic_file_spool):
        """Ensure the patches can be extracted."""
        index = basic_file_spool.get_contents()
        patches = [x for x in basic_file_spool]
        assert len(index) == len(patches)
        for patch in patches:
            assert isinstance(patch, dc.Patch)

    def test_str_(self, basic_file_spool):
        """Ensure the filespool has a useful str/repr."""
        out = str(basic_file_spool)
        # ensure the default str is not used.
        assert "object at" not in out


class TestSelect:
    """tests for subselecting data."""

    @pytest.fixture(scope="class")
    def spool_tag(self, basic_file_spool):
        """Return a string of a tag in the basic_file_spool."""
        contents = basic_file_spool.get_contents()
        tag = contents.loc[contents["tag"].astype(bool), "tag"].iloc[0]
        return tag

    def test_subselect_trims_patches(self, basic_file_spool):
        """Ensure sub-selecting trims start/end times on df and output patches."""
        current = basic_file_spool.get_contents()
        new_min = current["time_min"].min() + np.timedelta64(2, "s")
        new_max = current["time_max"].max() - np.timedelta64(2, "s")
        spool = basic_file_spool.select(time=(new_min, new_max))
        # the limits of rows which were intersected should have been trimmed.
        df = spool.get_contents()
        assert (df["time_min"] >= new_min).all()
        assert (df["time_max"] <= new_max).all()
        # as well as the patches produced
        for patch in spool:
            assert patch.attrs["time_min"] >= new_min
            assert patch.attrs["time_max"] <= new_max

    def test_sub_select_tag_equals(self, basic_file_spool, spool_tag):
        """Ensure selecting stations works."""
        new = basic_file_spool.select(tag=spool_tag)
        new_contents = new.get_contents()
        assert (new_contents["tag"] == spool_tag).all()

    def test_is_in_tag(self, basic_file_spool, spool_tag):
        """Ensure tags can also be selected from a collection."""
        tag_collection = {spool_tag, "bob", "bill"}
        out = basic_file_spool.select(tag=tag_collection).get_contents()
        assert out["tag"].isin(tag_collection).all()

    def test_multiple_selects(self, diverse_directory_spool):
        """Ensure selects can be stacked."""
        spool = diverse_directory_spool
        contents = spool.get_contents()
        duration = contents["time_max"] - contents["time_min"]
        new_max = (contents["time_min"] + duration.mean() / 2).median()
        out = (
            spool.select(network="das2").select(tag="ran*").select(time=(None, new_max))
        )
        assert len(out) > 0
        # first check content dataframe
        new_content = out.get_contents()
        assert len(new_content) == len(out)
        assert (new_content["network"] == "das2").all()
        assert (new_content["tag"].str.startswith("ran")).all()
        assert (new_content["time_max"] <= new_max).all()
        # then check patches
        for patch in out:
            attrs = patch.attrs
            assert attrs["network"] == "das2"
            assert attrs["tag"].startswith("ran")
            assert attrs["time_max"] <= new_max

    def test_select_time_tuple_with_string(self, basic_file_spool):
        """Ensure time tuples with strings still work."""
        time_str = "2017-09-18T00:00:04"
        dt = np.datetime64(time_str)
        spool1 = basic_file_spool.select(time=(None, dt))
        spool2 = basic_file_spool.select(time=(None, time_str))
        for pa1, pa2 in zip(spool1, spool2):
            assert pa1.attrs["time_max"] == pa2.attrs["time_max"]

    def test_select_non_zero_index(self, diverse_directory_spool):
        """
        A Bug caused the contents of the source dataframe to have
        non-zero based indices, thus spools didnt work. This fixes
        the issue.
        """
        contents = diverse_directory_spool.get_contents()
        end_time = contents["time_max"].min()
        sub = diverse_directory_spool.select(
            time=(None, end_time),
            distance=(100, 200),
        )
        assert len(sub) == 1
        patch = sub[0]
        assert isinstance(patch, dc.Patch)


class TestBasicChunk:
    """Tests for chunking filespool."""

    def test_directoy_path_doesnt_change(self, one_file_file_spool):
        """Chunking shouldn't change the path to the managed directory."""
        out = one_file_file_spool.chunk(time=1)
        assert out.spool_path == one_file_file_spool.spool_path

    def test_chunk_doesnt_modify_original(self, one_file_file_spool):
        """Chunking shouldn't modify original spool or its dfs."""
        spool = one_file_file_spool
        contents_before_chunk = spool.get_contents()
        _ = spool.chunk(time=2)
        contents_after_chunk = spool.get_contents()
        assert contents_before_chunk.equals(contents_after_chunk)

    def test_sub_chunk(self, one_file_file_spool):
        """Ensure the patches can be subdivided."""
        spool = one_file_file_spool
        contents = spool.get_contents()
        durations = contents["time_max"] - contents["time_min"]
        new_t_delta = (durations / 4).max()
        new_spool = spool.chunk(time=new_t_delta, keep_partial=True)
        # Ensure there are exactly 4x as many patches in spool after chunk
        new_contents = new_spool.get_contents()
        assert len(new_contents) == 4 * len(spool)
        # Ensure each spool can be iterated
        patch_list = list(new_spool)
        for patch in patch_list:
            assert isinstance(patch, dc.Patch)


class TestGetContents:
    """Tests for getting the contents of the spool."""

    def test_str_columns_in_dataframe(self, diverse_directory_spool):
        """Ensure all the string columns are in index."""
        df = diverse_directory_spool.get_contents()
        expected = HDFPatchIndexManager._min_itemsize
        assert set(df.columns).issuperset(set(expected))


class TestFileSpoolIntegrations:
    """Small integration tests for the file spool."""

    def test_one(self, diverse_spool_directory):
        """Small integration test with diverse spool."""
        network = "das2"
        endtime = np.datetime64("2022-01-01")
        duration = 3
        spool = (
            dc.spool(diverse_spool_directory)
            .select(network=network)  # sub-select das2 network
            .select(time=(None, endtime))  # unselect anything after 2022
            .chunk(time=duration, overlap=0.5)  # change the chunking of the patches
        )
        for patch in spool:
            assert isinstance(patch, dc.Patch)
            attrs = patch.attrs
            assert attrs["network"] == network
            assert attrs["time_max"] <= endtime
            patch_duration = (attrs["time_max"] - attrs["time_min"]) / ONE_SECOND
            diff = patch_duration - duration
            assert abs(diff) <= 1.5 * attrs["d_time"] / ONE_SECOND
