"""Tests for FileSpool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.examples
from dascore.clients.dirspool import DirectorySpool
from dascore.constants import ONE_SECOND
from dascore.exceptions import ParameterError
from dascore.io.core import PatchFileSummary
from dascore.utils.hdf5 import HDFPatchIndexManager
from dascore.utils.misc import register_func

DIRECTORY_SPOOLS = []


@pytest.fixture(scope="class")
@register_func(DIRECTORY_SPOOLS)
def dir_spool_index_out_of_order(random_spool, tmp_path_factory):
    """Create an index that isn't order chronologically."""
    path = tmp_path_factory.mktemp("out_of_order_index")
    spool = dc.spool(path)
    # sort patches by starttime
    patch_list = sorted(random_spool, key=lambda x: x.attrs.time_min)
    # write patches to disk out of order.
    patch_list[-1].io.write(path / "patch_3.h5", "dasdae")
    spool.update()
    patch_list[0].io.write(path / "patch_1.h5", "dasdae")
    spool.update()
    patch_list[1].io.write(path / "patch_2.h5", "dasdae")
    spool.update()
    return spool


@pytest.fixture(scope="class")
@register_func(DIRECTORY_SPOOLS)
def one_directory_spool(one_file_dir):
    """Create a directory with a single DAS file."""
    spool = DirectorySpool(one_file_dir)
    return spool.update()


@pytest.fixture(scope="class")
@register_func(DIRECTORY_SPOOLS)
def non_distance_dir_spool(tmp_path_factory):
    """Create a directory with a single DAS file."""
    # patch one simulates a patch that has time but no distance
    pa1 = dascore.examples.get_example_patch("random_das").rename_coords(
        distance="depth"
    )
    # patch2 has neither time nor distance but time in attrs.
    pa2 = (
        dascore.examples.get_example_patch("random_das")
        .update_attrs(time_min=pa1.attrs.time_max)
        .rename_coords(time="timey", distance="depth")
    )
    coord = pa2.get_coord("timey")
    pa2 = pa2.update_attrs(time_min=coord.min(), time_max=coord.max())
    spool = dc.spool([pa1, pa2])
    path = tmp_path_factory.mktemp("no_distance_spool")
    dascore.examples.spool_to_directory(spool, path)
    return dc.spool(path).update()


@pytest.fixture
def directory_spool_redundant_index(random_spool, tmp_path_factory):
    """Force a spool to be indexed many times with same files."""
    path = Path(tmp_path_factory.mktemp("redundant_index_spool"))
    dascore.examples.spool_to_directory(random_spool, path, "dasdae")
    spool = dc.spool(path).update()

    # Touch each file, re-index to saturate index with duplicates.
    for _ in range(12):
        for file_path in path.glob("*"):
            file_path.touch()
        spool = spool.update()
    return spool


@pytest.fixture(scope="class", params=DIRECTORY_SPOOLS)
def directory_spool(request):
    """Meta fixture for getting all file spools."""
    return request.getfixturevalue(request.param)


class TestDirectorySpoolBasics:
    """Basic tests for the directory spool."""

    def test_isinstance(self, directory_spool):
        """Simply ensure expected type was returned."""
        assert isinstance(directory_spool, DirectorySpool)

    def test_selected_str(self, diverse_directory_spool):
        """Ensure select kwargs show up in str."""
        new = diverse_directory_spool.select(station="big_gaps")
        contents = new.get_contents()
        assert (contents["station"] == "big_gaps").all()


class TestDirectoryIndex:
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
        spool = dc.spool(two_patch_directory)
        spool.indexer.index_path.unlink()
        df = spool.update().get_contents()
        bank_paths = list(Path(two_patch_directory).rglob("*hdf5"))
        assert isinstance(df, pd.DataFrame)
        assert len(bank_paths) == len(df)

    def test_index_columns(self, basic_index_df):
        """Ensure expected columns show up in the index."""
        schema_fields = list(PatchFileSummary.model_fields)
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

    def test_long_tags(self, random_patch, tmp_path):
        """Ensure a long tag still works."""
        new = random_patch.update_attrs(tag="hey" * 15)
        path = tmp_path / "test.h5"
        new.io.write(path, "dasdae")
        spool = dc.spool(path).update()
        isinstance(spool, dc.BaseSpool)

    def test_specify_index_path(self, random_patch, tmp_path_factory):
        """Ensure an external path can be specified for the index. See #129."""
        bank_path = tmp_path_factory.mktemp("bank")
        index_path = tmp_path_factory.mktemp("index") / "index.h5"
        random_patch.io.write(bank_path / "contents.h5", "dasdae")
        spool1 = dc.spool(bank_path, index_path=index_path)
        spool1.update()
        # ensure the index was created in the expected place
        assert spool1.indexer.index_path == index_path
        # ensure the default index file was not written
        default_index_path = bank_path / spool1.indexer._index_name
        assert not default_index_path.exists()
        # future banks should remember this path.
        spool2 = dc.spool(bank_path)
        assert spool2.indexer.index_path == spool1.indexer.index_path
        # next ensure the index path is used
        spool3 = dc.spool(bank_path, index_path=index_path)
        df = spool3.get_contents()
        assert len(df) == 1
        patch = spool3[0]
        assert isinstance(patch, dc.Patch)
        assert not default_index_path.exists()

    def test_nested_directories(self, diverse_spool, tmp_path_factory):
        """Ensure files in nested directories work up to 3 levels."""
        # split the spool into 3
        sp_len = len(diverse_spool)
        num = 3
        spools = [
            diverse_spool[int((x / num) * sp_len) : int(((x + 1) / num) * sp_len)]
            for x in range(num)
        ]
        # write each group to a different sub path
        base_path = tmp_path_factory.mktemp("nested_dir")
        path = base_path
        for num, spool in enumerate(spools):
            path = path / f"sub_{num}"
            path.mkdir(exist_ok=True, parents=True)
            dascore.examples.spool_to_directory(spool, path)
        df = dc.spool(base_path).update().get_contents()
        # ensure each sub-directory is represented
        paths = df["path"]
        assert any(paths.str.startswith("sub_0"))
        assert any(paths.str.startswith("sub_0/sub_1"))
        assert any(paths.str.startswith("sub_0/sub_1/sub_2"))


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
        # ensure raises when selecting off the end of the spool
        with pytest.raises(IndexError):
            out[len(new_content)]

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
        non-zero based indices, thus spools didn't work.
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

    def test_nice_error_message_bad_select(self, diverse_directory_spool):
        """Ensure a nice error message is raised for bad filter param."""
        with pytest.raises(ParameterError, match="must be a length 2 sequence"):
            _ = diverse_directory_spool.select(time=(None, None, None))[0]

    def test_select_correct_history_str(self, diverse_directory_spool):
        """Ensure no history string is added for selecting. See #142/#147."""
        spool = diverse_directory_spool
        t1 = spool[0].attrs.time_min
        dt = spool[0].attrs.time_step
        selected_spool = spool.select(time=(t1, t1 + 30 * dt))
        patch = selected_spool[0]
        history = patch.attrs.history
        assert len(history) <= 1


class TestBasicChunk:
    """Tests for chunking filespool."""

    @pytest.fixture(scope="class")
    def dir_spool_1_dim_patches(self, memory_spool_dim_1_patches, tmp_path_factory):
        """Create a directory with patches that have 1 dim in time."""
        path = tmp_path_factory.mktemp("dir_spool_1_dim_patches")
        out = dc.examples.spool_to_directory(memory_spool_dim_1_patches, path)
        return dc.spool(out).update()

    def test_directory_path_doesnt_change(self, one_file_directory_spool):
        """Chunking shouldn't change the path to the managed directory."""
        out = one_file_directory_spool.chunk(time=1)
        assert out.spool_path == one_file_directory_spool.spool_path

    def test_chunk_doesnt_modify_original(self, one_file_directory_spool):
        """Chunking shouldn't modify original spool or its dfs."""
        spool = one_file_directory_spool
        contents_before_chunk = spool.get_contents()
        _ = spool.chunk(time=2)
        contents_after_chunk = spool.get_contents()
        assert contents_before_chunk.equals(contents_after_chunk)

    def test_sub_chunk(self, one_file_directory_spool):
        """Ensure the patches can be subdivided."""
        spool = one_file_directory_spool
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

    def test_merge_1_dim_patches(self, dir_spool_1_dim_patches):
        """Ensure patches with one sample in time can be merged."""
        spool = dir_spool_1_dim_patches
        new = spool.chunk(time=None)
        assert len(new) == 1
        patch = new[0]
        content = spool.get_contents()
        assert patch.attrs.time_min == content["time_min"].min()
        assert patch.attrs.time_max == content["time_max"].max()
        assert patch.attrs.time_step == spool[0].attrs.time_step

    def test_chunk_out_of_order_index(self, dir_spool_index_out_of_order):
        """Ensure when the index isn't ordered chunk can still work."""
        spool = dir_spool_index_out_of_order
        time = 4.25
        chunk = spool.chunk(time=time)
        for patch in chunk:
            assert isinstance(patch, dc.Patch)
            dur = (patch.attrs.time_max - patch.attrs.time_min) / ONE_SECOND
            diff = np.abs(dur - time)
            # because we try to avoid overlaps, the segments can be up to 2
            # samples shorter than what was asked for. Maybe revisit this?
            assert diff <= 2 * (patch.attrs.time_step / ONE_SECOND)

    def test_chunk_redundant_index(self, directory_spool_redundant_index):
        """Ensure redundant indices are handled effectively with chunking"""
        spool = directory_spool_redundant_index.chunk(time=None)
        patch = spool[0]
        assert isinstance(patch, dc.Patch)


class TestGetContents:
    """Tests for getting the contents of the spool."""

    def test_str_columns_in_dataframe(self, diverse_directory_spool):
        """Ensure all the string columns are in index."""
        df = diverse_directory_spool.get_contents()
        expected = HDFPatchIndexManager._min_itemsize
        assert set(df.columns).issuperset(set(expected))


class TestIndexing:
    """Tests for indexing directory spool."""

    def test_slice_to_start(self, diverse_directory_spool):
        """Ensure a slice returns a subspool (shouldn't load data)."""
        out = diverse_directory_spool[0:2]
        assert isinstance(out, out.__class__)

    def test_slice_to_end(self, diverse_directory_spool):
        """Ensure a slice from the end returns a subspool."""
        out = diverse_directory_spool[-2:]
        assert isinstance(out, out.__class__)

    def test_sliced_spool_has_indexer(self, diverse_directory_spool):
        """Ensure the sliced spool still has its indexer."""
        out = diverse_directory_spool[1:3]
        assert hasattr(out, "indexer")
        assert out.indexer.path == diverse_directory_spool.indexer.path
        # ensure we can still load patches from sliced dirspool
        assert isinstance(out[0], dc.Patch)

    def test_chunked_sliced_spool_index(self, diverse_directory_spool):
        """Ensure chunked sliced spool can still be indexed and patches loaded."""
        out = diverse_directory_spool.chunk(time=4)
        middle_index = len(out) // 2
        sub = out[middle_index : middle_index + 3]
        for ind in range(len(sub)):
            patch = sub[ind]
            assert isinstance(patch, dc.Patch)


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
            assert abs(diff) <= 1.5 * attrs["time_step"] / ONE_SECOND

    def test_chunk_select(self, dir_spool_index_out_of_order):
        """Ensure chunking can be performed first, then selecting."""
        # get start/endtimes to encompass the last half of the first patch.
        # and the first half of the second patch.
        df = dir_spool_index_out_of_order.get_contents().sort_values("time_min")
        time = (df["time_max"] - df["time_min"]) / 2 + df["time_min"]
        time_tup = (time.iloc[0], time.iloc[1])
        # merge, then select, should still work.
        merged = dir_spool_index_out_of_order.chunk(time=...)
        assert len(merged) == 1
        select = merged.select(time=time_tup)
        assert len(select) == 1

    def test_doc_example(self, all_examples_spool):
        """Tests for quickstart."""
        spool = all_examples_spool.update()
        assert isinstance(spool, dc.BaseSpool)

    def test_patch_no_distance_coord(self, non_distance_dir_spool):
        """Ensure patches without distance coords still work."""
        # str should work
        assert str(non_distance_dir_spool)
        contents = non_distance_dir_spool.get_contents()
        assert len(contents) == 2
        for patch in non_distance_dir_spool:
            assert isinstance(patch, dc.Patch)

    def test_select_non_distance(self, non_distance_dir_spool):
        """We should be able to select on non-time/distance coords."""
        spool = non_distance_dir_spool
        depth_tup = (150, 250)
        selected_spool = spool.select(depth=depth_tup)
        for patch in selected_spool:
            # ensure depth has been trimmed.
            coord = patch.get_coord("depth")
            assert coord.min() >= depth_tup[0]
            assert coord.max() <= depth_tup[1]
