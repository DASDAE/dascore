"""Tests for directory-backed spools."""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.examples
from dascore.constants import ONE_SECOND
from dascore.core.spool import Spool
from dascore.exceptions import InvalidSpoolError, MissingPatchError, ParameterError
from dascore.utils.misc import suppress_warnings


@pytest.fixture(scope="module")
def dir_spool_index_out_of_order(random_spool, tmp_path_factory):
    """Create an index that isn't order chronologically."""
    path = tmp_path_factory.mktemp("out_of_order_index")
    spool = dc.spool(path)
    # sort patches by starttime
    patch_list = sorted(random_spool, key=lambda x: x.get_coord("time").min())
    # write patches to disk out of order.
    patch_list[-1].io.write(path / "patch_3.h5", "dasdae")
    spool.update()
    patch_list[0].io.write(path / "patch_1.h5", "dasdae")
    spool.update()
    patch_list[1].io.write(path / "patch_2.h5", "dasdae")
    spool.update()
    return spool


@pytest.fixture(scope="module")
def one_directory_spool(one_file_dir):
    """Create a directory with a single DAS file."""
    spool = Spool.from_directory(one_file_dir)
    return spool.update()


@pytest.fixture(scope="module")
def non_distance_dir_spool(tmp_path_factory):
    """Create a directory with a single DAS file."""
    # Simulate a patch that has time but no canonical distance coordinate.
    pa1 = dascore.examples.get_example_patch("random_das").rename_coords(
        distance="depth"
    )
    path = tmp_path_factory.mktemp("no_distance_spool")
    dc.write(pa1, path / "patch_1.h5", "dasdae")
    return dc.spool(path).update()


@pytest.fixture(scope="module")
def multi_patch_file_spool(tmp_path_factory):
    """Create a directory whose single file contains multiple patches."""
    path = tmp_path_factory.mktemp("multi_patch_file_spool")
    patch_1 = dascore.examples.get_example_patch("random_das")
    # Create a second patch contiguous with the first. Clear the history
    # so the two patches remain mergeable.
    time = patch_1.get_coord("time")
    patch_2 = patch_1.update_coords(time_min=time.max() + time.step)
    patch_1, patch_2 = (x.update_attrs(history=[]) for x in (patch_1, patch_2))
    dc.write(dc.spool([patch_1, patch_2]), path / "multi_patch.h5", "dasdae")
    return dc.spool(path).update()


@pytest.fixture
def directory_spool_redundant_index(random_spool, tmp_path_factory):
    """A spool re-indexed over files whose contents did not change."""
    path = Path(tmp_path_factory.mktemp("redundant_index_spool"))
    dascore.examples.spool_to_directory(random_spool, path, "dasdae")
    spool = dc.spool(path).update()
    # Touch, then re-index: the row count is the same after one round as
    # after twelve, so one is what this needs.
    for file_path in path.glob("*"):
        file_path.touch()
    return spool.update()


class TestDirectorySpoolBasics:
    """Basic tests for the directory spool."""

    def test_selected_str(self, diverse_directory_spool):
        """Ensure select kwargs show up in str."""
        new = diverse_directory_spool.select(tag="big_gaps")
        contents = new.get_contents()
        assert (contents["tag"] == "big_gaps").all()

    def test_sorted_multi_patch_uses_source_patch_key(self, tmp_path):
        """Sorted directory spool rows should reload the intended source patch."""
        path = tmp_path / "directory"
        path.mkdir()
        patch_2 = dc.get_example_patch()
        patch_1 = patch_2.update_coords(time=patch_2.coords.get_array("time") + 10)
        dc.write(dc.spool([patch_1, patch_2]), path / "multi_patch.h5", "dasdae")
        spool = Spool.from_directory(path).update().sort("time")
        patch = spool[0]
        assert patch.get_coord("time").min() == patch_2.get_coord("time").min()


class TestMultiPatchFile:
    """Tests for directories with files that contain multiple patches."""

    def test_iteration_returns_distinct_patches(self, multi_patch_file_spool):
        """Each row must map to its own patch, not just the file's first."""
        spool = multi_patch_file_spool
        contents = spool.get_contents()
        assert len(spool) == 2
        patches = list(spool)
        expected = set(contents["time_min"])
        returned = {x.get_coord("time").min() for x in patches}
        assert returned == expected

    def test_merge(self, multi_patch_file_spool):
        """Ensure patches from a single file can be merged."""
        spool = multi_patch_file_spool
        contents = spool.get_contents()
        merged = spool.chunk(time=None)
        assert len(merged) == 1
        patch = merged[0]
        time = patch.get_coord("time")
        assert time.min() == contents["time_min"].min()
        assert time.max() == contents["time_max"].max()


class TestLoadPatchFastPath:
    """FileResolver reads each row's patch through a single dc.read call."""

    def test_forwards_recorded_format_and_version(
        self, one_directory_spool, monkeypatch
    ):
        """The recorded format/version are forwarded so dc.read skips probing."""
        resolver = one_directory_spool._catalog.resolver
        calls = []

        def _fake_read(**kwargs):
            calls.append(kwargs)
            return object()

        monkeypatch.setattr("dascore.io.index.catalog.dc.read", _fake_read)
        row = {"source_format": "DASDAE", "source_version": "1"}
        resolver._read("path", row, {}, "")
        assert calls[-1]["file_format"] == "DASDAE"
        assert calls[-1]["file_version"] == "1"
        # empty format/version are simply omitted (dc.read detects them)
        resolver._read("path", {"source_format": ""}, {}, "")
        assert "file_format" not in calls[-1]

    def test_reads_file_once(self, one_directory_spool, monkeypatch):
        """A row's patch is read exactly once regardless of the reader's return."""
        calls = []

        def _fake_read(**kwargs):
            calls.append(kwargs)
            return ()  # an unusual (empty) reader return

        monkeypatch.setattr("dascore.io.index.catalog.dc.read", _fake_read)
        row = {"source_format": "DASDAE", "source_version": "1"}
        resolver = one_directory_spool._catalog.resolver
        resolver._read("path", row, {}, "")
        assert len(calls) == 1

    def test_multi_patch_resolves_identity_with_single_read(
        self, one_directory_spool, random_patch, monkeypatch
    ):
        """Multi-patch reads resolve source identity from one dc.read call."""
        patch_1 = random_patch.update_attrs(_source_patch_key="first")
        patch_2 = random_patch.update_attrs(_source_patch_key="second")
        reads = []

        def _fake_read(**kwargs):
            reads.append(kwargs)
            return dc.spool([patch_1, patch_2])

        monkeypatch.setattr("dascore.io.index.catalog.dc.read", _fake_read)
        row = {
            "source_path": "path",
            "source_format": "DASDAE",
            "source_version": "1",
            "source_patch_key": "second",
        }
        resolver = one_directory_spool._catalog.resolver
        patch = resolver.resolve(row)
        assert patch.attrs["_source_patch_key"] == "second"
        assert len(reads) == 1  # the file is read exactly once

    def test_positional_id_reads_whole_source(
        self, one_directory_spool, random_patch, monkeypatch
    ):
        """Positional ids must ignore trim hints; a trimmed read would shift them."""
        patch_2 = random_patch.update_attrs(tag="second")

        def _fake_read(**kwargs):
            assert "time" not in kwargs, "positional ids must read untrimmed"
            return dc.spool([random_patch, patch_2])

        monkeypatch.setattr("dascore.io.index.catalog.dc.read", _fake_read)
        row = {
            "source_path": "path",
            "source_format": "DASDAE",
            "source_version": "1",
            "source_patch_key": "1",
        }
        resolver = one_directory_spool._catalog.resolver
        patch = resolver.resolve(row, time=(None, None))
        assert patch.attrs["tag"] == "second"


class TestSelectedDirectorySpools:
    """Selection on directory spools (select_kwargs constructor removed)."""

    @pytest.fixture(scope="class")
    def spool_dir(self, random_spool, tmp_path_factory):
        """A directory holding the random spool, one file per patch."""
        path = tmp_path_factory.mktemp("select_kwargs_dir")
        for num, patch in enumerate(random_spool):
            patch.io.write(path / f"patch_{num}.h5", "dasdae")
        return path

    @pytest.fixture(scope="class")
    def first_patch_range(self, random_spool):
        """The time range of the chronologically first patch."""
        patch = sorted(random_spool, key=lambda x: x.get_coord("time").min())[0]
        time = patch.get_coord("time")
        return (time.min(), time.max())

    def test_contents_restricted(self, spool_dir, random_spool, first_patch_range):
        """Rows outside the requested range must not appear (regression)."""
        spool = Spool.from_directory(spool_dir).update().select(time=first_patch_range)
        assert 1 <= len(spool) < len(random_spool)
        contents = spool.get_contents()
        assert (contents["time_min"] <= first_patch_range[1]).all()
        assert (contents["time_max"] >= first_patch_range[0]).all()
        for patch in spool:
            time = patch.get_coord("time")
            assert time.min() >= first_patch_range[0]
            assert time.max() <= first_patch_range[1]

    def test_selected_spool_refuses_update(
        self, spool_dir, random_spool, first_patch_range
    ):
        """D1: any operation severs update()."""
        spool = Spool.from_directory(spool_dir).update().select(time=first_patch_range)
        with pytest.raises(InvalidSpoolError, match="root spool"):
            spool.update()

    def test_select_kwargs_parameter_removed(self, spool_dir):
        """The constructor no longer accepts select_kwargs."""
        with pytest.raises(TypeError, match="select_kwargs"):
            Spool.from_directory(spool_dir, select_kwargs={"tag": "x"})


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

    def test_index_len(self, random_patch, tmp_path):
        """Deleting and rebuilding the index reproduces the contents."""
        # own directory so no other spool holds the index file open
        dc.write(random_patch, tmp_path / "a.hdf5", "dasdae")
        dc.write(random_patch.update_attrs(tag="b"), tmp_path / "b.hdf5", "dasdae")
        spool = dc.spool(tmp_path)
        spool.get_contents()  # build the index
        # close the connection so the index file can be replaced (Windows
        # cannot delete a file with an open handle), then rebuild fresh.
        spool.indexer.close()
        spool.indexer.index_path.unlink()
        rebuilt = dc.spool(tmp_path).update()
        df = rebuilt.get_contents()
        rebuilt.indexer.close()
        bank_paths = list(Path(tmp_path).rglob("*hdf5"))
        assert isinstance(df, pd.DataFrame)
        assert len(bank_paths) == len(df)

    def test_index_columns(self, basic_index_df):
        """Ensure expected columns show up in the index."""
        expected = {
            "source_path",
            "source_format",
            "source_version",
            "dims",
            "time_min",
            "time_max",
            "time_step",
        }
        assert set(basic_index_df).issuperset(expected)

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
        assert isinstance(spool, dc.Spool)

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

    def test_nested_directories(self, random_spool, tmp_path_factory):
        """Ensure files in nested directories work up to 3 levels."""
        # One patch per level: what is under test is the walk, and the
        # diverse spool's 20-odd patches only made the writing slower.
        sp_len = len(random_spool)
        num = 3
        spools = [
            random_spool[int((x / num) * sp_len) : int(((x + 1) / num) * sp_len)]
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
        paths = df["source_path"]
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
            assert patch.get_coord("time").min() >= new_min
            assert patch.get_coord("time").max() <= new_max

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
            spool.select(acquisition_key="DAS2.*")
            .select(tag="ran*")
            .select(time=(None, new_max))
        )
        assert len(out) > 0
        # first check content dataframe
        new_content = out.get_contents()
        assert len(new_content) == len(out)
        assert (new_content["acquisition_key"] == "DAS2.R2D1..RAW").all()
        assert (new_content["tag"].str.startswith("ran")).all()
        assert (new_content["time_max"] <= new_max).all()
        # then check patches
        for patch in out:
            assert patch.attrs["acquisition_key"] == "DAS2.R2D1..RAW"
            assert patch.attrs["tag"].startswith("ran")
            assert patch.get_coord("time").max() <= new_max
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
            assert pa1.get_coord("time").max() == pa2.get_coord("time").max()

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
        time_coord = spool[0].get_coord("time")
        t1 = time_coord.min()
        dt = time_coord.step
        selected_spool = spool.select(time=(t1, t1 + 30 * dt))
        patch = selected_spool[0]
        history = patch.attrs.history
        assert len(history) <= 1


class TestBasicChunk:
    """Tests for chunking filespool."""

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

    def test_chunk_out_of_order_index(self, dir_spool_index_out_of_order):
        """Ensure when the index isn't ordered chunk can still work."""
        spool = dir_spool_index_out_of_order
        time = 4.25
        chunk = spool.chunk(time=time)
        for patch in chunk:
            assert isinstance(patch, dc.Patch)
            time_coord = patch.get_coord("time")
            dur = (time_coord.max() - time_coord.min()) / ONE_SECOND
            diff = np.abs(dur - time)
            # because we try to avoid overlaps, the segments can be up to 2
            # samples shorter than what was asked for. Maybe revisit this?
            assert diff <= 2 * (time_coord.step / ONE_SECOND)

    def test_chunk_redundant_index(self, directory_spool_redundant_index):
        """Ensure redundant indices are handled effectively with chunking"""
        spool = directory_spool_redundant_index.chunk(time=None)
        patch = spool[0]
        assert isinstance(patch, dc.Patch)


class TestGetContents:
    """Tests for getting the contents of the spool."""

    def test_str_columns_in_dataframe(self, diverse_directory_spool):
        """Ensure the conventional string columns are in the index."""
        df = diverse_directory_spool.get_contents()
        expected = {
            "source_path",
            "source_format",
            "source_version",
            "dims",
            "acquisition_key",
        }
        assert set(df.columns).issuperset(expected)


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

    def test_sorted_chunked_selected_spool_can_load_patches(
        self, diverse_directory_spool
    ):
        """Sorted chunked selections should still reload patches."""
        chunked = (
            diverse_directory_spool.select(distance=(100, 200))
            .chunk(time=None, conflict="keep_first")
            .sort("time_min")
        )
        assert len(chunked) > 0
        patch_time_mins = [patch.get_coord("time").min() for patch in chunked]
        assert patch_time_mins == sorted(patch_time_mins)
        assert isinstance(chunked[0], dc.Patch)
        assert all(isinstance(patch, dc.Patch) for patch in chunked)


class TestFileBackedSpoolIntegrations:
    """Small integration tests for the file spool."""

    @pytest.fixture(scope="class")
    def dist_differ_spool(self, tmp_path_factory, random_patch):
        """Setup conditions for testing #583"""
        out_path = tmp_path_factory.mktemp("multi_dis_dir")
        distance = random_patch.get_coord("distance")
        time = random_patch.get_coord("time")
        patch2 = random_patch.select(distance=(None, distance.max() / 4)).update_coords(
            time_min=time.max()
        )
        spool = dc.spool([random_patch, patch2])
        dascore.examples.spool_to_directory(spool, path=out_path)
        dist_range = (distance.max() / 2, ...)
        return dc.spool(out_path).update().select(distance=dist_range)

    def test_one(self, diverse_spool_directory):
        """Small integration test with diverse spool."""
        acquisition_key = "DAS2.R2D1..RAW"
        endtime = np.datetime64("2022-01-01")
        duration = 3
        spool = (
            dc.spool(diverse_spool_directory)
            .select(acquisition_key=acquisition_key)  # sub-select one data source
            .select(time=(None, endtime))  # unselect anything after 2022
            .chunk(time=duration, overlap=0.5)  # change the chunking of the patches
        )
        for patch in spool:
            assert isinstance(patch, dc.Patch)
            assert patch.attrs["acquisition_key"] == acquisition_key
            time_coord = patch.get_coord("time")
            assert time_coord.max() <= endtime
            patch_duration = (time_coord.max() - time_coord.min()) / ONE_SECOND
            diff = patch_duration - duration
            assert abs(diff) <= 1.5 * time_coord.step / ONE_SECOND

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
        assert len(contents) == 1
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

    def test_differing_distances(self, dist_differ_spool):
        """Iteration works cleanly under the conditions described in #583.

        The generic index stores per-file distance ranges, so the select
        already excluded the short-distance file; no patch needs the
        historic skip-with-warning workaround.
        """
        assert len(dist_differ_spool)
        for patch in dist_differ_spool:
            assert isinstance(patch, dc.Patch)

    def test_missing_patch_error_catchable_as_index_error(self, dist_differ_spool):
        """
        For backwards compatibility, MissingPatchError must remain
        catchable as an IndexError, which spool indexing used to raise.
        """
        assert issubclass(MissingPatchError, IndexError)
        with suppress_warnings(UserWarning):
            for ind in range(len(dist_differ_spool)):
                try:
                    dist_differ_spool[ind]
                except IndexError:
                    pass

    def test_selected_out_distance_shortens_spool(self, dist_differ_spool):
        """Selecting outside of distance range reduces spool length (#583)."""
        assert len(dist_differ_spool) == 1


def _patch_shape(patch):
    """Module-level helper (process pools need picklable functions)."""
    return patch.shape


class TestDirectorySpoolSerialization:
    """Directory spools must pickle (process-backed map depends on it)."""

    def test_pickle_round_trip(self, basic_file_spool):
        """A directory spool pickles and reopens its own connection."""
        loaded = pickle.loads(pickle.dumps(basic_file_spool))
        assert len(loaded) == len(basic_file_spool)
        assert loaded[0].shape == basic_file_spool[0].shape

    def test_pickle_selected_view(self, basic_file_spool):
        """Selected views keep their selection through pickling."""
        df = basic_file_spool.get_contents()
        sub = basic_file_spool.select(time=(df["time_min"].min(), None))
        loaded = pickle.loads(pickle.dumps(sub))
        assert len(loaded) == len(sub)

    @pytest.mark.concurrency
    def test_process_pool_map(self, basic_file_spool):
        """Spool.map works with a process pool executor."""
        with ProcessPoolExecutor(max_workers=1) as client:
            out = list(basic_file_spool.map(_patch_shape, client=client, progress=None))
        assert len(out) == len(basic_file_spool)
