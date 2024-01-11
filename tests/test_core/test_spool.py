"""Test for spool functions."""
from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.clients.filespool import FileSpool
from dascore.core.spool import BaseSpool, MemorySpool
from dascore.exceptions import (
    IncompatiblePatchError,
    InvalidSpoolError,
    ParameterError,
    PatchDimError,
)
from dascore.utils.time import to_datetime64, to_timedelta64


def _gigo(garbage):
    """Dummy func which can be serialized."""
    return garbage


class _SerialClient:
    """Serial client for testing mapping logic."""

    def map(self, func, iterable_thing, **kwargs):
        for thing in iterable_thing:
            yield func(thing, **kwargs)


@pytest.fixture(scope="session")
def random_spool_len_10():
    """Return a spool of length 10."""
    return dc.examples.get_example_spool(length=10)


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

    def test_updated_spool_eq(self, random_spool):
        """Ensure updating the spool doesnt change equality."""
        assert random_spool == random_spool.update()

    def test_empty_spool_str(self):
        """Ensure and empty spool has a string rep. See #295."""
        spool = dc.spool([])
        spool_str = str(spool)
        assert "Spool" in spool_str


class TestSpoolEquals:
    """Tests for spool equality."""

    def test_other_type(self, random_spool):
        """Ensure other types return false equality."""
        assert random_spool != 1
        assert random_spool != (1, 2)
        assert random_spool != {}
        assert {1: 2} != random_spool

    def test_chunked_differently(self, random_spool):
        """Spools with different chunking should !=."""
        sp1 = random_spool.chunk(time=1.12)
        assert sp1 != random_spool
        sp2 = random_spool.chunk(time=1.00)
        assert sp2 != sp1

    def test_eq_self(self, random_spool):
        """A spool should always eq itself."""
        assert random_spool == random_spool

    def test_unequal_attr(self, random_spool):
        """Simulate some attribute which isn't equal."""
        new1 = copy.deepcopy(random_spool)
        new1.__dict__["bad_attr"] = 1
        new2 = copy.deepcopy(random_spool)
        new2.__dict__["bad_attr"] = 2
        assert new1 != new2

    def test_unequal_dicts(self, random_spool):
        """Simulate some dicts which don't have the same values."""
        new1 = copy.deepcopy(random_spool)
        new1.__dict__["bad_attr"] = {1: 2}
        new2 = copy.deepcopy(random_spool)
        new2.__dict__["bad_attr"] = {2: 3}
        assert new1 != new2


class TestIndexing:
    """Tests for indexing spools to retrieve patches."""

    def test_simple_index(self, random_spool):
        """Ensure indexing a spool returns a patch."""
        for ind in range(len(random_spool)):
            patch = random_spool[ind]
            assert isinstance(patch, dc.Patch)

    def test_negative_index_random_spool(self, random_spool):
        """Like lists, negative index should start from end."""
        for ind in range(1, len(random_spool) + 1):
            patch1 = random_spool[-ind]
            patch2 = random_spool[-ind + len(random_spool)]
            assert isinstance(patch1, dc.Patch)
            assert patch1 == patch2

    def test_out_of_bounds_raises(self, random_spool):
        """Out of bounds queries to raise IndexError."""
        match = "out of bounds for spool"
        with pytest.raises(IndexError, match=match):
            _ = random_spool[len(random_spool)]


class TestSlicing:
    """Tests for slicing spools to get sub-spools."""

    slices = (
        slice(None, None),
        slice(1, 2),
        slice(1, -1),
        slice(2),
        slice(None, 2),
    )

    @pytest.mark.parametrize("sliz", slices)
    def test_slice_behaves_like_list(self, random_spool, sliz):
        """Ensure slicing as spool behaves like list."""
        patch_list = list(random_spool)[sliz]
        sub_spool = random_spool[sliz]
        assert len(sub_spool) == len(patch_list)
        for pa1, pa2 in zip(patch_list, sub_spool):
            assert pa1 == pa2

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


class TestSpoolIterable:
    """Tests for iterating Spools."""

    def test_len(self, random_spool):
        """Ensure the spool has a length."""
        assert len(random_spool) == len(list(random_spool))

    def test_index(self, random_spool):
        """Ensure the spool can be indexed."""
        assert isinstance(random_spool[0], dc.Patch)

    def test_list_o_patches(self, random_spool):
        """Ensure random_string can be iterated."""
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
        """Ensure the index returns the correct patch."""
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
        """Ensure multiple range selects can be used in one call."""
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

    def test_split_ellipses(self, diverse_spool):
        """Ensure ... can be used for an open interval."""
        spool1 = diverse_spool.select(time=(..., "2020-01-01"))
        spool2 = diverse_spool.select(time=(None, "2020-01-01"))
        assert spool1 == spool2


class TestSort:
    """Tests for sorting spools."""

    def test_base_spool_sort_raises(self, random_spool):
        """Ensure base spool's sort raises."""
        expected_str = "spool of type"
        with pytest.raises(NotImplementedError, match=expected_str):
            BaseSpool.sort(random_spool, "time")

    def test_sorting_attr_not_exists(self, diverse_spool):
        """Test sorting by an attribute that does not exist in the DataFrame."""
        expected_str = "Invalid attribute"
        with pytest.raises(IndexError, match=expected_str):
            diverse_spool.sort("dummy_attribute")

    def test_sorting_attr_exists(self, diverse_spool):
        """Test sorting by an attribute that exists in the DataFrame."""
        sorted_spool = diverse_spool.sort("time_min")
        df = sorted_spool.get_contents()
        assert df["time_min"].is_monotonic_increasing

    def test_sorting_attr_time(self, diverse_spool):
        """Test sorting by the 'time' attribute that that may not be in the df."""
        sorted_spool = diverse_spool.sort("time")
        df = sorted_spool.get_contents()
        assert df["time_min"].is_monotonic_increasing

    def test_sorting_attr_distance(self, diverse_spool):
        """Test sorting by the 'distance' attribute that may not exist in the df."""
        sorted_spool = diverse_spool.sort("distance")
        df = sorted_spool.get_contents()
        assert df["distance_min"].is_monotonic_increasing


class TestSplit:
    """Tests splitting spools into smaller spools."""

    @pytest.fixture(scope="class")
    def split_10(self, random_spool_len_10):
        """Split the spools using spool size."""
        spools = tuple(random_spool_len_10.split(size=3))
        return spools

    def test_both_parameters_raises(self, random_spool):
        """Ensure split raises when both spool_size and spool_count are defined."""
        msg = "requires either spool_count or spool_size"
        with pytest.raises(ParameterError, match=msg):
            list(random_spool.split(size=1, count=2))

    def test_spool_size(self, split_10):
        """Ensure spool size can be split."""
        # because there are 10 patches in the spool its len should be 4
        assert len(split_10) == 4
        for i in range(3):
            assert len(split_10[i]) == 3
        assert len(split_10[-1]) == 1

    def test_yielded_spools_indexable(self, split_10):
        """Ensure we can pull the first patch from each spool."""
        for spool in split_10:
            patch = spool[0]
            assert isinstance(patch, dc.Patch)

    def test_spool_count(self, random_spool):
        """Ensure we can split based on desired size of spool."""
        split = list(random_spool.split(size=2))
        assert len(split) == 2
        assert len(split[0]) == 2
        assert len(split[1]) == 1

    def test_base_split_raises(self, random_spool):
        """Ensure BaseSpool split raises NoteImplementedError."""
        msg = "has no split implementation"
        with pytest.raises(NotImplementedError, match=msg):
            BaseSpool.split(
                random_spool,
            )


class TestMap:
    """Test for mapping spool contents onto functions."""

    @pytest.fixture(scope="class")
    def thread_client(self):
        """A ThreadPoolExecutor."""
        return ThreadPoolExecutor()

    @pytest.fixture(scope="class")
    def proc_client(self):
        """A ProcessPoolExecutor."""
        return ProcessPoolExecutor()

    def test_simple(self, random_spool):
        """Simplest case for mapping a function on all patches."""
        out = list(random_spool.map(lambda x: x))
        assert len(out) == len(random_spool)
        assert dc.spool(out) == random_spool

    def test_non_patch_return(self, random_spool):
        """Ensure outputs don't have to be patches."""
        out = list(random_spool.map(lambda x: np.max(x.data)))
        for val in out:
            assert isinstance(val, np.float_)

    def test_dummy_client(self, random_spool):
        """Ensure a client arguments works."""
        out = list(random_spool.map(lambda x: x, client=_SerialClient()))
        assert len(out) == len(random_spool)
        assert dc.spool(out) == random_spool

    def test_thread_client(self, random_spool, thread_client):
        """Ensure a thread client works."""
        out = list(random_spool.map(lambda x: x, client=thread_client))
        assert len(out) == len(random_spool)
        assert dc.spool(out) == random_spool

    def test_process_client(self, random_spool, proc_client):
        """Ensure process pool also works."""
        out = list(random_spool.map(_gigo, client=proc_client))
        assert len(out) == len(random_spool)
        assert dc.spool(out) == random_spool

    def test_map_docstring(self, random_spool):
        """Ensure the docstring examples work."""
        results_list = list(
            random_spool.chunk(time=5).map(lambda x: np.std(x.data, axis=0))
        )
        out = np.stack(results_list, axis=-1)
        assert out.size

    def test_map_docs(self, random_spool):
        """Test the doc code for map."""

        def get_dist_max(patch):
            """Function which will be mapped to each patch in spool."""
            return patch.aggregate("time", "max")

        out = list(random_spool.chunk(time=5, overlap=1).map(get_dist_max))
        new_spool = dc.spool(out)
        merged = new_spool.chunk(time=None)
        assert merged
        assert isinstance(merged[0], dc.Patch)


class TestGetSpool:
    """Test getting spool from various sources."""

    def test_spool_from_spool(self, random_spool):
        """Ensure a spool is valid input to get spool."""
        out = dc.spool(random_spool)
        for p1, p2 in zip(out, random_spool):
            assert p1.equals(p2)

    def test_spool_from_patch_sequence(self, random_spool):
        """Ensure a list of patches returns a spool."""
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


class TestMisc:
    """Tests for misc. spool cases."""

    def test_changed_memory_spool(self, random_patch):
        """
        Calling spool on a patch that was returned from None results in
        the spool contents reverting to original patch.
        """
        # setup patch with simple history
        patch = random_patch.pass_filter(time=(10, 20))
        assert patch.attrs.history
        # create new patch with cleared history
        new_attrs = dict(patch.attrs)
        new_attrs["history"] = []
        new_patch = patch.update(attrs=new_attrs)
        assert not new_patch.attrs.history
        # add new patch (w/ no history) to spool, get first patch out.
        spool = dc.spool([new_patch])
        assert len(spool) == 1
        # get first patch, assert it has no history
        out = spool[0]
        assert not out.attrs.history

    def test_nice_non_exist_message(self):
        """Ensure a nice message is raised for nonexistent paths. See #126."""
        with pytest.raises(InvalidSpoolError, match="may not exist"):
            dc.spool("Bad/file/path.h5")

    def test_dft_patch_access(self, random_dft_patch):
        """Ensure a dft patch can be retrieved from as spool. See #303."""
        spool = dc.spool(random_dft_patch)
        patch = spool[0]
        assert isinstance(patch, dc.Patch)


class TestSpoolStack:
    """Tests for stacking (adding) spool content."""

    def test_stack_data(self):
        """
        Try the stack method on an example spool that has repeated
        copies of the same patch with different times. Check data.
        """
        # Grab the example spool which has repeats of the same patch
        # but with different time dimensions. Stack the patches.
        spool = dc.get_example_spool()
        stack_patch = spool.stack(dim_vary="time")
        # We expect the sum/stack to be same as a multiple of
        # the first patch's data.
        baseline = float(len(spool)) * spool[0].data
        assert np.allclose(baseline, stack_patch.data)

    def test_same_dim_different_shape(self, random_spool):
        """Ensure when stack dimensions have different shape an error is raised."""
        # Create a spool with two patches, each with time dim but with different
        # lengths.
        patch1, patch2 = random_spool[:2]
        patch2 = patch2.select(time=(1, 30), samples=True)
        spool = dc.spool([patch1, patch2])
        # Check that warnings/exceptions are raised.
        msg = "Patches are not compatible"
        with pytest.raises(IncompatiblePatchError, match=msg):
            spool.stack(dim_vary="time", check_behavior="raise")
        # Or a warning issued.
        with pytest.warns(UserWarning, match=msg):
            spool.stack(dim_vary="time", check_behavior="warn")

    def test_different_dimensions(self, random_spool):
        """Tests for when the spool has patches with different dimensions."""
        new_patch = random_spool[0].rename_coords(time="money")
        spool = dc.spool([random_spool[1], new_patch])
        msg = "because their dimensions are not equal"
        with pytest.warns(UserWarning, match=msg):
            spool.stack(dim_vary="time", check_behavior="warn")

    def test_bad_dim_vary(self, random_spool):
        """Ensure when dim_vary is not in patch an error is raised."""
        with pytest.raises(PatchDimError):
            random_spool.stack(dim_vary="money")

    def test_stack_coords(self):
        """
        Try the stack method on an example spool that has repeated
        copies of the same patch with different times. Check coords.
        """
        # Grab the example spool which has repeats of the same patch
        # but with different time dimensions. Stack the patches.
        spool = dc.get_example_spool()
        stack_patch = spool.stack(dim_vary="time")
        # check that 'time' coordinates has same step as original patch
        time_coords = stack_patch.coords.coord_map["time"]
        orig_time_coords = spool[0].coords.coord_map["time"]
        assert time_coords.step == orig_time_coords.step
        # check that distance coordinates are the same
        dist_coords = stack_patch.coords.coord_map["distance"]
        orig_dist_coords = spool[0].coords.coord_map["distance"]
        assert dist_coords.start == orig_dist_coords.start
        assert dist_coords.stop == orig_dist_coords.stop
        assert dist_coords.step == orig_dist_coords.step
        assert dist_coords.units == orig_dist_coords.units
