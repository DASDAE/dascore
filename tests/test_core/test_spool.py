"""Tests for spool function."""

from __future__ import annotations

import copy
import functools
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.utils.patch_assembly as assembly_mod
from dascore.core.spool import BaseSpool, Spool
from dascore.examples import ricker_moveout
from dascore.exceptions import (
    CoordMergeError,
    IncompatiblePatchError,
    InvalidSpoolError,
    InvalidSpoolQueryError,
    MissingOptionalDependencyError,
    MissingPatchError,
    ParameterError,
)
from dascore.io.index.planned import PlanResolver
from dascore.io.segy import SegyV1_0
from dascore.utils.downloader import fetch
from dascore.utils.misc import deep_equality_check, suppress_warnings
from dascore.utils.patch_assembly import _estimate_merge_samples, _get_varying_dim
from dascore.utils.time import to_datetime64, to_timedelta64


def _gigo(garbage):
    """Dummy func which can be serialized."""
    return garbage


class _CallableObject:
    """A callable object, which has no `__name__`."""

    def __call__(self, patch):
        return patch


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

    def test_spool_from_empty_sequence(self):
        """Ensure a spool can be created from empty list."""
        out = dc.spool([])
        assert isinstance(out, BaseSpool)
        assert len(out) == 0

    def test_updated_spool_eq(self, random_spool):
        """Ensure updating the spool doesn't change equality."""
        assert random_spool == random_spool.update()

    def test_empty_spool_str(self):
        """Ensure and empty spool has a string rep. See #295."""
        spool = dc.spool([])
        spool_str = str(spool)
        assert "Spool" in spool_str

    def test_spool_with_empty_patch_str(self):
        """A spool with an empty patch should have a str."""
        spool = dc.spool(dc.Patch())
        spool_str = str(spool)
        assert "Spool" in spool_str

    def test_invalid_input_raises(self):
        """A non-patch, non-spool input raises a clear error."""
        with pytest.raises(InvalidSpoolError, match="accepts a Patch"):
            Spool(42)

    def test_base_spool_alias(self):
        """BaseSpool is kept as an alias of the one spool class."""
        assert BaseSpool is Spool is dc.BaseSpool

    def test_uninitialized_subclass_raises(self, random_spool):
        """A subclass which skips Spool.__init__ is refused, not copied."""

        class SubSpool(Spool):
            """A subclass which never builds a catalog."""

            def __init__(self, patches):
                self._patches = list(patches)

        with pytest.raises(InvalidSpoolError, match="has no catalog"):
            Spool(SubSpool(random_spool))

    def test_viz_raises(self, random_spool):
        """Ensure Spool.viz raises AttributeError."""
        msg = "Apply 'viz' on a Patch object"
        with pytest.raises(AttributeError, match=msg):
            random_spool.viz.waterfall(random_spool)


class TestLiveSpoolLazy:
    """
    Tests for lazy behavior of in-memory spools.

    Spools created directly from patches shouldn't build their managing
    dataframes until an operation requires them.
    """

    @pytest.fixture()
    def patch_list(self):
        """Get a list of contiguous patches."""
        return list(dc.examples.get_example_spool(length=3))

    def test_simple_access_builds_no_dataframes(self, patch_list):
        """len, integer access, and iteration should not build dataframes."""
        spool = dc.spool(patch_list)
        assert len(spool) == len(patch_list)
        assert spool[0] == patch_list[0]
        assert spool[-1] == patch_list[-1]
        assert list(spool) == patch_list
        assert spool._catalog._backend is None

    def test_out_of_bounds_raises(self, patch_list):
        """The fast path must raise the same IndexError as the df path."""
        spool = dc.spool(patch_list)
        match = "out of bounds for spool"
        with pytest.raises(IndexError, match=match):
            _ = spool[len(patch_list)]
        assert spool._catalog._backend is None

    def test_access_unchanged_after_df_built(self, patch_list):
        """Patch access must return the same thing before/after df built."""
        spool = dc.spool(patch_list)
        lazy_patches = list(spool)
        _ = spool.get_contents()  # forces the flat relation to build
        assert spool._catalog._backend is not None
        assert list(spool) == lazy_patches
        assert spool[0] == lazy_patches[0]

    def test_equality_independent_of_access(self, patch_list):
        """Equal spools must stay equal regardless of what was accessed."""
        spool1, spool2 = dc.spool(patch_list), dc.spool(patch_list)
        _ = spool1.get_contents()  # build one spool's dataframes only.
        assert spool1 == spool2
        assert spool2 == spool1

    def test_input_mutation_does_not_change_spool(self, patch_list):
        """The spool contents are snapshotted at creation."""
        data = list(patch_list)
        spool = dc.spool(data)
        data.pop()
        assert len(spool) == len(patch_list)
        assert list(spool) == patch_list

    def test_derived_spools_use_df_machinery(self, patch_list):
        """Chunked/selected spools must go through the instruction dfs."""
        spool = dc.spool(patch_list)
        merged = spool.chunk(time=None)
        assert len(merged) == 1
        time_coord = merged[0].get_coord("time")
        expected_min = min(x.summary.get_coord_summary("time").min for x in patch_list)
        assert time_coord.min() == expected_min

    def test_derived_spool_is_own_catalog(self, patch_list):
        """A chunked spool is a fresh derived catalog sharing patches."""
        spool = dc.spool(patch_list)
        chunked = spool.chunk(time=1)
        assert chunked._catalog is not spool._catalog
        assert isinstance(chunked._catalog.resolver, PlanResolver)
        # member loading shares the parent's live patches, not copies
        registry = chunked._catalog.resolver.live_entries()
        assert {id(p) for p in registry.values()} <= {id(p) for p in patch_list}
        # no other patch containers exist on the instance
        assert "_patches" not in chunked.__dict__
        assert "_data" not in chunked.__dict__

    def test_single_patch_input_uses_lazy_storage(self, random_patch):
        """A single patch lands in the registry without realizing tables."""
        spool = Spool(random_patch)
        assert len(spool) == 1
        registry = spool._catalog.resolver.live_entries()
        assert tuple(registry.values()) == (random_patch,)
        # simple access never bootstrapped the index backend
        assert spool._catalog._backend is None

    def test_empty_memory_spool(self):
        """An empty Spool is a valid, iterable, zero-length spool."""
        spool = Spool()
        assert len(spool) == 0
        assert list(spool) == []


class TestSpoolHelpers:
    """Tests for helper functions used by spool implementations."""

    def test_get_varying_dim_ignores_missing_ranges(self):
        """Columns without min/max pairs should not count as varying dims."""
        df = pd.DataFrame(
            {"time_min": [0, 0], "time_max": [1, 1], "distance_min": [0, 1]}
        )
        assert _get_varying_dim(df) is None

    def test_estimate_merge_samples_missing_columns(self):
        """Missing range columns should disable streaming estimates."""
        df = pd.DataFrame({"time_min": [0], "time_max": [1]})
        assert _estimate_merge_samples(df, "time") is None

    def test_estimate_merge_samples_degenerate_step(self):
        """Non-finite sample counts should disable streaming estimates."""
        df = pd.DataFrame({"time_min": [0.0], "time_max": [1.0], "time_step": [0.0]})
        assert _estimate_merge_samples(df, "time") is None

    def test_estimate_merge_samples_negative_count(self):
        """Ranges with negative sample counts should disable streaming estimates."""
        df = pd.DataFrame({"time_min": [2.0], "time_max": [0.0], "time_step": [1.0]})
        assert _estimate_merge_samples(df, "time") is None


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

    def test_foreign_attrs_do_not_join_equality(self, random_spool):
        """Equality state is enumerated; stray instance attrs are ignored."""
        new1 = copy.deepcopy(random_spool)
        new1.__dict__["bad_attr"] = 1
        new2 = copy.deepcopy(random_spool)
        new2.__dict__["bad_attr"] = 2
        assert new1 == new2


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


class TestSpoolBoolArraySelect:
    """Tests for selecting patches using a boolean array."""

    def test_bool_all_true(self, random_spool):
        """All True should return an equal spool."""
        bool_array = np.ones(len(random_spool), dtype=np.bool_)
        out = random_spool[bool_array]
        assert out == random_spool

    def test_bool_all_false(self, random_spool):
        """All False should return an empty spool."""
        bool_array = np.zeros(len(random_spool), dtype=np.bool_)
        out = random_spool[bool_array]
        assert len(out) == 0

    def test_bool_some_true(self, random_spool):
        """Some true values should return a spool with some values."""
        bool_array = np.ones(len(random_spool), dtype=np.bool_)
        bool_array[1] = False
        out = random_spool[bool_array]
        assert len(out) == sum(bool_array)
        df1 = out.get_contents().reset_index(drop=True)
        df2 = random_spool.get_contents()[bool_array].reset_index(drop=True)
        assert df1.equals(df2)


class TestSpoolIntArraySelect:
    """Tests for selecting patches using an integer array."""

    def test_uniform(self, random_spool):
        """A uniform monotonic increasing array should return same spool."""
        array = np.arange(len(random_spool))
        spool = random_spool[array]
        assert spool == random_spool

    def test_out_of_bounds_raises(self, random_spool):
        """Ensure int values gt the spool len raises."""
        array = np.arange(len(random_spool))
        array[0] = len(random_spool) + 10
        with pytest.raises(IndexError):
            random_spool[array]

    def test_bad_array_type(self, random_spool):
        """Ensure a non-index or int array raises."""
        array = np.arange(len(random_spool)) + 0.01
        with pytest.raises(ValueError, match="Only bool or int dtypes"):
            random_spool[array]

    def test_rearrange(self, random_spool):
        """Ensure patch order can be changed."""
        array = np.array([len(random_spool) - 1, 0])
        out = random_spool[array]
        assert out[0] == random_spool[-1]
        assert out[-1] == random_spool[0]


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

    def test_contents_are_caller_owned(self, random_spool):
        """Mutating the returned dataframe must not change the spool."""
        df = random_spool.get_contents()
        df["tag"] = "modified"
        assert (random_spool.get_contents()["tag"] != "modified").all()


class TestSelect:
    """Tests for selecting/trimming spools."""

    def test_select_acquisition_key(self, diverse_spool):
        """Ensure a set can be used to select spools within data sources."""
        source_set = {"DAS2.R2D1..RAW", "DAS3.R2D1..RAW"}
        out = diverse_spool.select(acquisition_key=source_set)
        assert len(out), "an empty selection would pass the check below"
        for patch in out:
            assert patch.attrs["acquisition_key"] in source_set

    def test_select_tag_wildcard(self, diverse_spool):
        """Ensure wildcards can be used on str columns."""
        out = diverse_spool.select(tag="some*")
        assert len(out), "an empty selection would pass the check below"
        for patch in out:
            assert patch.attrs["tag"].startswith("some")

    def test_multiple_selects(self, diverse_spool):
        """Ensure selects can be stacked."""
        contents = diverse_spool.get_contents()
        duration = contents["time_max"] - contents["time_min"]
        new_max = (contents["time_min"] + duration / 2).max()
        out = (
            diverse_spool.select(acquisition_key="DAS2.*")
            .select(tag="ran*")
            .select(time=(None, new_max))
        )
        assert len(out)
        for patch in out:
            assert patch.attrs["acquisition_key"] == "DAS2.R2D1..RAW"
            assert patch.attrs["tag"].startswith("ran")
            assert patch.get_coord("time").max() <= new_max

    def test_multiple_range_selects(self, adjacent_spool_no_overlap):
        """Ensure multiple range selects can be used in one call."""
        spool = adjacent_spool_no_overlap
        contents = spool.get_contents()
        # get new time/distance ranges and select them
        time_min = to_datetime64(contents["time_min"].min() + to_timedelta64(4))
        time_max = to_datetime64(contents["time_max"].max() - to_timedelta64(4))
        distance_min = contents["distance_min"].min() + 50
        distance_max = contents["distance_max"].max() - 50
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
            assert patch.get_coord("time").min() >= time_min
            assert patch.get_coord("time").max() <= time_max
            assert patch.get_coord("distance").min() >= distance_min
            assert patch.get_coord("distance").max() <= distance_max

    def test_split_ellipses(self, diverse_spool):
        """Ensure ... can be used for an open interval."""
        spool1 = diverse_spool.select(time=(..., "2020-01-01"))
        spool2 = diverse_spool.select(time=(None, "2020-01-01"))
        assert spool1 == spool2

    def test_non_coord_patches(self, spool_with_non_coords):
        """Ensure non-coords still can be selected."""
        first = spool_with_non_coords[0]
        time_coord = first.get_coord("time")
        time_sel = (time_coord.min(), time_coord.max())
        out = spool_with_non_coords.select(time=time_sel)
        # Ensure all remaining patches have valid time coords.
        for patch in out:
            assert isinstance(patch, dc.Patch)
            assert not np.any(pd.isnull(patch.get_array("time")))


class TestUnselect:
    """Tests for removing the patches a selection would keep."""

    def test_complements_select(self, diverse_spool):
        """Every patch is in exactly one of the two halves."""
        kept = diverse_spool.select(tag="some_tag")
        dropped = diverse_spool.unselect(tag="some_tag")
        assert len(kept) + len(dropped) == len(diverse_spool)
        assert not set(kept.get_contents()["_patch_id"]) & set(
            dropped.get_contents()["_patch_id"]
        )

    def test_removes_the_matches(self, diverse_spool):
        """What comes back is what the selection would not have kept."""
        out = diverse_spool.unselect(tag="some_tag")
        assert len(out), "an empty result would pass the check below"
        for patch in out:
            assert patch.attrs["tag"] != "some_tag"

    def test_selector_shapes(self, diverse_spool):
        """A keyword means what it means in select, then is negated."""
        keys = {"DAS2.R2D1..RAW", "DAS3.R2D1..RAW"}
        out = diverse_spool.unselect(acquisition_key=keys)
        assert len(out), "an empty result would pass the check below"
        for patch in out:
            assert patch.attrs["acquisition_key"] not in keys
        wild = diverse_spool.unselect(tag="some*")
        for patch in wild:
            assert not patch.attrs["tag"].startswith("some")

    def test_attrs_namespace(self, diverse_spool):
        """The explicit namespace form works as it does in select."""
        out = diverse_spool.unselect(_attrs={"tag": "some_tag"})
        assert len(out) == len(diverse_spool.unselect(tag="some_tag"))

    def test_matching_nothing_keeps_everything(self, diverse_spool):
        """Removing what is not there removes nothing."""
        assert len(diverse_spool.unselect(tag="not_a_tag")) == len(diverse_spool)

    def test_matching_everything_keeps_nothing(self, random_spool):
        """And removing what is all of it leaves an empty spool."""
        assert len(random_spool.unselect(tag="*")) == 0

    def test_coordinates_raise(self, diverse_spool):
        """
        A coordinate complement is subdivision, which this cannot do yet.

        Selecting on a coordinate trims each patch, so removing a range
        cuts every patch into the pieces outside it -- one patch becoming
        two -- rather than choosing between patches.
        """
        with pytest.raises(InvalidSpoolQueryError, match="unselect cannot take"):
            diverse_spool.unselect(time=("2020-01-03", None))
        with pytest.raises(InvalidSpoolQueryError, match="unselect cannot take"):
            diverse_spool.unselect(_coords={"time": ("2020-01-03", None)})

    def test_unknown_name_raises(self, diverse_spool):
        """A misspelling is one here too."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            diverse_spool.unselect(not_a_name=1)

    def test_composes(self, diverse_spool):
        """A spool with patches removed is a spool."""
        tags = diverse_spool.get_contents()["tag"].tolist()
        removed = {"some_tag", "random"}
        out = diverse_spool.unselect(tag="some_tag").unselect(tag="random")
        assert len(out) == sum(x not in removed for x in tags)
        for patch in out:
            assert patch.attrs["tag"] not in removed

    def test_complements_within_a_window(self, diverse_spool):
        """
        A window fixes which rows are present, not which ones match.

        The complement of a selection over a windowed spool is the rest
        of the window, not nothing.
        """
        window = diverse_spool[2:8]
        tags = window.get_contents()["tag"].tolist()
        for tag in set(tags):
            expected = sum(x != tag for x in tags)
            assert len(window.unselect(tag=tag)) == expected
        assert len(window.unselect(tag="not_a_tag")) == len(window)

    def test_original_is_unchanged(self, diverse_spool):
        """As everywhere else, the spool it came from is left alone."""
        before = len(diverse_spool)
        diverse_spool.unselect(tag="some_tag")
        assert len(diverse_spool) == before

    def test_empty_spool_names_nothing(self):
        """A spool with no patches has no attrs, exactly as for select."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            dc.spool([]).unselect(tag="anything")

    def test_naming_nothing_raises(self, diverse_spool):
        """Emptying a spool is not something to guess at."""
        with pytest.raises(ParameterError, match="needs something to remove"):
            diverse_spool.unselect()

    def test_naming_only_a_no_op_raises(self, diverse_spool):
        """
        A no-op selector names nothing to remove either.

        `select(tag=None)` is the whole spool, so its complement is an
        empty one; that is the same guess bare `unselect()` refuses.
        """
        for empty in (None, ...):
            with pytest.raises(ParameterError, match="needs something to remove"):
                diverse_spool.unselect(tag=empty)

    def test_coords_tag_form_raises(self, diverse_spool):
        """The tag form names bare kwargs, and is refused the same way."""
        with pytest.raises(InvalidSpoolQueryError, match="unselect cannot take"):
            diverse_spool.unselect(_coords="time", time=("2020-01-03", None))


class TestSort:
    """Tests for sorting spools."""

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

    def test_sorted_spool_iteration_matches_sorted_contents(self, diverse_spool):
        """Sorting should reorder loaded patches, not just the contents dataframe."""
        sorted_spool = diverse_spool.sort("time_min")
        patch_time_mins = [patch.get_coord("time").min() for patch in sorted_spool]
        assert patch_time_mins == sorted(patch_time_mins)

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

    @pytest.mark.parametrize("kwargs", [{"size": 1, "count": 2}, {}])
    def test_needs_exactly_one_parameter(self, random_spool, kwargs):
        """Ensure split raises unless exactly one of size/count is given."""
        msg = "requires either spool_count or spool_size"
        with pytest.raises(ParameterError, match=msg):
            list(random_spool.split(**kwargs))

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

    def test_uneven_size(self, random_spool):
        """Ensure a size which doesn't divide evenly leaves a short last spool."""
        split = list(random_spool.split(size=2))
        assert len(split) == 2
        assert len(split[0]) == 2
        assert len(split[1]) == 1

    @pytest.mark.parametrize("kwargs", [{"size": 0}, {"size": -1}, {"count": 0}])
    def test_non_positive_raises(self, random_spool, kwargs):
        """A size or count of zero or less would never finish yielding."""
        msg = "requires a positive size or count"
        with pytest.raises(ParameterError, match=msg):
            list(random_spool.split(**kwargs))

    def test_non_integral_size(self, random_spool_len_10):
        """A size which isn't a whole number rounds up rather than raising."""
        split = list(random_spool_len_10.split(size=2.5))
        assert [len(x) for x in split] == [3, 3, 3, 1]

    def test_spool_count(self, random_spool_len_10):
        """Ensure we can split based on the desired number of spools."""
        split = list(random_spool_len_10.split(count=3))
        assert len(split) == 3
        assert sum(len(x) for x in split) == 10


class TestMap:
    """Test for mapping spool contents onto functions."""

    @pytest.fixture(scope="class")
    def thread_client(self):
        """A ThreadPoolExecutor."""
        return ThreadPoolExecutor()

    @pytest.fixture(scope="class")
    def proc_client(self):
        """A ProcessPoolExecutor."""
        try:
            return ProcessPoolExecutor()
        except (PermissionError, OSError, RuntimeError) as exc:
            pytest.skip(f"ProcessPoolExecutor unavailable: {exc}")

    @pytest.fixture(params=["partial", "callable_object", "patch_op"])
    def nameless_callable(self, request):
        """A callable with no `__name__`, of each kind DASCore makes."""
        callables = {
            "partial": functools.partial(_gigo),
            "callable_object": _CallableObject(),
            "patch_op": dc.proc.abs.op(),
        }
        return callables[request.param]

    def test_simple(self, random_spool):
        """Simplest case for mapping a function on all patches."""
        out = list(random_spool.map(lambda x: x))
        assert len(out) == len(random_spool)
        assert dc.spool(out) == random_spool

    def test_non_patch_return(self, random_spool):
        """Ensure outputs don't have to be patches."""
        out = list(random_spool.map(lambda x: np.max(x.data)))
        for val in out:
            assert isinstance(val, np.float64)

    def test_dummy_client(self, random_spool):
        """Ensure a client arguments works."""
        out = list(random_spool.map(lambda x: x, client=_SerialClient()))
        assert len(out) == len(random_spool)
        assert dc.spool(out) == random_spool

    @pytest.mark.concurrency
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

    def test_callable_without_name(self, random_spool, nameless_callable):
        """A callable with no `__name__` can be mapped."""
        out = list(random_spool.map(nameless_callable))
        assert len(out) == len(random_spool)

    def test_callable_without_name_client(self, random_spool, nameless_callable):
        """A client maps a nameless callable; the worker builds the label."""
        out = list(random_spool.map(nameless_callable, client=_SerialClient()))
        assert len(out) == len(random_spool)

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
            return patch.select(time=10, samples=True)

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
        if a format supports scanning, a lazy file-backed spool is
        returned. If it doesn't, all the file contents have to be loaded
        into memory, so the spool holds live patches.
        """
        path = tmp_path_factory.mktemp("file_spoolin")
        dasdae_path = path / "patch.h5"
        pickle_path = path / "patch.pkl"
        dc.write(random_spool, dasdae_path, "dasdae")
        dc.write(random_spool, pickle_path, "pickle")

        dasdae_spool = dc.spool(dasdae_path)
        assert not dasdae_spool.has_live_patches

        pickle_spool = dc.spool(pickle_path)
        assert pickle_spool.has_live_patches


class TestSpoolBehaviorOptionalImports:
    """
    Tests for spool behavior when handling optional formats which require
    optional dependencies.

    Essentially, if the spool is specific to the file (eg spool("file"))
    it should raise. If it is applied on a directory with such files
    (eg spool("directory/with/bad/files")) it should give a warning.
    """

    # The string to match against the warning/error.
    _msg = "found files that can be read if additional"

    @pytest.fixture(scope="function", autouse=True)
    def monkey_patch_segy(self, monkeypatch):
        """Monkey patch the name of the imported library for segy."""
        # TODO we should find a cleaner way to do this in the future.

        monkeypatch.setattr(SegyV1_0, "_package_name", "not_segyio_clearly")

    @pytest.fixture(scope="class")
    def segy_file_path(self, tmp_path_factory):
        """
        Create a directory structure like this:

        optional_import_test
        - h5_simple_1.h5
        - segy_only
          - small_channel_patch.sgy
        """
        dir_path = tmp_path_factory.mktemp("optional_import_test")
        simple_path = fetch("h5_simple_1.h5")
        shutil.copy(simple_path, dir_path)

        segy_only_path = dir_path / "segy_only"
        segy_only_path.mkdir(exist_ok=True, parents=True)
        segy_path = fetch("small_channel_patch.sgy")
        shutil.copy(segy_path, segy_only_path)
        return segy_only_path / segy_path.name

    def test_spool_on_directory_no_other_files(self, segy_file_path):
        """Ensure a directory with no other readable files raises."""
        with pytest.raises(MissingOptionalDependencyError, match=self._msg):
            dc.spool(segy_file_path.parent).update()

    def test_spool_on_single_file(self, segy_file_path):
        """Ensure a single file also raises."""
        with pytest.raises(MissingOptionalDependencyError, match=self._msg):
            dc.spool(segy_file_path).update()

    def test_spool_on_multiple_files(self, segy_file_path):
        """Ensure if other files exist the warning is issued."""
        top_level = segy_file_path.parent.parent
        with pytest.warns(UserWarning, match=self._msg):
            dc.spool(top_level).update()


class TestMisc:
    """Tests for misc. spool cases."""

    def test_chunk_timedelta_coord(self):
        """
        Chunking a spool whose time coordinate is timedelta64 (rather than
        datetime64) should work with a numeric chunk size (see #553).
        """
        patch = ricker_moveout()
        assert np.issubdtype(patch.get_coord("time").dtype, np.timedelta64)
        spool = dc.spool(patch)
        # Previously raised a TypeError comparing Timedelta with float.
        chunked = spool.chunk(time=0.02)
        assert len(chunked) > 1
        # The chunked patches retain the timedelta64 time coordinate.
        assert np.issubdtype(chunked[0].get_coord("time").dtype, np.timedelta64)
        # Overlap (also numeric) must work too.
        assert len(spool.chunk(time=0.02, overlap=0.005)) > 1

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


class TestDeepEqualityCheck:
    """Coverage for deep_equality_check branches (formerly via spool attrs)."""

    def test_non_dict_comparison(self):
        """Plain value comparison inside dicts."""
        assert deep_equality_check({"a": "hello"}, {"a": "hello"})
        assert not deep_equality_check({"a": "hello"}, {"a": "world"})

    def test_objects_with_dict(self):
        """Objects compare via recursive __dict__ comparison."""

        class TestObject:
            def __init__(self, value):
                self.value = value

        assert deep_equality_check({"o": TestObject(42)}, {"o": TestObject(42)})
        assert not deep_equality_check({"o": TestObject(1)}, {"o": TestObject(2)})

    def test_mixed_types(self):
        """Ints, lists, and numpy arrays compare by value."""
        d1 = {"i": 42, "l": [1, 2, 3], "a": np.array([1, 2, 3])}
        d2 = {"i": 42, "l": [1, 2, 3], "a": np.array([1, 2, 3])}
        assert deep_equality_check(d1, d2)
        d2["a"] = np.array([1, 2, 4])
        assert not deep_equality_check(d1, d2)

    def test_dataframes(self):
        """DataFrames compare via .equals."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert deep_equality_check({"df": df1}, {"df": df2})
        df3 = pd.DataFrame({"a": [1, 2, 4], "b": [4, 5, 6]})
        assert not deep_equality_check({"df": df1}, {"df": df3})

    def test_unequal_sub_dicts(self):
        """Nested dicts with different values are unequal."""
        assert not deep_equality_check({"d": {1: 2}}, {"d": {2: 3}})


class TestSpoolCoverageEdges:
    """Cover remaining spool-machinery branches with real operations."""

    @pytest.fixture(scope="class")
    def many_contiguous(self):
        """Twelve contiguous patches (for >10-row merge handling)."""
        t0 = np.datetime64("2020-01-01", "ns")
        patch = dc.get_example_patch(time_min=t0)
        step = patch.get_coord("time").step
        out = [patch]
        for _ in range(11):
            nxt = dc.get_example_patch(time_min=out[-1].get_coord("time").max() + step)
            out.append(nxt)
        return out

    def test_equality_and_repr(self):
        """Spool equality strips synthetic identity; repr shows a time span."""
        patch = dc.get_example_patch()
        left, right = dc.spool([patch]), dc.spool([patch])
        left.get_contents()  # realize so equality compares built frames
        right.get_contents()
        assert left == right
        assert "Time Span" in left.__rich__().__str__()

    def test_equality_of_empty_spools(self):
        """Empty spools (None frames) compare equal via the None-strip path."""
        assert Spool() == Spool()

    def test_repr_without_time_coordinate(self):
        """A spool whose patches have no time coord omits the time-span line."""
        data = np.random.default_rng().random((6, 4))
        coords = {"distance": np.arange(6), "frequency": np.arange(4.0)}
        patch = dc.Patch(data=data, coords=coords, dims=("distance", "frequency"))
        rendered = dc.spool([patch]).__rich__().__str__()
        assert "Spool" in rendered
        assert "Time Span" not in rendered  # no time coordinate to summarize

    def test_repr_with_time_coordinate(self):
        """A normal spool renders its time span."""
        assert "Time Span" in dc.spool([dc.get_example_patch()]).__rich__().__str__()

    def test_large_merge_dedups(self, many_contiguous):
        """Merging >10 sources into one patch exercises the de-dup branch."""
        merged = dc.spool(many_contiguous).chunk(time=None)
        assert len(merged) == 1
        # 12 contiguous patches merge into one continuous coordinate.
        assert merged[0].get_coord("time").size == sum(
            p.get_coord("time").size for p in many_contiguous
        )

    def test_union_of_scanless_spool(self, tmp_path):
        """A scanless (pickle) spool wraps its read patches in a live
        catalog; union shares them like any in-memory member.
        """
        dc.get_example_patch().io.write(tmp_path / "a.pkl", "pickle")
        pickle_spool = dc.spool(tmp_path / "a.pkl")
        combined = pickle_spool + dc.spool([dc.get_example_patch(tag="other")])
        assert len(combined) == 2

    def test_union_of_chunked_spool(self, many_contiguous):
        """A chunked spool is a derived catalog; unions compose it."""
        chunked = dc.spool(many_contiguous).chunk(time=None)
        assert isinstance(chunked._catalog.resolver, PlanResolver)
        combined = chunked + dc.spool([dc.get_example_patch(tag="other")])
        assert len(combined) == 2
        assert all(isinstance(p, dc.Patch) for p in combined)

    def test_iteration_skips_unresolvable_patch(self, monkeypatch):
        """A patch that fails to resolve is skipped with a #583 warning."""
        spool = dc.spool([dc.get_example_patch()])
        # Realize the relation so iteration resolves rows rather than
        # serving the live registry (which cannot fail to resolve).
        spool.get_contents()

        def _raise(*args, **kwargs):
            raise MissingPatchError("not available in this session")

        monkeypatch.setattr(spool._catalog, "resolve_row", _raise)
        with pytest.warns(UserWarning, match="Skipping patch"):
            assert list(spool) == []

    def test_derived_negative_and_bad_index(self):
        """Derived-catalog indexing handles negatives, raises out-of-bounds."""
        patches = list(dc.get_example_spool(length=2))
        derived = dc.spool(patches).concatenate(time=1)
        assert derived[-1] == derived[len(patches) - 1]
        with pytest.raises(IndexError, match="out of bounds"):
            _ = derived[len(patches)]
        with pytest.raises(IndexError, match="out of bounds"):
            _ = derived[-len(patches) - 1]

    def test_merge_buffer_grows_when_estimate_short(self, many_contiguous, monkeypatch):
        """An under-estimated merge buffer is grown to fit (uneven sampling)."""
        # Force the pre-merge sample estimate to be too small so the
        # streaming buffer must grow mid-merge.
        monkeypatch.setattr(assembly_mod, "_estimate_merge_samples", lambda *a, **k: 1)
        merged = dc.spool(many_contiguous).chunk(time=None)
        assert merged[0].get_coord("time").size == sum(
            p.get_coord("time").size for p in many_contiguous
        )

    def test_empty_memory_spool_len_iter_repr(self):
        """A bare Spool() is a valid empty spool."""
        empty = Spool()
        assert len(empty) == 0
        assert list(empty) == []
        assert "Spool" in str(empty)


class TestConcatenatePartitions:
    """Spool.concatenate partitions and polices as chunk does, then joins in order."""

    @pytest.fixture
    def pair(self):
        """Two same-kind patches, the second following the first in time."""
        first = dc.get_example_patch()
        time = first.get_coord("time")
        return first, first.update_coords(time_min=time.max() + time.step)

    def test_different_kinds_land_in_separate_outputs(self, pair):
        """Kind partitions; nothing is skipped and nothing raises."""
        first, other = pair
        other = other.update_attrs(tag="other")
        with suppress_warnings(action="error"):
            out = dc.spool([first, other]).concatenate(time=None)
        assert len(out) == 2
        assert sorted(out.get_contents()["tag"]) == sorted([first.attrs.tag, "other"])
        for patch, row in zip(out, out.get_contents().to_dict("records")):
            assert patch.attrs.tag == row["tag"]
            assert patch.shape == first.shape

    def test_check_behavior_is_deprecated(self, pair):
        """The old skip-or-raise knob warns and changes nothing."""
        first, other = pair
        mixed = [first, other.update_attrs(tag="other")]
        with pytest.warns(DeprecationWarning, match="separate outputs"):
            out = dc.spool(mixed).concatenate(time=None, check_behavior="raise")
        assert len(out) == 2  # neither skipped nor refused
        # ... spells None, as it does for chunk
        assert len(dc.spool(pair).concatenate(time=...)) == 1

    def test_missing_kind_value_filled_in_rows_and_patch(self, pair):
        """A member lacking the key matches, and the output carries the key."""
        first, other = pair
        other = other.update_attrs(acquisition_key="XX.R2D1..RAW")
        out = dc.spool([first, other]).concatenate(time=None)
        assert len(out) == 1
        assert out.get_contents()["acquisition_key"].iloc[0] == "XX.R2D1..RAW"
        assert out[0].attrs.acquisition_key == "XX.R2D1..RAW"
        assert out[0].shape[1] == 2 * first.shape[1]

    def test_data_units_policed_by_conflict(self, pair):
        """Known, different units conflict as in chunk; a missing one matches."""
        first, other = pair
        metres, km = first.set_units("m"), other.set_units("km")
        with pytest.raises(CoordMergeError, match="data_units"):
            dc.spool([metres, km]).concatenate(time=None)
        dropped = dc.spool([metres, km]).concatenate(time=None, conflict="drop")
        assert len(dropped) == 1
        assert dropped[0].attrs.data_units is None
        assert (
            dc.get_quantity(dropped.get_contents().get("data_units", [None])[0]) is None
        )
        bare = first.set_units(None)
        out = dc.spool([bare, km]).concatenate(time=None)
        assert len(out) == 1
        assert dc.get_quantity(out[0].attrs.data_units) == dc.get_quantity("km")
        # a missing unit never hides a conflict between two known ones
        time = other.get_coord("time")
        third = first.update_coords(time_min=time.max() + time.step).set_units("m")
        with pytest.raises(CoordMergeError, match="data_units"):
            dc.spool([bare, km, third]).concatenate(time=None)
        assert dc.get_quantity(out.get_contents()["data_units"].iloc[0]) == (
            dc.get_quantity("km")
        )

    def test_different_dims_partition(self, pair):
        """Other dimensions are another partition, not an error."""
        first, other = pair
        other = other.rename_coords(time="money")
        out = dc.spool([first, other]).concatenate(time=None)
        assert len(out) == 2

    def test_auxiliary_coordinates_policed_by_conflict(self, pair):
        """Non-dimensional coordinates are settled at assembly, as in chunk."""
        first, other = pair
        n = first.shape[first.get_axis("distance")]
        lat_a = first.update_coords(latitude=("distance", np.arange(n, dtype=float)))
        lat_b = other.update_coords(latitude=("distance", np.ones(n)))
        lat_a2 = other.update_coords(latitude=("distance", np.arange(n, dtype=float)))
        joined = dc.spool([lat_a, lat_a2]).concatenate(time=None)
        assert len(joined) == 1
        assert joined[0].shape[1] == 2 * first.shape[1]
        assert "latitude" in joined[0].coords.coord_map
        with pytest.raises(CoordMergeError, match="latitude"):
            dc.spool([lat_a, lat_b]).concatenate(time=None)[0]
        dropped = dc.spool([lat_a, lat_b]).concatenate(time=None, conflict="drop")[0]
        assert "latitude" not in dropped.coords.coord_map
        # values the envelope cannot tell apart are settled when assembling
        shuffled = np.arange(n, dtype=float)
        shuffled[1], shuffled[2] = 2.0, 1.0
        lat_c = other.update_coords(latitude=("distance", shuffled))
        planned = dc.spool([lat_a, lat_c]).concatenate(time=None)
        assert len(planned) == 1
        with pytest.raises(CoordMergeError, match="latitude"):
            planned[0]
        dropped = dc.spool([lat_a, lat_c]).concatenate(time=None, conflict="drop")[0]
        assert "latitude" not in dropped.coords.coord_map
        # a coordinate only one patch has rides along from the first member
        assert len(dc.spool([lat_a, other]).concatenate(time=None)) == 1

    def test_count_groups_within_partitions(self, pair):
        """Each partition's patches are grouped in order by the count."""
        first, other = pair
        time = other.get_coord("time")
        third = first.update_coords(time_min=time.max() + time.step)
        stranger = other.update_attrs(tag="other")
        out = dc.spool([first, stranger, third]).concatenate(time=2)
        assert len(out) == 2
        kinds = {(p.attrs.tag, p.shape[1]) for p in out}
        expected = {(first.attrs.tag, 2 * first.shape[1]), ("other", first.shape[1])}
        assert kinds == expected

    def test_each_output_carries_its_own_members(self, pair):
        """A value known only in another output is not borrowed."""
        first, other = pair
        blank, tagged = first.update_attrs(tag=""), other.update_attrs(tag="a")
        out = dc.spool([blank, tagged]).concatenate(time=1)
        assert [p.attrs.tag for p in out] == ["", "a"]
        assert list(out.get_contents()["tag"].fillna("")) == ["", "a"]

    def test_order_follows_the_dimension(self, pair):
        """Within a partition, patches join in order of the dimension."""
        first, other = pair
        patch = dc.spool([other, first]).concatenate(time=None)[0]
        coord = patch.get_coord("time")
        assert coord.min() == first.get_coord("time").min()
        assert np.all(np.diff(coord.values.astype("int64")) > 0)

    def test_selection_before_concatenate_stays_apart(self):
        """Coordinates a pending selection might reconcile are not guessed."""
        base = dc.get_example_patch()
        time = base.get_coord("time")
        shifted = base.update_coords(time_min=time.max() + time.step, distance_min=5)
        selected = dc.spool([base, shifted]).select(distance=(5, 299))
        with suppress_warnings(action="error"):
            out = selected.concatenate(time=None)
        assert len(out) == 2
        # loading the selected patches settles their coordinates
        loaded = dc.spool(list(selected)).concatenate(time=None)
        assert len(loaded) == 1
        assert loaded[0].shape[1] == 2 * base.shape[1]

    def test_plan_keeps_its_kind_rule(self, pair):
        """A plan made under one kind rule assembles under it later."""
        first, other = pair
        other = other.update_attrs(tag="other")
        # with tag out of the kind, it is an ordinary attr the policy folds
        with dc.config_context(patch_kind_attrs=("acquisition_key",)):
            out = dc.spool([first, other]).concatenate(time=None, conflict="keep_first")
            assert len(out) == 1
        assert out[0].shape[1] == 2 * first.shape[1]
        assert out[0].attrs.tag == first.attrs.tag

    def test_positional_check_behavior_still_works(self, pair):
        """The deprecated argument keeps its old positional slot."""
        with pytest.warns(DeprecationWarning, match="separate outputs"):
            out = dc.spool(pair).concatenate("warn", time=None)
        assert len(out) == 1

    def test_convertible_units_plan_together(self, pair):
        """Metres and centimetres along the dimension are one partition."""
        first, _ = pair
        distance = first.get_coord("distance")
        shifted = first.update_coords(distance_min=distance.max() + distance.step)
        in_cm = shifted.convert_units(distance="cm")
        out = dc.spool([first, in_cm]).concatenate(distance=None)
        assert len(out) == 1
        patch = out[0]
        assert patch.shape[0] == 2 * first.shape[0]
        assert np.all(np.diff(patch.get_coord("distance").values) > 0)

    def test_descending_data_keep_their_step(self, pair):
        """Contiguous descending members concatenate in their own direction."""
        first, _ = pair
        n = first.shape[first.get_axis("distance")]
        high = first.update_coords(distance=np.arange(2 * n - 1, n - 1, -1.0))
        low = first.update_coords(distance=np.arange(n - 1, -1, -1.0))
        out = dc.spool([low, high]).concatenate(distance=None)
        coord = out[0].get_coord("distance")
        assert coord.max() == 2 * n - 1 and coord.min() == 0
        assert np.all(np.diff(coord.values) < 0)
        assert out.get_contents()["distance_step"].iloc[0] == -1

    def test_auxiliary_name_becomes_a_dimension(self, pair):
        """Concatenating along a non-dimensional coordinate's name adds a dimension."""
        first, _ = pair
        n = first.shape[first.get_axis("distance")]
        aux = first.update_coords(sensor=("distance", np.arange(n, dtype=float)))
        out = dc.spool([aux, aux.new()]).concatenate(sensor=None)
        patch = out[0]
        assert "sensor" in patch.dims
        assert patch.shape[patch.get_axis("sensor")] == 2
        row = out.get_contents().iloc[0]
        assert "sensor" in str(row["dims"]).split(",")
        # a dimension without values claims no envelope
        assert "sensor_min" not in row.index or pd.isnull(row["sensor_min"])

    def test_dropped_coordinate_leaves_the_catalog(self, pair):
        """A coordinate dropped for conflicting is not advertised either."""
        first, other = pair
        n = first.shape[first.get_axis("distance")]
        lat_a = first.update_coords(latitude=("distance", np.arange(n, dtype=float)))
        lat_b = other.update_coords(latitude=("distance", np.ones(n)))
        out = dc.spool([lat_a, lat_b]).concatenate(time=None, conflict="drop")
        contents = out.get_contents()
        assert "latitude" not in out[0].coords.coord_map
        assert "latitude_min" not in contents.columns or pd.isnull(
            contents["latitude_min"].iloc[0]
        )

    def test_string_dimension_keeps_input_order(self):
        """Labels have no orientation, so a string dimension keeps spool order."""
        base = dc.get_example_patch()
        data = base.data[:2]
        coords = {"station": np.array(["c", "d"]), "time": base.get_coord("time")}
        later = dc.Patch(data=data, coords=coords, dims=("station", "time"))
        early = later.update_coords(station=np.array(["a", "b"]))
        out = dc.spool([later, early]).concatenate(station=None)
        assert list(out[0].get_coord("station").values) == ["c", "d", "a", "b"]
        assert "station_step" not in out.get_contents().columns or pd.isnull(
            out.get_contents()["station_step"].iloc[0]
        )

    def test_irregular_descending_data_keep_input_order(self):
        """Without a step the envelopes cannot tell orientation; order is kept."""
        base = dc.get_example_patch()
        n = base.shape[base.get_axis("distance")]
        rng = np.random.default_rng(0)
        gaps = np.sort(rng.uniform(0.5, 1.5, n))[::-1].cumsum()
        high = base.update_coords(distance=2 * n + 10 - gaps)
        low = base.update_coords(distance=-gaps)
        out = dc.spool([high, low]).concatenate(distance=None)
        values = out[0].get_coord("distance").values
        assert np.all(np.diff(values) < 0)
        assert np.allclose(values[:n], high.get_coord("distance").values)

    def test_new_dimension_over_auxiliary_name_keeps_input_order(self, pair):
        """The vanishing coordinate's values do not order the new dimension."""
        first, _ = pair
        n = first.shape[first.get_axis("distance")]
        high = first.update_coords(sensor=("distance", np.arange(n) + 100.0))
        low = first.update_coords(sensor=("distance", np.arange(n) * 1.0))
        out = dc.spool([high, low]).concatenate(sensor=None)
        patch = out[0]
        axis = patch.get_axis("sensor")
        assert np.allclose(np.take(patch.data, 0, axis=axis), high.data)
        assert np.allclose(np.take(patch.data, 1, axis=axis), low.data)

    def test_later_only_coordinate_rides_along_unless_it_rides_the_dimension(
        self, pair
    ):
        """The catalog and the patch agree on a coordinate the first member lacks."""
        first, other = pair
        nd = first.shape[first.get_axis("distance")]
        nt = first.shape[first.get_axis("time")]
        lat = other.update_coords(latitude=("distance", np.arange(nd) * 1.0))
        out = dc.spool([first, lat]).concatenate(time=None)
        assert "latitude" in out[0].coords.coord_map
        assert out.get_contents()["latitude_min"].notna().all()
        clock = other.update_coords(clock=("time", np.arange(nt) * 1.0))
        out = dc.spool([first, clock]).concatenate(time=None)
        assert "clock" not in out[0].coords.coord_map
        contents = out.get_contents()
        assert "clock_min" not in contents.columns or contents["clock_min"].isna().all()
        # later members which disagree are a conflict, first member or not
        lat2 = other.update_coords(latitude=("distance", np.ones(nd)))
        with pytest.raises(CoordMergeError, match="latitude"):
            dc.spool([first, lat, lat2]).concatenate(time=None)[0]
        dropped = dc.spool([first, lat, lat2]).concatenate(time=None, conflict="drop")
        assert "latitude" not in dropped[0].coords.coord_map

    def test_nanosecond_neighbours_keep_dimension_order(self):
        """Starts a float64 cannot tell apart still order the members."""
        base = dc.get_example_patch().select(time=(0, 10), samples=True)
        t0 = np.datetime64("2020-01-01T00:00:00.000000000")
        ticks = np.arange(10) * np.timedelta64(1, "ns")
        early = base.update_coords(time=t0 + ticks)
        late = base.update_coords(time=t0 + np.timedelta64(10, "ns") + ticks)
        out = dc.spool([late, early]).concatenate(time=None)
        values = out[0].get_coord("time").values
        assert values[0] == t0
        assert np.all(np.diff(values) > np.timedelta64(0, "ns"))

    def test_attrs_named_for_a_new_dimension_are_refused(self, pair):
        """An attr spelled like the new dimension's envelope cannot survive it."""
        first, _ = pair
        named = first.update_attrs(batch_step=3)
        with pytest.raises(ParameterError, match="batch_step"):
            dc.spool([named, named.new()]).concatenate(batch=None)

    def test_implicit_coordinate_drop_is_lossy(self, pair):
        """Re-planning a view which lost a coordinate does not resurrect it."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        clock = other.update_coords(clock=("time", np.arange(nt) * 1.0))
        out = dc.spool([first, clock]).concatenate(time=None)
        again = out.concatenate(time=1)
        assert len(again) == 1
        assert "clock" not in again[0].coords.coord_map

    def test_value_kinds_partition(self, pair):
        """A datetime coordinate and a numeric one of the same name stay apart."""
        first, _ = pair
        dated = first.rename_coords(time="epoch")
        nt = dated.shape[dated.get_axis("epoch")]
        numeric = dated.update_coords(epoch=np.arange(nt) * 1.0)
        out = dc.spool([dated, numeric]).concatenate(epoch=None)
        assert len(out) == 2
        kinds = {x.get_coord("epoch").values.dtype.kind for x in out}
        assert kinds == {"M", "f"}

    def test_coordinate_riding_the_dimension_is_joined(self, pair):
        """A coordinate every member carries along the dimension follows it."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        a = first.update_coords(clock=("time", np.arange(nt) * 1.0))
        b = other.update_coords(clock=("time", np.arange(nt) * 1.0 + nt))
        out = dc.spool([a, b]).concatenate(time=None)
        patch = out[0]
        assert patch.coords.dim_map["clock"] == ("time",)
        assert np.array_equal(patch.get_array("clock"), np.arange(2 * nt) * 1.0)
        assert out.get_contents()["clock_max"].iloc[0] == 2 * nt - 1
        direct = dc.utils.patch.concatenate_patches([a, b], time=None)[0]
        assert np.array_equal(direct.get_array("clock"), np.arange(2 * nt) * 1.0)

    def test_first_row_without_step_keeps_input_order(self):
        """A later row's step does not lend the partition an orientation."""
        base = dc.get_example_patch()
        n = base.shape[base.get_axis("distance")]
        rng = np.random.default_rng(1)
        gaps = np.sort(rng.uniform(0.5, 1.5, n))[::-1].cumsum()
        irregular = base.update_coords(distance=2 * n + 10 - gaps)
        regular = base.update_coords(distance=np.arange(n)[::-1] * 1.0)
        out = dc.spool([irregular, regular]).concatenate(distance=None)
        values = out[0].get_coord("distance").values
        assert np.allclose(values[:n], irregular.get_coord("distance").values)
        assert np.all(np.diff(values) < 0)

    def test_coordinate_attached_differently_conflicts(self):
        """Equal values on different dimensions are different coordinates."""
        base = dc.get_example_patch()
        square = base.select(distance=(0, 50), samples=True)
        square = square.select(time=(0, 50), samples=True)
        t = square.get_coord("time")
        later = square.update_coords(time_min=t.max() + t.step)
        a = square.update_coords(quality=("distance", np.arange(50) * 1.0))
        b = later.update_coords(quality=("time", np.arange(50) * 1.0))
        for conflict in ("raise", "drop"):
            sp = dc.spool([a, b]).concatenate(time=None, conflict=conflict)
            with pytest.raises(CoordMergeError, match="quality"):
                sp[0]
        with pytest.raises(IncompatiblePatchError, match="quality"):
            dc.utils.patch.concatenate_patches(
                [a, b], time=None, check_behavior="raise"
            )

    def test_numeric_units_normalize_beside_another_kind(self, pair):
        """A text coordinate of the same name does not stop metres meeting cm."""
        first, _ = pair
        distance = first.get_coord("distance")
        shifted = first.update_coords(distance_min=distance.max() + distance.step)
        metres = first.rename_coords(distance="range")
        cm = shifted.convert_units(distance="cm").rename_coords(distance="range")
        n = first.shape[first.get_axis("distance")]
        labels = np.array([f"s{i:03d}" for i in range(n)])
        text = first.update_coords(distance=labels).rename_coords(distance="range")
        out = dc.spool([metres, cm, text]).concatenate(range=None)
        assert len(out) == 2
        joined = [x for x in out if x.shape[x.get_axis("range")] == 2 * n]
        assert len(joined) == 1

    def test_rider_only_on_the_first_is_left_out_directly(self, pair):
        """The direct function drops a rider a later patch lacks, as the plan does."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        a = first.update_coords(clock=("time", np.arange(nt) * 1.0))
        out = dc.utils.patch.concatenate_patches([a, other], time=None)[0]
        assert "clock" not in out.coords.coord_map
        assert out.shape[out.get_axis("time")] == 2 * nt

    def test_rider_units_survive(self, pair):
        """A unitful rider keeps its units; members in other spellings convert."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        a = first.update_coords(clock=("time", np.arange(nt) * 1.0))
        a = a.convert_units(clock="s")
        b = other.update_coords(clock=("time", (np.arange(nt) + nt) * 1000.0))
        b = b.convert_units(clock="ms")
        out = dc.spool([a, b]).concatenate(time=None)[0]
        clock = out.get_coord("clock")
        assert clock.units == dc.get_quantity("s")
        assert np.allclose(clock.values, np.arange(2 * nt))
        # a unitless first member adopts the first stated units
        bare = first.update_coords(clock=("time", np.arange(nt) * 1.0))
        out = dc.spool([bare, b]).concatenate(time=None)[0]
        clock = out.get_coord("clock")
        assert clock.units == dc.get_quantity("ms")
        expected = np.concatenate([np.arange(nt), (np.arange(nt) + nt) * 1000.0])
        assert np.allclose(clock.values, expected)

    def test_riders_survive_non_raise_policies(self, pair):
        """Differing rider identities are joined, not dropped, under drop."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        a = first.update_coords(clock=("time", np.arange(nt) * 1.0))
        b = other.update_coords(clock=("time", np.arange(nt) * 1.0 + nt))
        for conflict in ("drop", "keep_first"):
            out = dc.spool([a, b]).concatenate(time=None, conflict=conflict)
            assert "clock" in out[0].coords.coord_map
            assert out.get_contents()["clock_max"].iloc[0] == 2 * nt - 1

    def test_rider_catalog_spelling_matches_assembly(self, pair):
        """The catalog describes a joined rider in the spelling it is joined in."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        a = first.update_coords(clock=("time", np.arange(nt) * 1.0)).convert_units(
            clock="s"
        )
        b = other.update_coords(
            clock=("time", (np.arange(nt) + nt) * 1000.0)
        ).convert_units(clock="ms")
        out = dc.spool([a, b]).concatenate(time=None)
        row = out.get_contents().iloc[0]
        assert row["clock_max"] == 2 * nt - 1
        assert dc.get_quantity(row["clock_units"]) == dc.get_quantity("s")
        assert out[0].get_coord("clock").units == dc.get_quantity("s")

    def test_rider_spelling_chosen_per_output(self, pair):
        """Each output picks its own rider spelling, in catalog and patch alike."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        ticks = np.arange(nt) * 1.0
        s_then_ms = [
            first.update_coords(clock=("time", ticks)).convert_units(clock="s"),
            other.update_coords(clock=("time", (ticks + nt) * 1000)).convert_units(
                clock="ms"
            ),
        ]
        ms_then_s = [
            x.update_attrs(tag="b")
            for x in (
                first.update_coords(clock=("time", ticks * 1000)).convert_units(
                    clock="ms"
                ),
                other.update_coords(clock=("time", ticks + nt)).convert_units(
                    clock="s"
                ),
            )
        ]
        out = dc.spool([*s_then_ms, *ms_then_s]).concatenate(time=None)
        assert len(out) == 2
        contents = out.get_contents().set_index("tag")
        for tag, expected in (("random", "s"), ("b", "ms")):
            patch = out.select(tag=tag)[0]
            assert patch.get_coord("clock").units == dc.get_quantity(expected)
            row = contents.loc[tag]
            assert dc.get_quantity(row["clock_units"]) == dc.get_quantity(expected)
            assert row["clock_max"] == patch.get_coord("clock").max()

    def test_incompatible_rider_units_follow_the_policy(self, pair):
        """Seconds beside metres cannot join: dropped under drop, refused otherwise."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        ticks = np.arange(nt) * 1.0
        a = first.update_coords(clock=("time", ticks)).convert_units(clock="s")
        b = other.update_coords(clock=("time", ticks + nt)).convert_units(clock="m")
        dropped = dc.spool([a, b]).concatenate(time=None, conflict="drop")
        assert "clock" not in dropped[0].coords.coord_map
        assert "clock_max" not in dropped.get_contents().columns or pd.isnull(
            dropped.get_contents()["clock_max"].iloc[0]
        )
        with pytest.raises(CoordMergeError, match="clock"):
            dc.spool([a, b]).concatenate(time=None)[0]

    def test_direct_concatenation_picks_the_lowest_spelling(self, pair):
        """Without a plan, the rider joins in the units of its lowest member."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        ticks = np.arange(nt) * 1.0
        low = first.update_coords(clock=("time", ticks)).convert_units(clock="s")
        high = other.update_coords(clock=("time", (ticks + nt) * 1000)).convert_units(
            clock="ms"
        )
        out = dc.utils.patch.concatenate_patches([high, low], time=None)[0]
        clock = out.get_coord("clock")
        assert clock.units == dc.get_quantity("s")
        assert np.allclose(np.sort(clock.values), np.arange(2 * nt))

    def test_contiguous_rider_keeps_its_step(self, pair):
        """A rider whose segments meet end to end is a range in the catalog too."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        ticks = np.arange(nt) * 1.0
        a = first.update_coords(clock=("time", ticks))
        b = other.update_coords(clock=("time", ticks + nt))
        out = dc.spool([a, b]).concatenate(time=None)
        assert out.get_contents()["clock_step"].iloc[0] == 1.0
        assert out[0].get_coord("clock").step == 1.0
        # a gap between the segments leaves the joined coordinate irregular
        far = other.update_coords(clock=("time", ticks + 10 * nt))
        gapped = dc.spool([a, far]).concatenate(time=None)
        assert pd.isnull(gapped.get_contents()["clock_step"].iloc[0])
        assert gapped[0].get_coord("clock").step is None

    def test_drop_settles_incomparable_coordinates(self, pair):
        """A coordinate numeric on one member and text on another drops cleanly."""
        first, other = pair
        n = first.shape[first.get_axis("distance")]
        numbers = first.update_coords(latitude=("distance", np.arange(n) * 1.0))
        labels = np.array([f"p{i:03d}" for i in range(n)])
        text = other.update_coords(latitude=("distance", labels))
        out = dc.spool([numbers, text]).concatenate(time=None, conflict="drop")
        assert len(out) == 1
        assert "latitude" not in out[0].coords.coord_map
        contents = out.get_contents()
        assert "latitude_min" not in contents.columns or pd.isnull(
            contents["latitude_min"].iloc[0]
        )

    def test_rider_units_normalize_beside_another_kind(self, pair):
        """A text rider elsewhere does not stop seconds meeting milliseconds."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        ticks = np.arange(nt) * 1.0
        a = first.update_coords(clock=("time", ticks)).convert_units(clock="s")
        b = other.update_coords(clock=("time", (ticks + nt) * 1000)).convert_units(
            clock="ms"
        )
        words = np.array([f"t{i:04d}" for i in range(nt)])
        c = first.update_attrs(tag="b").update_coords(clock=("time", words))
        d = other.update_attrs(tag="b").update_coords(
            clock=("time", np.array([f"u{i:04d}" for i in range(nt)]))
        )
        out = dc.spool([a, b, c, d]).concatenate(time=None)
        row = out.get_contents().set_index("tag").loc["random"]
        assert dc.get_quantity(row["clock_units"]) == dc.get_quantity("s")
        assert row["clock_max"] == 2 * nt - 1
        assert out.select(tag="random")[0].get_coord("clock").units == dc.get_quantity(
            "s"
        )

    def test_keep_first_converts_the_data_it_labels(self, pair):
        """Keeping the first data units converts the members stated otherwise."""
        first, other = pair
        metres = first.set_units("m")
        km = other.set_units("km")
        out = dc.spool([metres, km]).concatenate(time=None, conflict="keep_first")[0]
        assert out.attrs.data_units == dc.get_quantity("m")
        nt = first.shape[first.get_axis("time")]
        assert np.allclose(out.data[:, nt:], km.data * 1000)
        assert np.allclose(out.data[:, :nt], metres.data)

    def test_rider_of_two_kinds_still_concatenates(self, pair):
        """Numeric and text rider segments join, the catalog claiming no envelope."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        a = first.update_coords(clock=("time", np.arange(nt) * 1.0))
        words = np.array([f"t{i:04d}" for i in range(nt)])
        b = other.update_coords(clock=("time", words))
        out = dc.spool([a, b]).concatenate(time=None)
        contents = out.get_contents()
        assert "clock_min" not in contents.columns or pd.isnull(
            contents["clock_min"].iloc[0]
        )
        assert out[0].get_array("clock").shape == (2 * nt,)

    def test_descending_rider_keeps_its_sign(self, pair):
        """A descending rider is described as descending, not as ascending."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        ticks = np.arange(nt) * 1.0
        a = first.update_coords(clock=("time", (2 * nt - 1) - ticks))
        b = other.update_coords(clock=("time", (nt - 1) - ticks))
        out = dc.spool([a, b]).concatenate(time=None)
        assert out.get_contents()["clock_step"].iloc[0] == -1.0
        assert out[0].get_coord("clock").step == -1.0

    def test_all_null_rider_is_not_partial(self, pair):
        """A rider every member states, though all its values are NaN, stays."""
        first, other = pair
        nt = first.shape[first.get_axis("time")]
        a = first.update_coords(clock=("time", np.full(nt, np.nan)))
        b = other.update_coords(clock=("time", np.full(nt, np.nan)))
        out = dc.spool([a, b]).concatenate(time=None)
        assert "clock" in out[0].coords.coord_map
        contents = out.get_contents()
        assert (
            "_clock_def_key" not in contents or contents["_clock_def_key"].notna().all()
        )

    def test_stack_refuses_differing_riders(self, pair):
        """Stacking keeps one coordinate manager, so riders must agree."""
        first, _ = pair
        nt = first.shape[first.get_axis("time")]
        a = first.update_coords(clock=("time", np.arange(nt) * 1.0))
        b = first.update_coords(clock=("time", np.arange(nt) * 2.0))
        with pytest.raises(IncompatiblePatchError, match="clock"):
            dc.utils.patch.stack_patches(
                [a, b], dim_vary="time", check_behavior="raise"
            )

    def test_dimensional_envelope_kept_beside_an_auxiliary_role(self, pair):
        """A name auxiliary elsewhere keeps its envelope where it is a dimension."""
        first, other = pair
        n = first.shape[first.get_axis("distance")]
        as_dim = first.rename_coords(distance="sensor")
        as_dim2 = other.rename_coords(distance="sensor")
        aux = first.update_coords(sensor=("distance", np.arange(n) * 1.0))
        out = dc.spool([as_dim, as_dim2, aux]).concatenate(time=None)
        contents = out.get_contents()
        dimensional = contents[contents["dims"].str.contains("sensor")]
        assert len(dimensional) == 1
        assert dimensional["sensor_min"].notna().all()
        assert dimensional["sensor_max"].iloc[0] == n - 1

    def test_vanishing_coordinate_kinds_do_not_partition(self, pair):
        """Auxiliaries the new dimension replaces do not split it by kind."""
        first, _ = pair
        n = first.shape[first.get_axis("distance")]
        numbers = first.update_coords(sensor=("distance", np.arange(n) * 1.0))
        labels = np.array([f"s{i}" for i in range(n)])
        text = first.update_coords(sensor=("distance", labels))
        out = dc.spool([numbers, text]).concatenate(sensor=None)
        assert len(out) == 1

    def test_numeric_order_beside_a_text_partition(self, pair):
        """Numbers rank as numbers even when labels share the dimension name."""
        first, _ = pair
        n = first.shape[first.get_axis("distance")]
        two = first.update_coords(distance=np.arange(n) + 2.0).rename_coords(
            distance="range"
        )
        ten = first.update_coords(distance=np.arange(n) + 2.0 + n).rename_coords(
            distance="range"
        )
        labels = np.array([f"s{i:03d}" for i in range(n)])
        text = first.update_coords(distance=labels).rename_coords(distance="range")
        out = dc.spool([ten, two, text]).concatenate(range=None)
        numeric = [x for x in out if x.get_coord("range").values.dtype.kind == "f"]
        assert len(numeric) == 1
        values = numeric[0].get_coord("range").values
        assert values[0] == 2.0 and np.all(np.diff(values) > 0)
        contents = out.get_contents()
        assert contents["range_step"].notna().sum() == 1

    def test_created_dimension_keeps_its_length_apart(self, pair):
        """Outputs of two and of one member do not merge along another dimension."""
        first, _ = pair
        trio = [first, first.new(), first.new()]
        ranked = dc.spool(trio).concatenate(rank=2)
        assert len(ranked) == 2
        again = ranked.concatenate(distance=None)
        assert len(again) == 2
        assert sorted(x.shape[x.get_axis("rank")] for x in again) == [1, 2]

    def test_vanishing_coordinate_units_do_not_partition(self, pair):
        """Auxiliary coordinates the new dimension replaces do not split by units."""
        first, _ = pair
        n = first.shape[first.get_axis("distance")]
        metres = first.update_coords(sensor=("distance", np.arange(n) * 1.0))
        metres = metres.convert_units(sensor="m")
        seconds = first.update_coords(sensor=("distance", np.arange(n) * 1.0))
        seconds = seconds.convert_units(sensor="s")
        out = dc.spool([metres, seconds]).concatenate(sensor=None)
        assert len(out) == 1
        assert out[0].shape[out[0].get_axis("sensor")] == 2

    def test_coordinate_identity_partitions_only_where_dimensional(self, pair):
        """A name dimensional in one patch is still auxiliary in the others."""
        first, other = pair
        n = first.shape[first.get_axis("distance")]
        aux_a = first.update_coords(sensor=("distance", np.arange(n) * 1.0))
        aux_b = other.update_coords(sensor=("distance", np.ones(n)))
        as_dim = first.rename_coords(distance="sensor")
        out = dc.spool([aux_a, aux_b, as_dim]).concatenate(time=None, conflict="drop")
        assert len(out) == 2
        joined = out[0]
        assert joined.shape[joined.get_axis("time")] == 2 * first.shape[1]
        assert "sensor" not in joined.coords.coord_map
        assert out[1].dims == as_dim.dims

    def test_singleton_outputs_keep_weak_coordinates(self, pair):
        """An output of one member has nothing to conflict with, so nothing drops."""
        first, _ = pair
        n = first.shape[first.get_axis("distance")]
        irregular = np.arange(n, dtype=float)
        irregular[1], irregular[2] = 2.0, 1.0
        aux = first.update_coords(latitude=("distance", irregular))
        out = dc.spool([aux, aux.new()]).concatenate(time=1, conflict="drop")
        assert len(out) == 2
        assert all("latitude" in x.coords.coord_map for x in out)

    def test_replanning_after_a_drop_keeps_the_drop(self, pair):
        """Concatenating again along the same dimension does not resurrect metadata."""
        first, other = pair
        metres, km = first.set_units("m"), other.set_units("km")
        dropped = dc.spool([metres, km]).concatenate(time=None, conflict="drop")
        again = dropped.concatenate(time=None)
        assert len(again) == 1
        assert again[0].attrs.data_units is None

    def test_conflict_policy_is_part_of_the_identity(self, pair):
        """Drop and keep_first give different patches, so different ids."""
        first, other = pair
        a = first.update_attrs(foo="a")
        b = other.update_attrs(foo="b")
        kept = dc.spool([a, b]).concatenate(time=None, conflict="keep_first")[0]
        dropped = dc.spool([a, b]).concatenate(time=None, conflict="drop")[0]
        assert kept.attrs.processing_id != dropped.attrs.processing_id

    def test_new_dimension(self, pair):
        """A dimension no patch has is added, one sample per patch."""
        first, _ = pair
        # along a new dimension the existing coordinates must agree
        out = dc.spool([first, first.new()]).concatenate(wave_rank=None)
        assert len(out) == 1
        patch = out[0]
        assert "wave_rank" in patch.dims
        assert patch.shape[patch.get_axis("wave_rank")] == 2
        # patches whose time ranges differ cannot share a new dimension
        assert len(dc.spool(pair).concatenate(wave_rank=None)) == 2


class TestEmptyConcatenate:
    """Concatenating an empty spool returns an empty spool (F7)."""

    @pytest.mark.parametrize("kwargs", [{"time": None}, {"time": 2}, {"new_dim": None}])
    def test_empty_returns_empty(self, kwargs):
        """Empty in, empty out — matching chunk's behavior."""
        out = dc.spool([]).concatenate(**kwargs)
        assert len(out) == 0
        assert list(out) == []
