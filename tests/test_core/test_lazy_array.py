"""Tests for LazyArray, the recipe for an array read a block at a time."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.lazy_array import (
    AXIS_FIELDS,
    NEW_AXIS,
    LazyArray,
    LazyTable,
    concat,
    stack,
)
from dascore.core.source import ArraySource
from dascore.exceptions import ParameterError
from dascore.io import core as io_core
from dascore.utils.array_api import backend_name
from dascore.utils.downloader import fetch
from dascore.utils.identity import H


@pytest.fixture(scope="module")
def path():
    """A DASDAE file holding one two dimensional patch."""
    return fetch("example_dasdae_event_1.h5")


@pytest.fixture(scope="module")
def patch(path):
    """The patch read from the file; its axes have different lengths."""
    return dc.read(path)[0]


@pytest.fixture(scope="module")
def source(patch):
    """The source the framework gave the patch."""
    return patch._source


@pytest.fixture(scope="module")
def lazy(source):
    """The whole of the stored array, as one member."""
    return LazyArray.from_source(source)


@pytest.fixture(scope="module")
def two_files(patch, tmp_path_factory):
    """Two DASDAE files holding windows of different lengths."""
    directory = tmp_path_factory.mktemp("lazy_array")
    dim = patch.dims[0]
    out = []
    for index, span in enumerate(((0, 100), (100, None))):
        window = patch.select(**{dim: span}, samples=True)
        path = directory / f"part_{index}.h5"
        dc.write(window, path, "dasdae")
        out.append(path)
    return out


@pytest.fixture(scope="module")
def two_sources(two_files):
    """The source of each of the two files."""
    return [dc.read(x)[0]._source for x in two_files]


@pytest.fixture(scope="module")
def joined(two_sources):
    """The two files, joined along their first axis."""
    return LazyArray.from_sources(two_sources)


@pytest.fixture()
def reads(monkeypatch):
    """Count the arrays the loader reads."""
    calls = []
    original = io_core._load_array_source
    monkeypatch.setattr(
        io_core, "_load_array_source", lambda x: calls.append(x) or original(x)
    )
    return calls


@pytest.fixture(scope="module")
def parts():
    """Three arrays of one shape, each a different constant."""
    return [constant((2, 3, 4), float(x)) for x in range(3)]


def constant(shape, value=1.0, dtype=None):
    """Return a lazy array of one constant block."""
    return LazyArray.from_source(ArraySource.full(shape, value, dtype))


def broadcast_array(shape=(2, 3), length=4):
    """Return an array whose first axis is broadcast from a stored one."""
    array = constant(shape)
    frame = stack([array], axis=0).to_frame()
    new = frame["out_axis"] == 0
    frame.loc[new, "out_stop"] = length
    return LazyArray.from_frame(frame, (length, *shape), array.dtype)


class TestDescription:
    """An array knows what it is without reading anything."""

    def test_header(self, lazy, patch):
        """Shape, ndim, dtype and size come from the header, not the members."""
        assert lazy.shape == patch.shape
        assert (lazy.ndim, lazy.size) == (patch.data.ndim, patch.data.size)
        assert lazy.dtype == patch.data.dtype
        assert len(lazy) == 1

    def test_repr(self, lazy):
        """The repr says the shape, dtype and how many members there are."""
        assert "shape" in repr(lazy) and "members" in repr(lazy)

    def test_nothing_read(self, joined, reads):
        """Building, slicing, joining and naming an array open no file."""
        array = concat([joined[0:50], joined[50:601]], axis=0)
        assert array.data_id and array[10:20].shape == (10, 1001)
        assert array.to_frame() is not None
        assert not reads

    def test_empty_slice_keeps_header(self, lazy):
        """A slice which selects nothing is still an array of its own."""
        empty = lazy[0:0]
        assert empty.shape == (0, lazy.shape[1])
        assert empty.dtype == lazy.dtype
        assert len(empty) == 0
        assert empty.load().shape == empty.shape
        assert empty.validate() is empty

    def test_needs_a_source(self):
        """An array is made of at least one source."""
        with pytest.raises(ParameterError, match="at least one source"):
            LazyArray.from_sources([])

    def test_refuses_unloadable(self):
        """A source which describes no array cannot be a member."""
        with pytest.raises(ParameterError, match="does not say enough"):
            LazyArray.from_source(ArraySource(key="a"))

    def test_refuses_scalar(self):
        """An array has at least one axis."""
        source = ArraySource.full((1,), 1.0).describe((), np.float64)
        with pytest.raises(ParameterError, match="at least one dimension"):
            LazyArray.from_source(source)

    def test_refuses_unknown_extent(self):
        """A member must know how big the whole stored array is."""
        source = ArraySource(
            path="a", format="b", windows=((0, 2),), shape=(2,), dtype=np.float64
        )
        with pytest.raises(ParameterError, match="extent"):
            LazyArray.from_source(source)

    def test_refuses_ragged_sources(self):
        """Sources of one array all have the same ndim."""
        sources = [ArraySource.full((2, 2), 1.0), ArraySource.full((2,), 1.0)]
        with pytest.raises(ParameterError, match="same ndim"):
            LazyArray.from_sources(sources)


class TestSlice:
    """Slicing clips the boxes and moves the windows; nothing is read."""

    def test_matches_numpy(self, lazy, patch):
        """A window of the array is the same window of the loaded data."""
        window = lazy[10:20, 5:9]
        assert window.shape == (10, 4)
        assert np.array_equal(window.load(), patch.data[10:20, 5:9])

    def test_twice_is_once(self, joined, two_sources):
        """Slicing twice gives the data and the id of slicing once."""
        twice = joined[100:400][50:100]
        once = joined[150:200]
        assert np.array_equal(twice.load(), once.load())
        assert twice.data_id == once.data_id

    def test_trailing_axes_are_whole(self, lazy):
        """An axis left out of the request is taken whole."""
        assert lazy[0:5].shape == (5, lazy.shape[1])
        assert lazy[0:5, :].shape == (5, lazy.shape[1])

    def test_whole_array_is_the_same_view(self, lazy):
        """A request which clips nothing returns the array itself."""
        assert lazy[:] is lazy
        assert lazy[0 : lazy.shape[0]] is lazy

    def test_negative_bounds(self, lazy, patch):
        """Negative bounds count back from the end, as numpy does."""
        assert np.array_equal(lazy[-5:].load(), patch.data[-5:])

    def test_only_touched_members(self, joined):
        """The result holds the members the request reaches, and no others."""
        assert len(joined) == 2
        assert len(joined[0:10]) == 1
        assert len(joined[90:110]) == 2
        assert len(joined[200:300]) == 1

    def test_scan_when_not_a_stack(self, lazy):
        """An array which is not a stack of blocks is scanned instead."""
        pieces = [constant((2, 3), float(x)) for x in range(4)]
        starts = np.array([[0, 0], [0, 3], [2, 0], [2, 3]])
        grid = LazyArray.from_sources([x.source(0) for x in pieces], starts=starts)
        assert grid.shape == (4, 6)
        assert np.array_equal(grid[0:2].load(), grid.load()[0:2])
        assert len(grid[0:2]) == 2

    @pytest.mark.parametrize("bad", [slice(None, None, 2), 3, [1, 2], Ellipsis, None])
    def test_refuses_other_indexes(self, lazy, bad):
        """Steps, integers, ellipses and index arrays are refused."""
        with pytest.raises(ParameterError, match="step of one"):
            lazy[bad]

    def test_refuses_too_many_axes(self, lazy):
        """An index cannot name more axes than the array has."""
        with pytest.raises(ParameterError, match="dimensional array"):
            lazy[0:1, 0:1, 0:1]


class TestCombine:
    """Arrays are joined in one pass, along old axes and new ones."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_concat_matches_numpy(self, parts, axis):
        """Joining agrees with numpy on every axis, the middle included."""
        joined = concat(parts, axis=axis)
        expected = np.concatenate([x.load() for x in parts], axis=axis)
        assert joined.shape == expected.shape
        assert np.array_equal(joined.load(), expected)
        joined.validate()

    @pytest.mark.parametrize("axis", [0, 1, 2, 3])
    def test_stack_matches_numpy(self, parts, axis):
        """Stacking adds an axis nothing feeds, and agrees with numpy."""
        stacked = stack(parts, axis=axis)
        expected = np.stack([x.load() for x in parts], axis=axis)
        assert stacked.shape == expected.shape
        assert np.array_equal(stacked.load(), expected)
        stacked.validate()
        assert (stacked.placement["src_axis"][:, axis] == NEW_AXIS).all()

    def test_concat_files(self, joined, patch, two_sources):
        """Two files join into the array they were cut from."""
        assert joined.shape == patch.shape
        assert np.array_equal(joined.load(), patch.data)
        assert len(joined.sources) == 2
        assert joined.sources[0].path == str(two_sources[0].path)

    def test_negative_axis(self, parts):
        """An axis may be counted from the end."""
        assert concat(parts, axis=-1).shape == (2, 3, 12)

    def test_promotes_dtype(self):
        """The output takes the dtype every member fits in."""
        parts = [constant((2, 2), 1, np.int32), constant((2, 2), 1.5, np.float32)]
        joined = concat(parts, axis=0)
        assert joined.dtype == np.result_type(np.int32, np.float32)
        assert np.array_equal(joined.load(), np.concatenate([x.load() for x in parts]))

    def test_refuses_mismatched_shapes(self, parts):
        """Arrays which disagree away from the joined axis are refused."""
        with pytest.raises(ParameterError, match="cannot be joined"):
            concat([parts[0], constant((2, 5, 4))], axis=0)

    def test_refuses_mismatched_stack(self, parts):
        """Stacking takes arrays of one shape."""
        with pytest.raises(ParameterError, match="cannot be stacked"):
            stack([parts[0], constant((2, 5, 4))])

    def test_refuses_mixed_ndim(self, parts):
        """Arrays of different ndim cannot be combined."""
        with pytest.raises(ParameterError, match="cannot be combined"):
            concat([parts[0], constant((2, 3))])

    def test_middle_axis_reorders(self):
        """Joining away from the first axis puts the members back in order."""
        pair = [constant((2, 3), 1.0), constant((2, 3), 2.0)]
        stacks = [concat(pair, axis=0), concat(pair, axis=0)]
        joined = concat(stacks, axis=1)
        joined.validate()
        assert np.array_equal(
            joined.load(), np.concatenate([x.load() for x in stacks], axis=1)
        )

    def test_refuses_nothing(self):
        """There must be something to combine."""
        with pytest.raises(ParameterError, match="At least one array"):
            concat([])


class TestTranspose:
    """A member whose axes are swapped still loads the right way up."""

    def test_matches_numpy(self, lazy, patch):
        """The whole array transposes."""
        moved = lazy.transpose()
        assert moved.shape == patch.data.T.shape
        assert np.array_equal(moved.load(), patch.data.T)

    def test_window_of_a_transposed_member(self, lazy, patch):
        """A window of a transposed array reads the right samples."""
        moved = lazy.transpose()[100:110, 3:9]
        assert np.array_equal(moved.load(), patch.data.T[100:110, 3:9])

    def test_explicit_order(self):
        """An order says where each old axis goes."""
        array = constant((2, 3, 4))
        moved = array.transpose((1, 2, 0))
        assert moved.shape == (3, 4, 2)
        assert np.array_equal(moved.load(), np.transpose(array.load(), (1, 2, 0)))

    def test_transposed_concat(self, joined, patch):
        """Transposing keeps the members in canonical order."""
        moved = joined.transpose()
        moved.validate()
        assert np.array_equal(moved.load(), patch.data.T)

    def test_refuses_bad_order(self, lazy):
        """The order must be a permutation of the axes."""
        with pytest.raises(ParameterError, match="permutation"):
            lazy.transpose((0, 0))
        with pytest.raises(ParameterError, match="outside an array"):
            lazy.transpose((0, 5))


class TestConstants:
    """A block which stores no data generates it instead."""

    def test_fills_its_box(self, lazy, patch):
        """A constant member fills the box it is placed in."""
        pad = constant((10, patch.shape[1]), 3.0)
        array = concat([lazy, pad], axis=0)
        loaded = array.load()
        assert np.array_equal(loaded[: patch.shape[0]], patch.data)
        assert (loaded[patch.shape[0] :] == 3.0).all()

    def test_nan(self):
        """A nan constant loads as nan, and two of them are one array."""
        array = concat([constant((2, 3), np.nan), constant((2, 3), np.nan)], axis=0)
        assert np.isnan(array.load()).all()
        assert array.data_id == constant((4, 3), np.nan).data_id

    def test_hole_is_a_member(self):
        """A hole is filled by a constant rather than left out."""
        array = concat([constant((2, 3), 1.0), constant((2, 3), np.nan)], axis=0)
        array.validate()
        assert np.isnan(array.load()[2:]).all()

    def test_window_of_a_constant(self):
        """A clipped constant is the constant of the smaller shape."""
        array = constant((10,), 2.0)[2:5]
        assert array.data_id == ArraySource.full((3,), 2.0).data_id
        assert np.array_equal(array.load(), np.full(3, 2.0))

    def test_integer_constant(self):
        """A constant keeps the dtype it was given."""
        array = constant((3, 2), 4, np.int16)
        assert array.dtype == np.int16
        assert array.load().dtype == np.int16


class TestValidate:
    """The rules are checked on demand, not on construction."""

    def test_valid(self, joined):
        """A well formed array validates and returns itself."""
        assert joined.validate() is joined

    def test_hole(self):
        """A box left uncovered is refused."""
        array = LazyArray.from_sources(
            [ArraySource.full((2, 3), 1.0)], starts=np.zeros((1, 2)), shape=(4, 3)
        )
        with pytest.raises(ParameterError, match="leave a hole"):
            array.validate()

    def test_overlap_by_volume(self):
        """Boxes which cover too many samples are refused."""
        source = ArraySource.full((4, 3), 1.0)
        starts = np.array([[0, 0], [1, 0]])
        array = LazyArray.from_sources([source] * 2, starts=starts, shape=(5, 3))
        with pytest.raises(ParameterError, match="so they overlap"):
            array.validate()

    def test_overlap_with_a_hole(self):
        """Boxes which overlap and leave a hole are refused as well."""
        source = ArraySource.full((2, 2), 1.0)
        starts = np.array([[0, 0], [0, 1]])
        array = LazyArray.from_sources([source] * 2, starts=starts, shape=(2, 4))
        with pytest.raises(ParameterError, match="overlap"):
            array.validate()

    def test_box_outside_the_array(self):
        """A box must be inside the array it fills."""
        source = ArraySource.full((4, 3), 1.0)
        array = LazyArray.from_sources([source], starts=np.zeros((1, 2)), shape=(2, 3))
        with pytest.raises(ParameterError, match="outside an array"):
            array.validate()

    def test_negative_shape(self, lazy):
        """An array cannot have a negative shape."""
        frame = lazy.to_frame()
        array = LazyArray.from_frame(frame, (-1, 1001), lazy.dtype)
        with pytest.raises(ParameterError, match="cannot have shape"):
            array.validate()

    def test_window_outside_the_source(self, lazy):
        """A window must be inside the array it reads."""
        frame = lazy.to_frame()
        frame.loc[0, "src_start"] = 10
        array = LazyArray.from_frame(frame, lazy.shape, lazy.dtype)
        with pytest.raises(ParameterError, match="outside the source"):
            array.validate()

    def test_stored_axis_outside_the_array(self, lazy):
        """A stored axis cannot be read onto an axis which is not there."""
        frame = lazy.to_frame()
        frame.loc[0, "src_axis"] = 5
        array = LazyArray.from_frame(frame, lazy.shape, lazy.dtype)
        with pytest.raises(ParameterError, match="outside the array"):
            array.validate()

    def test_repeated_stored_axis(self, lazy):
        """One stored axis cannot feed two output axes."""
        frame = lazy.to_frame()
        frame.loc[1, "src_axis"] = 0
        array = LazyArray.from_frame(frame, lazy.shape, lazy.dtype)
        with pytest.raises(ParameterError, match="two output axes"):
            array.validate()

    def test_out_of_order(self, joined):
        """Members which are not in placement order are refused."""
        frame = joined.to_frame()
        frame["ordinal"] = 1 - frame["ordinal"]
        array = LazyArray.from_frame(
            frame.sort_values(["ordinal", "out_axis"]), joined.shape, joined.dtype
        )
        with pytest.raises(ParameterError, match="canonical placement order"):
            array.validate()

    def test_members_of_an_empty_array(self, lazy):
        """An array of no samples has no box a member could fill."""
        array = LazyArray.from_frame(lazy.to_frame(), (0, 1001), lazy.dtype)
        with pytest.raises(ParameterError, match="outside an array"):
            array.validate()

    def test_grid_of_blocks(self):
        """Blocks which tile a two dimensional grid are accepted."""
        source = ArraySource.full((2, 3), 1.0)
        starts = np.array([[0, 0], [0, 3], [2, 0], [2, 3]])
        grid = LazyArray.from_sources([source] * 4, starts=starts, shape=(4, 6))
        assert grid.validate() is grid
        assert np.array_equal(grid.load(), np.ones((4, 6)))

    def test_slab_missing_from_a_grid(self):
        """Blocks which cover the right number of samples in the wrong place."""
        source = ArraySource.full((2, 2), 1.0)
        starts = np.array([[2, 0], [2, 0], [2, 2], [2, 2]])
        array = LazyArray.from_sources([source] * 4, starts=starts, shape=(4, 4))
        with pytest.raises(ParameterError, match="do not tile"):
            array.validate()

    def test_broadcast_fields(self):
        """A broadcast axis reads nothing, so it states no window."""
        array = broadcast_array()
        array.validate()
        frame = array.to_frame()
        frame.loc[0, "src_extent"] = 3
        with pytest.raises(ParameterError, match="outside the source"):
            LazyArray.from_frame(frame, array.shape, array.dtype).validate()


class TestIdentity:
    """The id says which array this is, and nothing about where it is."""

    def test_agrees_with_the_source(self, lazy, source):
        """One member in identity placement is the source's own array."""
        assert lazy.data_id == source.data_id
        assert lazy[10:20].data_id == source[10:20].data_id

    def test_split_windows_coalesce(self, lazy, source, patch):
        """Cutting one window into adjacent blocks does not rename it."""
        length = patch.shape[0]
        parts = [source[0:7], source[7:length]]
        array = concat([LazyArray.from_source(x) for x in parts], axis=0)
        assert array.data_id == lazy.data_id

    def test_block_size_is_not_in_the_id(self, joined):
        """How an array was cut up cannot reach its id."""
        first = joined.rechunk([0, 200, 601])
        second = joined.rechunk([0, 500, 601])
        assert concat(list(first), axis=0).data_id == joined.data_id
        assert concat(list(second), axis=0).data_id == joined.data_id

    def test_paths_are_not_in_the_id(self, source, path):
        """The same members under another base uri are the same array."""
        base = str(path.parent) + "/"
        under = LazyArray.from_source(source, base_uri=base)
        assert under.data_id == LazyArray.from_source(source).data_id
        assert under.to_frame()["base_uri"].iloc[0] == base
        assert under.source(0) == source

    def test_placement_is_in_the_id(self, joined):
        """Two arrays which place the same members differently differ."""
        assert joined.transpose().data_id != joined.data_id
        assert stack([joined]).data_id != joined.data_id

    def test_shape_and_dtype_are_in_the_id(self):
        """The header the members are placed in is part of the id."""
        pair = [constant((2, 3), 1.0), constant((2, 3), 2.0)]
        along_rows = concat(pair, axis=0)
        assert along_rows.data_id != concat(pair, axis=1).data_id
        assert (
            constant((2, 3), 1.0).data_id != constant((2, 3), 1.0, np.float32).data_id
        )

    def test_cached(self, joined):
        """An id is worked out once and kept."""
        assert joined.data_id == joined.data_id
        assert joined.table._ids[joined.row] == joined.data_id

    def test_empty_arrays(self, lazy):
        """An array of no members is named by its header."""
        assert lazy[0:0].data_id == lazy[0:0].data_id
        assert lazy[0:0].data_id != lazy[:, 0:0].data_id

    def test_location_names_an_unnamed_member(self):
        """A member with no origin is named by where it is, as a source is."""
        source = ArraySource(path="/a/b.h5", format="DASDAE", version="1").describe(
            (3, 2), np.float32
        )
        array = LazyArray.from_source(source)
        assert array.data_id == source.data_id
        assert array[0:2].data_id == source[0:2].data_id

    def test_odd_origin_id(self, lazy):
        """An origin which is not 32 hex characters is folded into the id."""
        frame = lazy.to_frame()
        frame["origin_id"] = "a stored name"
        array = LazyArray.from_frame(frame, lazy.shape, lazy.dtype)
        pair = concat([array[0:10], array[10:601]], axis=0)
        assert pair.data_id == array.data_id
        assert array.data_id == array.source(0).data_id

    def test_windows_of_many_files(self, joined, two_sources):
        """An array of several partial windows is named by all of them."""
        first, second = two_sources
        array = joined[50:150]
        assert len(array) == 2
        assert array.data_id == joined[50:150].data_id
        assert array.data_id != joined[51:151].data_id
        assert (
            array.data_id
            != concat(
                [
                    LazyArray.from_source(second[0:50]),
                    LazyArray.from_source(first[50:100]),
                ],
                axis=0,
            ).data_id
        )

    def test_staggered_members_do_not_merge(self):
        """Members which abut on one axis but are offset on another stay two."""
        whole = ArraySource(
            path="/a/b.h5", format="DASDAE", version="1", origin_id="b" * 32
        ).describe((4, 6), np.float32)
        parts = [whole[0:2, 0:3], whole[2:4, 3:6]]
        starts = np.array([[0, 0], [2, 3]])
        array = LazyArray.from_sources(parts, starts=starts, shape=(4, 6))
        assert array.data_id != LazyArray.from_source(whole).data_id
        assert (
            array.data_id
            == LazyArray.from_sources(parts, starts=starts, shape=(4, 6)).data_id
        )

    def test_constants_among_the_members(self, lazy, patch):
        """A constant which cannot merge is named by its own value."""
        width = patch.shape[1]
        first = concat([lazy, constant((5, width), 1.0)], axis=0)
        second = concat([lazy, constant((5, width), 2.0)], axis=0)
        assert first.data_id != second.data_id
        pair = concat([constant((2, 3), 1.0), constant((2, 3), 2.0)], axis=0)
        other = concat([constant((2, 3), 1.0), constant((2, 3), 3.0)], axis=0)
        assert pair.data_id != other.data_id

    def test_unnamed_members(self):
        """Members with no origin are named by where they are, as sources are."""
        source = ArraySource(path="/a/b.h5", format="DASDAE", version="1")
        source = source.describe((10, 3), np.float32)
        array = LazyArray.from_source(source)
        pair = concat([array[0:2], array[5:7]], axis=0)
        assert pair.data_id == concat([array[0:2], array[5:7]], axis=0).data_id
        assert pair.data_id != concat([array[0:2], array[6:8]], axis=0).data_id

    @pytest.mark.parametrize("origin", ["a stored name", "abcd", "z" * 32])
    def test_folded_origin_ids(self, lazy, origin):
        """An origin which is not 32 hex characters still names its member."""
        frame = lazy.to_frame()
        frame["origin_id"] = origin
        array = LazyArray.from_frame(frame, lazy.shape, lazy.dtype)
        pair = concat([array[0:10], array[100:110]], axis=0)
        assert pair.data_id == concat([array[0:10], array[100:110]], axis=0).data_id
        assert pair.data_id != concat([array[0:10], array[101:111]], axis=0).data_id

    def test_is_an_operation_parameter(self, lazy):
        """An array carries its id where an operation hashes its arguments."""
        assert H("operation", {"array": lazy}) == H("operation", {"array": lazy})
        assert H("operation", {"array": lazy}) != H("operation", {"array": lazy[0:5]})


class TestSources:
    """A member and a source say the same thing."""

    def test_round_trip(self, source, lazy):
        """The member a source made gives that source back."""
        assert lazy.source(0) == source
        assert lazy.sources == (source,)

    def test_base_uri_is_joined(self, source, path):
        """A path stored in two parts is resolved as it is read out."""
        base = str(path.parent) + "/"
        array = LazyArray.from_source(source, base_uri=base)
        assert array.source(0).path == source.path

    def test_path_outside_the_base(self, source):
        """A path which does not start with the base uri is stored whole."""
        array = LazyArray.from_source(source, base_uri="/somewhere/else/")
        assert array.to_frame()["base_uri"].iloc[0] == ""
        assert array.source(0) == source

    def test_window_of_a_member(self, lazy, source):
        """A clipped member reads the window it was clipped to."""
        assert lazy[10:20].source(0) == source[10:20]

    def test_constant_source(self):
        """A constant member gives the constant source back."""
        source = ArraySource.full((3, 2), 5.0)
        assert constant((3, 2), 5.0).source(0) == source


class TestLoad:
    """Loading reads each member and puts it in its box."""

    def test_many_files(self, joined, patch, reads):
        """Each member is read once, through its own source."""
        assert np.array_equal(joined.load(), patch.data)
        assert len(reads) == 2

    def test_numpy_asks_for_it(self, lazy, patch):
        """Numpy can ask for the array, with or without a dtype."""
        assert np.array_equal(np.asarray(lazy), patch.data)
        assert np.asarray(lazy, dtype=np.float32).dtype == np.float32

    def test_broadcast_member(self):
        """A member with no stored axis fills the axis it was given."""
        array = broadcast_array(length=4)
        loaded = array.load()
        assert loaded.shape == (4, 2, 3)
        assert np.array_equal(loaded, np.broadcast_to(loaded[0], (4, 2, 3)))

    def test_transposed_member(self, lazy, patch):
        """A member whose axes are swapped is put back the right way."""
        moved = lazy.transpose()
        assert np.array_equal(moved.load(), patch.data.T)


class TestTable:
    """One table holds many arrays and owns all the storage."""

    def test_views_share_storage(self, joined):
        """An array's matrices are views of the table's own arrays."""
        table = joined.table
        for name in AXIS_FIELDS:
            assert np.shares_memory(joined.placement[name], table.axes[name])

    def test_slicing_copies_no_more_than_it_selects(self, joined):
        """A window holds the members it selected, and no other rows."""
        window = joined[0:10]
        assert window.table is not joined.table
        assert window.table.n_members == 1
        assert not np.shares_memory(
            window.placement["out_start"], joined.table.axes["out_start"]
        )

    def test_mixed_ndim(self, joined):
        """Arrays of different ndim sit in one table."""
        table = LazyTable.from_arrays([joined, stack([joined, joined])])
        assert [x.ndim for x in table] == [2, 3]
        assert [x.shape for x in table] == [joined.shape, (2, *joined.shape)]
        assert table.n_members == len(joined) + 2 * len(joined)
        for array in table:
            assert np.shares_memory(
                array.placement["out_start"], table.axes["out_start"]
            )

    def test_indexing(self, joined):
        """A table is indexed by row, from either end."""
        table = LazyTable.from_arrays([joined, joined[0:10]])
        assert len(table) == 2
        assert table[-1].shape == (10, joined.shape[1])
        with pytest.raises(IndexError, match="outside a table"):
            table[5]

    def test_nbytes(self, joined):
        """The table says how much storage it takes."""
        assert joined.table.nbytes > 0


class TestFrames:
    """The frames are the database tables, and read back as they are."""

    def test_array_round_trip(self, joined, patch):
        """One array round trips through its long frame."""
        frame = joined.to_frame()
        back = LazyArray.from_frame(frame, joined.shape, joined.dtype)
        assert back.data_id == joined.data_id
        assert np.array_equal(back.load(), patch.data)
        assert len(frame) == len(joined) * joined.ndim

    def test_array_frame_columns(self, joined):
        """The columns are the placement and what each member reads."""
        columns = set(joined.to_frame().columns)
        assert set(AXIS_FIELDS) <= columns
        assert {"ordinal", "out_axis", "base_uri", "path", "origin_id"} <= columns

    def test_out_of_order_frame(self, joined):
        """A frame which was shuffled is put back in order."""
        frame = joined.to_frame()
        back = LazyArray.from_frame(frame.iloc[::-1], joined.shape, joined.dtype)
        assert back.data_id == joined.data_id

    def test_table_round_trip(self, joined):
        """A whole table round trips through the database frames."""
        table = LazyTable.from_arrays([joined, stack([joined, joined])])
        frames = table.to_frames()
        back = LazyTable.from_frames(frames)
        assert [x.data_id for x in back] == [x.data_id for x in table]
        assert [x.shape for x in back] == [x.shape for x in table]

    def test_table_frame_names(self, joined):
        """The frames are named and keyed as the database tables are."""
        frames = LazyTable.from_arrays([joined]).to_frames()
        assert set(frames) == {
            "sources",
            "lazy_arrays",
            "lazy_members",
            "lazy_member_axes",
        }
        assert next(iter(frames["sources"].columns)) == "source_row"
        assert "source_row" in frames["lazy_members"].columns
        assert set(frames["lazy_arrays"].columns) == {
            "array_row",
            "data_id",
            "ndim",
            "shape",
            "dtype",
        }

    def test_empty_table(self):
        """A table of no arrays round trips as well."""
        table = LazyTable.from_arrays([])
        assert len(table) == 0 and table.n_members == 0
        frames = table.to_frames()
        assert len(LazyTable.from_frames(frames)) == 0
        assert isinstance(frames["lazy_members"], pd.DataFrame)

    def test_constant_round_trip(self):
        """A constant member keeps its value and dtype through a frame."""
        array = concat([constant((2, 3), np.nan), constant((2, 3), 1.0)], axis=0)
        back = LazyArray.from_frame(array.to_frame(), array.shape, array.dtype)
        assert back.data_id == array.data_id
        assert np.array_equal(back.load(), array.load(), equal_nan=True)


class TestRechunk:
    """Cutting an array at new bounds gives many arrays in one table."""

    def test_pieces_tile_the_parent(self, joined, patch):
        """The pieces are the parent, cut where it was asked for."""
        table = joined.rechunk([0, 150, 400, 601])
        assert [x.shape[0] for x in table] == [150, 250, 201]
        loaded = [x.validate().load() for x in table]
        assert np.array_equal(np.concatenate(loaded), patch.data)

    def test_pieces_keep_their_ids(self, joined, two_sources):
        """A piece which is one window is named as that window is."""
        table = joined.rechunk([0, 50, 100])
        assert table[0].data_id == two_sources[0][0:50].data_id
        assert table[1].data_id == two_sources[0][50:100].data_id

    def test_merging(self, joined, patch):
        """Bounds coarser than the members merge them."""
        table = joined.rechunk([0, 601])
        assert len(table) == 1 and table[0].shape == patch.shape
        assert np.array_equal(table[0].load(), patch.data)

    def test_dropping_samples(self, joined, patch):
        """Samples outside the bounds are left out."""
        table = joined.rechunk([50, 150])
        assert len(table) == 1 and table[0].shape[0] == 100
        assert np.array_equal(table[0].load(), patch.data[50:150])

    def test_other_axis(self, patch):
        """Any axis can be cut along."""
        array = concat([constant((3, 4), 1.0), constant((3, 6), 2.0)], axis=1)
        table = array.rechunk([0, 5, 10], axis=1)
        assert [x.shape for x in table] == [(3, 5), (3, 5)]
        loaded = np.concatenate([x.load() for x in table], axis=1)
        assert np.array_equal(loaded, array.load())

    def test_constants(self):
        """A constant cut in two is two constants."""
        table = constant((10, 3), 2.0).rechunk([0, 4, 10])
        assert table[0].data_id == ArraySource.full((4, 3), 2.0).data_id
        assert np.array_equal(table[1].load(), np.full((6, 3), 2.0))

    def test_refuses_a_grid(self):
        """An array which is not a stack of blocks is not rechunked yet."""
        source = ArraySource.full((2, 3), 1.0)
        starts = np.array([[0, 0], [0, 3], [2, 0], [2, 3]])
        grid = LazyArray.from_sources([source] * 4, starts=starts, shape=(4, 6))
        with pytest.raises(NotImplementedError, match="full width"):
            grid.rechunk([0, 2, 4])

    def test_refuses_bad_bounds(self, joined):
        """Bounds must ascend and stay inside the array."""
        with pytest.raises(ParameterError, match="at least two bounds"):
            joined.rechunk([10])
        with pytest.raises(NotImplementedError, match="overlapping or empty"):
            joined.rechunk([10, 10, 20])
        with pytest.raises(ParameterError, match="outside the array"):
            joined.rechunk([0, 10_000])


class TestArrayApi:
    """A lazy array says which backend owns it."""

    def test_backend_name(self, lazy):
        """The array api reports the lazy backend."""
        assert backend_name(lazy) == "lazy"

    def test_namespace(self, lazy):
        """The namespace is the one the backend name comes from."""
        assert lazy.__array_namespace__().__name__ == "lazy"
