"""Tests for coordinate object."""
from __future__ import annotations

import pickle
from functools import partial
from io import BytesIO
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import rich.text
from pydantic import ValidationError

import dascore as dc
from dascore.core.coords import (
    BaseCoord,
    CoordDegenerate,
    CoordMonotonicArray,
    CoordRange,
    CoordSummary,
    get_coord,
)
from dascore.exceptions import CoordError, ParameterError
from dascore.units import get_quantity
from dascore.utils.misc import all_close, register_func
from dascore.utils.time import dtype_time_like, is_datetime64, is_timedelta64, to_float

COORDS = []


@pytest.fixture(scope="session")
@register_func(COORDS)
def evenly_sampled_coord():
    """Create coordinates which are evenly sampled."""
    ar = np.arange(0, 100, 1)
    return get_coord(data=ar)


@pytest.fixture(scope="session")
@register_func(COORDS)
def evenly_sampled_reversed_coord():
    """Create coordinates which are evenly sampled."""
    ar = np.arange(0, -100, -1)
    return get_coord(data=ar)


@pytest.fixture(scope="session")
@register_func(COORDS)
def evenly_sampled_float_coord_with_units():
    """Create coordinates which are evenly sampled and have units."""
    ar = np.arange(1, 100, 1 / 3)
    return get_coord(data=ar, units="m")


@pytest.fixture(scope="session")
@register_func(COORDS)
def evenly_sampled_float_reverse_units_coord():
    """Create coordinates which are evenly sampled and have units."""
    ar = np.arange(-1, -100, -1 / 3)
    return get_coord(data=ar, units="m")


@pytest.fixture(scope="session")
@register_func(COORDS)
def evenly_sampled_date_coord():
    """Create coordinates which are evenly sampled."""
    ar = dc.to_datetime64(np.arange(1, 100_000, 1_000))
    return get_coord(data=ar)


@pytest.fixture(scope="session")
@register_func(COORDS)
def evenly_sampled_time_delta_coord():
    """Create coordinates which are evenly sampled."""
    ar = dc.to_timedelta64(np.arange(1, 100_000, 1_000))
    return get_coord(data=ar)


@pytest.fixture(scope="session")
@register_func(COORDS)
def monotonic_float_coord():
    """Create coordinates which are evenly sampled."""
    ar = np.cumsum(np.abs(np.random.rand(100)))
    return get_coord(data=ar)


@pytest.fixture(scope="session")
@register_func(COORDS)
def reverse_monotonic_float_coord():
    """Create coordinates which are evenly sampled."""
    ar = np.cumsum(np.abs(np.random.rand(100)))[::-1]
    return get_coord(data=ar)


@pytest.fixture(scope="session")
@register_func(COORDS)
def monotonic_datetime_coord():
    """Create coordinates which are evenly sampled."""
    ar = np.cumsum(np.abs(np.random.rand(100) * 1_000))
    return get_coord(data=dc.to_datetime64(ar))


@pytest.fixture(scope="session")
@register_func(COORDS)
def random_coord():
    """Create coordinates which are evenly sampled."""
    ar = np.random.rand(100) * 1_000
    return get_coord(data=ar)


@pytest.fixture(scope="session")
@register_func(COORDS)
def random_date_coord():
    """Create coordinates which are evenly sampled."""
    ar = np.random.rand(100) * 1_000
    return get_coord(data=dc.to_datetime64(ar))


@pytest.fixture(scope="session", params=COORDS)
def coord(request) -> BaseCoord:
    """Meta-fixture for returning all coords."""
    return request.getfixturevalue(request.param)


@pytest.fixture()
def degenerate_time_coord():
    """Return a simply degenerate coord."""
    ar = np.empty((0, 10), dtype="datetime64[ns]")
    return get_coord(data=ar)


@pytest.fixture(scope="session")
def two_d_coord():
    """Return a 2D coordinate."""
    ar = np.ones((10, 10))
    return get_coord(data=ar, units="m/s")


def assert_value_in_one_step(coord, index, value, greater=True):
    """Ensure value at index is within one step of value."""
    # reverse greater for reverse monotonic
    coord_value = coord.values[index]
    if greater:
        assert value <= coord_value
    else:
        assert value >= coord_value
    # next we ensure value is in the correct index. This is a little
    # bit complicated by reverse order coords.
    val1 = coord.values[index + 1]
    val2 = coord.values[index - 1]
    assert (val1 <= value <= val2) or (val2 <= value <= val1)


class TestBasics:
    """A suite of basic tests for coordinates."""

    def test_coord_init(self, coord):
        """Simply run to insure all coords initialize."""
        assert isinstance(coord, BaseCoord)

    def test_bad_init(self):
        """Ensure no parameters raises error."""
        with pytest.raises(CoordError):
            get_coord()

    def test_value(self, coord):
        """All coords should return an array with the values attr."""
        ar = coord.values
        assert isinstance(ar, np.ndarray)

    def test_len_one_array(self):
        """Ensure one length array returns coord."""
        ar = np.array([10])
        coord = get_coord(data=ar)
        assert isinstance(coord, BaseCoord)
        assert len(coord) == 1

    def test_empty_array(self):
        """An empty array should also be a valid coordinate."""
        ar = np.array([])
        coord = get_coord(data=ar)
        assert isinstance(coord, BaseCoord)
        assert len(coord) == 0

    def test_coord_input(self, coord):
        """A coordinate should be valid input."""
        assert get_coord(data=coord) == coord

    def test_cant_add_extra_fields(self, evenly_sampled_coord):
        """Ensure coordinates are immutable."""
        with pytest.raises(ValueError, match="Instance is frozen"):
            evenly_sampled_coord.bob = 1

    def test_sort(self, coord):
        """Ensure every coord can be sorted."""
        if coord.degenerate or len(coord.shape) > 1:
            return  # no need to test degenerate coord or multidim
        data = coord.data
        new, indexer = coord.sort()
        assert new.dtype == coord.dtype
        assert new.sorted
        assert not new.degenerate
        assert not new.reverse_sorted
        assert len(new) == len(coord)
        assert data[indexer] is not None

    def test_reverse_sort(self, coord):
        """Ensure every coord can be reverse sorted."""
        if coord.degenerate or len(coord.shape) > 1:
            return  # no need to test degenerate coord or multidim
        data = coord.data
        new, indexer = coord.sort(reverse=True)
        assert new.dtype == coord.dtype
        assert not new.sorted
        assert new.reverse_sorted
        assert len(new) == len(coord)
        assert data[indexer] is not None

    def test_immutable(self, evenly_sampled_coord):
        """Fields can't change once created."""
        with pytest.raises((TypeError, ValueError), match="Instance is frozen"):
            evenly_sampled_coord.start = 10

    def test_values_immutable(self, coord):
        """Values should all be immutable arrays."""
        with pytest.raises(ValueError, match="assignment destination is read-only"):
            coord.data[0] = coord.data[1]

    def test_str(self, coord):
        """All coords should be convertible to str."""
        out = str(coord)
        assert isinstance(out, str)

    def test_rich(self, coord):
        """Each coord should have nice rich printing."""
        out = coord.__rich__()
        assert isinstance(out, rich.text.Text)

    def test_snap(self, coord):
        """Tests for snapping coordinates."""
        out = coord.snap()
        assert isinstance(out, BaseCoord)
        assert out.shape == coord.shape
        # sort order should stay the same
        if coord.reverse_sorted:
            assert out.reverse_sorted
        if coord.sorted:
            assert out.sorted

    def test_default_units(self):
        """Default units should be None."""
        out = get_coord(data=np.arange(10))
        assert out.units is None

    def test_min_max_align_with_array(self, coord):
        """Min/max values should match the same operation on data."""
        if coord.degenerate:
            return
        values = coord.values
        assert coord.min() == np.min(values)
        assert coord.max() == np.max(values)
        assert coord.max() > coord.min()

    def test_can_pickle(self, coord):
        """Ensure each coord can be pickled/unpickled, both should be equal."""
        len(coord)
        bio = BytesIO()
        pickle.dump(coord, bio)
        bio.seek(0)
        new = pickle.load(bio)
        assert new == coord

    def test_like_coords_have_s_units(self, coord):
        """Time like coordinates should have units of s set automatically."""
        if dtype_time_like(coord.dtype):
            assert coord.units == get_quantity("s")

    def test_convert_simple_units(self, coord):
        """Ensure units can be converted/set."""
        # time always units of s
        if dtype_time_like(coord.dtype):
            pytest.skip("Time units cannot be converted.")
        out_m = coord.convert_units("m")
        assert dc.get_unit(out_m.units) == dc.get_unit("m")
        out_mm = out_m.convert_units("mm")
        assert dc.get_unit(out_mm.units) == dc.get_unit("mm")
        assert len(out_mm) == len(out_m)
        assert np.allclose(out_mm.values / 1000, out_m.values)

    def test_convert_offset_units(self):
        """Ensure units can be converted/set for offset units."""
        array = np.arange(10)
        f_array = array * (9 / 5) + 32.0
        coord = get_coord(data=array, units="degC")
        out = coord.convert_units("degF")
        assert out.shape == f_array.shape
        assert np.allclose(f_array, out.values)

    def test_unit_str(self, evenly_sampled_coord):
        """Ensure the unit string returns a string."""
        coord = evenly_sampled_coord.set_units("m/s")
        assert isinstance(coord.unit_str, str)

    def test_out_of_range_raises(self, evenly_sampled_coord):
        """Accessing a value out of the range of array should raise."""
        with pytest.raises(IndexError, match="exceeds coord length"):
            _ = evenly_sampled_coord[len(evenly_sampled_coord)]

    def test_to_array(self, coord):
        """Ensure np.array converts coord to an array."""
        ar = np.array(coord)
        assert isinstance(ar, np.ndarray)
        assert len(ar) == len(coord)

    def test_bad_range_step(self):
        """Ensure start, stop, step are consistent. See #321."""
        msg = "Sign of step must match"
        with pytest.raises(ValidationError, match=msg):
            get_coord(start=1, stop=5, step=-1)
        with pytest.raises(ValidationError, match=msg):
            get_coord(start=1, stop=-5, step=1)

    def test_both_values_and_data_raises(self):
        """Cannot specify both values and data."""
        data = np.empty(10)
        msg = "Cannot specify both data and values"
        with pytest.raises(CoordError, match=msg):
            get_coord(data=data, values=data)


class TestCoordSummary:
    """tests for converting to and from summary coords."""

    cast_data_list: ClassVar = [
        {"min": 400.0, "step": 1.0, "units": "", "max": 1000.0},
        {"min": 0, "max": 10},
        {"min": dc.to_datetime64(10), "max": dc.to_datetime64(100)},
        {"min": dc.to_timedelta64(100), "max": dc.to_timedelta64(200)},
    ]

    @pytest.fixture(scope="session")
    def summary(self, coord) -> CoordSummary:
        """Convert each coord to a summary."""
        return coord.to_summary()

    def test_dtype_consistent(self, summary, coord):
        """Ensure the datatype is preserved."""
        dtype = np.dtype(summary.dtype)
        min_ar = np.array(summary.min)
        max_ar = np.array(summary.max)
        assert np.issubdtype(min_ar.dtype, dtype)
        assert np.issubdtype(max_ar.dtype, dtype)

    def test_to_summary(self, coord):
        """Ensure all coords can be converted to a summary."""
        out = coord.to_summary()
        assert isinstance(out, CoordSummary)

    def test_coord_range_round_trip(self, coord):
        """Coord ranges should round-trip to summaries and back."""
        if not coord.evenly_sampled:
            return
        summary = coord.to_summary()
        back = summary.to_coord()
        assert back == coord

        if not back.to_summary() == summary:
            coord.to_summary()

        assert back.to_summary() == summary

    def test_to_summary_raises(self, random_coord):
        """Ensure to_summary raises if not evenly sampled."""
        match = "Cannot convert summary which is not evenly sampled"
        with pytest.raises(CoordError, match=match):
            random_coord.to_summary().to_coord()

    @pytest.mark.parametrize("data", cast_data_list)
    def test_dtype_inferred(self, data):
        """Ensure the dtypes are correctly determined if not specified."""
        out = CoordSummary(**data)
        assert np.dtype(out.dtype) == np.dtype(type(data["min"]))


class TestGetSliceTuple:
    """Tests for getting slice tuples."""

    def test_get_slice_tuple(self, coord):
        """Basic tests for range of coords."""
        start = coord.min()
        end = coord.max()
        out = coord.get_slice_tuple((start, end))
        assert out == (start, end)
        assert coord.get_slice_tuple((start, None)) == (start, None)
        assert coord.get_slice_tuple((None, end)) == (None, end)
        # if units aren't none, ensure they work.
        if not (unit := coord.units):
            return
        # since we can't multiply dt by quant, convert to float
        if dtype_time_like(coord):
            start, end = to_float(start), to_float(end)
        start_unit = start * get_quantity(unit)
        end_unit = end * get_quantity(unit)
        # check that first slice is close to original
        sliced = coord.get_slice_tuple((start_unit, None))
        lims = (coord.min(), None)
        assert all_close([to_float(lims[0])], [to_float(sliced[0])])
        # same for second slice.
        lims = to_float(np.array((coord.min(), coord.max())))
        sliced = to_float(np.array(coord.get_slice_tuple((start_unit, end_unit))))
        assert all_close(lims, sliced)

    def test_get_slice_range_bad_values(self, evenly_sampled_coord):
        """Ensure bad values raise helpful error."""
        with pytest.raises(ParameterError, match="must be a length 2"):
            evenly_sampled_coord.get_slice_tuple((1, 2, 3))

    def test_slice_with_step_raises(self, evenly_sampled_coord):
        """A slice with a step shouldn't work."""
        match = "Step not supported"
        with pytest.raises(ParameterError, match=match):
            evenly_sampled_coord.select(slice(1, 10, 2))

    def test_slice_with_step(self, evenly_sampled_coord):
        """Ensure slice works like tuple."""
        coord = evenly_sampled_coord
        vmin, vmax = coord.min(), coord.max()
        sli = slice(vmin, vmax + 1)
        out_sli = evenly_sampled_coord.get_slice_tuple(sli)
        assert out_sli == (None, None) or out_sli == (0, len(coord))


class TestNDCoords:
    """Tests for multidimensional coordinates."""

    def test_empty(self, two_d_coord):
        """Ensure 2D coordinate can be emptied out."""
        out = two_d_coord.empty()
        assert out.degenerate
        assert tuple([0] * 2) == out.shape

    def test_empty_axis(self, two_d_coord):
        """Ensure 2D coordinate can be emptied out."""
        out = two_d_coord.empty(axes=1)
        assert out.degenerate
        assert (out.shape[0], 0) == out.shape


class TestSelect:
    """Generic tests for selecting values from coords."""

    def test_select_end_end_time(self, coord):
        """Ensure when time range is == (end, end) that dim has len 1."""
        out = coord.select((coord.max(), coord.max()))[0]
        assert len(out) == 1

    def test_intra_sample_select(self, coord):
        """
        Selecting ranges that fall within samples should raise.

        This is consistent with pandas indices.
        """
        values = coord.values
        # get value between first/second sample
        arg = values[0] + (values[1] - values[0]) / 2
        assert arg not in np.unique(values)
        out, indexer = coord.select((arg, arg))
        assert isinstance(out, CoordDegenerate)
        assert np.size(coord.data[indexer]) == 0

    def test_select_start_start_time(self, coord):
        """Ensure when time range is == (start, start) that dim has len 1."""
        out = coord.select((coord.min(), coord.min()))[0]
        assert len(out) == 1

    def test_select_inclusive(self, coord):
        """Ensure selecting is inclusive on both ends."""
        # These tests are only for coords with len > 7
        if len(coord) < 7:
            return
        values = np.sort(coord.values)
        value_set = set(values)
        assert len(values) > 7
        args = (values[3], values[-3])
        new, _ = coord.select(args)
        # Check min and max values conform to args
        new_values = new.values
        new_min, new_max = np.min(new_values), np.max(new_values)
        assert (new_min >= args[0]) or np.isclose(new_min, args[0])
        assert (new_max <= args[1]) or np.isclose(new_max, args[1])
        # There can often be a bit of "slop" in float indices. We may
        # need to rethink this but this is good enough for now.
        if not np.issubdtype(new.dtype, np.float_):
            assert {values[3], values[-3]}.issubset(value_set)

    def test_select_bounds(self, coord):
        """Ensure selecting bounds are honored."""
        # These tests are only for coords with len > 7
        if len(coord) < 7:
            return
        values = np.sort(coord.values)
        # get a little more than and little less than 2 and -2 index vals
        val1 = values[2] + (values[3] - values[2]) / 2
        val2 = values[-2] - (values[-2] - values[-3]) / 2
        new, _ = coord.select((val1, val2))
        new_values = new.values
        new_min, new_max = np.min(new_values), np.max(new_values)
        assert (new_min >= val1) or np.isclose(new_min, val1)
        assert (new_max <= val2) or np.isclose(new_max, val2)
        if not np.issubdtype(new.dtype, np.float_):
            assert set(values).issuperset(set(new.values))

    def test_select_out_of_bounds_too_early(self, coord):
        """Applying a select out of bounds (too early) should raise an Error."""
        diff = (coord.max() - coord.min()) / (len(coord) - 1)
        # get a range which is for sure before data.
        v1 = coord.min() - np.abs(100 * diff)
        v2 = v1 + np.abs(30 * diff)
        # it should return a degenerate because range not contained by coord.
        new, indexer = coord.select((v1, v2))
        assert new.degenerate
        assert np.size(coord.data[indexer]) == 0
        # Same thing if end time is too early
        new, indexer = coord.select((None, v2))
        assert new.degenerate
        assert np.size(coord.data[indexer]) == 0
        # but this should be fine
        new = coord.select((v1, None))[0]
        assert new == coord

    def test_select_out_of_bounds_too_late(self, coord):
        """Applying a select out of bounds (too late) should raise an Error."""
        diff = (coord.max() - coord.min()) / (len(coord) - 1)
        # get a range which is for sure after data.
        v1 = coord.max() + np.abs(100 * diff)
        v2 = v1 + np.abs(30 * diff)
        # it should return a degenerate since out of range
        new, indexer = coord.select((v1, v2))
        assert new.degenerate
        assert np.size(coord.data[indexer]) == 0
        # Same thing if start time is too late
        new, indexer = coord.select((v1, None))
        assert new.degenerate
        assert np.size(coord.data[indexer]) == 0
        assert coord.select((None, v2))[0] == coord

    def test_wide_select_bounds(self, coord):
        """Wide (lt and gt limits) select bounds should be fine."""

        def _is_equal(coord1, coord2, indexer):
            """Assert coord and indexer are equal."""
            assert coord1 == coord2
            if isinstance(indexer, slice):
                assert indexer == slice(None, None)
            elif isinstance(indexer, np.ndarray):
                assert np.all(indexer)

        diff = (coord.max() - coord.min()) / (len(coord) - 1)
        v1 = coord.min() - 10 * diff
        v2 = coord.max() + 10 * diff
        out, slice_thing = coord.select((v1, v2))
        _is_equal(coord, out, slice_thing)
        out, slice_thing = coord.select((None, v2))
        _is_equal(coord, out, slice_thing)
        out, slice_thing = coord.select((v1, None))
        _is_equal(coord, out, slice_thing)

    def test_select_ordered_coords(self, coord):
        """Basic select tests all coords should pass."""
        if not (coord.sorted or coord.reverse_sorted):
            return
        assert len(coord.values) == len(coord)
        value1 = coord.values[10]
        new, sliced = coord.select((value1, ...))
        # this accounts for reverse sorted coords
        slice_value = sliced.start if coord.sorted else sliced.stop - 1
        assert_value_in_one_step(coord, slice_value, value1, greater=True)
        assert new.values is not coord.values, "new data should be made"
        assert slice_value == 10
        # now test exact value
        value2 = coord.values[20]
        new, sliced = coord.select((value2, ...))
        slice_value = sliced.start if coord.sorted else sliced.stop - 1
        assert_value_in_one_step(coord, slice_value, value2, greater=True)
        assert slice_value == 20
        # test end index
        new, sliced = coord.select((..., value1))
        assert sliced.stop == 11 if new.sorted else sliced.start == 10
        new, sliced = coord.select((..., value2))
        assert sliced.stop == 21 if new.sorted else sliced.start == 20
        # test range
        new, sliced = coord.select((value1, value2))
        assert len(new.values) == 11 == len(new)
        assert sliced == slice(10, 21)

    def test_select_relative(self, coord):
        """Test relative select from start/end."""
        dist = (coord.max() - coord.min()) / 4
        abs1 = coord.min() + dist
        abs2 = coord.max() - dist
        out1, ind_1 = coord.select((abs1, abs2))
        out2, ind_2 = coord.select((dist, -dist), relative=True)
        assert np.all(out1 == out2)
        if isinstance(ind_1, np.ndarray):
            assert np.all(ind_1 == ind_2)
        else:
            assert ind_1 == ind_2

    def test_select_changes_len(self, coord):
        """Ensure select changes the length of the coordinate."""
        if len(coord) < 3 or not coord.sorted:
            return
        data = coord.data
        new, dslice = coord.select((data[2], data[-2]))
        # plus 3 because end value is inclusive
        new_len = dslice.stop - dslice.start
        assert len(new) == new_len


class TestEqual:
    """Tests for comparing coord equality."""

    def test_different_units_not_equal(self, coord):
        """When different units are set the coords should not be equal."""
        if coord.units == "furlongs":
            return
        if dtype_time_like(coord.dtype):
            pytest.skip("Cant set time-like units")
        new = coord.set_units("furlongs")
        assert new != coord

    def test_compare_non_coord_coord_range(self, evenly_sampled_coord):
        """Ensure non-coords compare false."""
        assert evenly_sampled_coord is not None
        assert not evenly_sampled_coord == {1, 2}

    def test_compare_non_coord_coord_array(self, random_coord):
        """Ensure non-coords compare false."""
        assert random_coord is not None
        assert not random_coord == {1, 2}


class TestCoordRange:
    """Tests for coords from array."""

    def test_init_array(self, evenly_sampled_coord):
        """Ensure the array can be initialized."""
        assert evenly_sampled_coord.step == 1.0
        assert evenly_sampled_coord.units is None

    def test_values(self, evenly_sampled_coord):
        """Ensure we can get an array with values attribute."""
        vals = evenly_sampled_coord.values
        assert len(vals) == len(evenly_sampled_coord)

    def test_set_units(self, evenly_sampled_coord):
        """Ensure units can be set."""
        out = evenly_sampled_coord.set_units("m")
        assert out.units == get_quantity("m")
        assert out is not evenly_sampled_coord

    def test_set_quantity(self, evenly_sampled_coord):
        """Ensure units can be set."""
        out = evenly_sampled_coord.set_units("10 m")
        assert out.units == get_quantity("10 m")
        assert out is not evenly_sampled_coord

    def test_dtype(self, evenly_sampled_coord, evenly_sampled_date_coord):
        """Ensure datatypes are sensible."""
        dtype1 = evenly_sampled_coord.dtype
        assert np.issubdtype(dtype1, np.int_)
        dtype2 = evenly_sampled_date_coord.dtype
        assert np.issubdtype(dtype2, np.datetime64)

    def test_select_tuple_ints(self, evenly_sampled_coord):
        """Ensure a tuple works as a limit."""
        assert evenly_sampled_coord.select((50, None))[1] == slice(50, None)
        assert evenly_sampled_coord.select((0, None))[1] == slice(None, None)
        assert evenly_sampled_coord.select((-10, None))[1] == slice(None, None)
        assert evenly_sampled_coord.select((None, None))[1] == slice(None, None)
        assert evenly_sampled_coord.select((None, 1_000_000))[1] == slice(None, None)

    def test_identity_slice_ints(self, evenly_sampled_coord):
        """Ensure slice with exact start/end gives same coord."""
        coord = evenly_sampled_coord
        new, sliced = coord.select((coord.start, coord.stop))
        assert new == coord

    def test_identity_slice_floats(self, evenly_sampled_float_coord_with_units):
        """Ensure slice with exact start/end gives same coord."""
        coord = evenly_sampled_float_coord_with_units
        new, sliced = coord.select((coord.start, coord.stop))
        assert new == coord

    def test_float_basics(self):
        """Ensure floating point range coords work."""
        ar = np.arange(1, 100, 1 / 3)
        out = get_coord(data=ar, units="m")
        assert len(out) == len(ar)
        assert np.allclose(out.values, ar)

    def test_select_units(self, evenly_sampled_float_coord_with_units):
        """Ensure units can work in select."""
        coord = evenly_sampled_float_coord_with_units
        value_ft = 100
        value_m = value_ft * 0.3048
        new, sliced = coord.select((value_ft * dc.get_unit("ft"), ...))
        # ensure value is surrounded.
        assert new.start + new.step >= value_m
        assert new.start - new.step <= value_m
        # units should also not change
        assert new.units == coord.units

    def test_set_convert_units(self, coord):
        """Simple test for setting and converting units."""
        if dtype_time_like(coord.dtype):
            pytest.skip("Cant set time-like units")
        factor = get_quantity("m").to(get_quantity("ft")).magnitude
        c1 = coord.set_units("m")
        c2 = c1.convert_units("ft")
        assert c2.units == get_quantity("ft")
        if c2.degenerate:
            return
        values1 = c1.values
        values2 = c2.values
        # for datetime and timedelta no conversion should happen
        if is_datetime64(values1[0]) or is_timedelta64(values1[0]):
            assert np.all(np.equal(values1, values2))
        else:
            np.allclose(values1, values2 / factor)

    def test_select_date_string(self, evenly_sampled_date_coord):
        """Ensure string selection works with datetime objects."""
        coord = evenly_sampled_date_coord
        date_str = "1970-01-01T12"
        new, sliced = coord.select((date_str, ...))
        datetime = dc.to_datetime64(date_str)
        assert new.start + new.step >= datetime
        assert new.start - new.step <= datetime

    def test_sort(self, evenly_sampled_coord):
        """Ensure sort returns equal coord."""
        out, _ = evenly_sampled_coord.sort()
        assert out == evenly_sampled_coord

    def test_update_limits_too_many_params(self, evenly_sampled_coord):
        """Update limits should raise if too many parameters are specified."""
        with pytest.raises(ValueError, match="At most two parameters"):
            evenly_sampled_coord.update_limits(1, 10, 1)

    def test_update_limits_start(self, evenly_sampled_coord):
        """Ensure start can be updated."""
        new_start = evenly_sampled_coord.start + 2 * evenly_sampled_coord.step
        new = evenly_sampled_coord.update_limits(min=new_start)
        assert len(new) == len(evenly_sampled_coord)
        assert new.step == evenly_sampled_coord.step
        assert new.start == new_start

    def test_update_limits_end(self, evenly_sampled_coord):
        """Ensure end can be updated."""
        new_stop = evenly_sampled_coord.start - 2 * evenly_sampled_coord.step
        step = evenly_sampled_coord.step
        new = evenly_sampled_coord.update_limits(max=new_stop - step)
        assert len(new) == len(evenly_sampled_coord)
        assert new.step == evenly_sampled_coord.step
        assert new.stop == new_stop

    def test_update_limits_step(self, evenly_sampled_coord):
        """Ensure the step can be updated."""
        coord = evenly_sampled_coord
        new_step = coord.step * 2
        new = coord.update_limits(step=new_step)
        assert np.all(new.values == coord.values * 2)

    def test_update_limits_start_stop(self, evenly_sampled_coord):
        """Ensure both start/stop can be updated."""
        coord = evenly_sampled_coord
        start, stop = coord.values[2], coord.values[-2]
        new = coord.update_limits(min=start, max=stop)
        assert len(new) == len(coord)
        assert new.start == start
        assert new.stop == stop

    def test_update_limits_no_args(self, evenly_sampled_coord):
        """Ensure both start/stop can be updated."""
        coords = evenly_sampled_coord
        assert coords == coords.update_limits()

    def test_update_limits_timedelta64_different_precision(self):
        """Update limits with datetime64 that have different precisions."""
        start, stop = np.datetime64("2023-01-01"), np.datetime64("2023-01-01T01")
        step = np.timedelta64(60, "s")
        coord = get_coord(start=start, stop=stop, step=step)
        # change maximum value, keeping length the same and chaning step
        coord_new = coord.update_limits(max=stop + 10 * step)
        assert coord_new.max() == stop + 10 * step

    def test_test_select_end_floats(self, evenly_sampled_float_coord_with_units):
        """Ensure we can select right up to the end of the array."""
        coord = evenly_sampled_float_coord_with_units
        new_coord, out = coord.select((coord.min(), coord.max()))
        assert len(new_coord) == len(coord)
        assert np.allclose(new_coord.values, coord.values)

    def test_empty(self, coord):
        """Ensure coords can be emptied out."""
        new = coord.empty()
        assert isinstance(new, CoordDegenerate)
        assert new.dtype == coord.dtype

    def test_init_length_one(self):
        """Ensure len 1 coord can be inited provided step is supplied."""
        time = dc.to_datetime64(["2020-01-01"])
        dt = dc.to_timedelta64(0.09999)
        coord1 = get_coord(start=time[0], stop=time[0] + dt, step=dt)
        coord2 = get_coord(data=time, step=dt)
        assert isinstance(coord1, CoordRange)
        assert coord1 == coord2
        # both length and shape should be 1
        assert len(coord1) == 1
        assert coord1.shape == (1,)

    def test_monotonic_with_sampling(self):
        """Ensure init'ing monotonic array with sampling also works."""
        sample_rate = 1_000
        t_array = np.linspace(0.0, 1, 1000)
        sample_rate = 1 / sample_rate
        out = get_coord(data=t_array, step=sample_rate)
        assert out.shape == out.data.shape == (len(out),)

    def test_properties_coord_range(self, evenly_sampled_coord):
        """Test key properties for evenly sampled."""
        coord = evenly_sampled_coord
        assert coord.evenly_sampled
        assert coord.sorted
        assert not coord.reverse_sorted
        assert not coord.degenerate

    def test_coord_range_len_1(self):
        """Ensure coord range can have step size of 0."""
        new = CoordRange(start=0, stop=0, step=0)
        assert len(new) == 1
        assert np.all(new.values == np.zeros(0))

    def test_reversed_coord(self, evenly_sampled_reversed_coord):
        """Ensure reverse sampling works for evenly sampled coord."""
        coord = evenly_sampled_reversed_coord
        assert isinstance(coord, CoordRange)
        assert coord.evenly_sampled
        assert coord.reverse_sorted
        assert not coord.sorted

    def test_arrange_equiv(self):
        """Ensure np.arange produces same len as get_coord, start, stop, step."""
        start = -120.02095199999974
        stop = 1055.2182894582486
        step = 1.0209523838714072
        ar = np.arange(start, stop, step)
        coord = get_coord(start=start, stop=stop, step=step)
        assert coord.shape == ar.shape

    def test_unchanged_len(self):
        """Ensure an array converted to Coord range has same len. See #229."""
        t_array = np.linspace(0.0, 1, 44100)
        t_coord = get_coord(data=t_array)
        assert t_array.shape == t_coord.shape
        # now test datetime
        time = dc.to_timedelta64(t_array) + np.datetime64("2020-01-01")
        time_coord = get_coord(data=time)
        assert len(time) == len(time_coord)


class TestMonotonicCoord:
    """Tests for monotonic array coords."""

    def test_select_basic(self, monotonic_float_coord):
        """Basic select tests for monotonic array."""
        coord = monotonic_float_coord
        # test start only
        new, sliced = coord.select((5, ...))
        assert_value_in_one_step(coord, sliced.start, 5)
        assert new.min() >= 5
        # test stop only
        new, sliced = coord.select((..., 40))
        assert_value_in_one_step(coord, sliced.stop - 1, 40, greater=False)
        assert new.max() <= 40

    def test_sort(self, monotonic_float_coord):
        """Ensure sort returns equal coord."""
        out, _ = monotonic_float_coord.sort()
        assert out == monotonic_float_coord

    def test_reverse_monotonic(self):
        """Ensure reverse monotonic values can be handled."""
        ar = np.cumsum(np.abs(np.random.rand(100)))[::-1]
        coord = get_coord(data=ar)
        assert np.allclose(coord.values, ar)
        new, sliced = coord.select((ar[10], ar[20]))
        assert len(new) == 11 == len(new.values)
        assert np.allclose(new.values, coord.values[10:21])
        # test edges
        eps = 0.000000001
        new, sliced = coord.select((ar[0] + eps, ar[20] - eps))
        assert sliced.start is None
        # ensure messing with the ends keeps the order
        val1, val2 = ar[10] - eps, ar[20] + eps
        new, sliced = coord.select((val1, val2))
        assert sliced == slice(11, 20)
        assert new[0] <= val1
        assert new[-1] >= val2

    def test_index_with_string(self, monotonic_datetime_coord):
        """Ensure indexing works with date string."""
        coord = monotonic_datetime_coord
        value1 = coord.values[12] + np.timedelta64(1, "ns")
        new, sliced = coord.select((value1, ...))
        assert_value_in_one_step(coord, sliced.start, value1, greater=True)
        assert sliced.start == 13
        # now test exact value
        value2 = coord.values[12]
        new, sliced = coord.select((value2, ...))
        assert_value_in_one_step(coord, sliced.start, value2, greater=True)
        assert sliced.start == 12
        # test end index
        new, sliced = coord.select((..., value1))
        assert sliced.stop == 13
        new, sliced = coord.select((..., value2 - np.timedelta64(1, "ns")))
        assert sliced.stop == 12
        # test range
        new, sliced = coord.select((coord.values[10], coord.values[20]))
        new_vals = new.values
        assert coord.values[10] == new_vals[0]
        assert coord.values[20] == new_vals[-1]
        assert len(new) == 11
        assert slice(10, 21) == sliced

    def test_update_limits_too_many_params(self, monotonic_float_coord):
        """Update limits should raise if too many parameters are specified."""
        coords = monotonic_float_coord
        start, stop = coords.min() + 1, coords.max() - 1
        with pytest.raises(ValueError, match="At most one parameter"):
            coords.update_limits(start, stop)

    def test_update_limits_step(self, monotonic_float_coord):
        """Ensure step can be updated."""
        coord = monotonic_float_coord
        new = coord.update_limits(step=1.5)
        assert np.isclose(new.step, 1.5)
        assert new.min() == coord.min()

    def test_update_limits_end(self, monotonic_float_coord):
        """Ensure end can be updated."""
        coord = monotonic_float_coord
        new_stop = coord.max() - 10
        new = coord.update_limits(max=new_stop)
        assert len(new) == len(coord)
        assert new.max() == new_stop

    def test_update_limits_start(self, monotonic_float_coord):
        """Ensure the start can be updated."""
        coord = monotonic_float_coord
        new_start = coord.min() - 100
        new = coord.update_limits(min=new_start)
        assert len(new) == len(coord)
        assert np.isclose(new.min(), new_start)

    def test_update_limits_no_args(self, monotonic_float_coord):
        """Ensure both start/stop can be updated."""
        coords = monotonic_float_coord
        assert coords == coords.update_limits()

    def test_getitem_slice(self, monotonic_float_coord):
        """Ensure get_item_slice returns another coord."""
        new = monotonic_float_coord[slice(1, -1)]
        assert isinstance(new, monotonic_float_coord.__class__)
        assert (len(new) + 2) == len(monotonic_float_coord)

    def test_wide_filter_doesnt_change_size(self, monotonic_float_coord):
        """Ensure filtering outside data range doesn't change array size."""
        coord = monotonic_float_coord
        lims = coord.limits
        dur = lims[1] - lims[0]
        select_range = (lims[0] - 2 * dur, lims[1] + dur)
        wide_coord, inds = coord.select(select_range)
        # coord should be unchanged
        assert len(wide_coord) == len(coord)
        assert wide_coord == coord
        wide_coord, inds = coord.select((select_range[0], None))
        assert wide_coord == coord
        wide_coord, inds = coord.select((None, select_range[1]))
        assert wide_coord == coord

    def test_properties_mono(self, monotonic_float_coord):
        """Check a few properties for monotonic float coord."""
        coord = monotonic_float_coord
        assert coord.sorted
        assert not coord.evenly_sampled
        assert not coord.reverse_sorted
        assert not coord.degenerate

    def test_properties_reverse_mono(self, reverse_monotonic_float_coord):
        """Check a few properties for reverse monotonic float coord."""
        coord = reverse_monotonic_float_coord
        assert not coord.sorted
        assert not coord.evenly_sampled
        assert coord.reverse_sorted
        assert not coord.degenerate

    def test_max_degenerate(self, evenly_sampled_time_delta_coord):
        """Ensure time delta produces nullish value."""
        empty = evenly_sampled_time_delta_coord.empty()
        assert pd.isnull(empty.max())

    def test_single_value(self):
        """A single value coord should be considered sorted."""
        coord = get_coord(data=np.array([1]))
        assert coord.sorted
        assert not coord.reverse_sorted


class TestNonOrderedArrayCoords:
    """Tests for non-ordered array coords."""

    def test_select(self, random_coord):
        """Ensure selecting returns an ndarray."""
        min_v, max_v = np.min(random_coord.values), np.max(random_coord.values)
        dist = max_v - min_v
        val1, val2 = min_v + 0.2 * dist, max_v - 0.2 * dist
        new, bool_array = random_coord.select((val1, val2))
        assert np.all(new.values >= val1)
        assert np.all(new.values <= val2)
        assert bool_array.sum() == len(new)

    def test_sort(self, random_coord):
        """Ensure the coord can be ordered."""
        new, ordering = random_coord.sort()
        assert isinstance(new, CoordMonotonicArray)

    def test_snap(self, random_coord):
        """Ensure coords can be snapped to even sampling intervals."""
        out = random_coord.snap()
        assert isinstance(out, CoordRange)

    def test_snap_date(self, random_date_coord):
        """Ensure coords can be snapped to even sampling intervals."""
        out = random_date_coord.snap()
        assert np.issubdtype(out.dtype, np.datetime64)
        assert isinstance(out, CoordRange)

    def test_properties_random(self, random_coord):
        """Test key properties for evenly sampled."""
        coord = random_coord
        assert not coord.evenly_sampled
        assert not coord.sorted
        assert not coord.reverse_sorted
        assert not coord.degenerate


class TestDegenerateCoords:
    """Tests for degenerate coordinates."""

    def test_init_degen(self):
        """Ensure degen is inited by any sort of empty array."""
        arrays = [
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int_),
            np.empty((0, 10), dtype="datetime64[ns]"),
        ]
        for ar in arrays:
            out = get_coord(data=ar)
            assert isinstance(out, CoordDegenerate)
            assert out.dtype == ar.dtype
            assert len(out) == 0

    def test_select(self, degenerate_time_coord):
        """Selecting should simply return the same degenerate."""
        coord = degenerate_time_coord
        assert coord.select((10, 100))[0] == coord
        assert coord.select((None, 100))[0] == coord
        assert coord.select((10, None))[0] == coord
        assert coord.select((None, None))[0] == coord

    def test_empty(self, degenerate_time_coord):
        """Ensure empty just returns self."""
        assert degenerate_time_coord.empty() == degenerate_time_coord

    def test_min_max(self, degenerate_time_coord):
        """Ensure min/max are nullish."""
        assert pd.isnull(degenerate_time_coord.min())
        assert pd.isnull(degenerate_time_coord.max())

    def test_degenerate_with_step_from_array(self):
        """CoordRange should be possible empty."""
        ar = np.empty(0, dtype=int)
        coord = get_coord(data=ar, step=1)
        assert coord.step == 1
        assert coord.dtype == ar.dtype

    def test_properties_degenerate_from_coord_range(self, evenly_sampled_coord):
        """Test key properties for degenerate coord."""
        coord = evenly_sampled_coord.empty()
        assert not coord.evenly_sampled
        assert not coord.sorted
        assert not coord.reverse_sorted
        assert coord.degenerate


class TestCoercion:
    """Some data types should support coercion in selecting (eg dates)."""

    def test_date_str(self, evenly_sampled_date_coord):
        """Ensure date strings get coerced."""
        drange = ("1970-01-01T00:00:01", 10)
        out, indexer = evenly_sampled_date_coord.select(drange)
        assert isinstance(out, evenly_sampled_date_coord.__class__)
        assert out.dtype == evenly_sampled_date_coord.dtype

    def test_time_delta(self, evenly_sampled_time_delta_coord):
        """Ensure date strings get coerced."""
        coord = evenly_sampled_time_delta_coord
        drange = (10, 2_000)
        out, indexer = coord.select(drange)
        assert isinstance(out, coord.__class__)
        assert out.dtype == coord.dtype


class TestGetSampleCount:
    """Tests for getting the number of samples from a coordinate."""

    def test_uneven_raises(self, monotonic_float_coord):
        """Unevenly sampled counts should raise."""
        msg = "Coordinate is not evenly sampled"
        with pytest.raises(CoordError, match=msg):
            monotonic_float_coord.get_sample_count(10)

    def test_dt_lt_step(self, evenly_sampled_coord):
        """Ensure sample count is 1 when value < dt."""
        dt = evenly_sampled_coord.step * 0.1
        out = evenly_sampled_coord.get_sample_count(dt)
        assert out == 1

    def test_datetime(self, evenly_sampled_date_coord):
        """Ensure evenly sampled date coords work."""
        dt = evenly_sampled_date_coord.step
        out = evenly_sampled_date_coord.get_sample_count(12 * dt)
        assert out == 12

    def test_timedelta(self, evenly_sampled_time_delta_coord):
        """Ensure timedelta works."""
        dt = evenly_sampled_time_delta_coord.step
        out = evenly_sampled_time_delta_coord.get_sample_count(12 * dt)
        assert out == 12

    @pytest.mark.parametrize("sample", (0, 10, 100, 42, 13))
    def test_samples(self, evenly_sampled_coord, sample):
        """Ensure value is returned when samples==True."""
        assert len(evenly_sampled_coord) >= sample
        out = evenly_sampled_coord.get_sample_count(sample, samples=True)
        assert out == sample

    def test_overask_raises(self, evenly_sampled_coord):
        """Asking for more samples than length of coordinate should raise."""
        max_len = len(evenly_sampled_coord)
        duration = evenly_sampled_coord.max() - evenly_sampled_coord.min()
        step = evenly_sampled_coord.step
        # test for using samples kwargWindow or step size is larger than
        msg = "results in a window larger than coordinate"
        with pytest.raises(ParameterError, match=msg):
            evenly_sampled_coord.get_sample_count(max_len + 2, samples=True)
        # test for using normal mode
        with pytest.raises(ParameterError, match=msg):
            evenly_sampled_coord.get_sample_count(duration + 2 * step)

    def test_non_int_raises_with_samples(self, evenly_sampled_coord):
        """Non integer values should raise when sample=True."""
        match = "must be integers"
        with pytest.raises(ParameterError, match=match):
            evenly_sampled_coord.get_sample_count(1.2, samples=True)
        with pytest.raises(ParameterError, match=match):
            evenly_sampled_coord.get_sample_count("bob", samples=True)


class TestGetNextIndex:
    """Tests for finding the next index after a value."""

    def test_samples(self, evenly_sampled_coord):
        """Ensure samples argument allows samples to be passed in."""
        coord = evenly_sampled_coord
        assert coord.get_next_index(0, samples=True) == 0
        assert coord.get_next_index(-1.0, samples=True) == len(coord) - 1
        coord_len = len(coord)
        with pytest.raises(ValueError, match="is out of bounds"):
            coord.get_next_index(coord_len, samples=True)

    def test_samples_time(self, evenly_sampled_date_coord):
        """Ensure samples argument allows samples to be passed in."""
        coord = evenly_sampled_date_coord
        coord.get_next_index(0, samples=True)
        assert coord.get_next_index(0, samples=True) == 0
        assert coord.get_next_index(-1.0, samples=True) == len(coord) - 1

    def test_out_of_bounds_raises(self, evenly_sampled_coord):
        """Tests for out of bounds values raising exceptions."""
        msg = "is out of bounds"
        coord = evenly_sampled_coord
        step = coord.step
        max_value = coord.max()
        min_value = coord.min()
        # ensure out of bounds samples raises
        with pytest.raises(ValueError, match=msg):
            coord.get_next_index(len(coord) + 10, samples=True)
        with pytest.raises(ValueError, match=msg):
            coord.get_next_index(-len(coord) - 10, samples=True)
        # ensure out of bounds values raises
        with pytest.raises(ValueError, match=msg):
            coord.get_next_index(min_value - step)
        with pytest.raises(ValueError, match=msg):
            coord.get_next_index(max_value + step)

    def test_out_of_bounds_suppressed(self, evenly_sampled_coord):
        """Tests for out of bounds values returning when specified."""
        coord = evenly_sampled_coord
        step = coord.step
        max_value = coord.max()
        min_value = coord.min()
        func = partial(coord.get_next_index, allow_out_of_bounds=True)
        assert func(len(coord) + 10, samples=True) == len(coord) - 1
        assert func(-len(coord) - 10, samples=True) == 0
        assert func(min_value - step) == min_value
        assert func(max_value + step) == max_value

    def test_exact_values(self, evenly_sampled_coord):
        """Ensure using exact values contained in coord return index."""
        coord = evenly_sampled_coord
        for ind in [10, 1, 5]:
            assert coord.get_next_index(ind) == ind

    def test_between_values(self, evenly_sampled_coord):
        """Ensure values between coordinate values also work."""
        coord = evenly_sampled_coord
        val1, step = coord[1], coord.step
        ind1 = evenly_sampled_coord.get_next_index(val1 + step / 4)
        ind2 = evenly_sampled_coord.get_next_index(val1 + step * 3 / 4)
        assert ind1 == ind2 == 2

    def test_units(self, evenly_sampled_float_coord_with_units):
        """Ensure values with units work."""
        coord = evenly_sampled_float_coord_with_units
        val1 = 10 * get_quantity("m")
        val2 = val1.to(get_quantity("ft"))
        ind1 = coord.get_next_index(val1)
        ind2 = coord.get_next_index(val2)
        assert ind1 == ind2


class TestUpdate:
    """Tests for updating coordinates."""

    def test_update_data(self, coord):
        """Test updating data in the coordinate."""
        if coord.degenerate:
            return
        data = coord.data
        d_range = np.max(data) - np.min(data)
        new_data = data + d_range
        new_coord = coord.update(data=new_data)
        assert isinstance(new_coord, BaseCoord)
        assert all_close(new_coord.data, new_data)
        # using the keyword 'values' is deprecated but should also work.
        new_coord = coord.update_data(values=new_data)
        assert all_close(new_coord.data, new_data)


class TestIssues:
    """Tests for special issues related to coords."""

    @pytest.fixture(scope="class")
    def awkward_off_by_one(self):
        """Awkward array like one found in the wild which causes snap issues."""
        # Awkward, but this is a simplification of a time array found in the wild.
        # Unfortunately, these arrays have to be this large to trigger the off
        # by one error we are testing.
        dt1 = np.timedelta64(131457, "ns")
        dt2 = np.timedelta64(131456, "ns")

        t1 = dc.to_datetime64("2023-01-01T00:12:13.232123222")
        ar1 = t1 + np.arange(26926) * dt1

        t2 = t1.max() + 2.232323 * dt1
        ar2 = t2 + np.arange(45642 - len(ar1)) * dt2

        array = np.concatenate([ar1, ar2], axis=0)
        coord = dc.core.get_coord(data=array)
        return coord

    def test_event_1_gives_coord_range(self, event_patch_1):
        """Ensure patch1 gives the coord range."""
        time_vals = event_patch_1.coords.get_array("time")
        coord = get_coord(data=time_vals)
        assert coord.evenly_sampled

    def test_len_1_list(self):
        """A len 1 list should work as input to get_coords."""
        coord = get_coord(data=[0])
        assert isinstance(coord, BaseCoord)
        assert len(coord) == 1

    def test_update_degenerate_from_attrs(self, degenerate_time_coord):
        """Ensure updating dt on degenerate time coord doesn't fail."""
        # before this would cause infinite recursion.
        out = degenerate_time_coord.update(step=10)
        assert isinstance(out, BaseCoord)
        assert len(out) == 0

    def test_snap_awkward(self, awkward_off_by_one):
        """Ensure off by one error don't occur with snap."""
        out = awkward_off_by_one.snap()
        assert out.shape == awkward_off_by_one.shape
