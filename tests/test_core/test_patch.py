"""Tests for Trace2D object."""

from __future__ import annotations

import operator
import weakref

import numpy as np
import pandas as pd
import pytest
from rich.text import Text

import dascore as dc
from dascore.compat import random_state
from dascore.core import Patch
from dascore.core.coords import BaseCoord, CoordRange
from dascore.exceptions import CoordError, ParameterError
from dascore.proc.basic import apply_operator
from dascore.utils.misc import suppress_warnings


def get_simple_patch() -> Patch:
    """
    Return a small simple array for memory testing.

    Note: This cant be a fixture as pytest seems to hold a reference,
    even for function scoped outputs.
    """
    attrs = {"time_step": 1}
    pa = Patch(
        data=random_state.random((100, 100)),
        coords={"time": np.arange(100) * 0.01, "distance": np.arange(100) * 0.2},
        attrs=attrs,
        dims=("time", "distance"),
    )
    return pa


class TestInit:
    """Tests for init'ing Patch."""

    time1 = np.datetime64("2020-01-01")

    @pytest.fixture()
    def random_dt_coord(self):
        """Create a random patch with a datetime coord."""
        rand = np.random.RandomState(13)
        array = rand.random(size=(20, 200))
        attrs = dict(dx=1, time_step=1 / 250.0, category="DAS", id="test_data1")
        time_deltas = dc.to_timedelta64(np.arange(array.shape[1]) * attrs["time_step"])
        coords = dict(
            distance=np.arange(array.shape[0]) * attrs["dx"],
            time=self.time1 + time_deltas,
        )
        dims = tuple(coords)
        out = dict(data=array, coords=coords, attrs=attrs, dims=dims)
        return Patch(**out)

    @pytest.fixture(scope="class")
    def patch_complex_coords(self):
        """Create a patch with 'complex' (non-dimensional) coords."""
        rand = np.random.RandomState(13)
        array = rand.random(size=(20, 100))
        dt = 1 / 250.0
        attrs = dict(distance_step=1, time_step=dt, category="DAS", id="test_data1")
        time_deltas = dc.to_timedelta64(np.arange(array.shape[1]) * attrs["time_step"])
        coords = dict(
            distance=np.arange(array.shape[0]) * attrs["distance_step"],
            time=self.time1 + time_deltas,
            latitude=("distance", array[:, 0]),
            quality=(("distance", "time"), array),
        )
        dims = ("distance", "time")
        out = dict(data=array, coords=coords, attrs=attrs, dims=dims)
        return Patch(**out)

    @pytest.fixture(scope="class")
    def test_conflicting_attrs_coords_raises(self):
        """Patch for testing conflicting coordinates/attributes."""
        array = random_state.random((10, 10))
        # create attrs, these should all get overwritten by coords.
        attrs = dict(
            distance_step=10,
            time_step=dc.to_timedelta64(1),
            distance_min=1000,
            distance_max=2002,
            time_min=dc.to_datetime64("2017-01-01"),
            time_max=dc.to_datetime64("2019-01-01"),
        )
        # create coords
        coords = dict(
            time=dc.to_datetime64(np.cumsum(random_state.random(10))),
            distance=random_state.random(10),
        )
        # assemble and output.
        dims = ("distance", "time")
        out = dict(data=array, coords=coords, attrs=attrs, dims=dims)
        msg = "Coords and attrs are incompatible."
        with pytest.raises(ValueError, match=msg):
            dc.Patch(**out)

    def test_start_time_inferred_from_dt64_coords(self, random_dt_coord):
        """Ensure the time_min and time_max attrs can be inferred from coord time."""
        patch = random_dt_coord
        assert patch.attrs["time_min"] == self.time1

    def test_end_time_inferred_from_dt64_coords(self, random_dt_coord):
        """Ensure the time_min and time_max attrs can be inferred from coord time."""
        patch = random_dt_coord
        time = patch.coords.coord_map["time"].max()
        assert patch.attrs["time_max"] == time

    def test_init_from_array(self, random_patch):
        """Ensure a trace can be created from raw components; array, coords, attrs."""
        assert isinstance(random_patch, Patch)

    def test_max_time_populated(self, random_patch):
        """Ensure the time_max is populated when not explicitly given."""
        end_time = random_patch.attrs["time_max"]
        assert not pd.isnull(end_time)

    def test_min_max_populated(self, random_patch):
        """The min/max values of the distance attrs should have been populated."""
        attrs = random_patch.attrs
        attr_class = type(attrs)
        expected_filled_in = [
            x
            for x in list(attr_class.model_fields)
            if x.startswith("distance") and "units" not in x
        ]
        for attr in expected_filled_in:
            assert not pd.isnull(attrs[attr])

    def test_had_default_attrs(self, patch):
        """Test that all patches used in the test suite have default attrs."""
        attr_set = set(patch.attrs.model_dump())
        assert attr_set.issubset(set(patch.attrs.model_dump()))

    def test_no_attrs(self):
        """Ensure a patch with no attrs can be created."""
        pa = Patch(
            data=random_state.random((100, 100)),
            coords={
                "time": random_state.random(100),
                "distance": random_state.random(100),
            },
            dims=("time", "distance"),
        )
        assert isinstance(pa, Patch)

    def test_seconds_as_coords_time_no_dt(self):
        """Ensure seconds passed as coordinates with no attrs still works."""
        ar = get_simple_patch()
        assert not np.any(pd.isnull(ar.coords.get_array("time")))

    def test_shape(self, random_patch):
        """Ensure shape returns the shape of the data array."""
        assert random_patch.shape == random_patch.data.shape

    def test_init_with_complex_coordinates(self, patch_complex_coords):
        """Ensure complex coordinates work."""
        patch = patch_complex_coords
        assert isinstance(patch, Patch)
        assert "latitude" in patch.coords
        assert "quality" in patch.coords
        assert np.all(patch.coords.get_array("quality") == patch.data)

    def test_incomplete_raises(self):
        """An incomplete patch should raise an error."""
        data = np.ones((10, 10))
        with pytest.raises(ValueError, match="data, coords, and dims"):
            Patch(data=data)

    def test_coords_from_1_element_array(self):
        """Ensure CoordRange is still returned despite 1D array in time."""
        patch = dc.get_example_patch("random_das", shape=(100, 1))
        time_coord = patch.get_coord("time")
        assert isinstance(time_coord, CoordRange)

    def test_sin_wave_patch(self):
        """Ensure the sin wave patch is consistent with its coord dims."""
        # For some reason this combination can make coords with wrong shape.
        patch = dc.examples.sin_wave_patch(
            sample_rate=1000,
            frequency=[200, 10],
            channel_count=2,
        )
        assert patch.shape == patch.coords.shape == patch.data.shape
        time_shape = patch.shape[patch.get_axis("time")]
        assert time_shape == len(patch.coords.get_array("time"))
        assert time_shape == len(patch.coords.get_array("time"))

    def test_init_no_coords(self, random_patch):
        """Ensure a new patch can be inited from only attrs."""
        attrs = random_patch.attrs
        new = random_patch.__class__(attrs=attrs, data=random_patch.data)
        assert isinstance(new, dc.Patch)

    def test_init_with_patch(self, random_patch):
        """Ensure a patch inited on a patch just returns a patch."""
        new = dc.Patch(random_patch)
        assert new == random_patch

    def test_non_time_distance_dims(self):
        """Ensure dimensions other than time/distance work."""
        x = np.arange(10) * 2
        y = np.arange(10) * 3
        data = np.add.outer(y, x).astype(np.float64)
        patch = dc.Patch(data=data, coords={"x": x, "y": y}, dims=("x", "y"))
        assert isinstance(patch, dc.Patch)

    def test_patch_has_size(self, random_patch):
        """Ensure patches have same size as data."""
        assert random_patch.size == random_patch.data.size

    def test_new_patch_non_standard_dims(self):
        """Ensure a non-standard dimension has matching dims in attrs and coords."""
        data = random_state.rand(10, 5)
        coords = {"time": np.arange(10), "can": np.arange(5)}
        patch = dc.Patch(data=data, coords=coords, dims=("time", "can"))
        assert patch.dims == patch.attrs.dim_tuple

    def test_non_coord_dims(self):
        """Ensure non-coordinate dimensions can work and create non-coord."""
        data = random_state.rand(10, 5)
        coords = {"time": np.arange(10)}
        patch = dc.Patch(data=data, coords=coords, dims=("time", "money"))
        assert patch.dims == ("time", "money")


class TestNew:
    """Tests for `Patch.new` method."""

    def test_erase_history(self, random_patch):
        """Ensure new can erase history."""
        # do some processing, so history shows up.
        patch = random_patch.pass_filter(time=(None, 10)).decimate(time=5)
        assert patch.attrs.history
        new_attrs = dict(patch.attrs)
        new_attrs["history"] = []
        new_patch = patch.update(attrs=new_attrs)
        assert not new_patch.attrs.history

    def test_new_coord_dict_order(self, random_patch):
        """Ensure data can be init'ed with a dict of coords in any orientation."""
        patch = random_patch
        axis = patch.get_axis("time")
        data = np.std(patch.data, axis=axis, keepdims=True)
        new_time = patch.coords.get_array("time")[0:1]
        new_dist = patch.coords.get_array("distance")
        coords_1 = {"time": new_time, "distance": new_dist}
        coords_2 = {"distance": new_dist, "time": new_time}
        # the order the coords are defined shouldn't matter
        out_1 = patch.new(data=data, coords=coords_1)
        out_2 = patch.new(data=data, coords=coords_2)
        assert out_1 == out_2

    def test_attrs_preserved_when_not_specified(self, random_patch):
        """If attrs is not passed to new, old attrs should remain."""
        pa = random_patch.update_attrs(network="bob", tag="2", station="10")
        new_1 = pa.update(data=pa.data * 10)
        assert new_1.attrs == pa.attrs

    def test_new_dims_renames_dims(self, random_patch):
        """Ensure new can rename dimensions."""
        dims = ("tom", "jerry")
        out = random_patch.new(dims=dims)
        assert out.dims == dims


class TestDisplay:
    """Tests for displaying patches."""

    def test_str(self, patch):
        """All patches should have str rep."""
        out = str(patch)
        assert isinstance(out, str)
        assert len(out)

    def test_rich(self, patch):
        """Patches should also have rich representation."""
        out = patch.__rich__()
        assert isinstance(out, Text)

    def test_empty_patch_str(self):
        """An empty patch should have both a str and repr of nonzero length."""
        patch = Patch()
        assert len(str(patch))
        assert len(repr(patch))


class TestEmptyPatch:
    """Tests for empty patch objects."""

    def test_no_dims(self):
        """An empty test patch should have no dims or related attrs."""
        patch = Patch()
        attrs = patch.attrs
        assert len(attrs)
        assert not len(patch.dims)


class TestEquals:
    """Tests for checking equality."""

    def test_equal_self(self, random_patch):
        """Ensure a trace equals itself."""
        assert random_patch.equals(random_patch)

    def test_non_equal_array(self, random_patch):
        """Ensure the traces are not equal if the data are not equal."""
        new_data = random_patch.data + 1
        new = random_patch.new(data=new_data)
        assert not new.equals(random_patch)

    def test_coords_named_differently(self, random_patch):
        """Ensure if the coords are named differently patches are not equal."""
        dims = random_patch.dims
        with suppress_warnings():
            new_coords = dict(random_patch.coords)
        new_coords["bob"] = new_coords.pop(dims[-1])
        new_dims = tuple(list(dims)[:-1] + ["bob"])
        patch_2 = random_patch.new(coords=new_coords, dims=new_dims)
        assert not patch_2.equals(random_patch)

    def test_coords_not_equal(self, random_patch):
        """Ensure if the coords are not equal neither are the arrays."""
        with suppress_warnings():
            new_coords = dict(random_patch.coords)
        new_coords["distance"] = new_coords["distance"] + 10
        patch_2 = random_patch.new(coords=new_coords)
        assert not patch_2.equals(random_patch)

    def test_attrs_not_equal(self, random_patch):
        """Ensure if the attributes are not equal the arrays are not equal."""
        attrs = random_patch.attrs
        new_attr_dict = {}
        new_attr_dict["time_step"] = attrs["time_step"] * 0.80
        patch2 = random_patch.new(attrs=new_attr_dict)
        assert patch2.attrs.time_step == new_attr_dict["time_step"]
        assert not patch2.equals(random_patch)

    def test_one_null_value_in_attrs(self, random_patch):
        """Ensure setting a value to null in attrs doesn't eval to equal."""
        attrs = dict(random_patch.attrs)
        attrs["tag"] = ""
        patch2 = random_patch.new(attrs=attrs)
        patch2.equals(random_patch)
        assert not patch2.equals(random_patch)

    def test_extra_attrs(self, random_patch):
        """
        Ensure extra attrs in one patch still eval equal unless only_required
        is used.
        """
        attrs = dict(random_patch.attrs)
        attrs["new_label"] = "fun4eva"
        patch2 = random_patch.new(attrs=attrs)
        # they should be equal
        assert patch2.equals(random_patch)
        # until extra labels are allowed
        assert not patch2.equals(random_patch, only_required_attrs=False)

    def test_both_attrs_nan(self, random_patch):
        """If Both Attrs are some type of NaN the patches should be equal."""
        attrs1 = dict(random_patch.attrs)
        attrs1["label"] = np.nan
        patch1 = random_patch.new(attrs=attrs1)
        attrs2 = dict(attrs1)
        attrs2["label"] = None
        patch2 = random_patch.new(attrs=attrs2)
        assert patch1.equals(patch2)

    def test_transposed_patches_not_equal(self, random_patch):
        """Transposed patches are not considered equal."""
        transposed = random_patch.transpose()
        # make sure dims are NE
        assert transposed.attrs.dims != random_patch.attrs.dims
        assert transposed.dims != random_patch.dims
        # and test equality
        assert not random_patch.equals(transposed)

    def test_one_coord_not_equal(self, wacky_dim_patch):
        """Ensure coords being close but not equals fails equality."""
        patch = wacky_dim_patch
        coords = patch.coords
        coord_array = np.array(coords.coord_map["distance"].values)
        coord_array[20:30] *= 0.9
        assert not np.allclose(coord_array, coords.get_array("distance"))
        new_patch = patch.update_coords(distance=coord_array)
        new = patch.update(coords=new_patch.coords)
        assert new != patch

    def test_other_types(self, random_patch):
        """Ensure a non-patch is not equal."""
        assert not random_patch.equals(None)
        assert not random_patch.equals(1)
        assert not random_patch.equals(random_patch.data)

    def test_negative_equals(self, random_patch):
        """Ensure negative of random patch is equal to negative of same patch."""
        assert -random_patch == -random_patch
        assert (-random_patch).abs() == random_patch

    def test_close(self, random_patch):
        """Test the `close` parameter"""
        new = random_patch * 0.999999999999999
        assert not new.equals(random_patch, close=False)
        assert new.equals(random_patch, close=True)
        # Ensure equals returns false when arrays are not close.
        not_close = random_patch * 0.1
        assert not not_close.equals(random_patch, close=True)


class TestTranspose:
    """Tests for switching dimensions."""

    def test_simple_transpose(self, random_patch):
        """Just check dims and data shape are reversed."""
        dims = random_patch.dims
        dims_r = tuple(reversed(dims))
        pa1 = random_patch
        pa2 = pa1.transpose(*dims_r)
        assert pa2.dims == dims_r
        assert list(pa2.data.shape) == list(reversed(pa1.data.shape))

    def test_t_property(self, random_patch):
        """Ensure the .T property returns the same as the transpose."""
        assert random_patch.T == random_patch.transpose()


class TestUpdateAttrs:
    """
    Tests for updating the attrs.

    Note: Most of the testing for this functionality is actually done on
    dascore.util.patch.AttrsCoordsMixer.
    """

    def test_add_attr(self, random_patch):
        """Tests for adding an attribute."""
        new = random_patch.update_attrs(bob=1)
        assert "bob" in new.attrs.model_dump() and new.attrs["bob"] == 1

    def test_original_unchanged(self, random_patch):
        """Updating attributes shouldn't change original patch in any way."""
        old_attrs = random_patch.attrs.model_dump()
        _ = random_patch.update_attrs(bob=2)
        assert not hasattr(random_patch.attrs, "bob")
        assert random_patch.attrs.model_dump() == old_attrs

    def test_update_starttime1(self, random_patch):
        """Ensure coords are updated with attrs."""
        t1 = np.datetime64("2000-01-01")
        pa = random_patch.update_attrs(time_min=t1)
        assert pa.attrs["time_min"] == t1
        assert pa.coords.min("time") == t1

    def test_update_startttime2(self, random_patch):
        """Updating start time should update end time as well."""
        duration = random_patch.attrs["time_max"] - random_patch.attrs["time_min"]
        new_start = np.datetime64("2000-01-01")
        pa1 = random_patch.update_attrs(time_min=str(new_start))
        assert pa1.attrs["time_min"] == new_start
        assert pa1.attrs["time_max"] == new_start + duration

    def test_dt_is_datetime64(self, random_patch):
        """Ensure dt gets changed into timedelta64."""
        d_time = random_patch.attrs["time_step"]
        assert isinstance(d_time, np.timedelta64)
        new1 = random_patch.update_attrs(time_step=10)
        assert new1.attrs["time_step"] == dc.to_timedelta64(10)

    def test_update_non_sorted_coord(self, wacky_dim_patch):
        """Ensure update attrs updates non-sorted coordinates."""
        # test updating dist max
        pa = wacky_dim_patch.update_attrs(distance_max=10)
        assert pa.attrs.distance_max == 10
        assert not np.any(pd.isnull(pa.coords.get_array("distance")))
        # test update dist min
        pa = wacky_dim_patch.update_attrs(distance_min=10)
        assert pa.attrs.distance_min == 10
        assert not np.any(pd.isnull(pa.coords.get_array("distance")))

    def test_update_units(self, random_patch):
        """Ensure units can be updated in attrs."""
        new_dist = "ft"
        patch1 = random_patch.update_attrs(distance_units=new_dist)
        patch2 = random_patch.convert_units(distance=new_dist)
        coord1 = patch1.get_coord("distance")
        coord2 = patch2.get_coord("distance")
        coord3 = random_patch.get_coord("distance").convert_units(new_dist)
        assert coord1 == coord2 == coord3
        assert patch1 == patch2


class TestReleaseMemory:
    """Ensure memory is released when the patch is deleted."""

    def test_single_patch(self):
        """Ensure a single patch is gc'ed when it leaves scope."""
        simple_patch = get_simple_patch()
        wr = weakref.ref(simple_patch.data)
        # delete pa and ensure the array was collected.
        del simple_patch
        assert wr() is None

    def test_decimated_patch(self):
        """
        Ensure a process which changes the array lets the first array get gc'ed.

        This is very important since the main reason to decimate is to preserve
        memory.
        """
        patch = get_simple_patch()
        new = patch.decimate(time=10)
        wr = weakref.ref(patch.data)
        del patch
        assert isinstance(new, Patch)
        assert wr() is None

    def test_select(self):
        """A similar test to ensure select releases old memory."""
        patch = get_simple_patch()
        new = patch.select(time=[0.1, 10], copy=True)
        wr = weakref.ref(patch.data)
        del patch
        assert isinstance(new, Patch)
        assert wr() is None


class TestPipe:
    """Tests for piping Patch to other functions."""

    @staticmethod
    def pipe_func(patch, positional_arg, keyword_arg=None):
        """Just add passed arguments to stats, return new patch."""
        out = patch.update_attrs(positional_arg=positional_arg, keyword_arg=keyword_arg)
        return out

    def test_pipe_basic(self, random_patch):
        """Test simple application of pipe."""
        out = random_patch.pipe(self.pipe_func, 2)
        assert out.attrs["positional_arg"] == 2
        assert out.attrs["keyword_arg"] is None

    def test_keyword_arg(self, random_patch):
        """Ensure passing a keyword arg works."""
        out = random_patch.pipe(self.pipe_func, 2, keyword_arg="bob")
        assert out.attrs["positional_arg"] == 2
        assert out.attrs["keyword_arg"] == "bob"


class TestCoords:
    """Test various things about patch coordinates."""

    @pytest.fixture(scope="class")
    def random_patch_with_lat(self, random_patch):
        """Create a random patch with added lat/lon coordinates."""
        dist = random_patch.coords.get_array("distance")
        lat = np.arange(0, len(dist)) * 0.001 - 109.857952
        # add a single coord
        out = random_patch.update_coords(latitude=("distance", lat))
        return out

    def test_add_single_dim_one_coord(self, random_patch_with_lat):
        """Tests that one coordinate can be added to a patch."""
        assert "latitude" in random_patch_with_lat.coords

    def test_add_single_dim_two_coord2(self, random_patch_with_lat_lon):
        """Ensure multiple coords can be added to patch."""
        out2 = random_patch_with_lat_lon
        assert {"latitude", "longitude"}.issubset(set(out2.coords.coord_map))
        assert out2.coords.get_array("longitude").shape
        assert out2.coords.get_array("latitude").shape

    def test_add_multi_dim_coords(self, multi_dim_coords_patch):
        """Ensure coords with multiple dimensions works."""
        out1 = multi_dim_coords_patch
        assert "quality" in out1.coords
        assert out1.coords.get_array("quality").shape
        assert np.all(out1.coords.get_array("quality") == 1)

    def test_coord_time_narrow_select(self, multi_dim_coords_patch):
        """Ensure the coord type doesn't change in narrow slice."""
        patch = multi_dim_coords_patch
        time = patch.coords.coord_map["time"]
        new = patch.select(time=(time.min(), time.min()))
        assert 1 in new.shape
        new_coords = new.coords.coord_map
        assert isinstance(new_coords["time"], CoordRange)

    def test_seconds(self, random_patch_with_lat):
        """Ensure we can get number of seconds in the patch."""
        sampling_interval = random_patch_with_lat.attrs["time_step"] / np.timedelta64(
            1, "s"
        )
        expected = (
            random_patch_with_lat.attrs["time_max"]
            - random_patch_with_lat.attrs["time_min"]
        ) / np.timedelta64(1, "s") + sampling_interval
        assert random_patch_with_lat.seconds == expected

    def test_channel_count(self, random_patch_with_lat):
        """Ensure we can get number of channels in the patch."""
        expected = len(random_patch_with_lat.get_coord("distance"))
        assert random_patch_with_lat.channel_count == expected


class TestApplyOperator:
    """Tests for applying various ufunc-type operators."""

    ops = (
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.pow,
        operator.ge,
        operator.le,
        operator.gt,
        operator.lt,
    )
    bool_ops = (
        operator.and_,
        operator.or_,
    )

    @pytest.fixture(params=ops)
    def operation(self, request):
        """Parameterization to get operator."""
        return request.param

    def test_scalar1(self, random_patch, operation):
        """Tests for scalar operators."""
        new = operation(random_patch, 2)
        expected = operation(random_patch.data, 2)
        assert np.allclose(new.data, expected)

    def test_array_like(self, random_patch):
        """Ensure array-like operations work."""
        ones = np.ones(random_patch.shape)
        new = apply_operator(np.add, random_patch, ones)
        assert np.allclose(new.data, ones + random_patch.data)

    def test_comparison_ops(self, random_patch):
        """Simple tests for comparison operations."""
        out = random_patch > random_patch
        assert np.issubdtype(out.dtype, np.bool_)
        assert not out.all()

    def test_reverse_subtraction(self):
        """Test that reverse subtraction works correctly."""
        patch = dc.get_example_patch("random_das")
        scalar = 10.0
        # Test reverse subtraction: scalar - patch
        result = scalar - patch
        # The result should be scalar - patch.data, not patch.data - scalar
        expected = scalar - patch.data
        assert np.allclose(result.data, expected)

    def test_reverse_subtraction_with_array(self):
        """Test reverse subtraction with array."""
        patch = dc.get_example_patch("random_das")
        array = np.ones_like(patch.data) * 5.0
        # Test array - patch
        result = array - patch
        # Should be array - patch.data
        expected = array - patch.data
        assert np.allclose(result.data, expected)


class TestBool:
    """Tests for boolean operators on Patches."""

    def test_multi_dimensional_patch_raises(self, random_patch):
        """
        Much like numpy, the boolean conversion of an array is ambiguous.
        """
        with pytest.raises(ValueError, match="is ambiguous"):
            bool(random_patch)

    def test_single_value_returns_array_bool(self, random_patch):
        """An array with a single value should return that values truthiness."""
        truthy = (random_patch.abs() >= 0).all()
        falsey = (random_patch.abs() < 0).all()

        assert truthy
        assert not falsey

    def test_boolean_comparisons(self, random_patch):
        """Test that boolean comps work with number first and patch first."""
        pa = random_patch
        gt = pa > 0
        assert isinstance(gt, dc.Patch)
        assert gt.data.dtype == np.bool_
        rgt = 0 < pa
        assert rgt.equals(gt)
        # equality across self should be all True for <= and >=
        assert np.all((pa <= pa).data)
        assert np.all((pa >= pa).data)

    def test_boolean_comparisons_with_units(self, random_patch):
        """Ensure boolean comps work with units."""
        pa = random_patch.set_units("m/s")
        m = dc.get_quantity("m")
        s = dc.get_quantity("s")
        q = 10 * m / s
        out = pa > q
        assert isinstance(out, dc.Patch)
        assert out.data.dtype == np.bool_
        # Units should be gone
        assert dc.get_quantity(out.attrs.data_units) is None

    def test_patch_units_dropped(self, patch):
        """Ensure data units are dropped with boolean ops"""
        patch1 = patch < 0
        patch2 = patch.isinf()
        assert dc.get_quantity(patch1.attrs.data_units) is None
        assert dc.get_quantity(patch2.attrs.data_units) is None


class TestGetCoord:
    """Tests for retrieving coords and imposing requirements."""

    def test_simple(self, random_patch):
        """Ensure a simple coordinate is returned."""
        coord = random_patch.get_coord("time")
        assert coord == random_patch.coords.coord_map["time"]

    def test_raises_not_evenly_sampled(self, wacky_dim_patch):
        """Ensure an error is raised when coord is not evenly sampled, if specified."""
        patch = wacky_dim_patch
        assert isinstance(patch.get_coord("time"), BaseCoord)
        match = "Coordinate time is not evenly sampled"
        with pytest.raises(CoordError, match=match):
            patch.get_coord("time", require_evenly_sampled=True)

    def test_raises_not_sorted(self, wacky_dim_patch):
        """Ensure an error is raised when coord is not sorted, if specified."""
        patch = wacky_dim_patch
        assert isinstance(patch.get_coord("distance"), BaseCoord)
        match = "Coordinate distance is not sorted"
        with pytest.raises(CoordError, match=match):
            patch.get_coord("distance", require_sorted=True)


class TestDeprecations:
    """Ensure deprecations are issued."""

    def test_assign_coords_deprecated(self, random_patch):
        """assign_coords should issue dep. warning."""
        new_coord = random_patch.coords.get_array("time")

        with pytest.warns(DeprecationWarning):
            random_patch.assign_coords(new_time=("time", new_coord))


class TestSetDims:
    """Tests for setting dimensions."""

    # note: patch.set_dims just passes work down too CoordManager.set_dims
    # which is tested more thoroughly
    def test_set_dim(self, random_patch_with_lat_lon):
        """Simple test for set_dims."""
        patch = random_patch_with_lat_lon
        out = patch.set_dims(distance="longitude")
        assert "longitude" in out.dims
        assert "longitude" in out.coords.dim_map["distance"]

    def test_doctest_example(self, random_patch):
        """Ensure the doctest example works."""
        patch = random_patch
        my_coord = random_state.random(patch.coord_shapes["time"])
        out = patch.update_coords(my_coord=("time", my_coord)).set_dims(  # add my_coord
            time="my_coord"
        )  # set mycoord as dim (rather than time)
        assert "my_coord" in out.dims


class TestHistory:
    """Specific tests for tracking history of Patches."""

    def test_history_is_tuple(self, random_patch):
        """The history attribute should be immutable. See #417."""
        assert isinstance(random_patch.attrs.history, tuple)

    def test_history_tuple_after_operation(self, random_patch):
        """Ensure the history tuple remains after a patch operation."""
        patch = random_patch.pass_filter(time=(..., 20))
        assert isinstance(patch.attrs.history, tuple)

        old_len = len(random_patch.attrs.history)
        new_len = len(patch.attrs.history)
        assert old_len == new_len - 1

    def test_history_not_too_long(self, random_patch):
        """Ensure a single history entry doesnt get too long."""
        patch = random_patch
        dist = patch.get_array("distance")
        patch_new_dist = patch.update_coords(distance=(dist + 12))
        # this should use the contracted array rep
        hist = patch_new_dist.attrs.history
        entry = hist[-1]
        array_part = entry.split("'[")[-1].split("]'")[0]
        assert len(array_part) < 100


class TestGetPatchName:
    """Tests for getting the name of a patch."""

    def test_simple_get_name(self, random_patch):
        """Happy path test."""
        name = random_patch.get_patch_name()
        assert isinstance(name, str)


class TestNumpyFuncs:
    """Tests for apply numpy directly to patches."""

    def test_reducer_function(self, random_patch):
        """Ensure numpy functions can return patches."""
        out = np.min(random_patch, axis=1)
        assert isinstance(out, dc.Patch)
        # The functions should behave the same as the methods.
        assert random_patch.min(dim=random_patch.dims[1]).equals(out)

    def test_accumulator(self, random_patch):
        """Ensure accumulator method works."""
        out = np.add.accumulate(random_patch, axis=0)
        assert isinstance(out, dc.Patch)

    def test_reduce(self, random_patch):
        """Ensure reduce also works."""
        out = np.multiply.reduce(random_patch, axis=0)
        assert isinstance(out, dc.Patch)

    def test_non_reducer(self, random_patch):
        """Ensure a non-reducing function also works."""
        out = np.cumsum(random_patch, axis=0)
        assert isinstance(out, dc.Patch)
        assert out.shape == random_patch.shape

    def test_patch_on_patch(self, random_patch):
        """Ensure two patches can be passed to numpy functions."""
        funcs = [np.add, np.subtract, np.multiply, np.divide]
        for func in funcs:
            out = func(random_patch, random_patch)
            assert isinstance(out, dc.Patch)

    def test_at_raises(self, random_patch):
        """Ensure unupported ufuncs raise."""
        msg = "ufuncs"
        with pytest.raises(ParameterError, match=msg):
            np.multiply.at(random_patch, [1, 20], random_patch)

    def test_complete_reduction(self, random_patch):
        """Ensure a compete reduction works."""
        out = np.min(random_patch)
        assert isinstance(out, dc.Patch)
        assert out.size == 1
        assert out.ndim == random_patch.ndim

    def test_multiple_axes(self, random_patch):
        """Ensure multiple axes work."""
        out = np.min(random_patch, axis=(0, 1))
        assert isinstance(out, dc.Patch)

    @pytest.mark.parametrize("name", ("add", "subtract", "divide"))
    def test_some_binary_ufuncs(self, name, random_patch):
        """Ensure some binary ufuncs on the patch work."""
        func = getattr(random_patch, name)
        # Test ufunc against other patch.
        out = func(random_patch)
        assert isinstance(out, dc.Patch)
        # Test ufunc reduce
        time_ind = random_patch.get_axis("time")
        out = func.reduce("time")
        assert isinstance(out, dc.Patch)
        assert out.shape[time_ind] == 1
        # Test ufunc accumulate
        out = func.accumulate("time")
        assert isinstance(out, dc.Patch)
