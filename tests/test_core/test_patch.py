"""
Tests for Trace2D object.
"""
import operator
import weakref

import numpy as np
import pandas as pd
import pytest
from rich.text import Text

import dascore as dc
from dascore.core import Patch
from dascore.core.coords import CoordRange
from dascore.proc.basic import apply_operator


def get_simple_patch() -> Patch:
    """
    Return a small simple array for memory testing.

    Note: This cant be a fixture as pytest seems to hold a reference,
    even for function scoped outputs.
    """
    attrs = {"d_time": 1}
    pa = Patch(
        data=np.random.random((100, 100)),
        coords={"time": np.arange(100) * 0.01, "distance": np.arange(100) * 0.2},
        attrs=attrs,
        dims=("time", "distance"),
    )
    return pa


class TestInit:
    """Tests for init'ing Patch"""

    time1 = np.datetime64("2020-01-01")

    @pytest.fixture()
    def random_dt_coord(self):
        """Create a random patch with a datetime coord"""
        rand = np.random.RandomState(13)
        array = rand.random(size=(20, 200))
        attrs = dict(dx=1, d_time=1 / 250.0, category="DAS", id="test_data1")
        time_deltas = dc.to_timedelta64(np.arange(array.shape[1]) * attrs["d_time"])
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
        attrs = dict(d_distance=1, d_time=dt, category="DAS", id="test_data1")
        time_deltas = dc.to_timedelta64(np.arange(array.shape[1]) * attrs["d_time"])
        coords = dict(
            distance=np.arange(array.shape[0]) * attrs["d_distance"],
            time=self.time1 + time_deltas,
            latitude=("distance", array[:, 0]),
            quality=(("distance", "time"), array),
        )
        dims = ("distance", "time")
        out = dict(data=array, coords=coords, attrs=attrs, dims=dims)
        return Patch(**out)

    @pytest.fixture(scope="class")
    def patch_conflicting_attrs_coords(self):
        """Patch for testing conflicting coordinates/attributes"""
        array = np.random.random((10, 10))
        # create attrs, these should all get overwritten by coords.
        attrs = dict(
            d_distance=10,
            d_time=dc.to_timedelta64(1),
            distance_min=1000,
            distance_max=2002,
            time_min=dc.to_datetime64("2017-01-01"),
            time_max=dc.to_datetime64("2019-01-01"),
        )
        # create coords
        coords = dict(
            time=dc.to_datetime64(np.cumsum(np.random.random(10))),
            distance=np.random.random(10),
        )
        # assemble and output.
        dims = ("distance", "time")
        out = dict(data=array, coords=coords, attrs=attrs, dims=dims)
        patch = dc.Patch(**out)
        return patch

    def test_start_time_inferred_from_dt64_coords(self, random_dt_coord):
        """
        Ensure the time_min and time_max attrs can be inferred from coord time.
        """
        patch = random_dt_coord
        assert patch.attrs["time_min"] == self.time1

    def test_end_time_inferred_from_dt64_coords(self, random_dt_coord):
        """
        Ensure the time_min and time_max attrs can be inferred from coord time.
        """
        patch = random_dt_coord
        time = patch.coords.coord_map["time"].max()
        assert patch.attrs["time_max"] == time

    def test_init_from_array(self, random_patch):
        """Ensure a trace can be created from raw components; array, coords, attrs"""
        assert isinstance(random_patch, Patch)

    def test_max_time_populated(self, random_patch):
        """Ensure the time_max is populated when not explicitly given."""
        end_time = random_patch.attrs["time_max"]
        assert not pd.isnull(end_time)

    def test_min_max_populated(self, random_patch):
        """The min/max values of the distance attrs should have been populated."""
        attrs = random_patch.attrs
        expected_filled_in = [
            x
            for x in list(attrs.__fields__)
            if x.startswith("distance") and "units" not in x
        ]
        for attr in expected_filled_in:
            assert not pd.isnull(attrs[attr])

    def test_had_default_attrs(self, patch):
        """Test that all patches used in the test suite have default attrs."""
        attr_set = set(patch.attrs.dict())
        assert attr_set.issubset(set(patch.attrs.dict()))

    def test_no_attrs(self):
        """Ensure a patch with no attrs can be created."""
        pa = Patch(
            data=np.random.random((100, 100)),
            coords={"time": np.random.random(100), "distance": np.random.random(100)},
            dims=("time", "distance"),
        )
        assert isinstance(pa, Patch)

    def test_seconds_as_coords_time_no_dt(self):
        """Ensure seconds passed as coordinates with no attrs still works."""
        ar = get_simple_patch()
        assert not np.any(pd.isnull(ar.coords["time"]))

    def test_shape(self, random_patch):
        """Ensure shape returns the shape of the data array."""
        assert random_patch.shape == random_patch.data.shape

    def test_init_with_complex_coordinates(self, patch_complex_coords):
        """Ensure complex coordinates work."""
        patch = patch_complex_coords
        assert isinstance(patch, Patch)
        assert "latitude" in patch.coords
        assert "quality" in patch.coords
        assert np.all(patch.coords["quality"] == patch.data)

    def test_incomplete_raises(self):
        """An incomplete patch should raise an error."""
        data = np.ones((10, 10))
        with pytest.raises(ValueError, match="data, coords, and dims"):
            Patch(data=data)

    def test_coords_from_1_element_array(self):
        """Ensure CoordRange is still returned despite 1D array in time."""
        patch = dc.get_example_patch("random_das", shape=(100, 1))
        time_coord = patch.coords.coord_map["time"]
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
        time_shape = patch.shape[patch.dims.index("time")]
        assert time_shape == len(patch.coords["time"])
        assert time_shape == len(patch.coords["time"])

    def test_init_conflicting_coord_dims(self, patch_conflicting_attrs_coords):
        """Test initing a patch which has conflicting info in coords/dims."""
        patch = patch_conflicting_attrs_coords
        coords = patch.coords.coord_map
        attrs = patch.attrs
        for name, coord in coords.items():
            assert getattr(attrs, f"{name}_min") == coord.min()
            assert getattr(attrs, f"{name}_max") == coord.max()
            if pd.isnull(coord.step):
                assert pd.isnull(getattr(attrs, f"d_{name}"))
            else:
                assert getattr(attrs, f"d_{name}") == coord.step

    def test_init_no_coords(self, random_patch):
        """Ensure a new patch can be inited from only attrs."""
        attrs = random_patch.attrs
        new = random_patch.__class__(attrs=attrs, data=random_patch.data)
        assert isinstance(new, dc.Patch)

    def test_init_with_patch(self, random_patch):
        """Ensure a patch inited on a patch just returns a patch."""
        new = dc.Patch(random_patch)
        assert new == random_patch


class TestNew:
    """Tests for `Patch.new` method."""

    def test_erase_history(self, random_patch):
        """Ensure new can erase history."""
        # do some processing, so history shows up.
        patch = random_patch.pass_filter(time=(None, 10)).decimate(time=5)
        assert patch.attrs.history
        new_attrs = dict(patch.attrs)
        new_attrs["history"] = []
        new_patch = patch.new(attrs=new_attrs)
        assert not new_patch.attrs.history

    def test_new_coord_dict_order(self, random_patch):
        """Ensure data can be init'ed with a dict of coords in any orientation"""
        patch = random_patch
        axis = patch.dims.index("time")
        data = np.std(patch.data, axis=axis, keepdims=True)
        new_time = patch.coords["time"][0:1]
        new_dist = patch.coords["distance"]
        coords_1 = {"time": new_time, "distance": new_dist}
        coords_2 = {"distance": new_dist, "time": new_time}
        # the order the coords are defined shouldn't matter
        out_1 = patch.new(data=data, coords=coords_1)
        out_2 = patch.new(data=data, coords=coords_2)
        assert out_1 == out_2

    def test_attrs_preserved_when_not_specified(self, random_patch):
        """If attrs is not passed to new, old attrs should remain."""
        pa = random_patch.update_attrs(network="bob", tag="2", station="10")
        new_1 = pa.new(data=pa.data * 10)
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

    def test_has_attrs(self):
        """An empty test patch should have attrs with null(ish) values."""
        patch = Patch()
        attrs = patch.attrs
        assert len(attrs)
        # Check that default values exist but are null (of appropriate dtype)
        names = ["time", "distance"]
        fields = ["d_{x}", "{x}_min", "{x}_max"]
        for name in names:
            for field in fields:
                value = attrs[field.format(x=name)]
                assert pd.isnull(value)


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
        new_coords = dict(random_patch.coords)
        new_coords["bob"] = new_coords.pop(dims[-1])
        new_dims = tuple(list(dims)[:-1] + ["bob"])
        patch_2 = random_patch.new(coords=new_coords, dims=new_dims)
        assert not patch_2.equals(random_patch)

    def test_coords_not_equal(self, random_patch):
        """Ensure if the coords are not equal neither are the arrays."""
        new_coords = dict(random_patch.coords)
        new_coords["distance"] = new_coords["distance"] + 10
        patch_2 = random_patch.new(coords=new_coords)
        assert not patch_2.equals(random_patch)

    def test_attrs_not_equal(self, random_patch):
        """Ensure if the attributes are not equal the arrays are not equal"""
        attrs = dict(random_patch.attrs)
        attrs["d_time"] = attrs["d_time"] - np.timedelta64(10, "s")
        attrs.pop("time_max")  # need to remove time for this to be valid.
        patch2 = random_patch.new(attrs=attrs)
        assert not patch2.equals(random_patch)

    def test_one_null_value_in_attrs(self, random_patch):
        """
        Ensure setting a value to null in attrs doesn't eval to equal.
        """
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
        attrs1["label"] = np.NaN
        patch1 = random_patch.new(attrs=attrs1)
        attrs2 = dict(attrs1)
        attrs2["label"] = None
        patch2 = random_patch.new(attrs=attrs2)
        assert patch1.equals(patch2)

    def test_transposed_patches_not_equal(self, random_patch):
        """Transposed patches are not considered equal."""
        transposed = random_patch.transpose()
        assert transposed.dims != random_patch.dims
        assert not random_patch.equals(transposed)

    def test_one_coord_not_equal(self, wacky_dim_patch):
        """Ensure coords being close but not equals fails equality."""
        patch = wacky_dim_patch
        coords = patch.coords
        coord_array = np.array(coords.coord_map["distance"].values)
        coord_array[20:30] *= 0.9
        assert not np.allclose(coord_array, coords["distance"])
        new_coords = coords.update_coords(distance=coord_array)
        new = patch.new(coords=new_coords)
        assert new != patch


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


class TestUpdateAttrs:
    """
    Tests for updating the attrs.

    Note: Most of the testing for this functionality is actually done on
    dascore.util.patch.AttrsCoordsMixer.
    """

    def test_add_attr(self, random_patch):
        """Tests for adding an attribute."""
        new = random_patch.update_attrs(bob=1)
        assert "bob" in new.attrs.dict() and new.attrs["bob"] == 1

    def test_original_unchanged(self, random_patch):
        """Updating attributes shouldn't change original patch in any way."""
        old_attrs = dict(random_patch.attrs)
        _ = random_patch.update_attrs(bob=2)
        assert "bob" not in dict(random_patch.attrs)
        assert random_patch.attrs == old_attrs

    def test_update_starttime1(self, random_patch):
        """Ensure coords are updated with attrs."""
        t1 = np.datetime64("2000-01-01")
        pa = random_patch.update_attrs(time_min=t1)
        assert pa.attrs["time_min"] == t1
        assert pa.coords["time"].min() == t1

    def test_update_startttime2(self, random_patch):
        """Updating start time should update end time as well."""
        duration = random_patch.attrs["time_max"] - random_patch.attrs["time_min"]
        new_start = np.datetime64("2000-01-01")
        pa1 = random_patch.update_attrs(time_min=str(new_start))
        assert pa1.attrs["time_min"] == new_start
        assert pa1.attrs["time_max"] == new_start + duration

    def test_dt_is_datetime64(self, random_patch):
        """Ensure dt gets changed into timedelta64."""
        d_time = random_patch.attrs["d_time"]
        assert isinstance(d_time, np.timedelta64)
        new1 = random_patch.update_attrs(d_time=10)
        assert new1.attrs["d_time"] == dc.to_timedelta64(10)

    def test_update_non_sorted_coord(self, wacky_dim_patch):
        """Ensure update attrs updates non-sorted coordinates."""
        # test updating dist max
        pa = wacky_dim_patch.update_attrs(distance_max=10)
        assert pa.attrs.distance_max == 10
        assert not np.any(pd.isnull(pa.coords["distance"]))
        # test update dist min
        pa = wacky_dim_patch.update_attrs(distance_min=10)
        assert pa.attrs.distance_min == 10
        assert not np.any(pd.isnull(pa.coords["distance"]))

    def test_update_units(self, random_patch):
        """Ensure units can be updated in attrs."""
        new_dist = "ft"
        patch = random_patch.update_attrs(distance_units=new_dist)
        coord = patch.coords.coord_map["distance"]
        assert coord.units == new_dist
        patch2 = random_patch.convert_units(distance=new_dist)
        assert patch == patch2


class TestSqueeze:
    """Tests for squeeze."""

    def test_remove_dimension(self, random_patch):
        """Tests for removing random dimensions."""
        out = random_patch.aggregate("time").squeeze("time")
        assert "time" not in out.dims
        assert len(out.data.shape) == 1, "data should be 1d"


class TestReleaseMemory:
    """Ensure memory is released when the patch is deleted."""

    def test_single_patch(self):
        """
        Ensure a single patch is gc'ed when it leaves scope.
        """
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


class TestXarray:
    """Tests for xarray conversions."""

    pytest.importorskip("xarray")

    def test_convert_to_xarray(self, random_patch):
        """Tests for converting to xarray object."""
        import xarray as xr

        da = random_patch.to_xarray()
        assert isinstance(da, xr.DataArray)


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
        dist = random_patch.coords["distance"]
        lat = np.arange(0, len(dist)) * 0.001 - 109.857952
        # add a single coord
        out = random_patch.assign_coords(latitude=("distance", lat))
        return out

    def test_add_single_dim_one_coord(self, random_patch_with_lat):
        """Tests that one coordinate can be added to a patch."""
        assert "latitude" in random_patch_with_lat.coords

    def test_add_single_dim_two_coord2(self, random_patch_with_lat_lon):
        """Ensure multiple coords can be added to patch."""
        out2 = random_patch_with_lat_lon
        assert {"latitude", "longitude"}.issubset(set(out2.coords.coord_map))
        assert out2.coords["longitude"].shape
        assert out2.coords["latitude"].shape

    def test_add_multi_dim_coords(self, multi_dim_coords_patch):
        """Ensure coords with multiple dimensions works."""
        out1 = multi_dim_coords_patch
        assert "quality" in out1.coords
        assert out1.coords["quality"].shape
        assert np.all(out1.coords["quality"] == 1)

    def test_coord_time_narrow_select(self, multi_dim_coords_patch):
        """Ensure the coord type doesn't change in narrow slice."""
        patch = multi_dim_coords_patch
        time = patch.coords.coord_map["time"]
        new = patch.select(time=(time.min(), time.min()))
        assert 1 in new.shape
        new_coords = new.coords.coord_map
        assert isinstance(new_coords["time"], CoordRange)


class TestApplyOperator:
    """Tests for applying various ufunc-type operators."""

    ops = (
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.pow,
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
        new = apply_operator(random_patch, ones, np.add)
        assert np.allclose(new.data, ones + random_patch.data)
