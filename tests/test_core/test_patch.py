"""
Tests for Trace2D object.
"""
import weakref

import numpy as np
import pandas as pd
import pytest

import dascore
from dascore.core import Patch
from dascore.utils.time import to_timedelta64


def get_simple_patch() -> Patch:
    """
    Return a small simple array for memory testing.

    Note: This cant be a fixture as pytest holds a reference.
    """
    attrs = {"d_time": 1}
    pa = Patch(
        data=np.random.random((100, 100)),
        coords={"time": np.arange(100) * 0.01, "distance": np.arange(100) * 0.2},
        attrs=attrs,
    )
    return pa


class TestInit:
    """Tests for init'ing Trace2D"""

    time1 = np.datetime64("2020-01-01")

    @pytest.fixture()
    def random_dt_coord(self):
        """Create a random trace with a datetime coord"""
        rand = np.random.RandomState(13)
        array = rand.random(size=(20, 200))
        attrs = dict(dx=1, d_time=1 / 250.0, category="DAS", id="test_data1")
        time_deltas = to_timedelta64(np.arange(array.shape[1]) * attrs["d_time"])
        coords = dict(
            distance=np.arange(array.shape[0]) * attrs["dx"],
            time=self.time1 + time_deltas,
        )
        out = dict(data=array, coords=coords, attrs=attrs)
        return Patch(**out)

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
        time = patch.coords["time"].max()
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
        expected_filled_in = [x for x in attrs if x.startswith("distance")]
        for attr in expected_filled_in:
            assert not pd.isnull(attrs[attr])

    def test_dt_is_datetime64(self, random_patch):
        """Ensure dt gets changed into timedelta64."""
        d_time = random_patch.attrs["d_time"]
        assert isinstance(d_time, np.timedelta64)
        # test d_time from update_attrs
        new = random_patch.update_attrs(d_time=10)
        assert new.attrs["d_time"] == to_timedelta64(10)
        # test d_time in new Patch
        attrs = dict(random_patch.attrs)
        attrs["d_time"] = to_timedelta64(10)
        coords = random_patch.coords
        new = dascore.Patch(data=random_patch.data, attrs=attrs, coords=coords)
        assert new.attrs["d_time"] == to_timedelta64(10)

    def test_had_default_attrs(self, patch):
        """Test that all patches used in the test suite have default attrs."""
        attr_set = set(patch.attrs)
        assert attr_set.issubset(set(patch.attrs))

    def test_no_attrs(self):
        """Ensure a patch with no attrs can be created."""
        pa = Patch(
            data=np.random.random((100, 100)),
            coords={"time": np.random.random(100), "distance": np.random.random(100)},
        )
        assert isinstance(pa, Patch)

    def test_seconds_as_coords_time_no_dt(self):
        """Ensure seconds passed as coordinates with no attrs still works."""
        ar = get_simple_patch()
        assert not np.any(pd.isnull(ar.coords["time"]))


class TestEmptyPatch:
    """Tests for empty patch objects."""

    def test_empty_patch_str(self):
        """An empty patch should have both a str and repr of nonzero length."""
        patch = Patch()
        assert len(str(patch))
        assert len(repr(patch))

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
        """Ensure a trace equals itself"""
        assert random_patch.equals(random_patch)

    def test_non_equal_array(self, random_patch):
        """Ensure the traces are not equal if the data are not equal."""
        new_data = random_patch.data + 1
        new = random_patch.new(data=new_data)
        assert not new.equals(random_patch)

    def test_coords_named_differently(self, random_patch):
        """Ensure if the coords are not equal neither are the arrays."""
        dims = random_patch.dims
        new_coords = {x: random_patch.coords[x] for x in random_patch.coords}
        new_coords["bob"] = new_coords.pop(dims[-1])
        patch_2 = random_patch.new(coords=new_coords)
        assert not patch_2.equals(random_patch)

    def test_coords_not_equal(self, random_patch):
        """Ensure if the coords are not equal neither are the arrays."""
        new_coords = {x: random_patch.coords[x] for x in random_patch.coords}
        new_coords["distance"] = new_coords["distance"] + 10
        patch_2 = random_patch.new(coords=new_coords)
        assert not patch_2.equals(random_patch)

    def test_attrs_not_equal(self, random_patch):
        """Ensure if the attributes are not equal the arrays are not equal"""
        attrs = dict(random_patch.attrs)
        attrs["d_time"] = attrs["d_time"] - np.timedelta64(10, "s")
        patch2 = random_patch.new(attrs=attrs)
        assert not patch2.equals(random_patch)

    def test_one_null_value_in_attrs(self, random_patch):
        """
        Ensure setting a value to null in attrs doesn't eval to equal.
        """
        attrs = dict(random_patch.attrs)
        attrs["tag"] = None
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

    def test_transposed_patches_equal(self, random_patch):
        """Transposed patches are still equal."""
        transposed = random_patch.transpose()
        assert random_patch.equals(transposed)


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
        assert "bob" in new.attrs and new.attrs["bob"] == 1

    def test_original_unchanged(self, random_patch):
        """updating attributes shouldn't change original patch in any way."""
        old_attrs = dict(random_patch.attrs)
        _ = random_patch.update_attrs(bob=2)
        assert "bob" not in random_patch.attrs
        assert random_patch.attrs == old_attrs

    def test_update_starttime(self, random_patch):
        """Ensure coords are updated with attrs."""
        t1 = np.datetime64("2000-01-01")
        pa = random_patch.update_attrs(time_min=t1)
        assert pa.attrs["time_min"] == t1
        assert pa.coords["time"].min() == t1

    def test_update_startttime(self, random_patch):
        """Updating start time should update end time as well."""
        duration = random_patch.attrs["time_max"] - random_patch.attrs["time_min"]
        new_start = np.datetime64("2000-01-01")
        pa1 = random_patch.update_attrs(time_min=str(new_start))
        assert pa1.attrs["time_min"] == new_start
        assert pa1.attrs["time_max"] == new_start + duration


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
        simple_array = get_simple_patch()
        wr = weakref.ref(simple_array.data)
        # delete pa and ensure the array was collected.
        del simple_array
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
        """
        A similar test to ensure select releases old memory.
        """
        patch = get_simple_patch()
        new = patch.select(time=[0.1, 10], copy=True)
        wr = weakref.ref(patch.data)
        del patch
        assert isinstance(new, Patch)
        assert wr() is None


class TestXarray:
    """Tests"""

    pytest.importorskip("xarray")

    def test_convert_to_xarray(self, random_patch):
        """Tests for converting to xarray object."""
        import xarray as xr

        da = random_patch.to_xarray()
        assert isinstance(da, xr.DataArray)
