"""
Tests for Trace2D object.
"""
import weakref

import numpy as np
import pandas as pd
import pytest

import fios
from fios.core import Patch
from fios.utils.time import to_timedelta64


def get_simple_patch() -> Patch:
    """
    Return a small simple array for memory testing.

    Note: This cant be a fixture as pytest holds a reference.
    """
    pa = Patch(
        data=np.random.random((100, 100)),
        coords={"time": np.arange(100) * 0.01, "distance": np.arange(100) * 0.2},
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
        new = fios.Patch(data=random_patch.data, attrs=attrs, coords=coords)
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

    def test_coords_not_equal(self, random_patch):
        """Ensure if the coords are not equal neither is the array."""

    def test_attrs_not_equal(self, random_patch):
        """Ensure if the attributes are not equal the arrays are not equal"""

    def test_non_default_attrs(self, random_patch):
        """Ensure non default attrs don't effect equality unless specified."""


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
    """Tests for updating the attrs."""

    def test_add_attr(self, random_patch):
        """Tests for adding an attribute."""
        new = random_patch.update_attrs(bob=1)
        assert "bob" in new.attrs and new.attrs["bob"] == 1

    def test_original_unchanged(self, random_patch):
        """ "Updating starttime should also update endtime"""
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
        new = patch.proc.decimate(10, lowpass=False)
        wr = weakref.ref(patch.data)
        del patch
        assert isinstance(new, Patch)
        assert wr() is None

    def test_select(self):
        """
        A similar test to ensure select releases old memory.
        """
        patch = get_simple_patch()
        new = patch.proc.select(time=[0.1, 10], copy=True)
        wr = weakref.ref(patch.data)
        del patch
        assert isinstance(new, Patch)
        assert wr() is None
