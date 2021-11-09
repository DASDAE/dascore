"""
Tests for Trace2D object.
"""
import numpy as np
import pandas as pd
import pytest

from fios.core import Patch
from fios.utils.time import to_timedelta64
from fios.constants import DEFAULT_PATCH_ATTRS


class TestInit:
    """Tests for init'ing Trace2D"""

    time1 = np.datetime64("2020-01-01")

    @pytest.fixture()
    def random_dt_coord(self):
        """Create a random trace with a datetime coord"""
        rand = np.random.RandomState(13)
        array = rand.random(size=(20, 200))
        attrs = dict(dx=1, dt=1 / 250.0, category="DAS", id="test_data1")
        time_deltas = to_timedelta64(np.arange(array.shape[1]) * attrs["dt"])
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
        tds = patch.coords["time"]
        assert np.issubdtype(tds.dtype, np.timedelta64)

    def test_end_time_inferred_from_dt64_coords(self, random_dt_coord):
        """
        Ensure the time_min and time_max attrs can be inferred from coord time.
        """
        patch = random_dt_coord
        dt_max = patch.coords["time"].max()
        t1 = patch.attrs["time_min"]
        assert patch.attrs["time_max"] == t1 + dt_max
        tds = patch.coords["time"]
        assert np.issubdtype(tds.dtype, np.timedelta64)

    def test_init_from_array(self, random_patch):
        """Ensure a trace can be created from raw components; array, coords, attrs"""
        assert isinstance(random_patch, Patch)

    def test_max_time_populated(self, random_patch):
        """Ensure the endtime is populated when not explicitly given."""
        end_time = random_patch.attrs["time_max"]
        assert not pd.isnull(end_time)

    def test_min_max_populated(self, random_patch):
        """The min/max values of the distance attrss should have been populated."""
        attrs = random_patch.attrs
        expected_filled_in = [x for x in attrs if x.startswith("distance")]
        for attr in expected_filled_in:
            assert not pd.isnull(attrs[attr])

    def test_had_default_attrs(self, patch):
        """Test that all patches used in the test suite have default attrs."""
        attr_set = set(patch.attrs)
        assert attr_set.issubset(set(patch.attrs))


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


class TestSelect:
    """Tests for selecting data from Trace."""

    def test_select_by_distance(self, random_patch):
        """
        Ensure distance can be used to filter trace.
        """
        dmin, dmax = 100, 200
        tr = random_patch.select(distance=(dmin, dmax))
        assert tr.data.shape < random_patch.data.shape
        # the attrs should have updated as well
        assert tr.attrs["distance_min"] >= 100
        assert tr.attrs["distance_max"] <= 200

    def test_select_by_absolute_time(self, random_patch):
        """
        Ensure the data can be sub-selected using absolute time.
        """
        shape = random_patch.data.shape
        t1 = random_patch.attrs['time_min'] + np.timedelta64(1, 's')
        t2 = t1 + np.timedelta64(3, "s")

        pa1 = random_patch.select(time=(None, t1))
        assert pa1.attrs["time_max"] <= t1
        assert pa1.data.shape < shape

        pa2 = random_patch.select(time=(t1, None))
        assert pa2.attrs["time_min"] >= t1
        assert pa2.data.shape < shape

        breakpoint()
        tr3 = random_patch.select(time=(t1, t2))
        assert tr3.attrs["time_min"] >= t1
        assert tr3.attrs["time_max"] <= t2
        assert tr3.data.shape < shape

    def test_select_by_positive_float(self, random_patch):
        """Floats in time dim should usable to reference start of the trace."""
        shape = random_patch.data.shape
        pa1 = random_patch.select(time=(1, None))
        expected_start = np.datetime64("1970-01-01T00:00:01")
        assert pa1.attrs["time_min"] <= expected_start
        assert pa1.data.shape < shape


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
        """Ensure updating the attrs doesn't modify original."""
