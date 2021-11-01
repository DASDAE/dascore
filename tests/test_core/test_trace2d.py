"""
Tests for Trace2D object.
"""
import pandas as pd
import numpy as np

from fios.core import Trace2D


class TestInit:
    """Tests for init'ing Trace2D"""

    def test_init_from_array(self, random_das_array):
        """Ensure a trace can be created from raw components; array, coords, attrs"""
        assert isinstance(random_das_array, Trace2D)

    def test_min_max_populated(self, random_das_array):
        """The min/max values of the atts should have been populated. """
        attrs = random_das_array.attrs
        expected_filled_in = [
            x for x in attrs if x.startswith("time") or x.startswith("distance")
        ]
        for attr in expected_filled_in:
            assert not pd.isnull(attrs[attr])


class TestEquals:
    """Tests for checking equality."""

    def test_equal_self(self, random_das_array):
        """Ensure a trace equals itself"""
        assert random_das_array.equals(random_das_array)

    def test_non_equal_array(self, random_das_array):
        """Ensure the traces are not equal if the data are not equal."""
        new_data = random_das_array.data + 1
        new = random_das_array.new(data=new_data)
        assert not new.equals(random_das_array)

    def test_coords_not_equal(self, random_das_array):
        """Ensure if the coords are not equal neither is the array."""

    def test_attrs_not_equal(self, random_das_array):
        """Ensure if the attributes are not equal the arrays are not equal"""

    def test_non_default_attrs(self, random_das_array):
        """Ensure non default attrs don't effect equality unless specified."""


class TestSelect:
    """Tests for selecting data from Trace."""

    def test_select_by_distance(self, random_das_array):
        """
        Ensure distance can be used to filter trace.
        """
        dmin, dmax = 100, 200
        tr = random_das_array.select(distance=(dmin, dmax))
        assert tr.data.shape < random_das_array.data.shape
        # the attrs should have updated as well
        assert tr.attrs["distance_min"] >= 100
        assert tr.attrs["distance_max"] <= 200

    def test_select_by_absolute_time(self, random_das_array):
        """
        Ensure the data can be sub-selected using absolute time.
        """
        shape = random_das_array.data.shape
        t1 = np.datetime64("1970-01-01T00:00:01")
        t2 = t1 + np.timedelta64(3, "s")

        tr1 = random_das_array.select(time=(None, t1))
        assert tr1.attrs["time_max"] <= t1
        assert tr1.data.shape < shape

        tr2 = random_das_array.select(time=(t1, None))
        assert tr2.attrs["time_min"] >= t1
        assert tr2.data.shape < shape

        tr3 = random_das_array.select(time=(t1, t2))
        assert tr3.attrs["time_min"] >= t1
        assert tr3.attrs["time_max"] <= t2
        assert tr3.data.shape < shape

    def test_select_by_positive_float(self, random_das_array):
        """Floats in time dim should usable to reference start of the trace."""
        shape = random_das_array.data.shape
        tr1 = random_das_array.select(time=(1, None))
        expected_start = np.datetime64("1970-01-01T00:00:01")
        assert tr1.attrs["time_min"] <= expected_start
        assert tr1.data.shape < shape
