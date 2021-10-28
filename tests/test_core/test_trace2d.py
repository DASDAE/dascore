"""
Tests for Trace2D object.
"""
import numpy as np

from fios.core import Trace2D


class TestInit:
    """Tests for init'ing Trace2D"""

    def test_init_from_array(self, random_das_array):
        """"""
        assert isinstance(random_das_array, Trace2D)


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
        dar = random_das_array.select(distance=(100, 300))
        assert dar.data.shape < random_das_array.data.shape
