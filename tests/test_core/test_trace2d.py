"""
Tests for Trace2D object.
"""
import numpy as np

from dfs.core import Trace2D


class TestInit:
    """Tests for init'ing Trace2D"""

    def test_init_from_array(self, random_das_array):
        """"""
        assert isinstance(random_das_array, Trace2D)


class TestEquals:
    """Tests for checking equality."""

    def test_equal_self(self, random_das_array):
        """Ensure a trace"""
