"""
Tests for creating/registering accessors.
"""
from dfs.core import trim_by_time


class TestDFSBasics:
    """Test the basic DFS namespace functions."""

    def test_namespace_exists(self, terra15_das_array):
        """Just ensure namespace returns accessor."""
        assert hasattr(terra15_das_array, "dfs"), "accessor not registered"
        acc = terra15_das_array.dfs
        name = trim_by_time.__name__
        out = getattr(acc, name)
        assert callable(out)
