"""
Tests for creating/registering accessors.
"""


class TestDFSBasics:
    """Test the basic DFS namespace functions."""

    def test_namespace_exists(self, terra15_das):
        """Just ensure namespace returns accessor."""
        assert hasattr(terra15_das, 'dfs'), 'accessor not registered'
        acc = terra15_das.dfs
        breakpoint()
