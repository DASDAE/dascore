"""
Test for reading files.
"""
import dfs


class TestRead:
    """Basic tests for reading files."""

    def test_read_terra15(self, terra_15_path):
        """Ensure terra15 can be read."""
        out = dfs.read(terra_15_path)
