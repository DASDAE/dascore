"""
Test for basic IO and related functions.
"""
import dfs
import pytest

from dfs.exceptions import UnknownFiberFormat

class TestGetFormat:
    """Tests to ensure formats can be retreived."""
    def test_terra_15(self, terra15_path):
        out = dfs.get_format(terra15_path)
        assert out.upper() == 'TERRA15_V2'

    def test_not_known(self, dummy_text_file):
        """Ensure a non-path/str object raises."""
        with pytest.raises(UnknownFiberFormat):
            assert dfs.get_format(dummy_text_file)


class TestRead:
    """Basic tests for reading files."""

    def test_read_terra15(self, terra15_path, terra15_das_array):
        """Ensure terra15 can be read."""
        out = dfs.read(terra15_path)
        assert isinstance(out, dfs.Stream)
        assert len(out) == 1
        assert out[0].equals(terra15_das_array)


class TestScan:
    """Tests for scanning fiber files."""
    @pytest.fixture()
    def expected_summary(self, terra15_das_array):
        """Return the expected summary from the terra15_das_array."""

    def test_scan_terra15(self, terra15_path):
        """"""