"""
Test for basic IO and related functions.
"""
import pytest

import dascore
from dascore.exceptions import UnknownFiberFormat


class TestGetFormat:
    """Tests to ensure formats can be retrieved."""

    def test_terra_15(self, terra15_das_example_path):
        """Ensure terra15 v2 can be read"""
        out = dascore.get_format(terra15_das_example_path)
        vtuple = (out[0].upper(), out[1])
        assert vtuple == ("TERRA15", "4")

    def test_not_known(self, dummy_text_file):
        """Ensure a non-path/str object raises."""
        with pytest.raises(UnknownFiberFormat):
            assert dascore.get_format(dummy_text_file)


class TestRead:
    """Basic tests for reading files."""

    def test_read_terra15(self, terra15_das_example_path, terra15_das_patch):
        """Ensure terra15 can be read."""
        out = dascore.read(terra15_das_example_path)
        assert isinstance(out, dascore.Stream)
        assert len(out) == 1
        assert out[0].equals(terra15_das_patch)


class TestScan:
    """Tests for scanning fiber files."""

    def test_scan_terra15(self, terra15_das_example_path):
        """Ensure terra15 format can be automatically determined."""
        out = dascore.scan_file(terra15_das_example_path)
        assert isinstance(out, list)
        assert len(out)
