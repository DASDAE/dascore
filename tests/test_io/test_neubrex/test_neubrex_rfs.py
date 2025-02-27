"""
Tests for Rayleigh Frequency Shift format.
"""

import pytest

import dascore as dc


class TestWrite:
    """Tests for writing to the NeubrexRFS format."""

    @pytest.fixture(scope="class")
    def written_rfs(self, random_patch, tmp_path_factory):
        """Write the random patch, return path."""
        path = tmp_path_factory.mktemp("rfs") / "random_rfs.h5"
        random_patch.io.write(path, "NeubrexRFS")
        return path

    def test_multi_patch_raises(self, random_spool, tmp_path):
        """Trying to save multiple patch should raise."""
        msg = "only supports writing a single patch"
        with pytest.raises(ValueError, match=msg):
            dc.write(random_spool, tmp_path / "bad_file.h5", "NeubrexRFS")

    def test_format(self, written_rfs):
        """Ensure the returned format type is correct."""
        format_name = dc.get_format(written_rfs)
        assert "NeubrexRFS" in format_name
