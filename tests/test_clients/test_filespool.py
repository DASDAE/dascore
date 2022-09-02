"""
Tests for the file spool.
"""
import pytest

import dascore as dc
from dascore.clients.filespool import FileSpool


@pytest.fixture(scope="class")
def terra15_file_spool(terra15_v5_path):
    """A file spool for terra15."""
    return dc.spool(terra15_v5_path)


class TestBasic:
    """Basic tests for the filespool."""

    def test_type(self, terra15_file_spool, terra15_v5_path):
        """Ensure a file spool was returned."""
        assert isinstance(terra15_file_spool, FileSpool)
        assert len(terra15_file_spool) == len(dc.scan_to_df(terra15_v5_path))

    def test_get_patch(self, terra15_file_spool):
        """Ensure the patch is returned."""
        patch = terra15_file_spool[0]
        assert isinstance(patch, dc.Patch)

    def test_str(self, terra15_file_spool):
        """Ensure file spool works."""
        out = str(terra15_file_spool)
        assert "FileSpool" in out
