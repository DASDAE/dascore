"""
Tests for the file spool.
"""
import pytest

import dascore as dc
from dascore.clients.filespool import FileSpool
from dascore.utils.hdf5 import HDFPatchIndexManager


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

    def test_update(self, tmp_path_factory, random_patch):
        """Update should trigger indexing on formats that support it."""
        path = tmp_path_factory.mktemp("update_test") / "random.h5"
        dc.write(random_patch, path, "dasdae", "1")
        # pre-update
        spool = dc.spool(path)
        contents = spool.get_contents()
        assert not HDFPatchIndexManager(path).has_index
        new_spool = spool.update()
        assert HDFPatchIndexManager(path).has_index
        new_contents = new_spool.get_contents()
        assert contents.equals(new_contents)
