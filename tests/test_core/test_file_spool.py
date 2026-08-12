"""Tests for the file spool."""

from __future__ import annotations

import pytest
from upath import UPath

import dascore as dc
from dascore.core.spool import Spool
from dascore.exceptions import PatchAttributeError


class TestBasic:
    """Basic tests for the filespool."""

    def test_type(self, terra15_file_spool, terra15_v5_path):
        """Ensure a file spool was returned."""
        assert isinstance(terra15_file_spool, Spool)
        assert len(terra15_file_spool) == len(dc.scan_to_df(terra15_v5_path))

    def test_get_patch(self, terra15_file_spool):
        """Ensure the patch is returned."""
        patch = terra15_file_spool[0]
        assert isinstance(patch, dc.Patch)

    def test_init_from_filespool(self, terra15_file_spool):
        """Ensure a spool can copy-construct from another spool."""
        new = Spool(terra15_file_spool)
        assert isinstance(new, Spool)

    def test_str(self, terra15_file_spool):
        """Ensure file spool works."""
        out = str(terra15_file_spool)
        assert "Spool" in out

    def test_update(self, tmp_path_factory, random_patch):
        """Update should preserve contents even when a format index hook is a no-op."""
        path = tmp_path_factory.mktemp("update_test") / "random.h5"
        dc.write(random_patch, path, "dasdae", "1")
        spool = dc.spool(path)
        contents = spool.get_contents()
        new_spool = spool.update()
        new_contents = new_spool.get_contents()
        assert contents.equals(new_contents)

    def test_raises_bad_file(self):
        """Simply ensures a bad file will raise."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            Spool.from_file("/not/a/directory")

    def test_local_upath_file(self, terra15_v5_path):
        """Ensure from_file accepts local UPath inputs."""
        spool = Spool.from_file(UPath(terra15_v5_path))
        assert isinstance(spool, Spool)
        assert len(spool)

    def test_chunk(self, terra15_file_spool):
        """Ensure chunking along time axis works on a file spool."""
        spool = terra15_file_spool
        time_coord = spool[0].get_coord("time")
        duration = time_coord.max() - time_coord.min()
        dt = duration / 3
        spool = terra15_file_spool.chunk(time=dt, keep_partial=True)
        assert len(spool) == 3
        for loaded_patch in spool:
            assert isinstance(loaded_patch, dc.Patch)

    def test_sorted_multi_patch_uses_source_patch_id(self, tmp_path):
        """Sorting should not change which source patch gets reloaded."""
        path = tmp_path / "multi_patch.h5"
        patch_2 = dc.get_example_patch()
        patch_1 = patch_2.update_coords(time=patch_2.coords.get_array("time") + 10)
        dc.write(dc.spool([patch_1, patch_2]), path, "dasdae", file_version="1")
        spool = Spool.from_file(path).sort("time")
        loaded_patch = spool[0]
        assert loaded_patch.get_coord("time").min() == patch_2.get_coord("time").min()

    def test_multi_patch_without_source_patch_id_raises(self, tmp_path):
        """Multi-patch reload should fail instead of guessing from row index."""
        path = tmp_path / "multi_patch.h5"
        spool = dc.examples.get_example_spool("random_das", length=2)
        dc.write(spool, path, "dasdae", file_version="1")
        file_spool = Spool.from_file(path)
        kwargs = {
            "source_path": str(path),
            "source_format": "DASDAE",
            "source_version": "1",
        }
        with pytest.raises(PatchAttributeError, match="uniquely resolved"):
            file_spool._catalog.resolver.resolve(kwargs)
