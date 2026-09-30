"""Tests for an indexed file holding a patch whose coordinate has a hole."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import PatchAttributeError
from dascore.io.dasdae.utils import _save_patch
from tests.conftest import join_patches

MS = np.timedelta64(1, "ms")


@pytest.fixture(scope="module")
def gapped_file_spool(tmp_path_factory):
    """A directory whose file holds a gapped patch whole, written around dc.write."""
    patch = dc.get_example_patch()
    t0 = patch.get_coord("time").min()
    gapped = join_patches(
        [
            patch.select(time=(None, t0 + 1000 * MS)),
            patch.select(time=(t0 + 1012 * MS, None)),
        ]
    )
    path = tmp_path_factory.mktemp("gapped") / "gapped.h5"
    dc.write(patch, path, "dasdae")
    with h5py.File(path, "a") as h5:
        _save_patch(gapped, h5["waveforms"], "gapped", compact=True)
    return dc.spool(path.parent).update()


class TestGappedFile:
    """The gapped patch is indexed by its envelope and cannot be loaded."""

    def test_one_stepless_row_that_cannot_load(self, gapped_file_spool):
        """The reader splits the patch at its hole, so its key names two patches."""
        contents = gapped_file_spool.get_contents()
        assert len(contents) == 2
        assert contents["time_step"].isna().sum() == 1
        with pytest.raises(PatchAttributeError, match="uniquely resolved"):
            list(gapped_file_spool)
