"""Tests for an indexed file holding a patch whose coordinate has a hole."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import dascore as dc
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
    """The gapped patch is indexed with its runs and loads whole."""

    def test_one_row_with_its_runs(self, gapped_file_spool):
        """The index holds the patch as one row whose coordinate has two runs."""
        contents = gapped_file_spool.get_contents()
        assert len(contents) == 2
        # the flat contents state the step of a range only
        assert contents["time_step"].isna().sum() == 1
        backend = gapped_file_spool._catalog.backend
        runs = backend._fetch_df("SELECT * FROM coord_runs")
        assert runs["count"].tolist() == [251, 1747]

    def test_loads(self, gapped_file_spool):
        """Reading no longer splits the patch, so its key names one (#1284)."""
        patches = list(gapped_file_spool)
        assert len(patches) == 2
        (gapped,) = [x for x in patches if x.get_coord("time").gapped]
        assert gapped.get_coord("time").runs_count == 2
