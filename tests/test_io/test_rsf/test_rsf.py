"""Tests for RSF format."""

import os
from pathlib import Path

import numpy as np

import dascore as dc
from dascore.io.rsf import RSFV1


class TestRsfWrite:
    """testing the rSF write out function."""

    def test_write_nopath(self, random_patch, tmp_path):
        """Test write function."""
        spool = dc.spool(random_patch)
        path = tmp_path / "test_hdrdata.rsf"
        RSFV1().write(spool, path)

        assert path.exists()
        test_data = random_patch.data.astype(np.float32)
        dtype = np.dtype(test_data.dtype)
        file_esize = dtype.itemsize
        datasize = test_data.size * file_esize
        assert os.path.getsize(path) >= datasize

    def test_write_path(self, random_patch, tmp_path):
        """Test write function."""
        spool = dc.spool(random_patch)
        path = tmp_path / "test_hdr.rsf"
        datapath = tmp_path / "binary/test_data.rsf"
        RSFV1().write(spool, path, data_path=datapath)

        assert path.exists()
        newdatapath = Path(str(datapath) + "@")
        assert newdatapath.exists()
        test_data = random_patch.data.astype(np.float32)
        dtype = np.dtype(test_data.dtype)
        file_esize = dtype.itemsize
        datasize = test_data.size * file_esize
        assert os.path.getsize(newdatapath) == datasize
