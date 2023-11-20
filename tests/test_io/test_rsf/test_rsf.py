"""Tests for RSF format."""

import os
from pathlib import Path

import numpy as np

from dascore.io.rsf import RSFV1


class TestRsfWrite:
    """testing the rSF write out function."""

    def test_write_nopath(self, random_patch, tmp_path):
        """Test write function."""
        path = tmp_path / "test_hdrdata.rsf"
        RSFV1().write(random_patch, path)

        assert path.exists()
        dtype = np.dtype(random_patch.data.dtype)
        file_esize = dtype.itemsize
        datasize = random_patch.data.size * file_esize
        assert os.path.getsize(path) >= datasize

    def test_write_path(self, random_patch, tmp_path):
        """Test write function."""
        path = tmp_path / "test_hdr.rsf"
        datapath = tmp_path / "binary/test_data.rsf"
        RSFV1().write(random_patch, path, data_path=datapath)

        assert path.exists()
        newdatapath = Path(str(datapath) + "@")
        assert newdatapath.exists()
        dtype = np.dtype(random_patch.data.dtype)
        file_esize = dtype.itemsize
        datasize = random_patch.data.size * file_esize
        assert os.path.getsize(newdatapath) == datasize
