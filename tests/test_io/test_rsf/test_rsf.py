"""Tests for RSF format."""

import os
from pathlib import Path

import numpy as np
import pytest
from upath import UPath

import dascore as dc
from dascore.io.rsf import RSFV1


class TestRsfWrite:
    """testing the rSF write out function."""

    def test_write_nopath(self, random_patch, tmp_path):
        """
        Test write function with no binary path specified.
        Data and header are combined.
        """
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
        """Test write function with different binary data path specified."""
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

    def test_write_cmplx(self, random_patch, tmp_path, **kwargs):
        """Test write function for non-int and non-float values.
        if should fail and return:
        ValueError("Data format is not integer or floating.").
        """
        complex_patch = random_patch.dft("time")
        spool = dc.spool(complex_patch)
        path = tmp_path / "test_hdrcmplx.rsf"
        with pytest.raises(ValueError):
            RSFV1().write(spool, path)

    def test_write_int(self, random_patch, tmp_path, **kwargs):
        """Test write function for int values.
        if should return float values of the integer numbers.
        """
        data = np.ones_like(random_patch.data, dtype=np.int32)
        int_patch = random_patch.new(data=data)
        spool = dc.spool(int_patch)
        path = tmp_path / "test_hdrint.rsf"
        datapath = tmp_path / "binary/test_int.rsf"
        RSFV1().write(spool, path, data_path=datapath)

        assert path.exists()
        newdatapath = Path(str(datapath) + "@")
        assert newdatapath.exists()
        test_data = data.astype(np.float32)
        dtype = np.dtype(test_data.dtype)
        file_esize = dtype.itemsize
        datasize = test_data.size * file_esize
        assert os.path.getsize(newdatapath) == datasize

    def test_write_remote_memory_combined(self, random_patch):
        """Combined RSF output should support remote UPath destinations."""
        spool = dc.spool(random_patch)
        path = UPath("memory://dascore/test_hdrdata_remote.rsf")
        RSFV1().write(spool, path)
        assert path.exists()
        test_data = random_patch.data.astype(np.float32)
        datasize = test_data.size * np.dtype(test_data.dtype).itemsize
        assert path.stat().st_size >= datasize

    def test_write_remote_memory_split(self, random_patch):
        """Split RSF output should support remote UPath destinations."""
        spool = dc.spool(random_patch)
        path = UPath("memory://dascore/test_hdr_remote.rsf")
        datapath = UPath("memory://dascore/binary/test_data_remote.rsf")
        RSFV1().write(spool, path, data_path=datapath)
        assert path.exists()
        newdatapath = UPath(str(datapath) + "@")
        assert newdatapath.exists()
        test_data = random_patch.data.astype(np.float32)
        datasize = test_data.size * np.dtype(test_data.dtype).itemsize
        assert newdatapath.stat().st_size == datasize
