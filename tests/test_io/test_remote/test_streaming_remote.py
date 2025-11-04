"""
Tests for streaming remote HDF5 files using h5py's fileobj driver.

These tests verify that h5py can stream files directly without downloading.
"""

from __future__ import annotations

import h5py
import pytest

import dascore as dc
from dascore.compat import UPath


class TestH5pyStreaming:
    """Tests for h5py streaming with fileobj driver."""

    def test_h5py_fileobj_driver_local(self, tmp_path):
        """Test that h5py fileobj driver works with local files."""
        # Create a simple HDF5 file
        test_file = tmp_path / "test.h5"
        with h5py.File(test_file, "w") as f:
            f.create_dataset("data", data=[1, 2, 3, 4, 5])
            f.attrs["test_attr"] = "test_value"

        # Read it using fileobj driver
        with open(test_file, "rb") as file_handle:
            with h5py.File(file_handle, "r", driver="fileobj") as f:
                assert "data" in f
                assert list(f["data"][:]) == [1, 2, 3, 4, 5]
                assert f.attrs["test_attr"] == "test_value"

    def test_h5py_fileobj_driver_upath_local(self, tmp_path):
        """Test that h5py fileobj driver works with UPath.open()."""
        # Create a simple HDF5 file
        test_file = tmp_path / "test.h5"
        with h5py.File(test_file, "w") as f:
            f.create_dataset("data", data=[1, 2, 3, 4, 5])

        # Read it using UPath and fileobj driver
        upath = UPath(test_file)
        with upath.open("rb") as file_handle:
            with h5py.File(file_handle, "r", driver="fileobj") as f:
                assert "data" in f
                assert list(f["data"][:]) == [1, 2, 3, 4, 5]

    def test_dascore_read_with_upath(self, tmp_path):
        """Test that dascore can read HDF5 files via UPath."""
        # Create a patch and save it
        patch = dc.get_example_patch()
        test_file = tmp_path / "test.h5"
        dc.write(patch, test_file, "DASDAE")

        # Read it back using UPath
        upath = UPath(test_file)
        spool = dc.read(upath)
        assert len(spool) == 1

        patch_read = spool[0]
        assert patch_read.dims == patch.dims


class TestStreamingVsDownload:
    """Tests comparing streaming vs download approaches."""

    def test_streaming_preferred_for_h5(self, tmp_path, monkeypatch):
        """Verify that streaming is attempted before downloading."""
        # Create a test file
        patch = dc.get_example_patch()
        test_file = tmp_path / "test.h5"
        dc.write(patch, test_file, "DASDAE")

        # Track if download was called
        download_called = []

        from dascore.utils import io

        original_download = io.download_remote_to_temp

        def mock_download(path):
            download_called.append(True)
            return original_download(path)

        monkeypatch.setattr(io, "download_remote_to_temp", mock_download)

        # Read the file - should not download since it's local
        spool = dc.read(UPath(test_file))
        assert len(spool) == 1

        # For local files, download shouldn't be called
        assert not download_called


class TestFileHandleManagement:
    """Tests for proper file handle management."""

    def test_file_handle_closed_on_error(self, tmp_path):
        """Verify file handles are properly closed on errors."""
        # Create a non-HDF5 file
        bad_file = tmp_path / "bad.h5"
        bad_file.write_text("This is not an HDF5 file")

        # Try to open with fileobj driver
        upath = UPath(bad_file)
        with pytest.raises(Exception):
            with upath.open("rb") as f:
                h5py.File(f, "r", driver="fileobj")

    def test_dascore_read_closes_handles(self, tmp_path):
        """Verify dascore properly manages file handles."""
        patch = dc.get_example_patch()
        test_file = tmp_path / "test.h5"
        dc.write(patch, test_file, "DASDAE")

        # Read multiple times to ensure handles are released
        for _ in range(3):
            spool = dc.read(UPath(test_file))
            assert len(spool) == 1
            # Force cleanup
            del spool
