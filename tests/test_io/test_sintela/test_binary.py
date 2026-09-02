"""
Tests for sintela binary format.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.core import FiberIO
from dascore.io.sintela import SintelaBinaryV3
from dascore.utils.downloader import fetch


class TestScanSintelaBinary:
    """Tests for scanning a binary file."""

    @pytest.fixture(scope="class")
    def extra_bytes_file(self, tmp_path_factory):
        """Create a sintela binary file with extra bytes."""
        tmp = tmp_path_factory.mktemp("sintela_binary")
        binary_path = Path(fetch("sintela_binary_v3_test_1.raw"))
        new = tmp / "extra_bytes.raw"
        shutil.copy(binary_path, new)

        with open(new, "ab") as fi:
            fi.write(b"some_bytes_des_is")

        return new

    def test_extra_bytes_raises(self, extra_bytes_file):
        """Ensure a file with extra bytes raises an exception."""
        fiber_io = SintelaBinaryV3()
        with pytest.raises(InvalidFiberFileError):
            fiber_io.scan(extra_bytes_file)


class TestReadArray:
    """Tests for slicing the mapped packet payloads."""

    @pytest.fixture(scope="class")
    def binary_path(self):
        """The example Sintela binary recording."""
        return fetch("sintela_binary_v3_test_1.raw")

    def test_matches_default(self, binary_path):
        """The override returns what the read-and-trim default returns."""
        io = SintelaBinaryV3()
        patch = dc.spool(binary_path)[0]
        windows = {"time": (3, 11), "distance": (2, 6)}
        out = io.read_array(binary_path, windows)
        expected = FiberIO.read_array(io, binary_path, windows)
        assert out.dtype == expected.dtype
        assert np.array_equal(out, expected)
        assert np.array_equal(out, patch.data[3:11, 2:6])

    def test_empty_window(self, binary_path):
        """A window past the end is empty, with the right width and dtype."""
        io = SintelaBinaryV3()
        whole = io.read_array(binary_path, {})
        out = io.read_array(binary_path, {"time": (10**9, 10**9 + 5)})
        assert out.shape == (0, whole.shape[1])
        assert out.dtype == whole.dtype
