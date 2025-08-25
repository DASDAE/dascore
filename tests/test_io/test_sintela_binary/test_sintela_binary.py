"""
Tests for sintela binary format.
"""

import shutil
from pathlib import Path

import pytest

from dascore.exceptions import InvalidFiberFileError
from dascore.io.sintela_binary import SintelaBinaryV3
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
