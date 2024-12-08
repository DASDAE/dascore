"""Tests for SEGY format."""

import pytest

from dascore.io.segy.core import SegyV1_0


class TestSegy:
    """Tests for SEGY format."""

    @pytest.fixture(scope="class")
    def small_file(self, tmp_path_factory):
        """Creates a small file with only a few bytes."""
        parent = tmp_path_factory.mktemp("small_file")
        path = parent / "test_file.segy"
        with path.open("wb") as f:
            f.write(b"abd")
        return path

    def test_small_file(self, small_file):
        """
        Ensure a file that is too small to contain segy header doesn't throw
        an error.
        """
        segy = SegyV1_0()
        out = segy.get_format(small_file)
        assert out is False  # we actually want to make sure its False.
