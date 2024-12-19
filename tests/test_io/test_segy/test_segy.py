"""Tests for SEGY format."""

import numpy as np
import pytest

import dascore as dc
from dascore.io.segy.core import SegyV1_0


class TestSegyGetFormat:
    """Tests for getting format codes of SEGY files."""

    @pytest.fixture(scope="class")
    def small_file(self, tmp_path_factory):
        """Creates a small file with only a few bytes."""
        parent = tmp_path_factory.mktemp("small_file")
        path = parent / "test_file.segy"
        with path.open("wb") as f:
            f.write(b"abd")
        return path

    def test_get_formate_small_file(self, small_file):
        """
        Ensure a file that is too small to contain segy header doesn't throw
        an error.
        """
        segy = SegyV1_0()
        out = segy.get_format(small_file)
        assert out is False  # we actually want to make sure its False.


class TestSegyWrite:
    """Tests for writing segy files."""

    @pytest.fixture(scope="class")
    def test_segy_directory(self, tmp_path_factory):
        """Make a tmp directory for saving files."""
        path = tmp_path_factory.mktemp("test_segy_write_directory")
        return path

    @pytest.fixture(scope="class")
    def channel_patch(self, random_patch):
        """Get a patch with channels rather than distance."""
        distance = random_patch.get_coord("distance")
        new = random_patch.rename_coords(distance="channel").update_coords(
            **{"channel": np.arange(len(distance))}
        )
        return new

    @pytest.fixture(scope="class")
    def channel_patch_path(self, channel_patch, test_segy_directory):
        """Write the channel patch to disk."""
        path = test_segy_directory / "patch_with_channel_coord.segy"
        channel_patch.io.write(path, "segy")
        return path

    def test_can_get_format(self, channel_patch_path):
        """Ensure we can get the correct format/version."""
        segy = SegyV1_0()
        out = segy.get_format(channel_patch_path)
        assert out, "Failed to detect written segy file."
        assert out[0] == segy.name

    def test_channel_patch_round_trip(self, channel_patch_path, channel_patch):
        """The channel patch should round trip."""
        patch1 = channel_patch
        patch2 = dc.spool(channel_patch_path)[0].transpose(*patch1.dims)
        # We really don't have a way to transport attributes yet, so we
        # just check that data and coords are equal.
        assert np.allclose(patch1.data, patch2.data)
        assert patch1.coords == patch2.coords
