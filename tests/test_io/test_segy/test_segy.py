"""Tests for SEGY format."""

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import (
    InvalidSpoolError,
    PatchError,
)
from dascore.io.segy.core import SegyV1_0
from dascore.utils.misc import suppress_warnings


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
        pytest.importorskip("segyio")
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

    def test_write_non_channel_path(self, random_patch, tmp_path_factory):
        """Ensure a 'normal' patch can be written."""
        pytest.importorskip("segyio")
        path = tmp_path_factory.mktemp("test_write_segy") / "temppath.segy"
        match = "non-time dimension"
        with pytest.warns(match=match):
            random_patch.io.write(path, "segy")
        assert path.exists()
        patch2 = dc.spool(path)[0]
        assert set(random_patch.shape) == set(patch2.shape)

    def test_loss_of_precision_raises(self, random_patch, tmp_path_factory):
        """Ensure that loss of precision raises a PatchError."""
        pytest.importorskip("segyio")
        path = tmp_path_factory.mktemp("test_loss_of_precision") / "temppath.segy"
        patch = random_patch.update_coords(time_step=np.timedelta64(10, "ns"))
        match = "will result in a loss of precision"
        with pytest.raises(PatchError, match=match):
            with suppress_warnings():
                patch.io.write(path, "segy")

    def test_bad_dims_raises(self, random_patch, tmp_path):
        """Ensure a bad dimension name raises."""
        pytest.importorskip("segyio")
        patch = random_patch.rename_coords(distance="bad_dim")
        with pytest.raises(PatchError, match="Can only save 2D patches"):
            patch.io.write(tmp_path, "segy")

    def test_multi_patch_spool_raises(self, random_spool, tmp_path):
        """Spools with more than one patch cant be written."""
        pytest.importorskip("segyio")
        segy = SegyV1_0()
        with pytest.raises(InvalidSpoolError):
            segy.write(random_spool, tmp_path)
