"""Tests for example fetching."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.examples import EXAMPLE_PATCHES
from dascore.exceptions import UnknownExampleError
from dascore.utils.time import to_float


class TestGetExamplePatch:
    """Test suite for `get_example_patch`."""

    def test_default(self):
        """Ensure calling get_example_patch with no args returns patch."""
        patch = dc.get_example_patch()
        assert isinstance(patch, dc.Patch)

    def test_raises_on_bad_key(self):
        """Ensure a bad key raises expected error."""
        with pytest.raises(UnknownExampleError, match="No example patch"):
            dc.get_example_patch("NotAnExampleRight????")

    def test_data_file_name(self):
        """Ensure get_example_spool works on a datafile."""
        spool = dc.get_example_spool("dispersion_event.h5")
        assert isinstance(spool, dc.BaseSpool)

    @pytest.mark.parametrize("name", EXAMPLE_PATCHES)
    def test_load_example_patch(self, name):
        """Ensure the registered example patches can all be loaded."""
        patch = dc.get_example_patch(name)
        assert isinstance(patch, dc.Patch)


class TestGetExampleSpool:
    """Test suite for `get_example_spool`."""

    def test_default(self):
        """Ensure calling get_example_spool with no args returns a Spool."""
        patch = dc.get_example_spool()
        assert isinstance(patch, dc.BaseSpool)

    def test_raises_on_bad_key(self):
        """Ensure a bad key raises expected error."""
        with pytest.raises(UnknownExampleError, match="No example spool"):
            dc.get_example_spool("NotAnExampleRight????")

    def test_data_file_name(self):
        """Ensure get_example_spool works on a datafile."""
        spool = dc.get_example_spool("dispersion_event.h5")
        assert isinstance(spool, dc.BaseSpool)


class TestRickerMoveout:
    """Tests for Ricker moveout patch."""

    def test_moveout(self):
        """Ensure peaks of ricker wavelet line up with expected moveout."""
        velocity = 100
        patch = dc.get_example_patch("ricker_moveout", velocity=velocity)
        argmaxes = np.argmax(patch.data, axis=0)
        peak_times = patch.get_coord("time").values[argmaxes]
        moveout = to_float(peak_times - np.min(peak_times))
        distances = patch.get_coord("distance").values
        expected_moveout = distances / velocity
        assert np.allclose(moveout, expected_moveout)


class TestDeltaPatch:
    """Tests for the delta_patch example."""

    @pytest.mark.parametrize("invalid_dim", ["inv_dim", "", None, 123, 1.1])
    def test_delta_patch_invalid_dim(self, invalid_dim):
        """
        Test that passing an invalid dimension value raises a ValueError.
        """
        msg = "with 'time' and 'distance'"
        with pytest.raises(ValueError, match=msg):
            dc.get_example_patch("delta_patch", dim=invalid_dim)

    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_delta_patch_structure(self, dim):
        """Test that the delta_patch returns a Patch with correct structure."""
        patch = dc.get_example_patch("delta_patch", dim=dim)
        assert isinstance(patch, dc.Patch), "delta_patch should return a Patch instance"

        dims = patch.dims
        assert (
            "time" in dims and "distance" in dims
        ), "Patch must have 'time' and 'distance' dimensions"

    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_delta_patch_delta_location(self, dim):
        """
        Ensures the delta is at the center of the chosen dimension and zeros elsewhere.
        """
        # The default shape from the function signature: shape=(10, 200)
        # If dim="time", we end up with a single (distance=0) trace => shape (200,)
        # If dim="distance", we end up with a single (time=0) trace => shape (10,)
        patch = dc.get_example_patch("delta_patch", dim=dim)
        data = patch.squeeze().data

        # The expected midpoint and verify single delta at center
        mid_idx = len(data) // 2

        assert data[mid_idx] == 1.0, "Expected a unit delta at the center"
        # Check all other samples are zero
        # Replace the center value with zero and ensure all zeros remain
        test_data = np.copy(data)
        test_data[mid_idx] = 0
        assert np.allclose(
            test_data, 0
        ), "All other samples should be zero except the center"

    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_delta_patch_with_patch(self, dim):
        """Test passing an existing patch to delta_patch and ensure delta is applied."""
        # Create a base patch
        base_patch = dc.get_example_patch("random_das", shape=(5, 50))
        # Apply the delta_patch function with the existing patch
        delta_applied_patch = dc.get_example_patch(
            "delta_patch", dim=dim, patch=base_patch
        )

        assert isinstance(delta_applied_patch, dc.Patch), "Should return a Patch"
        data = delta_applied_patch.squeeze().data

        # Check that only the center value is one and others are zero
        mid_idx = len(data) // 2
        assert data[mid_idx] == 1.0, "Center sample should be 1.0"
        test_data = np.copy(data)
        test_data[mid_idx] = 0
        assert np.allclose(
            test_data, 0
        ), "All other samples should be zero except the center"

    @pytest.mark.parametrize("dim", ["lag_time", "distance"])
    def test_delta_patch_with_3d_patch(self, dim):
        """Test passing a 3D patch."""
        # Create a base patch
        base_patch = dc.get_example_patch("sin_wav")
        base_patch_3d = base_patch.correlate(distance=[2], samples=True)
        # Apply the delta_patch function with the existing patch
        delta_applied_patch = dc.get_example_patch(
            "delta_patch", dim=dim, patch=base_patch_3d
        )
        assert isinstance(delta_applied_patch, dc.Patch), "Should return a Patch"
        data = delta_applied_patch.squeeze().data
        # Check that only the center value is one and others are zero
        mid_idx = len(data) // 2
        assert data[mid_idx] == 1.0, "Center sample should be 1.0"
        test_data = np.copy(data)
        test_data[mid_idx] = 0
        assert np.allclose(
            test_data, 0
        ), "All other samples should be zero except the center"
