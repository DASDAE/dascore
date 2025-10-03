"""
Tests for alignment functionality.
"""

from typing import ClassVar

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError


@pytest.fixture(scope="class")
def patch_with_align_coord_2d():
    """Create a patch with a non-dimensional coordinate for alignment."""
    patch = dc.get_example_patch()
    time = patch.get_array("time")
    distance = patch.get_array("distance")
    # Create three potential coords for shifting
    shift_time_samples = np.arange(len(distance))
    shift_time_absolute = time[shift_time_samples]
    shift_time_relative = shift_time_absolute - shift_time_absolute.min()

    return patch.update_coords(
        shift_time_absolute=("distance", shift_time_absolute),
        shift_time_samples=("distance", shift_time_samples),
        shift_time_relative=("distance", shift_time_relative),
    )


@pytest.fixture(scope="class")
def patch_with_align_coord_4d():
    """Create a patch with 1D non-dimensional coordinate for alignment."""
    patch = dc.get_example_patch()
    time = patch.get_array("time")
    distance = patch.get_array("distance")

    data1 = np.stack([patch.data, patch.data], axis=-1)
    data2 = np.stack([data1, data1], axis=-1)

    # Create shift coordinate
    shift_time_samples = np.arange(len(distance))
    shift_time_absolute = time[shift_time_samples]
    shift_time_relative = shift_time_absolute - shift_time_absolute.min()

    shift_time_2d = np.stack([shift_time_absolute, shift_time_absolute], axis=-1)
    shift_time_3d = np.stack([shift_time_2d, shift_time_2d], axis=-1)

    coords = {
        "time": time,
        "distance": distance,
        "dim3": [1, 2],
        "dim4": [2, 3],
        "shift_time_absolute_1d": ("distance", shift_time_absolute),
        "shift_time_samples_1d": ("distance", shift_time_absolute),
        "shift_time_relative_1d": ("distance", shift_time_relative),
        "shift_time_absolute_2d": (("distance", "dim3"), shift_time_2d),
        "shift_time_absolute_3d": (("distance", "dim3", "dim4"), shift_time_3d),
    }
    dims = (*patch.dims, "dim3", "dim4")
    return dc.Patch(data=data2, coords=coords, dims=dims)


class TestAlignToCoordValidation:
    """Tests for align_to_coord validation logic."""

    def test_no_kwargs_raises(self, patch_with_align_coord_2d):
        """Test that calling with no kwargs raises ParameterError."""
        with pytest.raises(ParameterError, match="exactly one keyword"):
            patch_with_align_coord_2d.align_to_coord()

    def test_multiple_kwargs_raises(self, patch_with_align_coord_2d):
        """Test that calling with multiple kwargs raises ParameterError."""
        with pytest.raises(ParameterError, match="exactly one keyword"):
            patch_with_align_coord_2d.align_to_coord(
                time="ref_time", distance="some_coord"
            )

    def test_invalid_dimension_raises(self, patch_with_align_coord_2d):
        """Test that invalid dimension name raises ParameterError."""
        with pytest.raises(ParameterError, match="not one of the patch"):
            patch_with_align_coord_2d.align_to_coord(invalid_dim="ref_time")

    def test_invalid_coord_name_raises(self, patch_with_align_coord_2d):
        """Test that invalid coordinate name raises ParameterError."""
        with pytest.raises(ParameterError, match="not a coordinate"):
            patch_with_align_coord_2d.align_to_coord(time="nonexistent_coord")

    def test_coord_is_dimension_raises(self, patch_with_align_coord_2d):
        """Test that coord name that is also a dimension raises error."""
        with pytest.raises(ParameterError, match="not a coordinate"):
            patch_with_align_coord_2d.align_to_coord(time="distance")

    def test_coord_depends_on_dimension_raises(self, patch_with_align_coord_2d):
        """Test that coord depending on selected dimension raises error."""
        patch = patch_with_align_coord_2d.update_coords(
            time_dep_coord=(
                "time",
                np.arange(len(patch_with_align_coord_2d.coords.get_array("time"))),
            )
        )
        with pytest.raises(ParameterError, match="not depend on the selected"):
            patch.align_to_coord(time="time_dep_coord")


class TestAlignToCoord:
    """Tests for aligning patches."""

    def _test_mode_same(
        self,
        patch,
        shifted_patch,
        shift_dim="time",
        shift_coord="shift_time",
    ):
        """Helper function to test mode=same."""
        assert shifted_patch.shape == patch.shape
        # The shift value should be the same as the number of NaN in each
        # trace (due to how the shift_time coord was built).
        time = shifted_patch.get_coord(shift_dim)
        axis = shifted_patch.get_axis(shift_dim)
        nan_count = np.isnan(shifted_patch.data).sum(axis=axis, keepdims=True)
        shift_values = shifted_patch.get_array(shift_coord)
        # Handle different shift coordinate types
        if shift_coord.endswith("_samples"):
            # Shift coord is already in samples
            shift = shift_values
        elif shift_coord.endswith("_relative"):
            # Shift coord is relative to start, just divide by step
            shift = shift_values / time.step
        else:
            # Shift coord is in absolute units, convert to samples
            shift = (shift_values - time.min()) / time.step
        # Need to broadcast up shift.
        if nan_count.ndim > shift.ndim:
            dims = shifted_patch.coords.dim_map[shift_coord]
            missing = set(shifted_patch.dims) - set(dims)
            args = [slice(None)] * nan_count.ndim
            for dim in missing:
                args[shifted_patch.get_axis(dim)] = None
            shift = shift[tuple(args)]
        assert np.allclose(nan_count, shift)

    def _test_mode_full(
        self,
        patch,
        shifted_patch,
        shift_dim="time",
        shift_coord="shift_time",
    ):
        """Tests for full mode shifts."""
        # The number of NaN should be the same for all channels.
        axis = shifted_patch.get_axis(shift_dim)
        nan_count = np.isnan(shifted_patch.data).sum(axis=axis)
        assert len(np.unique(nan_count)) == 1

    def _test_valid_mode(
        self,
        patch,
        shifted_patch,
        shift_dim="time",
        shift_coord="shift_time",
    ):
        """Tests for when the valid mode is used."""
        # Valid mode should produce smaller or equal sized output
        axis = shifted_patch.get_axis(shift_dim)
        assert shifted_patch.shape[axis] <= patch.shape[axis]
        # There should be no NaN values in valid mode
        assert np.isnan(shifted_patch.data).sum() == 0

    _validators: ClassVar = {
        "full": _test_mode_full,
        "same": _test_mode_same,
        "valid": _test_valid_mode,
    }

    @pytest.mark.parametrize("mode", list(_validators))
    def test_shift_2d_absolute(self, patch_with_align_coord_2d, mode):
        """Test the 2D case with various modes."""
        patch = patch_with_align_coord_2d
        shift_dim = "time"
        shift_coord = "shift_time_absolute"
        shifted_patch = patch.align_to_coord(
            **{shift_dim: shift_coord},
            mode=mode,
        )
        validator = self._validators[mode].__get__(self)
        validator(patch, shifted_patch, shift_dim=shift_dim, shift_coord=shift_coord)

    @pytest.mark.parametrize("mode", list(_validators))
    def test_shift_2d_samples(self, patch_with_align_coord_2d, mode):
        """Test the 2D case with various modes."""
        patch = patch_with_align_coord_2d
        shift_dim = "time"
        shift_coord = "shift_time_samples"
        shifted_patch = patch.align_to_coord(
            **{shift_dim: shift_coord}, samples=True, mode=mode
        )
        validator = self._validators[mode].__get__(self)
        validator(patch, shifted_patch, shift_dim=shift_dim, shift_coord=shift_coord)

    @pytest.mark.parametrize("mode", list(_validators))
    def test_shift_2d_relative(self, patch_with_align_coord_2d, mode):
        """Test the 2D case with various modes."""
        patch = patch_with_align_coord_2d
        shift_dim = "time"
        shift_coord = "shift_time_relative"
        shifted_patch = patch.align_to_coord(
            **{shift_dim: shift_coord},
            relative=True,
            mode=mode,
        )
        validator = self._validators[mode].__get__(self)
        validator(patch, shifted_patch, shift_dim=shift_dim, shift_coord=shift_coord)

    @pytest.mark.parametrize("mode", list(_validators))
    def test_4d_patch_1d_shift(self, patch_with_align_coord_4d, mode):
        """Test properties of the same mode."""
        patch = patch_with_align_coord_4d
        shift_dim = "time"
        shift_coord = "shift_time_absolute_1d"
        shifted_patch = patch.align_to_coord(
            **{shift_dim: shift_coord},
            mode=mode,
        )
        validator = self._validators[mode].__get__(self)
        validator(patch, shifted_patch, shift_dim=shift_dim, shift_coord=shift_coord)

    @pytest.mark.parametrize("mode", list(_validators))
    def test_4d_patch_2d_shift(self, patch_with_align_coord_4d, mode):
        """Test properties of the same mode."""
        patch = patch_with_align_coord_4d
        shift_dim = "time"
        shift_coord = "shift_time_absolute_2d"
        shifted_patch = patch.align_to_coord(
            **{shift_dim: shift_coord},
            mode=mode,
        )
        validator = self._validators[mode].__get__(self)
        validator(patch, shifted_patch, shift_dim=shift_dim, shift_coord=shift_coord)
