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
        "shift_time_samples_1d": ("distance", shift_time_samples),
        "shift_time_relative_1d": ("distance", shift_time_relative),
        "shift_time_absolute_2d": (("distance", "dim3"), shift_time_2d),
        "shift_time_absolute_3d": (("distance", "dim3", "dim4"), shift_time_3d),
    }
    dims = (*patch.dims, "dim3", "dim4")
    return dc.Patch(data=data2, coords=coords, dims=dims)


@pytest.fixture(scope="class")
def simple_patch_for_alignment():
    """Create a simple patch with predictable data for alignment testing."""
    # Create small patch: 10 time samples, 5 distance channels
    time_vals = np.arange(10) * 0.1  # 0.0, 0.1, 0.2, ..., 0.9
    dist_vals = np.arange(5) * 10.0  # 0, 10, 20, 30, 40

    # Create data where each trace has constant value equal to its channel index
    # trace 0: all 0s, trace 1: all 1s, trace 2: all 2s, etc.
    data = np.zeros((10, 5))
    for i in range(5):
        data[:, i] = i

    coords = {
        "time": time_vals,
        "distance": dist_vals,
    }
    dims = ("time", "distance")
    return dc.Patch(data=data, coords=coords, dims=dims)


@pytest.fixture(scope="class")
def patch_with_zero_shifts(simple_patch_for_alignment):
    """Patch with zero shifts for all channels."""
    patch = simple_patch_for_alignment
    distance = patch.get_array("distance")
    zero_shifts = np.zeros(len(distance))
    return patch.update_coords(shift_time_zero=("distance", zero_shifts))


@pytest.fixture(scope="class")
def patch_with_known_shifts(simple_patch_for_alignment):
    """Patch with known positive shifts: [0, 1, 2, 3, 4] samples."""
    patch = simple_patch_for_alignment
    distance = patch.get_array("distance")
    known_shifts = np.arange(len(distance))  # [0, 1, 2, 3, 4]
    return patch.update_coords(shift_time_known=("distance", known_shifts))


@pytest.fixture(scope="class")
def patch_with_mixed_shifts(simple_patch_for_alignment):
    """Patch with mixed positive and negative shifts: [-2, -1, 0, 1, 2]."""
    patch = simple_patch_for_alignment
    distance = patch.get_array("distance")
    mixed_shifts = np.arange(len(distance)) - 2  # [-2, -1, 0, 1, 2]
    return patch.update_coords(shift_time_mixed=("distance", mixed_shifts))


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

    def test_raises_non_strings(self, patch_with_align_coord_2d):
        """Test that calling with non-string kwargs raises ParameterError."""
        patch = patch_with_align_coord_2d
        with pytest.raises(ParameterError, match="to be strings"):
            patch.align_to_coord(time=None)


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

    def test_correctness_no_shift(self, patch_with_zero_shifts):
        """Test that data values remain unchanged when shifts are zero."""
        patch = patch_with_zero_shifts

        # Test all modes with zero shifts
        for mode in ["same", "full", "valid"]:
            shifted_patch = patch.align_to_coord(
                time="shift_time_zero", mode=mode, samples=True
            )

            # Data should be identical to original
            assert shifted_patch.shape == patch.shape
            np.testing.assert_array_equal(shifted_patch.data, patch.data)

            # Each trace should still have its original constant value
            for i in range(5):  # 5 distance channels
                trace_data = shifted_patch.data[:, i]
                assert np.all(trace_data == i), f"Trace {i} should have value {i}"

    def test_positive_shifts_same_mode(self, patch_with_known_shifts):
        """Test data correctness with positive shifts in same mode."""
        patch = patch_with_known_shifts
        shifted_patch = patch.align_to_coord(
            time="shift_time_known", mode="same", samples=True
        )

        # Shape should remain the same
        assert shifted_patch.shape == patch.shape

        # Check each trace: trace i should be shifted by i samples
        for i in range(5):
            trace_data = shifted_patch.data[:, i]

            # First i samples should be NaN (due to shift)
            assert np.all(
                np.isnan(trace_data[:i])
            ), f"First {i} samples of trace {i} should be NaN"

            # Remaining samples should have value i
            remaining_samples = trace_data[i:]
            expected_length = 10 - i  # 10 total samples minus i shifted samples
            assert len(remaining_samples) == expected_length
            assert np.all(
                remaining_samples == i
            ), f"Remaining samples of trace {i} should be {i}"

    def test_mixed_shifts_full_mode(self, patch_with_known_shifts):
        """Test data correctness with varying shifts in full mode."""
        patch = patch_with_known_shifts
        shifted_patch = patch.align_to_coord(
            time="shift_time_known", mode="full", samples=True
        )

        # Full mode should expand the time dimension to accommodate all shifts
        # Original: 10 samples, shifts: [0, 1, 2, 3, 4], range = 4
        # New size should be: 10 + 4 = 14
        assert shifted_patch.shape == (14, 5)

        # Check that all original data is preserved
        # Each trace should appear at the correct shifted position
        shifts = np.array([0, 1, 2, 3, 4])
        min_shift = shifts.min()  # 0

        for i in range(5):
            trace_data = shifted_patch.data[:, i]
            shift = shifts[i]

            # Calculate where this trace's data should start in the output
            start_pos = shift - min_shift  # This gives us [0, 1, 2, 3, 4]
            end_pos = start_pos + 10  # Original data length is 10

            # The trace data at the correct position should have value i
            trace_segment = trace_data[start_pos:end_pos]
            assert np.all(
                trace_segment == i
            ), f"Trace {i} data should be {i} at positions {start_pos}:{end_pos}"

            # All other positions should be NaN
            before_segment = trace_data[:start_pos]
            after_segment = trace_data[end_pos:]
            if len(before_segment) > 0:
                assert np.all(
                    np.isnan(before_segment)
                ), f"Before segment of trace {i} should be NaN"
            if len(after_segment) > 0:
                assert np.all(
                    np.isnan(after_segment)
                ), f"After segment of trace {i} should be NaN"

    def test_valid_mode_overlap(self, patch_with_known_shifts):
        """Test data correctness in valid mode with overlapping regions."""
        patch = patch_with_known_shifts
        shifted_patch = patch.align_to_coord(
            time="shift_time_known", mode="valid", samples=True
        )

        # Valid mode should return only the overlapping region
        # Shifts: [0, 1, 2, 3, 4], original length: 10
        # Max shift: 4, so valid region is 10 - 4 = 6 samples
        assert shifted_patch.shape == (6, 5)

        # In valid mode, no NaN values should exist
        assert not np.any(np.isnan(shifted_patch.data))

        # Each trace should still have its original constant value
        for i in range(5):
            trace_data = shifted_patch.data[:, i]
            assert np.all(
                trace_data == i
            ), f"Trace {i} should have value {i} in valid mode"

    def test_round_trip_with_reverse(self, patch_with_known_shifts):
        """Test that alignment can be reversed using reverse parameter."""
        patch = patch_with_known_shifts

        # Apply forward alignment with full mode to preserve all data
        aligned = patch.align_to_coord(
            time="shift_time_known", mode="full", samples=True
        )

        # Apply reverse alignment to undo the shift
        reversed_patch = aligned.align_to_coord(
            time="shift_time_known", mode="full", samples=True, reverse=True
        )

        # Drop NaN values reversed patch
        reversed_clean = reversed_patch.dropna("time")

        # Verify shapes, coords and data are the same after dropping nan.
        assert reversed_clean.shape == patch.shape
        assert reversed_clean.equals(patch)

    def test_reverse_with_zero_shifts(self, patch_with_zero_shifts):
        """Test reverse parameter with zero shifts (should be identity)."""
        patch = patch_with_zero_shifts

        # Apply forward alignment with zero shifts
        aligned = patch.align_to_coord(
            time="shift_time_zero", mode="full", samples=True
        )

        # Apply reverse alignment (should be identity)
        reversed_patch = aligned.align_to_coord(
            time="shift_time_zero", mode="full", samples=True, reverse=True
        )

        # Both patches should be identical to original
        np.testing.assert_array_equal(aligned.data, patch.data)
        np.testing.assert_array_equal(reversed_patch.data, patch.data)
        assert aligned.shape == patch.shape
        assert reversed_patch.shape == patch.shape
