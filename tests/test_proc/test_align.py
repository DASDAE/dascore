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
    bad_type_coord = dc.to_datetime64(distance)

    return patch.update_coords(
        shift_time_samples=("distance", shift_time_samples),
        shift_time_relative=("distance", shift_time_relative),
        bad_type_shift=("distance", bad_type_coord),
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
        "shift_time_samples_1d": ("distance", shift_time_samples),
        "shift_time_1d": ("distance", shift_time_relative),
        "shift_time_2d": (("distance", "dim3"), shift_time_2d - shift_time_2d.min()),
        "shift_time_3d": (
            ("distance", "dim3", "dim4"),
            shift_time_3d - shift_time_3d.min(),
        ),
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
    zero_shifts = np.zeros(len(distance), dtype=np.int64)
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


@pytest.fixture(scope="class")
def patch_non_overlapping_shifts(random_patch):
    """Patch with shifts that cause some traces to not overlap."""
    sub = (
        random_patch.select(distance=slice(0, 3), samples=True)
        .select(time=slice(0, 10), samples=True)
        .update_coords(shift=("distance", np.array([0, 8, 16])))
    )
    return sub


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

    def test_invalid_mode_raises(self, patch_with_align_coord_2d):
        """Test that invalid mode raises ParameterError."""
        patch = patch_with_align_coord_2d
        with pytest.raises(ParameterError, match="mode must be one of"):
            patch.align_to_coord(
                time="shift_time_samples", mode="invalid", samples=True
            )

    def test_samples_non_int_raises(self, patch_with_align_coord_2d):
        """Ensure non int coords raise when using samples=True"""
        patch = patch_with_align_coord_2d
        msg = "samples=True"
        with pytest.raises(ParameterError, match=msg):
            patch.align_to_coord(time="shift_time_relative", samples=True)

    def test_bad_alignment_dim(self, patch_with_align_coord_2d):
        """Ensure incompatible coord/dim dtypes raises."""
        patch = patch_with_align_coord_2d
        msg = "Incompatible dtype"
        with pytest.raises(ParameterError, match=msg):
            patch.align_to_coord(time="bad_type_shift")


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
        else:
            shift = shift_values / time.step
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
            mode=mode,
        )
        validator = self._validators[mode].__get__(self)
        validator(patch, shifted_patch, shift_dim=shift_dim, shift_coord=shift_coord)

    @pytest.mark.parametrize("mode", list(_validators))
    def test_4d_patch_1d_shift(self, patch_with_align_coord_4d, mode):
        """Test 4D patch with 1D shift coordinate."""
        patch = patch_with_align_coord_4d
        shift_dim = "time"
        shift_coord = "shift_time_1d"
        shifted_patch = patch.align_to_coord(
            **{shift_dim: shift_coord},
            mode=mode,
        )
        validator = self._validators[mode].__get__(self)
        validator(patch, shifted_patch, shift_dim=shift_dim, shift_coord=shift_coord)

    @pytest.mark.parametrize(
        "fixture_name,shift_coord,kwargs",
        [
            ("patch_with_known_shifts", "shift_time_known", {"samples": True}),
            ("patch_with_zero_shifts", "shift_time_zero", {"samples": True}),
            ("patch_with_mixed_shifts", "shift_time_mixed", {"samples": True}),
        ],
    )
    def test_round_trip_with_reverse(self, fixture_name, shift_coord, kwargs, request):
        """Test that alignment can be reversed using reverse parameter."""
        patch = request.getfixturevalue(fixture_name)

        # Apply forward alignment with full mode to preserve all data
        aligned = patch.align_to_coord(time=shift_coord, mode="full", **kwargs)

        # Apply reverse alignment to undo the shift
        reversed_patch = aligned.align_to_coord(
            time=shift_coord, mode="full", reverse=True, **kwargs
        )

        # Verify round-trip - check data values match where not NaN
        for i in range(5):
            orig_trace = patch.data[:, i]
            rev_trace_full = reversed_patch.data[:, i]
            # Find non-NaN indices
            non_nan = ~np.isnan(rev_trace_full)
            # Verify the non-NaN data matches original
            np.testing.assert_array_equal(rev_trace_full[non_nan], orig_trace)

    def test_valid_mode_reverse(self, patch_with_known_shifts):
        """Test reverse with valid mode (data loss expected)."""
        patch = patch_with_known_shifts

        # Forward alignment
        aligned = patch.align_to_coord(
            time="shift_time_known", samples=True, mode="valid"
        )

        # Reverse won't fully restore, but shouldn't crash
        reversed_patch = aligned.align_to_coord(
            time="shift_time_known", samples=True, mode="valid", reverse=True
        )

        # Verify it produces valid output with same number of traces
        assert reversed_patch.shape[1] == patch.shape[1]

    def test_align_round_trip(self, simple_patch_for_alignment):
        """Test round_trip with relative values."""
        patch = simple_patch_for_alignment
        time = patch.get_array("time")

        # Create relative time shifts (relative to start)
        relative_shifts = time[[0, 1, 2, 3, 4]]  # First 5 time values
        relative_shifts = relative_shifts - relative_shifts.min()
        patch = patch.update_coords(rel_shifts=("distance", relative_shifts))

        # Forward
        aligned = patch.align_to_coord(time="rel_shifts", mode="full")

        # Reverse
        reversed_patch = aligned.align_to_coord(
            time="rel_shifts", mode="full", reverse=True
        )

        # Verify round-trip
        reversed_clean = reversed_patch.dropna("time")
        assert reversed_clean.shape == patch.shape

    def test_reverse_coordinate_accuracy(self, patch_with_known_shifts):
        """Verify coordinate step is preserved after reverse."""
        patch = patch_with_known_shifts
        original_time_coord = patch.get_coord("time")

        # Forward alignment
        aligned = patch.align_to_coord(
            time="shift_time_known", samples=True, mode="full"
        )

        # Reverse alignment
        reversed_patch = aligned.align_to_coord(
            time="shift_time_known", samples=True, mode="full", reverse=True
        )

        # Coordinates should have same step
        assert np.isclose(
            reversed_patch.get_coord("time").step, original_time_coord.step
        )

    def test_align_no_overlap(self, patch_non_overlapping_shifts):
        """Test that a patch with shifts that cause traces to not overlap."""
        patch = patch_non_overlapping_shifts
        msg = "some traces with no overlaps"
        # Mode = valid and same should fail.
        with pytest.raises(ParameterError, match=msg):
            patch.align_to_coord(time="shift", mode="valid")
        with pytest.raises(ParameterError, match=msg):
            patch.align_to_coord(time="shift", mode="same")
        # But full should work.
        new = patch.align_to_coord(time="shift", mode="full")
        assert new.dropna("time").data.size == 0
