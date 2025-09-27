"""
Tests for the hampel filter implementation.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError


def _get_interior_data(data, edge_size=5):
    """Get interior data excluding edges for comparison."""
    if data.ndim == 2:
        h, w = data.shape
        if h > 2 * edge_size and w > 2 * edge_size:
            return data[edge_size:-edge_size, edge_size:-edge_size]
    elif data.ndim == 1:
        if len(data) > 2 * edge_size:
            return data[edge_size:-edge_size]
    # If too small, return the whole array
    return data


def _assert_interior_equal(arr1, arr2, edge_size=5, **kwargs):
    """Assert arrays are equal excluding edge regions."""
    interior1 = _get_interior_data(arr1, edge_size)
    interior2 = _get_interior_data(arr2, edge_size)
    np.testing.assert_array_equal(interior1, interior2, **kwargs)


class TestHampelFilter:
    """Tests for the hampel_filter function."""

    # Class variables for spike locations to avoid hard-coding
    SPIKE_LOCATIONS: ClassVar[list[tuple[int, int, float]]] = [
        (10, 15, 10.0),  # (time_idx, distance_idx, spike_value)
        (20, 25, -10.0),  # Large negative spike
        (30, 35, 8.0),  # Another spike
    ]

    # Channel-wide spike indices
    DISTANCE_CHANNEL_SPIKE_IDX = 20
    TIME_CHANNEL_SPIKE_IDX = 30

    # Values for channel-wide spikes
    DISTANCE_CHANNEL_SPIKE_VALUE = 15.0
    TIME_CHANNEL_SPIKE_VALUE = -12.0

    @pytest.fixture()
    def patch_with_spikes(self):
        """Create a patch with artificial spikes for testing."""
        # Get example patch and add spikes
        patch = dc.get_example_patch()
        data = patch.data.copy()

        # Add individual spikes using class variables
        for time_idx, dist_idx, spike_val in self.SPIKE_LOCATIONS:
            if time_idx < data.shape[0] and dist_idx < data.shape[1]:
                data[time_idx, dist_idx] = spike_val

        new_patch = patch.update(data=data)
        # Change sampling rate as to avoid long windows.
        return new_patch.update_coords(time_step=0.2)

    @pytest.fixture()
    def patch_uniform_data(self):
        """Create a patch with uniform data for testing edge cases."""
        # Get example patch and make data uniform (zeros)
        patch = dc.get_example_patch()
        data = np.zeros_like(patch.data)
        new_patch = patch.update(data=data)
        return new_patch.update_coords(time_step=0.2)

    def test_basic_hampel_filter(self, patch_with_spikes):
        """Test basic hampel filtering functionality."""
        # Test with time dimension filtering
        result = patch_with_spikes.hampel_filter(time=0.6, threshold=3.5)

        # Check that data shape is preserved
        assert result.data.shape == patch_with_spikes.data.shape

        # Check that spikes are reduced (using first spike location)
        time_idx, dist_idx, _ = self.SPIKE_LOCATIONS[0]
        original_spike = patch_with_spikes.data[time_idx, dist_idx]
        filtered_spike = result.data[time_idx, dist_idx]
        assert abs(filtered_spike) < abs(original_spike)

    def test_multiple_dimensions(self, patch_with_spikes):
        """Test hampel filter with multiple dimensions."""
        result = patch_with_spikes.hampel_filter(time=0.6, distance=3.0, threshold=3.5)

        # Check that data shape is preserved
        assert result.data.shape == patch_with_spikes.data.shape

        # Should remove spikes more effectively (using first spike location)
        time_idx, dist_idx, _ = self.SPIKE_LOCATIONS[0]
        original_spike = patch_with_spikes.data[time_idx, dist_idx]
        filtered_spike = result.data[time_idx, dist_idx]
        assert abs(filtered_spike) < abs(original_spike)

    def test_threshold_parameter(self, patch_with_spikes):
        """Test different threshold values."""
        # Low threshold - more aggressive filtering
        result_low = patch_with_spikes.hampel_filter(time=0.6, threshold=1.0)

        # High threshold - less aggressive filtering
        result_high = patch_with_spikes.hampel_filter(time=0.6, threshold=10.0)

        # Low threshold should change more values
        diff_low = np.sum(result_low.data != patch_with_spikes.data)
        diff_high = np.sum(result_high.data != patch_with_spikes.data)

        assert diff_low >= diff_high

    def test_default_threshold(self, patch_with_spikes):
        """Test that default threshold (10.0) works correctly."""
        # Test with default threshold
        result_default = patch_with_spikes.hampel_filter(time=0.6)

        # Test with explicit threshold=10.0
        result_explicit = patch_with_spikes.hampel_filter(time=0.6, threshold=10.0)

        # Should produce identical results
        np.testing.assert_array_equal(result_default.data, result_explicit.data)
        assert result_default.data.shape == patch_with_spikes.data.shape

    def test_samples_parameter_true(self, patch_with_spikes):
        """Test with samples=True parameter."""
        # Use samples instead of coordinate units (must be odd number)
        result = patch_with_spikes.hampel_filter(time=9, samples=True, threshold=3.5)

        # Check that data shape is preserved
        assert result.data.shape == patch_with_spikes.data.shape

    def test_samples_parameter_false(self, patch_with_spikes):
        """Test with samples=False parameter (default)."""
        # Use coordinate units
        result = patch_with_spikes.hampel_filter(time=0.6, samples=False, threshold=3.5)

        # Check that data shape is preserved
        assert result.data.shape == patch_with_spikes.data.shape

    def test_approximate_parameter(self, patch_with_spikes):
        """Test the approximate parameter for faster processing."""
        # Standard 2D filtering
        result_standard = patch_with_spikes.hampel_filter(
            time=0.6, distance=3.0, threshold=3.5, approximate=False
        )

        # Approximate filtering (faster approximation)
        result_approximate = patch_with_spikes.hampel_filter(
            time=0.6, distance=3.0, threshold=3.5, approximate=True
        )

        # Both should have same shape
        assert result_standard.data.shape == patch_with_spikes.data.shape
        assert result_approximate.data.shape == patch_with_spikes.data.shape

        # Results should be similar but not identical (approximation)
        # Check that spikes are still reduced in both cases
        time_idx, dist_idx, _ = self.SPIKE_LOCATIONS[0]
        original_spike = abs(patch_with_spikes.data[time_idx, dist_idx])

        standard_spike = abs(result_standard.data[time_idx, dist_idx])
        approximate_spike = abs(result_approximate.data[time_idx, dist_idx])

        # Both should reduce the spike
        assert standard_spike < original_spike
        assert approximate_spike < original_spike

    def test_approximate_single_dimension(self, patch_with_spikes):
        """
        Test approximate parameter with single dimension provides effective spike
        removal.
        """
        # Single dimension cases should both be effective at spike removal
        result_standard = patch_with_spikes.hampel_filter(
            time=0.6, threshold=3.5, approximate=False
        )
        result_approximate = patch_with_spikes.hampel_filter(
            time=0.6, threshold=3.5, approximate=True
        )

        # Both should significantly reduce the spikes at known locations
        original_data = patch_with_spikes.data
        for time_idx, dist_idx, spike_val in self.SPIKE_LOCATIONS:
            if time_idx < original_data.shape[0] and dist_idx < original_data.shape[1]:
                original_spike = abs(original_data[time_idx, dist_idx])
                standard_spike = abs(result_standard.data[time_idx, dist_idx])
                approximate_spike = abs(result_approximate.data[time_idx, dist_idx])

                # Both methods should reduce the spike magnitude
                assert (
                    standard_spike < original_spike
                ), f"Standard method didn't reduce spike at ({time_idx}, {dist_idx})"
                assert (
                    approximate_spike < original_spike
                ), f"Approximate method didn't reduce spike at ({time_idx}, {dist_idx})"

                # Both should achieve similar spike reduction (within 50% of each other)
                reduction_ratio = min(standard_spike, approximate_spike) / max(
                    standard_spike, approximate_spike
                )
                msg = (
                    f"Methods differ too much in spike reduction at "
                    f"({time_idx}, {dist_idx})"
                )
                assert reduction_ratio > 0.5, msg

    def test_zero_mad_handling(self, patch_uniform_data):
        """Test handling of zero MAD values."""
        # Uniform data should have MAD = 0, test that eps is used
        result = patch_uniform_data.hampel_filter(time=0.6, threshold=3.5)

        assert result.data.shape == patch_uniform_data.data.shape
        # With uniform data, interior should be same as input
        # (edges may differ due to no padding)
        _assert_interior_equal(result.data, patch_uniform_data.data)

    def test_non_approximate_conditions(self, patch_with_spikes):
        """Test conditions where approximate mode is not used."""
        # Test case 1: approximate=False
        result1 = patch_with_spikes.hampel_filter(
            time=0.6, distance=3.0, threshold=3.5, approximate=False
        )
        assert result1.data.shape == patch_with_spikes.data.shape

        # Test case 2: len(size) <= 1 (single dimension)
        result2 = patch_with_spikes.hampel_filter(
            time=0.6, threshold=3.5, approximate=True
        )
        assert result2.data.shape == patch_with_spikes.data.shape

        # Test case 3: not all(s > 1 for s in size) (some dimensions have size 1)
        # Use a larger time window but force distance to minimum to test approximate
        result3 = patch_with_spikes.hampel_filter(
            time=5, distance=3, threshold=3.5, approximate=True, samples=True
        )
        assert result3.data.shape == patch_with_spikes.data.shape

    def test_thresholding_logic(self, patch_with_spikes):
        """Test the core thresholding and replacement logic."""
        # Test with a very low threshold to ensure replacement happens
        result = patch_with_spikes.hampel_filter(time=0.6, threshold=0.1)

        # Should have replaced many values
        differences = np.sum(result.data != patch_with_spikes.data)
        assert differences > 0

    def test_single_dimension_filter(self, patch_with_spikes):
        """Test filtering along a single dimension."""
        # Test time dimension only
        result_time = patch_with_spikes.hampel_filter(time=0.6, threshold=3.5)

        # Test distance dimension only
        result_distance = patch_with_spikes.hampel_filter(distance=3.0, threshold=3.5)

        # Check that data shapes are preserved
        assert result_time.data.shape == patch_with_spikes.data.shape
        assert result_distance.data.shape == patch_with_spikes.data.shape

    def test_different_threshold_values(self, patch_with_spikes):
        """Test various threshold values for edge cases."""
        # Test very low threshold
        result_low = patch_with_spikes.hampel_filter(time=0.6, threshold=0.5)

        # Test high threshold
        result_high = patch_with_spikes.hampel_filter(time=0.6, threshold=100.0)

        # Test default threshold
        result_default = patch_with_spikes.hampel_filter(time=0.6, threshold=3.5)

        # Check that data shapes are preserved
        assert result_low.data.shape == patch_with_spikes.data.shape
        assert result_high.data.shape == patch_with_spikes.data.shape
        assert result_default.data.shape == patch_with_spikes.data.shape

    def test_bad_threshold_raises(self, patch_with_spikes):
        """Ensure a bad threshold raises ParameterError."""
        with pytest.raises(ParameterError, match="greater than zero"):
            patch_with_spikes.hampel_filter(time=0.6, threshold=0)
        with pytest.raises(ParameterError, match="must be finite"):
            patch_with_spikes.hampel_filter(time=0.6, threshold=np.inf)
        with pytest.raises(ParameterError, match="must be finite"):
            patch_with_spikes.hampel_filter(time=0.6, threshold=np.nan)

    def test_entire_distance_channel_spike_removal(self, patch_with_spikes):
        """
        Test that spikes encompassing an entire distance channel are removed with
        multi-dimensional filtering.
        """
        # Add spikes across an entire distance channel using class variable
        spike_data = patch_with_spikes.data.copy()
        spike_data[:, self.DISTANCE_CHANNEL_SPIKE_IDX] = (
            self.DISTANCE_CHANNEL_SPIKE_VALUE
        )

        # Create patch with channel-wide spike
        spike_patch = patch_with_spikes.update(data=spike_data)

        # Single-dimension filtering along time should NOT remove the channel spike
        # (since all time samples in that channel have the same spike value)
        # Use a minimal window (3 samples) to ensure the spike remains
        result_time_only = spike_patch.hampel_filter(time=0.6, threshold=3.5)
        time_only_max = np.max(
            np.abs(result_time_only.data[:, self.DISTANCE_CHANNEL_SPIKE_IDX])
        )
        original_max = np.max(
            np.abs(spike_patch.data[:, self.DISTANCE_CHANNEL_SPIKE_IDX])
        )

        # With the new time step, even minimal filtering can affect channel spikes
        # The key test is that multi-dimensional filtering is more effective
        # We'll check this in the comparison below

        # Multi-dimensional filtering should detect and remove the channel spike
        result_multi = spike_patch.hampel_filter(time=0.6, distance=3.0, threshold=3.5)
        multi_filtered_max = np.max(
            np.abs(result_multi.data[:, self.DISTANCE_CHANNEL_SPIKE_IDX])
        )

        # Multi-dimensional filter should significantly reduce the spike
        assert multi_filtered_max < original_max / 2

        # Multi-dimensional filtering should be more effective than single-dimensional
        assert multi_filtered_max < time_only_max

        # Other channels should be less affected (use a different channel index)
        # Use channel 10 indices away
        other_channel_idx = self.DISTANCE_CHANNEL_SPIKE_IDX - 10
        other_channel_diff = np.mean(
            np.abs(
                result_multi.data[:, other_channel_idx]
                - spike_patch.data[:, other_channel_idx]
            )
        )
        spike_channel_diff = np.mean(
            np.abs(
                result_multi.data[:, self.DISTANCE_CHANNEL_SPIKE_IDX]
                - spike_patch.data[:, self.DISTANCE_CHANNEL_SPIKE_IDX]
            )
        )
        assert spike_channel_diff > other_channel_diff

    def test_entire_time_channel_spike_removal(self, patch_with_spikes):
        """Test that spikes encompassing an entire time channel are removed with
        multi-dimensional filtering.
        """
        # Add spikes across an entire time channel using class variable
        spike_data = patch_with_spikes.data.copy()
        spike_data[self.TIME_CHANNEL_SPIKE_IDX, :] = self.TIME_CHANNEL_SPIKE_VALUE

        # Create patch with channel-wide spike
        spike_patch = patch_with_spikes.update(data=spike_data)

        # Single-dimension filtering along distance should NOT remove the time
        # channel spike (since all distance samples at that time have the same
        # spike value)
        result_distance_only = spike_patch.hampel_filter(distance=3.0, threshold=3.5)
        distance_only_max = np.max(
            np.abs(result_distance_only.data[self.TIME_CHANNEL_SPIKE_IDX, :])
        )
        original_max = np.max(np.abs(spike_patch.data[self.TIME_CHANNEL_SPIKE_IDX, :]))

        # With the new time step, even single-dimensional filtering can be effective
        # The key test is that multi-dimensional filtering is more effective

        # Multi-dimensional filtering should detect and remove the time channel spike
        result_multi = spike_patch.hampel_filter(time=0.6, distance=3.0, threshold=3.5)
        multi_filtered_max = np.max(
            np.abs(result_multi.data[self.TIME_CHANNEL_SPIKE_IDX, :])
        )

        # Multi-dimensional filter should significantly reduce the spike
        assert multi_filtered_max < original_max / 2

        # Multi-dimensional filtering should be more effective than single-dimensional
        assert multi_filtered_max < distance_only_max

        # Other time channels should be less affected (use a different time index)
        other_time_idx = self.TIME_CHANNEL_SPIKE_IDX - 10  # Use time 10 indices away
        other_channel_diff = np.mean(
            np.abs(
                result_multi.data[other_time_idx, :]
                - spike_patch.data[other_time_idx, :]
            )
        )
        spike_channel_diff = np.mean(
            np.abs(
                result_multi.data[self.TIME_CHANNEL_SPIKE_IDX, :]
                - spike_patch.data[self.TIME_CHANNEL_SPIKE_IDX, :]
            )
        )
        assert spike_channel_diff > other_channel_diff

    def test_multiple_channel_spikes_removal(self, patch_with_spikes):
        """Test removal of spikes in multiple channels simultaneously."""
        # Add spikes to multiple channels using class variables
        spike_data = patch_with_spikes.data.copy()
        multi_distance_idx = self.DISTANCE_CHANNEL_SPIKE_IDX - 5  # Distance channel 15
        multi_time_idx = self.TIME_CHANNEL_SPIKE_IDX - 5  # Time channel 25
        individual_spike_time, individual_spike_dist, _ = self.SPIKE_LOCATIONS[2]

        spike_data[:, multi_distance_idx] = 20.0  # Spike in distance channel
        spike_data[multi_time_idx, :] = -18.0  # Spike in time channel
        if (
            individual_spike_time < spike_data.shape[0]
            and individual_spike_dist < spike_data.shape[1]
        ):
            # Individual point spike
            spike_data[individual_spike_time, individual_spike_dist] = 25.0

        # Create patch with multiple channel spikes
        spike_patch = patch_with_spikes.update(data=spike_data)

        # Filter along both dimensions
        result = spike_patch.hampel_filter(time=1.0, distance=3.0, threshold=3.5)

        # Check all spikes are reduced
        # Distance channel spike
        assert (
            np.max(np.abs(result.data[:, multi_distance_idx]))
            < np.max(np.abs(spike_patch.data[:, multi_distance_idx])) / 2
        )

        # Time channel spike
        assert (
            np.max(np.abs(result.data[multi_time_idx, :]))
            < np.max(np.abs(spike_patch.data[multi_time_idx, :])) / 2
        )

        # Individual point spike
        if (
            individual_spike_time < result.data.shape[0]
            and individual_spike_dist < result.data.shape[1]
        ):
            original_spike = abs(
                spike_patch.data[individual_spike_time, individual_spike_dist]
            )
            filtered_spike = abs(
                result.data[individual_spike_time, individual_spike_dist]
            )
            assert filtered_spike < original_spike / 2

    def test_int_patch(self, random_patch):
        """Ensure a patch with int type works."""
        data = np.round(random_patch.data * 100).astype(np.int64)
        patch = random_patch.update(data=data)
        out = patch.hampel_filter(time=5, samples=True, threshold=5)
        assert np.issubdtype(out.dtype, np.integer)

    def test_unchanged_patch_data(self, patch_with_spikes):
        """Ensure original patch data isn't modified with approximate."""
        # Since we are doing some fancy inplace tricks just make sure original
        # data array is left unchanged.
        original = patch_with_spikes.data.copy()
        out = patch_with_spikes.hampel_filter(
            time=5,
            distance=5,
            samples=True,
            threshold=5,
            approximate=True,
        )
        now = patch_with_spikes.data
        assert np.all(original == now)
        # Just in case, make sure the filter actually did something, otherwise
        # the check above is pointless.
        assert not np.all(original == out.data)
