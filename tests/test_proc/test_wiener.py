"""
Tests for the Wiener filter implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError


class TestWienerFilter:
    """Tests for the wiener_filter function."""

    @pytest.fixture()
    def noisy_patch(self):
        """Create a patch with artificial noise for testing."""
        # Get example patch and add Gaussian noise
        patch = dc.get_example_patch()
        rnd = np.random.RandomState(18)
        noise = rnd.normal(0, 0.1, patch.data.shape)
        noisy_data = patch.data + noise
        return patch.update(data=noisy_data)

    @pytest.fixture()
    def uniform_patch(self):
        """Create a patch with uniform data for testing edge cases."""
        patch = dc.get_example_patch()
        uniform_data = np.ones_like(patch.data) * 5.0
        return patch.update(data=uniform_data)

    @pytest.fixture()
    def zero_patch(self):
        """Create a patch with zero data for testing edge cases."""
        patch = dc.get_example_patch()
        zero_data = np.zeros_like(patch.data)
        return patch.update(data=zero_data)

    def test_basic_wiener_filter(self, noisy_patch):
        """Test basic Wiener filtering functionality."""
        # Test with dimension-specific parameters
        result = noisy_patch.wiener_filter(time=5, samples=True)

        # Check that data shape is preserved
        assert result.data.shape == noisy_patch.data.shape

        # Check that filtering reduces variance (noise)
        original_var = np.var(noisy_patch.data)
        filtered_var = np.var(result.data)
        assert filtered_var < original_var

        # Check that result is a Patch
        assert isinstance(result, dc.Patch)

    def test_dimension_specific_filtering(self, noisy_patch):
        """Test Wiener filter with dimension-specific window sizes."""
        # Test filtering along time dimension
        result_time = noisy_patch.wiener_filter(time=5, samples=True)
        assert result_time.data.shape == noisy_patch.data.shape

        # Test filtering along distance dimension
        result_distance = noisy_patch.wiener_filter(distance=3, samples=True)
        assert result_distance.data.shape == noisy_patch.data.shape

        # Test filtering along both dimensions
        result_both = noisy_patch.wiener_filter(time=5, distance=3, samples=True)
        assert result_both.data.shape == noisy_patch.data.shape

    def test_coordinate_units_vs_samples(self, noisy_patch):
        """Test filtering with coordinate units vs samples."""
        # Get coordinate info for conversion
        time_coord = noisy_patch.get_coord("time")
        time_step = time_coord.step

        # Test with samples=True
        result_samples = noisy_patch.wiener_filter(time=5, samples=True)

        # Test with coordinate units (samples=False)
        time_value = time_step * 5
        result_coords = noisy_patch.wiener_filter(time=time_value, samples=False)

        # Results should be similar (may not be identical due to rounding)
        assert result_samples.data.shape == result_coords.data.shape

    def test_noise_parameter(self, noisy_patch):
        """Test different noise parameter values."""
        # Test with default noise estimation
        result_default = noisy_patch.wiener_filter(time=5, samples=True)

        # Test with explicit noise value
        result_noise = noisy_patch.wiener_filter(time=5, samples=True, noise=0.01)

        # Both should have same shape
        assert result_default.data.shape == result_noise.data.shape
        assert result_noise.data.shape == noisy_patch.data.shape

        # Results should be different
        assert not np.array_equal(result_default.data, result_noise.data)

    def test_parameter_validation(self, noisy_patch):
        """Test parameter validation."""
        # Test no parameters
        with pytest.raises(ParameterError, match="you must specify"):
            noisy_patch.wiener_filter()

    def test_uniform_data(self, uniform_patch):
        """Test Wiener filter on uniform data."""
        result = uniform_patch.wiener_filter(time=5, samples=True)

        # Check that shape is preserved
        assert result.data.shape == uniform_patch.data.shape

        # For uniform data, Wiener filter may produce NaN due to zero variance
        # This is expected behavior from scipy.signal.wiener
        assert isinstance(result, dc.Patch)

    def test_zero_data(self, zero_patch):
        """Test Wiener filter on zero data."""
        result = zero_patch.wiener_filter(time=5, samples=True)

        # Check that shape is preserved
        assert result.data.shape == zero_patch.data.shape

        # Zero data with Wiener filter may produce NaN due to zero variance
        # This is expected behavior from scipy.signal.wiener
        assert isinstance(result, dc.Patch)

    def test_coordinates_preserved(self, noisy_patch):
        """Test that coordinates are preserved after filtering."""
        result = noisy_patch.wiener_filter(time=5, samples=True)

        # Check that dimensions are preserved
        assert result.dims == noisy_patch.dims

        # Check that coordinate arrays are preserved
        for dim in noisy_patch.dims:
            original_coord = noisy_patch.get_array(dim)
            result_coord = result.get_array(dim)
            np.testing.assert_array_equal(original_coord, result_coord)

    def test_metadata_preserved(self, noisy_patch):
        """Test that metadata is preserved after filtering."""
        result = noisy_patch.wiener_filter(time=5, samples=True)

        # Check that core attributes are preserved (excluding history which
        # gets updated)
        original_attrs = dict(noisy_patch.attrs)
        result_attrs = dict(result.attrs)

        # Remove history from comparison as it gets updated by patch_function decorator
        original_attrs.pop("history", None)
        result_attrs.pop("history", None)

        assert result_attrs == original_attrs

    def test_different_window_sizes(self, noisy_patch):
        """Test various window sizes."""
        window_sizes = [3, 5, 7, 9]

        for window_size in window_sizes:
            result = noisy_patch.wiener_filter(time=window_size, samples=True)
            assert result.data.shape == noisy_patch.data.shape

            # Larger windows should generally provide more smoothing
            variance = np.var(result.data)
            assert variance <= np.var(noisy_patch.data)

    def test_multi_dimensional_windows(self, noisy_patch):
        """Test filtering with different window sizes per dimension."""
        result = noisy_patch.wiener_filter(time=7, distance=3, samples=True)

        assert result.data.shape == noisy_patch.data.shape

        # Should reduce noise
        original_var = np.var(noisy_patch.data)
        filtered_var = np.var(result.data)
        assert filtered_var < original_var

    def test_single_sample_dimension(self, noisy_patch):
        """Test behavior when one dimension has window size 1."""
        # Select a single distance to create a 1D-like patch
        reduced_patch = noisy_patch.select(distance=0, samples=True)

        # This should work (essentially no filtering along distance dimension)
        result = reduced_patch.wiener_filter(time=5, samples=True)
        assert result.data.shape == reduced_patch.data.shape

    def test_noise_reduction_effectiveness(self):
        """Test that the filter actually reduces noise."""
        # Create a simple test with a known signal
        # Use a simple constant signal with added noise
        rand = np.random.RandomState(123)
        simple_signal = np.ones((50, 50)) * 5.0
        noise = rand.normal(0, 0.5, simple_signal.shape)
        noisy_signal = simple_signal + noise

        # Create patch
        test_patch = dc.Patch(
            data=noisy_signal,
            coords={"time": np.arange(50), "distance": np.arange(50)},
            dims=("time", "distance"),
        )

        # Apply filter with larger window for better noise reduction
        filtered = test_patch.wiener_filter(time=7, distance=7, samples=True)

        # Check that variance is reduced (indicating noise reduction)
        original_variance = np.var(test_patch.data)
        filtered_variance = np.var(filtered.data)

        # Filtered version should have less variance (indicating
        # smoothing/noise reduction)
        assert filtered_variance <= original_variance

    def test_edge_case_small_patch(self):
        """Test with very small patch."""
        # Create a small patch
        rand = np.random.RandomState(242)
        small_data = rand.rand(3, 3)
        patch = dc.Patch(
            data=small_data,
            coords={"time": np.arange(3), "distance": np.arange(3)},
            dims=("time", "distance"),
        )

        # Should work with small window
        result = patch.wiener_filter(time=3, samples=True)
        assert result.data.shape == patch.data.shape
