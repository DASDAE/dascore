"""Tests for moving window operations."""

from __future__ import annotations

import numpy as np
import pytest

from dascore.exceptions import ParameterError
from dascore.utils.moving import (
    OPERATION_REGISTRY,
    _get_available_engines,
    move_max,
    move_mean,
    move_median,
    move_min,
    move_sum,
    moving_window,
)

# Test configuration
TEST_OPERATIONS = list(OPERATION_REGISTRY.keys())
TEST_ENGINES = ["scipy", "bottleneck", "auto"]
TEST_CONVENIENCE_FUNCS = {
    "median": move_median,
    "mean": move_mean,
    "sum": move_sum,
    "min": move_min,
    "max": move_max,
}


class TestMovingWindow:
    """Test moving window operations with different engines."""

    # Helper methods
    def _validate_basic_result(self, result, original_data):
        """Validate basic properties of results."""
        assert isinstance(result, np.ndarray)
        assert result.shape == original_data.shape
        assert result.dtype.kind in ["f", "i", "c"]  # float, int, or complex

    def _test_finite_properties(self, results, data):
        """Test mathematical properties of operations."""
        # Find where all results are finite
        all_finite = np.ones(len(data), dtype=bool)
        for result in results.values():
            all_finite &= np.isfinite(result)

        if not np.any(all_finite):
            return  # Skip if no finite values

        finite_indices = np.where(all_finite)[0]

        # Test properties only where values are finite
        mean_vals = results["mean"][finite_indices]
        sum_vals = results["sum"][finite_indices]
        min_vals = results["min"][finite_indices]
        max_vals = results["max"][finite_indices]

        # Sum should be >= mean (for positive window size)
        assert np.all(sum_vals >= mean_vals)
        # Min should be <= max
        assert np.all(min_vals <= max_vals)

    def _compare_engine_results(self, result1, result2, window):
        """Compare results from different engines."""
        # Compare interior values (avoiding edge effects)
        interior_slice = slice(window // 2, -window // 2 if window > 2 else None)

        interior1 = result1[interior_slice]
        interior2 = result2[interior_slice]

        # Both should be finite in interior
        finite1 = np.isfinite(interior1)
        finite2 = np.isfinite(interior2)

        if np.any(finite1 & finite2):
            common_finite = finite1 & finite2
            vals1 = interior1[common_finite]
            vals2 = interior2[common_finite]

            # Check that both give reasonable results (within data range)
            data_min, data_max = -10, 10  # Reasonable range
            assert np.all((vals1 >= data_min) & (vals1 <= data_max))
            assert np.all((vals2 >= data_min) & (vals2 <= data_max))

            # For most operations, results should be similar
            # (allowing for different edge handling)
            if len(vals1) > 0:
                mean1, mean2 = np.mean(vals1), np.mean(vals2)
                if abs(mean1) > 1e-10 and abs(mean2) > 1e-10:
                    rel_diff = abs(mean1 - mean2) / max(abs(mean1), abs(mean2))
                    # Allow 50% difference for edge handling differences
                    assert rel_diff < 0.5

    # Fixtures

    @pytest.fixture(scope="class")
    def test_data(self):
        """Test data fixtures."""
        return {
            "1d": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            "2d": np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]),
            "large": np.random.default_rng(42).random(1000),
            "single": np.array([5.0]),
            "constant": np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
        }

    @pytest.fixture(scope="class")
    def available_engines(self):
        """Get available engines."""
        return _get_available_engines()

    # Core functionality tests
    @pytest.mark.parametrize("operation", TEST_OPERATIONS)
    @pytest.mark.parametrize("engine", TEST_ENGINES)
    def test_operation_engine_combinations(
        self, test_data, operation, engine, available_engines
    ):
        """Test all operation/engine combinations."""
        if engine == "bottleneck" and "bottleneck" not in available_engines:
            pytest.skip("Bottleneck not available")

        data = test_data["1d"]
        window = 3

        try:
            result = moving_window(data, window, operation, engine=engine)
            self._validate_basic_result(result, data)
        except ParameterError as e:
            if "not available" in str(e):
                pytest.skip(f"Operation {operation} not available in {engine}")
            else:
                raise

    def test_convenience_functions(self, test_data):
        """Test that convenience functions work correctly."""
        data = test_data["1d"]
        window = 3

        for operation, func in TEST_CONVENIENCE_FUNCS.items():
            result = func(data, window)
            self._validate_basic_result(result, data)

            # Check equivalence with generic function
            expected = moving_window(data, window, operation)
            np.testing.assert_array_equal(result, expected)

    def test_multi_axis_operations(self, test_data):
        """Test operations on different axes."""
        data = test_data["2d"]
        window = 2

        for axis in [0, 1]:
            result = move_median(data, window, axis=axis)
            assert result.shape == data.shape

    # Input validation tests
    @pytest.mark.parametrize("invalid_window", [0, -1])
    def test_invalid_window_size(self, test_data, invalid_window):
        """Test invalid window sizes."""
        with pytest.raises(ParameterError, match="Window size must be positive"):
            moving_window(test_data["1d"], invalid_window, "median")

    def test_invalid_operation(self, test_data):
        """Test invalid operation name."""
        with pytest.raises(ParameterError, match="Unknown operation"):
            moving_window(test_data["1d"], 3, "invalid_operation")

    def test_unavailable_engine_warns(self, test_data):
        """Test unavailable engine warns."""
        with pytest.warns(UserWarning, match="not available"):
            moving_window(test_data["1d"], 3, "median", engine="invalid_engine")

    def test_large_window_warning(self, test_data):
        """Test warning for window larger than data."""
        data = test_data["1d"]
        large_window = len(data) + 1
        with pytest.raises(ParameterError, match="larger than data size"):
            result = moving_window(data, large_window, "median")
            assert isinstance(result, np.ndarray)

    # Edge cases
    def test_edge_cases(self, test_data):
        """Test various edge cases."""
        edge_cases = [
            ("single", 1, "Single element array"),
            ("constant", 3, "Constant values"),
        ]

        for data_key, window, _ in edge_cases:
            data = test_data[data_key]

            result = move_median(data, window)
            self._validate_basic_result(result, data)

            # For window size 1, result should equal input
            if window == 1:
                np.testing.assert_array_equal(result, data)

    def test_different_dtypes(self, test_data):
        """Test with different data types."""
        base_data = test_data["1d"][:5]  # Smaller for faster testing

        dtypes = [np.int32, np.float32, np.float64]
        for dtype in dtypes:
            data = base_data.astype(dtype)
            result = move_mean(data, 3)
            assert isinstance(result, np.ndarray)

    # Numerical properties tests
    def test_operation_properties(self, test_data):
        """Test mathematical properties of operations."""
        data = test_data["1d"]
        window = 3

        results = {}
        for operation in ["median", "mean", "sum", "min", "max"]:
            results[operation] = moving_window(data, window, operation)

        # Test properties where both results are finite
        self._test_finite_properties(results, data)

    def test_numerical_accuracy(self):
        """Test numerical accuracy with known results."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window = 3

        # Test median accuracy
        result_median = move_median(data, window, engine="scipy")
        # Interior elements should be exact
        assert result_median[2] == 3.0  # median of [2,3,4]

        # Test mean accuracy
        result_mean = move_mean(data, window, engine="scipy")
        expected_mean = np.array([2.0, 3.0, 4.0])  # means of [1,2,3], [2,3,4], [3,4,5]
        # Check interior values
        np.testing.assert_allclose(result_mean[1:4], expected_mean, rtol=1e-12)

    def test_engine_consistency(self, test_data, available_engines):
        """Test consistency between engines where applicable."""
        if "bottleneck" not in available_engines:
            pytest.skip("Bottleneck not available for comparison")

        data = test_data["large"][:100]  # Manageable size
        window = 10

        operations_to_test = ["median", "mean", "sum", "min", "max"]

        for operation in operations_to_test:
            try:
                result_scipy = moving_window(data, window, operation, engine="scipy")
                result_bn = moving_window(data, window, operation, engine="bottleneck")

                # Compare interior values (avoiding edge effects)
                self._compare_engine_results(result_scipy, result_bn, window)

            except ParameterError:
                # Skip if operation not available in one engine
                continue

    def test_bottleneck_std(self, test_data):
        """
        Ensure the bottleneck std works (current scipy engine doesnt have this).
        """
        pytest.importorskip("bottleneck")
        out = moving_window(test_data["1d"], 10, "std", axis=0)
        assert out.shape == test_data["1d"].shape
