"""
Tests for trimming data.
"""
import pytest
from dfs.core import trim_by_time, trim_by_distance

import numpy as np


class TestTrimByTime:
    """Tests for trimming in time."""

    @pytest.fixture()
    def half_trim(self, terra15_das_array):
        """Tests for trimming terra15 in half (by time)."""
        endtime = terra15_das_array["time"].mean().values
        return trim_by_time(terra15_das_array, end_time=endtime)

    def test_new_data(self, terra15_das_array, half_trim):
        """Ensure a new data array is created from trim operation."""
        assert terra15_das_array is not half_trim
        assert len(terra15_das_array["time"]) > len(half_trim["time"])

    def test_empty_trim(self, terra15_das_array):
        """Ensure trimming with no start/stop params doesnt change array."""
        out = trim_by_time(terra15_das_array)
        assert out.equals(terra15_das_array)

    def test_one_sample_start(self, terra15_das_array):
        """Ensure trimming is inclusive on first sample."""
        start_time = terra15_das_array["time"].min()
        trim1 = trim_by_time(terra15_das_array, start_time=start_time)
        assert len(trim1["time"]) == len(terra15_das_array["time"])
        # increment starttime by 1 ns and test that it gets trimed out
        new_start = start_time + np.timedelta64(1, "ns")
        trim2 = trim_by_time(terra15_das_array, start_time=new_start)
        # the trim should have removed exactly one sample
        assert (len(trim2["time"]) + 1) == len(terra15_das_array["time"])

    def test_one_sample_end(self, terra15_das_array):
        """Ensure trimming is inclusive on first sample."""
        end_time = terra15_das_array["time"].max()
        trim1 = trim_by_time(terra15_das_array, end_time=end_time)
        assert len(trim1["time"]) == len(terra15_das_array["time"])
        # decrease endtime by 1 ns and test that it gets trimed out
        new_end = end_time - np.timedelta64(1, "ns")
        trim2 = trim_by_time(terra15_das_array, end_time=new_end)
        # the trim should have removed exactly one sample
        assert (len(trim2["time"]) + 1) == len(terra15_das_array["time"])


class TestTrimDistance:
    """Tests for triming in space."""

    @pytest.fixture()
    def half_dist_trim(self, terra15_das_array):
        """Tests for trimming terra15 in half (by time)."""
        end_distance = terra15_das_array["distance"].mean().values
        return trim_by_distance(terra15_das_array, start_distance=end_distance)

    def test_new_data(self, terra15_das_array, half_dist_trim):
        """Ensure a new data array is created from trim operation."""
        assert terra15_das_array is not half_dist_trim
        assert len(terra15_das_array["distance"]) > len(half_dist_trim["distance"])

    def test_empty_trim(self, terra15_das_array):
        """Ensure trimming with no start/stop params doesnt change array."""
        out = trim_by_distance(terra15_das_array)
        assert out.equals(terra15_das_array)
