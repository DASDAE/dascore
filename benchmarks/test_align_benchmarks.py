"""Benchmarks for patch alignment using pytest-codspeed."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc


@pytest.fixture(scope="module")
def patch_2d_with_1d_shift():
    """Create a 2D patch with 1D shift coordinate."""
    patch = dc.get_example_patch()
    time = patch.get_array("time")
    distance = patch.get_array("distance")
    shift_time_samples = np.arange(len(distance))
    shift_time_absolute = time[shift_time_samples]
    return patch.update_coords(
        shift_time_absolute=("distance", shift_time_absolute),
    )


@pytest.fixture(scope="module")
def patch_4d_with_3d_shift():
    """Create a 4D patch with 3D shift coordinate."""
    patch = dc.get_example_patch()
    time = patch.get_array("time")
    distance = patch.get_array("distance")
    # Create 4D patch
    data1 = np.stack([patch.data, patch.data], axis=-1)
    data2 = np.stack([data1, data1], axis=-1)
    # Create 3D shift coordinate
    shift_time_samples = np.arange(len(distance))
    shift_time_absolute = time[shift_time_samples]
    shift_time_2d = np.stack([shift_time_absolute, shift_time_absolute], axis=-1)
    shift_time_3d = np.stack([shift_time_2d, shift_time_2d], axis=-1)
    coords = {
        "time": time,
        "distance": distance,
        "dim3": [1, 2],
        "dim4": [2, 3],
        "shift_time_absolute_3d": (("distance", "dim3", "dim4"), shift_time_3d),
    }
    dims = (*patch.dims, "dim3", "dim4")
    return dc.Patch(data=data2, coords=coords, dims=dims)


class TestAlignBenchmarks:
    """Benchmarks for align_to_coord operation."""

    @pytest.mark.benchmark
    def test_align_1d_shift_full(self, patch_2d_with_1d_shift):
        """Benchmark 2D patch with 1D shift coordinate (300 shifts), full mode."""
        patch = patch_2d_with_1d_shift
        patch.align_to_coord(time="shift_time_absolute", mode="full")

    @pytest.mark.benchmark
    def test_align_1d_shift_same(self, patch_2d_with_1d_shift):
        """Benchmark 2D patch with 1D shift coordinate (300 shifts), same mode."""
        patch = patch_2d_with_1d_shift
        patch.align_to_coord(time="shift_time_absolute", mode="same")

    @pytest.mark.benchmark
    def test_align_1d_shift_valid(self, patch_2d_with_1d_shift):
        """Benchmark 2D patch with 1D shift coordinate (300 shifts), valid mode."""
        patch = patch_2d_with_1d_shift
        patch.align_to_coord(time="shift_time_absolute", mode="valid")

    @pytest.mark.benchmark
    def test_align_3d_shift_full(self, patch_4d_with_3d_shift):
        """Benchmark 4D patch with 3D shift coordinate (1200 shifts), full mode."""
        patch = patch_4d_with_3d_shift
        patch.align_to_coord(time="shift_time_absolute_3d", mode="full")

    @pytest.mark.benchmark
    def test_align_3d_shift_same(self, patch_4d_with_3d_shift):
        """Benchmark 4D patch with 3D shift coordinate (1200 shifts), same mode."""
        patch = patch_4d_with_3d_shift
        patch.align_to_coord(time="shift_time_absolute_3d", mode="same")

    @pytest.mark.benchmark
    def test_align_3d_shift_valid(self, patch_4d_with_3d_shift):
        """Benchmark 4D patch with 3D shift coordinate (1200 shifts), valid mode."""
        patch = patch_4d_with_3d_shift
        patch.align_to_coord(time="shift_time_absolute_3d", mode="valid")
