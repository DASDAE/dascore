"""Tests for mute processing function."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError


def _get_testable_coord_values(coord, relative=False):
    """Get some values in the coordinate for testing different ranges."""
    start_ind, stop_ind = len(coord) // 4, 3 * len(coord) // 4
    start, stop = coord[start_ind], coord[stop_ind]
    if relative:
        start, stop = start - coord.min(), stop - coord.min()
    return (dc.to_float(start), dc.to_float(stop))


def _assert_coord_ranges(
    patch,
    dim,
    zero_ranges,
    one_ranges,
    relative=True,
):
    """Assert that the expected values occur in the patch."""
    for zrange in zero_ranges:
        sub = patch.select(**{dim: zrange}, relative=relative)
        assert np.allclose(sub.data, 0)
    for orange in one_ranges:
        sub = patch.select(**{dim: orange}, relative=relative)
        assert np.allclose(sub.data, 1)


def _assert_point_values(
    patch,
    dims=None,
    points=(),
    expected_values=(),
    relative=True,
):
    """Assert points in the patch are 0 or 1."""
    assert len(points) == len(expected_values)
    dims = dims if dims is not None else patch.dims
    axes = tuple(patch.get_axis(x) for x in dims)

    for vals, expected in zip(points, expected_values, strict=True):
        # Get the index (in the data array) of the expected 0.
        inds = [slice(None)] * patch.ndim
        for val, dim, ax in zip(vals, dims, axes, strict=True):
            ax = patch.get_axis(dim)
            coord = patch.get_coord(dim)
            inds[ax] = coord.get_next_index(val, relative=relative)

        patch_value = patch.data[tuple(inds)]
        assert np.isclose(patch_value, expected)


@pytest.fixture(scope="session")
def patch_ones(random_patch):
    """Return a patch filled with ones."""
    return random_patch.new(data=np.ones_like(random_patch.data))


@pytest.fixture(scope="module")
def patch_ones_3d():
    """Return a 3D patch filled with ones for testing."""
    data = np.ones((20, 30, 10))
    coords = {
        "time": np.arange(20) * 0.1,
        "distance": np.arange(30) * 10.0,
        "depth": np.arange(10) * 5.0,
    }
    dims = ("time", "distance", "depth")
    return dc.Patch(data=data, coords=coords, dims=dims)


@pytest.fixture(scope="module")
def patch_ones_4d():
    """Return a 4D patch filled with ones for testing."""
    data = np.ones((15, 20, 8, 6))
    coords = {
        "time": np.arange(15) * 0.1,
        "distance": np.arange(20) * 10.0,
        "depth": np.arange(8) * 5.0,
        "angle": np.arange(6) * 15.0,
    }
    dims = ("time", "distance", "depth", "angle")
    return dc.Patch(data=data, coords=coords, dims=dims)


class TestMuteBasics:
    """Basic tests for mute functionality."""

    def test_mute_no_kwargs_raises(self, random_patch):
        """Mute without dimension specifications should raise."""
        with pytest.raises(ParameterError, match="one or two keyword"):
            random_patch.mute()

    def test_not_tuple_raises(self, random_patch):
        """Boundary must be tuple of length 2."""
        with pytest.raises(ParameterError, match="two boundaries when using"):
            random_patch.mute(time=5)

    def test_tuple_wrong_length_raises(self, random_patch):
        """Tuple must have exactly 2 elements."""
        with pytest.raises(ParameterError, match="two boundaries when using"):
            random_patch.mute(time=(1, 2, 3))

    def test_smooth_bad_float(self, random_patch):
        """If a floating point value is used, it must be between 0 and 1."""
        with pytest.raises(ParameterError, match="smooth parameter for"):
            random_patch.mute(time=(1, 2), smooth=1.1)
        with pytest.raises(ParameterError, match="smooth parameter for"):
            random_patch.mute(time=(1, 2), smooth=-0.01)


class Test1DMute:
    """Test 1D block mutes (single dimension)."""

    def test_1d_mute_no_taper(self, patch_ones):
        """Mute first portion of time dimension."""
        # Use relative=True (default) with numeric offset
        coord = patch_ones.get_coord("time")
        v1, v2 = _get_testable_coord_values(coord, relative=True)
        muted = patch_ones.mute(time=(v1, v2), relative=True)
        step = dc.to_float(coord.step)
        _assert_coord_ranges(
            patch=muted,
            dim="time",
            zero_ranges=[(v1, v2)],
            one_ranges=[(..., v1 - step), (v2 + step, ...)],
            relative=True,
        )

    def test_mute_open_interval(self, patch_ones):
        """Mute using None for interval ends."""
        coord = patch_ones.get_coord("distance")
        v1, v2 = _get_testable_coord_values(coord, relative=True)
        muted1 = patch_ones.mute(distance=(v2, ...), relative=True)
        step = dc.to_float(coord.step)
        _assert_coord_ranges(
            patch=muted1,
            dim="distance",
            zero_ranges=[(v2, ...)],
            one_ranges=[(..., v2 - step)],
            relative=True,
        )

    def test_mute_absolute(self, patch_ones):
        """Test absolute coordinates work."""
        coord = patch_ones.get_coord("distance")
        v1, v2 = _get_testable_coord_values(coord, relative=False)
        muted1 = patch_ones.mute(distance=(v1, v2), relative=False)
        step = dc.to_float(coord.step)
        _assert_coord_ranges(
            patch=muted1,
            dim="distance",
            zero_ranges=[(v1, v2)],
            one_ranges=[(..., v1 - step), (v2 + step, ...)],
            relative=False,
        )

    # Test smoothing parameters for 1D case.
    def test_single_float_smooth(self, patch_ones):
        """Test that taper mute works with a single floating point value."""
        coord = patch_ones.get_coord("time")
        v1, v2 = _get_testable_coord_values(coord, relative=True)
        muted1 = patch_ones.mute(time=(v1, v2), relative=True, smooth=0.01)
        # With taper, we can't assert exact zeros/ones, just check shape
        assert muted1.shape == patch_ones.shape
        # Check that middle region has been modified (not all ones)
        sub = muted1.select(time=(v1, v2), relative=True)
        assert not np.allclose(sub.data, 1)

    def test_single_int_smooth(self, patch_ones):
        """Test that smooth can be a single integer which means samples."""
        coord = patch_ones.get_coord("time")
        v1, v2 = _get_testable_coord_values(coord, relative=True)
        muted1 = patch_ones.mute(time=(v1, v2), relative=True, smooth=5)
        # With taper, we can't assert exact zeros/ones, just check shape
        assert muted1.shape == patch_ones.shape
        # Check that middle region has been modified (not all ones)
        sub = muted1.select(time=(v1, v2), relative=True)
        assert not np.allclose(sub.data, 1)

    def test_single_quantity(self, patch_ones):
        """A single quantity should work with a specified dimension."""
        coord = patch_ones.get_coord("time")
        v1, v2 = _get_testable_coord_values(coord, relative=True)
        smooth_val = 0.01 * coord.units
        muted1 = patch_ones.mute(time=(v1, v2), relative=True, smooth=smooth_val)
        # With taper, we can't assert exact zeros/ones, just check shape
        assert muted1.shape == patch_ones.shape
        # Check that middle region has been modified (not all ones)
        sub = muted1.select(time=(v1, v2), relative=True)
        assert not np.allclose(sub.data, 1)

    def test_mute_strips(self, patch_ones):
        """Mute strips along each dimension."""
        # Mute a time strip
        muted_time = patch_ones.mute(time=(0.5, 1.0))
        # All distances should be affected equally
        assert np.allclose(muted_time.data[:, 10], muted_time.data[0, 10])

        # Mute a distance strip
        muted_dist = patch_ones.mute(distance=(10, 20))
        # All times should be affected equally
        assert np.allclose(muted_dist.data[15, :], muted_dist.data[15, 0])


class TestMuteLines:
    """Tests for muting between lines."""

    _dims = ("distance", "time")

    def test_point_raise(self, patch_ones):
        """A degenerate line (point) should raise."""
        msg = "is degenerate"
        with pytest.raises(ParameterError, match=msg):
            patch_ones.mute(
                time=([0, 0], [0, 0.25]),
                distance=([0, 0], [0, 300]),
            )

    def test_not_same_direction(self, patch_ones):
        """Create lines which do not point in the same directions."""
        match = "point in the same direction"

        with pytest.raises(ParameterError, match=match):
            patch_ones.mute(
                time=[[0, 0], [1, -1]],
                distance=[[1, -1], [-1, 1]],
            )

    def test_mute_lines(self, patch_ones):
        """Test for muting non-parallel lines."""
        time = ([0, 2], [0, 4])
        distance = ([0, 100], [0, 100])
        muted = patch_ones.mute(time=time, distance=distance)
        inverted = patch_ones.mute(time=time, distance=distance, invert=True)

        points = [(145, 4), (182, 6), (100, 3), (180, 3), (60, 6), (50, 1)]
        expected = np.array([0, 0, 0, 1, 1, 1])
        _assert_point_values(muted, self._dims, points=points, expected_values=expected)
        _assert_point_values(
            inverted, self._dims, points=points, expected_values=-expected + 1
        )

    def test_mute_lines_parallel(self, patch_ones):
        """Test for muting parallel lines."""
        time = ([0, 7.0], [1, 8.0])
        distance = ([0, 300], [1, 301])
        muted = patch_ones.mute(
            time=time,
            distance=distance,
        )
        inverted = patch_ones.mute(time=time, distance=distance, invert=True)

        points = [(110, 3), (271, 7), (23, 1), (50, 6), (250, 1), (170, 3)]
        expected = np.array([0, 0, 0, 1, 1, 1])
        _assert_point_values(muted, self._dims, points=points, expected_values=expected)
        _assert_point_values(
            inverted, self._dims, points=points, expected_values=-expected + 1
        )

    def test_implicit_with_positive_line(self, patch_ones):
        """
        Ensure None can be used to specify vertical/horizontal lines with
        another line with positive direction.
        """
        time = (0, [0, 8.0])
        distance = (None, [0, 301])
        muted = patch_ones.mute(
            time=time,
            distance=distance,
        )
        inverted = patch_ones.mute(time=time, distance=distance, invert=True)

        points = [(50, 6), (250, 2)]
        expected = np.array([1, 0])
        _assert_point_values(muted, self._dims, points=points, expected_values=expected)
        _assert_point_values(
            inverted, self._dims, points=points, expected_values=-expected + 1
        )

    def test_implicit_with_negative_line(self, patch_ones):
        """
        Ensure None can be used to specify vertical/horizontal lines with
        another line pointing in negative direction.
        """
        time = (4, [4.0, 0])
        distance = (None, [150, 0])
        muted = patch_ones.mute(
            time=time,
            distance=distance,
        )
        inverted = patch_ones.mute(time=time, distance=distance, invert=True)

        points = [(50, 3), (10, 3), (50, 2), (100, 2)]
        expected = np.array([0, 0, 0, 1])
        _assert_point_values(muted, self._dims, points=points, expected_values=expected)
        _assert_point_values(
            inverted, self._dims, points=points, expected_values=-expected + 1
        )

    def test_two_implicit_parallel_lines(self, patch_ones):
        """Ensure two implicit values can define parallel lines."""
        time = (0, 2)
        distance = (None, None)
        muted = patch_ones.mute(
            time=time,
            distance=distance,
        )
        sub = muted.select(time=(2.1, ...), relative=True)
        assert np.allclose(sub.data, 1)

        inverted = patch_ones.mute(time=time, distance=distance, invert=True)
        sub = inverted.select(time=(2.1, ...), relative=True)
        assert np.allclose(sub.data, 0)

    def test_two_implicit_orthogonal_lines(self, patch_ones):
        """Ensure two implicit values can define orthogonal lines."""
        time = (None, 2)
        distance = (100, None)
        muted = patch_ones.mute(
            time=time,
            distance=distance,
        )
        sub = muted.select(time=(2.1, ...), distance=(101, ...), relative=True)
        assert np.allclose(sub.data, 0)

        inverted = patch_ones.mute(time=time, distance=distance, invert=True)
        sub = inverted.select(time=(2.1, ...), distance=(101, ...), relative=True)
        assert np.allclose(sub.data, 1)

    def test_mute_lines_absolute(self, patch_ones_3d):
        """Test 2D line mute with absolute coordinates (relative=False)."""
        # Use 3D patch which has simple float coordinates (time: 0-1.9, distance: 0-290)
        # Get the actual coordinate values from the patch
        time_coord = patch_ones_3d.get_coord("time")
        dist_coord = patch_ones_3d.get_coord("distance")

        time_min = dc.to_float(time_coord.min())
        dist_min = dc.to_float(dist_coord.min())

        # Define two non-parallel lines using absolute coordinates
        # Line 1: from (t=0, d=0) to (t=1.0, d=100)
        # Line 2: from (t=0, d=0) to (t=1.5, d=100)
        time = ([time_min, time_min + 1.0], [time_min, time_min + 1.5])
        distance = ([dist_min, dist_min + 100], [dist_min, dist_min + 100])

        muted = patch_ones_3d.mute(time=time, distance=distance, relative=False)
        inverted = patch_ones_3d.mute(
            time=time, distance=distance, relative=False, invert=True
        )

        # Verify muting worked
        assert np.any(muted.data == 0)
        assert np.any(muted.data == 1)
        assert np.any(inverted.data == 0)
        assert np.any(inverted.data == 1)

        # Test some specific points using absolute coordinates
        # Points that should be inside the muted region (between the two lines)
        # Format: (distance, time)
        # For 3D patch, selecting 2D point gives array along 3rd dimension
        point_inside = (dist_min + 100, time_min + 1.2)
        point_outside = (dist_min + 150, time_min + 0.5)

        # Get indices for inside point
        dist_idx_in = dist_coord.get_next_index(point_inside[0], relative=False)
        time_idx_in = time_coord.get_next_index(point_inside[1], relative=False)
        # For 3D patch with dims (time, distance, depth), mute on time-distance
        # affects all depths
        assert np.allclose(muted.data[time_idx_in, dist_idx_in, :], 0)
        assert np.allclose(inverted.data[time_idx_in, dist_idx_in, :], 1)

        # Get indices for outside point
        dist_idx_out = dist_coord.get_next_index(point_outside[0], relative=False)
        time_idx_out = time_coord.get_next_index(point_outside[1], relative=False)
        assert np.allclose(muted.data[time_idx_out, dist_idx_out, :], 1)
        assert np.allclose(inverted.data[time_idx_out, dist_idx_out, :], 0)


class TestMuteSmoothing:
    """Tests for smoothing functionality in mute."""

    def test_dict_smooth_2d_lines(self, patch_ones):
        """Test smoothing with dict parameter for dimension-specific values."""
        time = ([0, 2], [0, 4])
        distance = ([0, 100], [0, 100])
        smooth = {"time": 0.02, "distance": 5}
        muted = patch_ones.mute(time=time, distance=distance, smooth=smooth)

        # With smoothing, muted region should have intermediate values
        assert muted.shape == patch_ones.shape
        # Check that we have values between 0 and 1 (not just sharp cutoff)
        assert np.any((muted.data > 0.01) & (muted.data < 0.99))

    def test_dict_smooth_with_none(self, patch_ones):
        """Test dict with None value for one dimension."""
        time = ([0, 2], [0, 4])
        distance = ([0, 100], [0, 100])
        smooth = {"time": 0.02, "distance": None}
        muted = patch_ones.mute(time=time, distance=distance, smooth=smooth)
        assert muted.shape == patch_ones.shape

    def test_dict_smooth_mismatched_keys_raises(self, patch_ones):
        """Test that dict with mismatched keys raises appropriate error."""
        time = ([0, 2], [0, 4])
        distance = ([0, 100], [0, 100])
        smooth = {"time": 0.02, "oriely": 0.2}  # Missing distance key
        msg = "a smooth dictionary"
        with pytest.raises(ParameterError, match=msg):
            patch_ones.mute(time=time, distance=distance, smooth=smooth)

    def test_smooth_creates_gradual_transition(self, patch_ones):
        """Verify smoothing creates gradual transitions with intermediate values."""
        coord = patch_ones.get_coord("time")
        v1, v2 = _get_testable_coord_values(coord, relative=True)
        muted = patch_ones.mute(time=(v1, v2), relative=True, smooth=0.05)

        # Check that the transition region has values between 0 and 1
        transition_region = muted.select(time=(v1 - 0.2, v2 + 0.2), relative=True).data
        # There should be gradual transition, not just 0s and 1s
        unique_vals = np.unique(transition_region)
        assert len(unique_vals) > 10  # Many intermediate values
        assert np.any((transition_region > 0.01) & (transition_region < 0.99))

    def test_smooth_with_invert(self, patch_ones):
        """Test that smoothing works with invert=True."""
        time = ([0, 2], [0, 4])
        distance = ([0, 100], [0, 100])
        smooth = 0.03
        muted = patch_ones.mute(
            time=time, distance=distance, smooth=smooth, invert=True
        )

        # With smoothing and invert, should have intermediate values
        assert muted.shape == patch_ones.shape
        assert np.any((muted.data > 0.01) & (muted.data < 0.99))

    def test_smooth_parallel_lines(self, patch_ones):
        """Test smoothing with parallel lines."""
        time = ([0, 7.0], [1, 8.0])
        distance = ([0, 300], [1, 301])
        smooth = {"time": 5, "distance": 10}
        muted = patch_ones.mute(time=time, distance=distance, smooth=smooth)

        assert muted.shape == patch_ones.shape
        # Check for smooth transition
        assert np.any((muted.data > 0.01) & (muted.data < 0.99))

    def test_smooth_very_small_value(self, patch_ones):
        """Test edge case with very small smooth value (1 sample)."""
        coord = patch_ones.get_coord("time")
        v1, v2 = _get_testable_coord_values(coord, relative=True)
        muted = patch_ones.mute(time=(v1, v2), relative=True, smooth=1)
        # Should still work, just with minimal smoothing
        assert muted.shape == patch_ones.shape

    def test_smooth_preserves_shape(self, patch_ones):
        """Test that smoothing preserves patch shape."""
        time = ([0, 2], [0, 4])
        distance = ([0, 100], [0, 100])
        smooth = 0.05
        muted = patch_ones.mute(time=time, distance=distance, smooth=smooth)
        assert muted.shape == patch_ones.shape


class TestMute3D:
    """Tests for muting on 3D patches."""

    def test_1d_mute_time(self, patch_ones_3d):
        """Test 1D mute along time dimension."""
        v1, v2 = 0.3, 0.8
        muted = patch_ones_3d.mute(time=(v1, v2), relative=True)

        # Check that the muted region is zeroed
        sub_muted = muted.select(time=(v1, v2), relative=True)
        assert np.allclose(sub_muted.data, 0)

        # Check that outside region is still ones
        sub_kept = muted.select(time=(v2 + 0.1, ...), relative=True)
        assert np.allclose(sub_kept.data, 1)

    def test_2d_line_mute(self, patch_ones_3d):
        """Test 2D line mute using time-distance plane."""
        time = ([0, 1.0], [0, 1.5])
        distance = ([0, 100], [0, 100])
        muted = patch_ones_3d.mute(time=time, distance=distance)

        # Should maintain 3D shape
        assert muted.shape == patch_ones_3d.shape
        assert muted.ndim == 3

        # Check that some region is muted
        assert np.any(muted.data == 0)
        # Check that some region is not muted
        assert np.any(muted.data == 1)

    def test_2d_line_mute_distance_depth(self, patch_ones_3d):
        """Test 2D line mute using distance-depth plane."""
        distance = ([0, 100], [0, 200])
        depth = ([0, 20], [0, 30])
        muted = patch_ones_3d.mute(distance=distance, depth=depth)

        assert muted.shape == patch_ones_3d.shape
        assert np.any(muted.data == 0)
        assert np.any(muted.data == 1)

    def test_smoothing_3d(self, patch_ones_3d):
        """Test that smooth parameter works correctly in 3D."""
        v1, v2 = 0.3, 0.8
        muted = patch_ones_3d.mute(time=(v1, v2), relative=True, smooth=0.05)
        # With smoothing, should have intermediate values
        assert np.any((muted.data > 0.01) & (muted.data < 0.99))

    def test_invert_3d(self, patch_ones_3d):
        """Test invert parameter in 3D."""
        v1, v2 = 0.3, 0.8
        muted = patch_ones_3d.mute(time=(v1, v2), relative=True, invert=True)

        # With invert, the selected region should be kept
        sub_kept = muted.select(time=(v1, v2), relative=True)
        assert np.allclose(sub_kept.data, 1)

        # Outside region should be zeroed
        sub_zeroed = muted.select(time=(v2 + 0.1, ...), relative=True)
        assert np.allclose(sub_zeroed.data, 0)


class TestMute4D:
    """Tests for muting on 4D patches."""

    def test_1d_mute_time(self, patch_ones_4d):
        """Test 1D mute along time dimension in 4D patch."""
        v1, v2 = 0.3, 0.8
        muted = patch_ones_4d.mute(time=(v1, v2), relative=True)

        # Check that the muted region is zeroed
        sub_muted = muted.select(time=(v1, v2), relative=True)
        assert np.allclose(sub_muted.data, 0)

        # Check that outside region is still ones
        sub_kept = muted.select(time=(v2 + 0.1, ...), relative=True)
        assert np.allclose(sub_kept.data, 1)

    def test_1d_mute_angle(self, patch_ones_4d):
        """Test 1D mute along angle dimension."""
        muted = patch_ones_4d.mute(angle=(20, 60), relative=True)
        sub_muted = muted.select(angle=(20, 60), relative=True)
        assert np.allclose(sub_muted.data, 0)

    def test_2d_line_mute_depth_angle(self, patch_ones_4d):
        """Test 2D line mute using depth-angle plane."""
        depth = ([0, 15], [0, 25])
        angle = ([0, 30], [0, 60])
        muted = patch_ones_4d.mute(depth=depth, angle=angle)

        assert muted.shape == patch_ones_4d.shape
        assert np.any(muted.data == 0)
        assert np.any(muted.data == 1)

    def test_smoothing_4d_with_dict(self, patch_ones_4d):
        """Test smooth parameter with dict in 4D."""
        time = ([0, 0.8], [0, 1.2])
        distance = ([0, 80], [0, 80])
        smooth = {"time": 0.02, "distance": 5}
        muted = patch_ones_4d.mute(time=time, distance=distance, smooth=smooth)
        # With smoothing, should have intermediate values
        assert muted.shape == patch_ones_4d.shape
        assert np.any((muted.data > 0.01) & (muted.data < 0.99))

    def test_smoothing_1d_mute_in_4d(self, patch_ones_4d):
        """Test smoothing with 1D mute in 4D patch."""
        v1, v2 = 0.3, 0.8
        muted = patch_ones_4d.mute(time=(v1, v2), relative=True, smooth=0.05)
        # Check for gradual transition
        assert np.any((muted.data > 0.01) & (muted.data < 0.99))

    def test_invert_4d(self, patch_ones_4d):
        """Test invert parameter in 4D."""
        v1, v2 = 0.3, 0.8
        muted = patch_ones_4d.mute(time=(v1, v2), relative=True, invert=True)

        # With invert, the selected region should be kept
        sub_kept = muted.select(time=(v1, v2), relative=True)
        assert np.allclose(sub_kept.data, 1)

        # Outside region should be zeroed
        sub_zeroed = muted.select(time=(v2 + 0.1, ...), relative=True)
        assert np.allclose(sub_zeroed.data, 0)
