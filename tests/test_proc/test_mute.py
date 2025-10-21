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
    return (start, stop)


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
        inds = (
            patch.get_coord(dims[num]).get_next_index(
                vals[axes.index(num)], relative=relative
            )
            if num in axes
            else slice(None)
            for num in range(patch.ndim)
        )
        assert np.isclose(patch.data[tuple(inds)], expected)


@pytest.fixture(scope="session")
def patch_ones(random_patch):
    """Return a patch filled with ones."""
    return random_patch.new(data=np.ones_like(random_patch.data))


@pytest.fixture(scope="session")
def ricker_patch():
    """Return ricker moveout patch for velocity mute testing."""
    return dc.get_example_patch("ricker_moveout")


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
        _assert_coord_ranges(
            patch=muted,
            dim="time",
            zero_ranges=[(v1, v2)],
            one_ranges=[(..., v1 - coord.step), (v2 + coord.step, ...)],
            relative=True,
        )

    def test_mute_open_interval(self, patch_ones):
        """Mute using None for interval ends."""
        coord = patch_ones.get_coord("distance")
        v1, v2 = _get_testable_coord_values(coord, relative=True)
        muted1 = patch_ones.mute(distance=(v2, ...), relative=True)
        _assert_coord_ranges(
            patch=muted1,
            dim="distance",
            zero_ranges=[(v2, ...)],
            one_ranges=[(..., v2 - coord.step)],
            relative=True,
        )

    def test_mute_absolute(self, patch_ones):
        """Test absolute coordinates work."""
        coord = patch_ones.get_coord("distance")
        v1, v2 = _get_testable_coord_values(coord, relative=False)
        muted1 = patch_ones.mute(distance=(v1, v2), relative=False)
        _assert_coord_ranges(
            patch=muted1,
            dim="distance",
            zero_ranges=[(v1, v2)],
            one_ranges=[(..., v1 - coord.step), (v2 + coord.step, ...)],
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

        with pytest.raises(ValueError, match=match):
            patch_ones.mute(
                time=[[0, 0], [1, -1]],
                distance=[[1, -1], [-1, 1]],
            )

    def test_mute_lines(self, patch_ones):
        """Mute non-parallel lines."""
        muted = patch_ones.mute(
            time=([0, 2], [0, 4]),
            distance=([0, 100], [0, 100]),
        )
        points = [(145, 4), (182, 6), (100, 3), (180, 3), (60, 6), (50, 1)]
        expected = [1, 1, 1, 0, 0, 0]
        _assert_point_values(
            muted, ("time", "distance"), points=points, expected_values=expected
        )


#
