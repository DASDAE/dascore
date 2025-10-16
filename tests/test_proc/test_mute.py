"""Tests for mute processing function."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError


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

    def test_mute_exists(self, random_patch):
        """Ensure mute method exists on patch."""
        assert hasattr(random_patch, "mute")
        assert callable(random_patch.mute)

    def test_mute_no_kwargs_raises(self, random_patch):
        """Mute without dimension specifications should raise."""
        with pytest.raises(ParameterError, match="At least one dimension"):
            random_patch.mute()

    def test_invalid_dimension_raises(self, random_patch):
        """Mute with invalid dimension should raise."""
        with pytest.raises(ParameterError, match="not found"):
            random_patch.mute(invalid_dim=(0, 10))

    def test_not_tuple_raises(self, random_patch):
        """Boundary must be tuple of length 2."""
        with pytest.raises(ParameterError, match="must specify 2 boundaries"):
            random_patch.mute(time=5)

    def test_tuple_wrong_length_raises(self, random_patch):
        """Tuple must have exactly 2 elements."""
        with pytest.raises(ParameterError, match="must specify 2 boundaries"):
            random_patch.mute(time=(1, 2, 3))


class TestBlockMutes1D:
    """Test 1D block mutes (single dimension)."""

    def test_mute_start_of_time(self, patch_ones):
        """Mute first portion of time dimension."""
        # Use relative=True (default) with numeric offset
        muted = patch_ones.mute(time=(0, 0.5))

        # Check that first samples are zero
        assert np.allclose(muted.data[:, 0], 0)
        # Check that later samples are still 1
        assert np.allclose(muted.data[:, -1], 1)

    def test_mute_end_of_time(self, patch_ones):
        """Mute last portion of time dimension."""
        # Use relative with negative values (from end)
        # (-0.5, None) means from 0.5s before end to the end
        muted = patch_ones.mute(time=(-0.5, None))

        # Check that last samples are zero
        assert np.allclose(muted.data[:, -1], 0)
        # Check that early samples are still 1
        assert np.allclose(muted.data[:, 0], 1)

    def test_mute_middle_of_distance(self, patch_ones):
        """Mute middle portion of distance dimension."""
        dist = patch_ones.coords.get_array("distance")
        mid_start = dist[len(dist) // 4]
        mid_end = dist[3 * len(dist) // 4]

        muted = patch_ones.mute(distance=(mid_start, mid_end), relative=False)

        # Check middle is muted
        mid_idx = len(dist) // 2
        assert np.allclose(muted.data[mid_idx, :], 0)
        # Check edges are not muted
        assert np.allclose(muted.data[0, :], 1)
        assert np.allclose(muted.data[-1, :], 1)

    def test_mute_relative_default(self, patch_ones):
        """Test that relative=True is default."""
        # Should mute first second (relative to start)
        muted = patch_ones.mute(time=(0, 1))
        # First samples should be zero
        assert muted.data[:, 0].max() < 0.1

    def test_mute_relative_negative(self, patch_ones):
        """Test relative with negative values (from end)."""
        muted = patch_ones.mute(time=(-1, None))
        # Last samples should be zero
        assert muted.data[:, -1].max() < 0.1


class TestBlockMutes2D:
    """Test 2D block mutes (multiple dimensions)."""

    def test_mute_rectangular_region(self, patch_ones):
        """Mute a rectangular region in 2D."""
        time_coords = patch_ones.coords.get_array("time")
        dist_coords = patch_ones.coords.get_array("distance")

        t1, t2 = time_coords[10], time_coords[20]
        d1, d2 = dist_coords[5], dist_coords[15]

        muted = patch_ones.mute(
            time=(t1, t2),
            distance=(d1, d2),
            relative=False,
        )

        # Check interior is muted
        assert np.allclose(muted.data[10, 15], 0)
        # Check corners are not muted
        assert np.allclose(muted.data[0, 0], 1)
        assert np.allclose(muted.data[-1, -1], 1)

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


class TestMuteModes:
    """Test different mute modes."""

    def test_mode_union(self, patch_ones):
        """Test union mode (default, mutes inside region)."""
        muted = patch_ones.mute(time=(2.0, 6.0), mode="union")
        # Middle (at ~4s) should be zero
        mid_idx = len(patch_ones.coords.get_array("time")) // 2
        assert muted.data[:, mid_idx].max() < 0.1
        # Edges should be one
        assert muted.data[:, 0].min() > 0.9

    def test_mode_complement(self, patch_ones):
        """Test complement mode (mutes outside region)."""
        time_coords = patch_ones.coords.get_array("time")
        t1, t2 = time_coords[10], time_coords[20]

        muted = patch_ones.mute(time=(t1, t2), mode="complement", relative=False)

        # Edges should be zero
        assert muted.data[:, 0].max() < 0.1
        assert muted.data[:, -1].max() < 0.1
        # Middle should be one
        mid_idx = 15
        assert muted.data[:, mid_idx].min() > 0.9


class TestMuteTaper:
    """Test taper application in mutes."""

    def test_mute_with_taper(self, patch_ones):
        """Mute with taper creates gradual transition."""
        time_coords = patch_ones.coords.get_array("time")
        t1, t2 = time_coords[20], time_coords[40]

        # Mute without taper
        muted_no_taper = patch_ones.mute(
            time=(t1, t2),
            relative=False,
            taper=None,
        )

        # Mute with taper (5% of dimension range)
        muted_with_taper = patch_ones.mute(
            time=(t1, t2),
            relative=False,
            taper=0.05,
        )

        # Without taper should be sharp transition
        assert muted_no_taper.data[:, 19].min() > 0.9
        assert muted_no_taper.data[:, 20].max() < 0.1

        # With taper should have gradual transition
        # Values in taper region should be between 0 and 1
        taper_region = muted_with_taper.data[:, 15:20]
        assert (taper_region > 0).any()
        assert (taper_region < 1).any()

    def test_taper_dict(self, patch_ones):
        """Test dimension-specific taper using dict."""
        time_coords = patch_ones.coords.get_array("time")
        dist_coords = patch_ones.coords.get_array("distance")

        taper_dict = {
            "time": 0.05,  # 5% of time range
            "distance": 0,  # No taper on distance
        }

        muted = patch_ones.mute(
            time=(time_coords[20], time_coords[40]),
            distance=(dist_coords[10], dist_coords[30]),
            relative=False,
            taper=taper_dict,
        )

        # Should have taper in time, sharp in distance
        assert muted.shape == patch_ones.shape

    def test_taper_window_types(self, patch_ones):
        """Test different window types for taper."""
        time_coords = patch_ones.coords.get_array("time")
        t1, t2 = time_coords[20], time_coords[40]

        for window_type in ["hann", "hamming", "triang"]:
            muted = patch_ones.mute(
                time=(t1, t2),
                relative=False,
                taper=0.05,  # 5% of dimension range
                window_type=window_type,
            )
            assert isinstance(muted, dc.Patch)
            assert muted.shape == patch_ones.shape

    def test_taper_with_quantity(self, patch_ones):
        """Test taper with Quantity (absolute units)."""
        from dascore.units import s

        time_coords = patch_ones.coords.get_array("time")
        t1, t2 = time_coords[20], time_coords[40]

        # Use absolute time value with units (single dimension)
        muted = patch_ones.mute(
            time=(t1, t2),
            relative=False,
            taper={'time': 0.05 * s},
        )

        # Should have gradual transition
        taper_region = muted.data[:, 15:20]
        assert (taper_region > 0).any()
        assert (taper_region < 1).any()

    def test_taper_quantity_multiple_dims_raises(self, patch_ones):
        """Test that Quantity taper with multiple dims raises error."""
        from dascore.units import s

        time_coords = patch_ones.coords.get_array("time")
        dist_coords = patch_ones.coords.get_array("distance")

        # Should raise error if Quantity used without dict
        with pytest.raises(ParameterError, match="Cannot use Quantity"):
            patch_ones.mute(
                time=(time_coords[20], time_coords[40]),
                distance=(dist_coords[10], dist_coords[30]),
                relative=False,
                taper=0.05 * s,
            )

    def test_taper_mixed_quantity_fraction(self, patch_ones):
        """Test mixed taper: fraction for one dim, Quantity for another."""
        from dascore.units import s, m

        time_coords = patch_ones.coords.get_array("time")
        dist_coords = patch_ones.coords.get_array("distance")

        muted = patch_ones.mute(
            time=(time_coords[20], time_coords[40]),
            distance=(dist_coords[10], dist_coords[30]),
            relative=False,
            taper={'time': 0.05, 'distance': 10 * m},
        )

        assert muted.shape == patch_ones.shape

    def test_taper_custom_function(self, patch_ones):
        """Test custom taper function."""
        time_coords = patch_ones.coords.get_array("time")
        t1, t2 = time_coords[20], time_coords[40]

        # Define custom taper that squares the envelope
        def custom_taper(envelope):
            return envelope ** 2

        muted = patch_ones.mute(
            time=(t1, t2),
            relative=False,
            taper=custom_taper,
        )

        # Middle should still be zero
        assert muted.data[:, 30].max() < 0.1
        # Edges should still be one
        assert muted.data[:, 0].min() > 0.9

    def test_taper_custom_function_invalid_return(self, patch_ones):
        """Test that custom taper with invalid return raises error."""
        time_coords = patch_ones.coords.get_array("time")
        t1, t2 = time_coords[20], time_coords[40]

        # Function that returns wrong type
        def bad_taper(envelope):
            return list(envelope)

        with pytest.raises(ParameterError, match="must return a numpy array"):
            patch_ones.mute(
                time=(t1, t2),
                relative=False,
                taper=bad_taper,
            )

    def test_taper_custom_function_wrong_shape(self, patch_ones):
        """Test that custom taper with wrong shape raises error."""
        time_coords = patch_ones.coords.get_array("time")
        t1, t2 = time_coords[20], time_coords[40]

        # Function that returns wrong shape
        def bad_shape_taper(envelope):
            return np.ones((10, 10))

        with pytest.raises(ParameterError, match="must return array with shape"):
            patch_ones.mute(
                time=(t1, t2),
                relative=False,
                taper=bad_shape_taper,
            )


class TestMuteEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_mute_with_none_boundaries(self, patch_ones):
        """Test using None to reference coordinate edges."""
        # Mute from start to middle
        time_coords = patch_ones.coords.get_array("time")
        mid_time = time_coords[len(time_coords) // 2]

        muted = patch_ones.mute(time=(None, mid_time), relative=False)

        # First half should be zero
        assert muted.data[:, 0].max() < 0.1
        # Second half should be one
        assert muted.data[:, -1].min() > 0.9

    def test_mute_entire_dimension(self, patch_ones):
        """Mute entire dimension."""
        muted = patch_ones.mute(time=(None, None), relative=False)
        # Everything should be zero
        assert np.allclose(muted.data, 0)

    def test_mute_zero_width(self, patch_ones):
        """Mute with same start and end values."""
        time_coords = patch_ones.coords.get_array("time")
        t = time_coords[10]

        muted = patch_ones.mute(time=(t, t), relative=False)
        # Should mute just that one sample (or very close)
        assert muted.data[:, 10].max() < 0.1


class TestMuteWithSamples:
    """Test mute with samples=True."""

    def test_mute_samples_mode(self, patch_ones):
        """Test mute using sample indices."""
        # Mute samples 10 to 20
        muted = patch_ones.mute(time=(10, 20), samples=True)

        # Samples 10-20 should be zero
        assert muted.data[:, 15].max() < 0.1
        # Other samples should be one
        assert muted.data[:, 0].min() > 0.9
        assert muted.data[:, -1].min() > 0.9


class TestGeometricMutes:
    """Test geometric (array-based) mutes - Phase 2."""

    def test_array_boundaries_not_implemented(self, ricker_patch):
        """Array boundaries should raise error for mixing scalar/array."""
        # Currently we don't allow mixing scalars and arrays
        with pytest.raises(ParameterError, match="Cannot mix scalar and array"):
            ricker_patch.mute(
                time=(0, [0, 0.3]),
                distance=(0, [0, 300]),
                relative=False,
            )

    def test_mismatched_array_lengths_raises(self, random_patch):
        """Arrays with different lengths should raise error during parsing."""
        with pytest.raises(ParameterError, match="Cannot mix scalar and array"):
            # This currently raises mixing error before length checking
            random_patch.mute(
                time=(0, [0, 0.3, 0.5]),  # scalar and array
                distance=(0, [0, 300]),  # scalar and array
                relative=False,
            )


class TestMutePreservesMetadata:
    """Test that mute preserves patch metadata."""

    def test_preserves_coords(self, random_patch):
        """Mute should preserve coordinates."""
        muted = random_patch.mute(time=(0, 0.5))
        assert muted.coords == random_patch.coords

    def test_preserves_attrs(self, random_patch):
        """Mute should preserve attributes (except history)."""
        muted = random_patch.mute(time=(0, 0.5))
        # History will be different due to processing
        assert muted.attrs.data_type == random_patch.attrs.data_type
        assert muted.attrs.network == random_patch.attrs.network
        assert muted.attrs.station == random_patch.attrs.station

    def test_preserves_shape(self, random_patch):
        """Mute should preserve shape."""
        muted = random_patch.mute(time=(0, 0.5))
        assert muted.shape == random_patch.shape

    def test_preserves_dtype(self, random_patch):
        """Mute should preserve data type."""
        muted = random_patch.mute(time=(0, 0.5))
        # Allow for float conversion
        assert muted.dtype in (random_patch.dtype, np.float64, np.float32)
