"""Tests for taper processing function."""
from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.proc.taper import TAPER_FUNCTIONS, taper
from dascore.units import m
from dascore.utils.misc import broadcast_for_index


@pytest.fixture(scope="session")
def patch_ones(random_patch):
    """Return a patch filled with ones."""
    patch = random_patch.new(data=np.ones_like(random_patch.data))
    return patch


@pytest.fixture(scope="session", params=sorted(TAPER_FUNCTIONS))
def time_tapered_patch(request, patch_ones):
    """Return a tapered trace."""
    # first get a patch with all ones for easy testing
    patch = patch_ones.update(data=np.ones_like(patch_ones.data))
    out = taper(patch, time=0.05, window_type=request.param)
    return out


def _get_start_end_indices(patch, dim):
    """Helper function to get indices for slicing start/end of data."""
    axis = patch.dims.index(dim)
    n_dims = len(patch.dims)
    inds_start = broadcast_for_index(n_dims, axis, 0)
    inds_end = broadcast_for_index(n_dims, axis, -1)
    return inds_start, inds_end


class TestTaperBasics:
    """Ensure each taper runs."""

    def test_each_taper(self, time_tapered_patch, random_patch):
        """Ensure each taper type runs."""
        assert isinstance(time_tapered_patch, dc.Patch)
        assert time_tapered_patch.shape == random_patch.shape

    def test_time_dt_unchanged(self, time_tapered_patch, random_patch):
        """Ensure each taper type runs."""
        attrs1, attrs2 = random_patch.attrs, time_tapered_patch.attrs
        assert attrs1.time_units == attrs2.time_units
        assert attrs1.time_step == attrs2.time_step

    def test_ends_near_zero(self, time_tapered_patch):
        """Ensure the ends of the patch are near zero."""
        patch = time_tapered_patch
        data = patch.data
        inds_start, inds_end = _get_start_end_indices(patch, "time")
        assert np.all(data[inds_start] < 0.09)
        assert np.all(data[inds_end] < 0.09)

    def test_taper_start_only(self, patch_ones):
        """Ensure tapering only start works."""
        patch = patch_ones.taper(time=(0.05, None))
        data = patch.data
        inds_start, inds_end = _get_start_end_indices(patch, "time")
        assert np.all(data[inds_start] < 0.09)
        assert np.allclose(data[inds_end], 1)

    def test_taper_end_only(self, patch_ones):
        """Ensure tapering only the end works."""
        patch = patch_ones.taper(time=(None, 0.05))
        data = patch.data
        inds_start, inds_end = _get_start_end_indices(patch, "time")
        assert np.all(data[inds_end] < 0.09)
        assert np.allclose(data[inds_start], 1)

    def test_overlapping_windows_raises(self, patch_ones):
        """Overlapping tapers should raise."""
        with pytest.raises(ParameterError, match="cannot overlap"):
            patch_ones.taper(time=(0.51, 0.51))

    def test_too_big_raises(self, patch_ones):
        """Single tapers too large should raise."""
        with pytest.raises(ParameterError, match="taper lengths exceed"):
            patch_ones.taper(time=(None, 1.01))
        with pytest.raises(ParameterError, match="taper lengths exceed"):
            patch_ones.taper(time=(1.01, None))

    def test_doc_example(self, random_patch):
        """Tests the doc example case."""
        patch = random_patch
        patch_taper1 = patch.taper(time=0.05, window_type="hann")
        patch_taper2 = patch.taper(distance=(0.10, None), window_type="triang")
        assert patch_taper1.shape == patch_taper2.shape

    def test_bad_taper_str_raises(self, random_patch):
        """Test that an invalid taper function raises nice message."""
        with pytest.raises(ParameterError, match="not a known window"):
            random_patch.taper(time=(0.1, 0.1), window_type="windowsXP")
            # the only good one...

    def test_taper_with_units(self, patch_ones):
        """Ensure taper words with units specified."""
        value = 15 * m
        patch = patch_ones.taper(distance=value)
        data_new = patch.data
        data_old = patch_ones.data
        dim_len = patch.dims.index("distance")
        mid_dim = dim_len // 2

        assert not np.allclose(data_new, data_old)

        assert np.allclose(
            data_new[mid_dim - 10 : mid_dim + 10], data_old[mid_dim - 10 : mid_dim + 10]
        )
