"""Tests for taper processing function."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
import dascore.proc.coords
from dascore.exceptions import ParameterError
from dascore.proc.taper import taper
from dascore.units import m, percent
from dascore.utils.misc import broadcast_for_index
from dascore.utils.signal import WINDOW_FUNCTIONS

gen = np.random.default_rng(32)


@pytest.fixture(scope="session")
def patch_ones(random_patch):
    """Return a patch filled with ones."""
    patch = random_patch.new(data=np.ones_like(random_patch.data))
    return patch


@pytest.fixture(scope="session", params=sorted(WINDOW_FUNCTIONS))
def time_tapered_patch(request, patch_ones):
    """Return a tapered trace."""
    # first get a patch with all ones for easy testing
    patch = patch_ones.update(data=np.ones_like(patch_ones.data))
    out = taper(patch, time=0.05, window_type=request.param)
    return out


def _get_start_end_indices(patch, dim):
    """Helper function to get indices for slicing start/end of data."""
    axis = patch.get_axis(dim)
    n_dims = len(patch.dims)
    inds_start = broadcast_for_index(n_dims, axis, 0)
    inds_end = broadcast_for_index(n_dims, axis, -1)
    return inds_start, inds_end


@pytest.fixture(scope="class")
def patch_sorted_time(random_patch):
    """Return a patch with sorted but not evenly spaced time dim."""
    times = gen.random(len(random_patch.get_coord("time")))
    new_times = dc.to_datetime64(np.sort(times))
    return random_patch.update_coords(time=new_times)


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
        dim_len = patch.get_axis("distance")
        mid_dim = dim_len // 2

        assert not np.allclose(data_new, data_old)

        assert np.allclose(
            data_new[mid_dim - 10 : mid_dim + 10], data_old[mid_dim - 10 : mid_dim + 10]
        )

    def test_timedelta_taper(self, random_patch):
        """Test that a timedelta works for the taper argument. See #379."""
        time1 = dc.to_timedelta64(2)
        time2 = 2 * dc.get_quantity("seconds")
        patch1 = random_patch.taper(time=time1)
        patch2 = random_patch.taper(time=time2)
        assert patch1 == patch2

    def test_percentage_taper(self, patch_ones):
        """Ensure a percentage unit can be used in addition to fraction."""
        out1 = patch_ones.taper(time=(0.1, 0.2))
        out2 = patch_ones.taper(time=(10 * percent, 20 * percent))
        assert out1.equals(out2, close=True)

    def test_uneven_time_coord(self, patch_sorted_time):
        """Ensure taper works on patches without even sampling."""
        out = patch_sorted_time.taper(time=(0.1, 0.2))
        assert isinstance(out, dc.Patch)

    def test_full_taper(self, patch_ones):
        """Ensure a "full taper" works"""
        out = patch_ones.taper(time=0.5)
        # The taper should have covered all the data
        assert np.all(out.data < 1)


class TestTaperRange:
    """Test for tapering a range of values."""

    def test_dims(self, patch_ones):
        """Ensure both dimensions work."""
        for ax, dim in enumerate(patch_ones.dims):
            coord = patch_ones.get_coord(dim)
            clen = len(coord)
            ind1 = int(clen / 3)
            ind2 = int(2 * clen / 3)
            val1, val2 = coord[ind1], coord[ind2]
            out = patch_ones.taper_range(**{dim: (val1, val2)})
            data = out.data
            # Ensure the center of the envelope are untouched
            tapered_inds = broadcast_for_index(data.ndim, ax, slice(ind1, ind2))
            tapered = data[tapered_inds]
            assert np.allclose(tapered, 1)
            # Ensure the edges are near 0
            start_vals = data[broadcast_for_index(data.ndim, ax, 0)]
            assert np.allclose(start_vals, 0)
            end_vals = data[broadcast_for_index(data.ndim, ax, -1)]
            assert np.allclose(end_vals, 0)

    def test_invert(self, patch_ones):
        """Ensure inverting results in 0s near the peaks of taper."""
        dim, ax = "time", patch_ones.get_axis("time")
        coord = patch_ones.get_coord(dim)
        clen = len(coord)
        ind1 = int(clen / 3)
        ind2 = int(2 * clen / 3)
        val1, val2 = coord[ind1], coord[ind2]
        out = patch_ones.taper_range(**{dim: (val1, val2)}, invert=True)
        data = out.data
        # Ensure the center of the envelope are untouched
        tapered_inds = broadcast_for_index(data.ndim, ax, slice(ind1, ind2))
        tapered = data[tapered_inds]
        assert np.allclose(tapered, 0)
        # Ensure the edges are near 0
        start_vals = data[broadcast_for_index(data.ndim, ax, 0)]
        assert np.allclose(start_vals, 1)
        end_vals = data[broadcast_for_index(data.ndim, ax, -1)]
        assert np.allclose(end_vals, 1)

    def test_four_value_taper(self, patch_ones):
        """Ensure the range works with 4 values specifying taper limits."""
        dim = "time"
        coord = patch_ones.get_coord(dim)
        clen = len(coord)
        inds = int(clen / 6), int(2 * clen / 6), int(4 * clen / 6), int(5 * clen / 6)
        vals = [coord[x] for x in inds]
        out = patch_ones.taper_range(**{dim: vals}, invert=False)

        in_taper = out.select(time=(vals[1], vals[2]))
        assert np.allclose(in_taper.data, 1, rtol=1e-4)

        right_of_taper = out.select(time=(vals[3], ...))
        assert np.allclose(right_of_taper.data, 0)
        left_of_taper = out.select(time=(..., vals[0]))
        assert np.allclose(left_of_taper.data, 0)

    def test_samples_and_relative(self, random_patch):
        """Ensure samples and relative work with coord."""
        coord = random_patch.get_coord("time")
        values = coord.values
        ind = 200
        # normal style taper
        p1 = random_patch.taper_range(time=(values[ind], values[-ind]))
        # relative taper
        time_range = (values[ind] - values[0], values[-ind] - values[-1])
        p2 = random_patch.taper_range(time=time_range, relative=True)
        # samples taper
        p3 = random_patch.taper_range(time=(ind, -ind), samples=True)
        assert p1.equals(p2) and p2.equals(p3)

    def test_poorly_shaped_sequence_raises(self, random_patch):
        """Ensure sequences of wrong shape raise."""
        with pytest.raises(ParameterError, match="sequence is required"):
            random_patch.taper_range(time=1)
        with pytest.raises(ParameterError, match="sequence is required"):
            random_patch.taper_range(time=[1])
        with pytest.raises(ParameterError, match="sequence is required"):
            random_patch.taper_range(time=[1] * 10)

    def test_bad_use_of_none(self, random_patch):
        """Ensure bad use of None raises."""
        with pytest.raises(ParameterError, match="Cannot use ... or None"):
            random_patch.taper_range(time=(1, None), relative=True)

    def test_use_none(self, random_patch):
        """Ensure use of None specifies proper limits."""
        coord = random_patch.get_coord("time")
        time_span = coord.max() - coord.min()
        out1 = random_patch.taper_range(time=(None, 1, 4, None), relative=True)
        out2 = random_patch.taper_range(time=(0, 1, 4, time_span), relative=True)
        assert out1.equals(out2)

    def test_multiple_taper_values(self, patch_ones):
        """Ensure multiple values in taper are possible."""
        taper_range = ((25, 50, 100, 125), (150, 175, 200, 225))
        out = patch_ones.taper_range(distance=taper_range)

        # Check a few points inside and outside of the taper bounds
        assert np.allclose(out.select(distance=(..., 22)).data, 0)
        assert np.allclose(out.select(distance=(55, 90)).data, 1)
        assert np.allclose(out.select(distance=(126, 149)).data, 0)
        assert np.allclose(out.select(distance=(225, ...)).data, 0)

    def test_uneven_coords(self, patch_sorted_time):
        """Ensure uneven coords also work for tapering."""
        out = patch_sorted_time.taper_range(time=(0.4, 0.6), relative=True)
        assert isinstance(out, dc.Patch)

    def test_non_cosine_window(self, patch_ones):
        """Ensure other window funcs also work."""
        taper_range = ((25, 50, 100, 125), (150, 175, 200, 225))
        out = patch_ones.taper_range(distance=taper_range, window_type="ramp")
        # Check a few points inside and outside of the taper bounds
        assert np.allclose(out.select(distance=(..., 22)).data, 0)
        assert np.allclose(out.select(distance=(55, 90)).data, 1)
        assert np.allclose(out.select(distance=(126, 149)).data, 0)
        assert np.allclose(out.select(distance=(225, ...)).data, 0)
