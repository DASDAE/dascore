"""Tests for Fourier transforms."""
from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
import dascore.proc.coords
from dascore.transform.fourier import dft, idft
from dascore.units import get_quantity

F_0 = 2


@pytest.fixture(scope="session")
def sin_patch():
    """Get the sine wave patch, set units for testing."""
    patch = dc.get_example_patch("sin_wav", sample_rate=100, duration=3, frequency=F_0)
    out = patch.set_units(get_quantity("1.0 V"), time="s", distance="m")
    return out


@pytest.fixture(scope="session")
def fft_sin_patch_time(sin_patch):
    """Get the sine wave patch, set units for testing."""
    return dft(sin_patch, dim="time")


@pytest.fixture(scope="session")
def fft_sin_patch_all(sin_patch):
    """Get the sine wave patch, set units for testing."""
    return dft(sin_patch, dim=None)


@pytest.fixture(scope="session")
def ifft_sin_patch_time(fft_sin_patch_time):
    """Get the sine wave patch, set units for testing."""
    return idft(fft_sin_patch_time, dim="ft_time")


@pytest.fixture(scope="session")
def ifft_sin_patch_all(fft_sin_patch_all):
    """Get the sine wave patch, set units for testing."""
    return idft(fft_sin_patch_all, dim=None)


class TestDiscreteFourierTransform:
    """Forward DFT suite."""

    def test_max_frequency(self, fft_sin_patch_time):
        """Ensure when sin wave is input max freq is correct."""
        assert "ft_time" in fft_sin_patch_time.dims
        patch = fft_sin_patch_time
        freq_dim = patch.dims.index("ft_time")
        ar = np.argmax(np.abs(patch.data), freq_dim)
        assert np.allclose(ar, ar[0])
        freqs = patch.get_coord("ft_time").data
        max_freq = np.abs(freqs[ar[0]])
        assert np.isclose(max_freq, F_0, rtol=0.01)

    def test_units(self, fft_sin_patch_time, sin_patch):
        """Ensure units were transformed as expected."""
        time_units = get_quantity(fft_sin_patch_time.get_coord("time").units)
        ft_time_units = get_quantity(fft_sin_patch_time.get_coord("ft_time").units)
        assert 1 / time_units == ft_time_units
        old_data_units = get_quantity(sin_patch.attrs.data_units)
        new_data_units = get_quantity(fft_sin_patch_time.attrs.data_units)
        assert old_data_units * time_units == new_data_units

    def test_drop_non_dimensional_coordinates(self, random_patch_many_coords):
        """
        Non-dimensional coordinates associated with transformed axis should
        be dropped, but those associated with non-transformed axis should remain.
        """
        patch = random_patch_many_coords
        # every coord associated with time should be dropped in output.
        # but time itself should be kept as non-dimensional coord.
        coord_to_drop = set(patch.coords.dim_to_coord_map["time"]) - {"time"}
        coords_to_keep = set(patch.coords.coord_map) - coord_to_drop
        # do dft
        out = dft(patch, "time")
        # ensure kept coords are kept and dropped are dropped.
        new_coords = set(out.coords.coord_map)
        assert coord_to_drop.isdisjoint(new_coords)
        assert coords_to_keep.issubset(new_coords)
        # make sure time has no dimsensions
        assert out.coords.dim_map["time"] == ()

    def test_real_fft(self, sin_patch):
        """Ensure real fft works."""
        out = sin_patch.dft("time", real=True)
        coord = out.get_coord("ft_time")
        freq_ax = out.dims.index("ft_time")
        assert coord.min() == 0
        ar = np.argmax(np.abs(out.data), axis=freq_ax)
        assert np.allclose(ar, ar[0])
        max_freq = np.abs(coord.data[ar[0]])
        assert np.isclose(max_freq, F_0, rtol=0.01)
        # data shape should be less than before (since real fft)
        ft_shape = out.coord_shapes["ft_time"][0]
        time_shape = sin_patch.coord_shapes["time"][0]
        assert ft_shape == time_shape // 2 or ft_shape == (time_shape // 2 + 1)

    def test_all_dims(self, fft_sin_patch_all):
        """Ensure fft can be done on all axis."""
        patch = fft_sin_patch_all
        assert all(x.startswith("ft_") for x in patch.dims)

    def test_real_multiple_dims(self, sin_patch):
        """Ensure the real axis can be specified."""
        patch = sin_patch
        out = patch.dft(dim=("distance", "time"), real="distance")
        assert all(x.startswith("ft_") for x in out.dims)
        real_coord = out.get_coord("ft_distance")
        assert real_coord.min() == 0

    def test_parseval(self, sin_patch, fft_sin_patch_time):
        """
        Ensure parseval's theorem holds. This means we have scaled the
        transforms correctly.
        """
        pa1, pa2 = sin_patch, fft_sin_patch_time
        vals1 = (pa1**2).integrate("time", definite=True)
        vals2 = (pa2.abs() ** 2).integrate("ft_time", definite=True)
        assert np.allclose(vals1.data, vals2.data)


class TestInverseDiscreteFourierTransform:
    """Inverse DFT suite."""

    def _patches_about_equal(self, patch1, patch2):
        """Ensure patches are about equal in coord manager and data."""
        assert patch1.data.shape == patch2.data.shape
        assert np.allclose(patch1.data, patch2.data)
        cm1 = patch1.coords.drop_disassociated_coords()
        cm2 = patch2.coords.drop_disassociated_coords()
        assert cm1 == cm2

    def test_invertible_1d(self, sin_patch, ifft_sin_patch_time):
        """Ensure pre dft and idft(dft(patch)) are equal."""
        patch1 = sin_patch
        patch2 = ifft_sin_patch_time.real()
        self._patches_about_equal(patch1, patch2)

    def test_invertible_2d(self, sin_patch, ifft_sin_patch_all):
        """Ensure 2d patches are invertible."""
        patch1 = sin_patch
        patch2 = ifft_sin_patch_all.real()
        self._patches_about_equal(patch1, patch2)

    def test_undo_real_dft(self, sin_patch):
        """Ensure real dft is properly handled."""
        pa1 = sin_patch.dft(dim="time", real=True)
        pa2 = pa1.idft().real()
        self._patches_about_equal(sin_patch, pa2)

    def test_raises_on_untransformed_patch(self, sin_patch):
        """Only patches which have been first transformed can be idft'ed."""
        with pytest.raises(NotImplementedError):
            sin_patch.idft("time")

    def test_partial_inverse(self, fft_sin_patch_all, sin_patch):
        """Ensure inverse works on only a single axis."""
        # since we only reverse time it should be the same as forward distance.
        ift = fft_sin_patch_all.idft("time")
        dft = sin_patch.dft("distance")
        self._patches_about_equal(ift, dft)
        # and then if we reverse distance it should be the same as original
        full_inverse = ift.idft("distance")
        self._patches_about_equal(full_inverse, sin_patch)
