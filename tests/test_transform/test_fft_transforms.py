"""
Tests for Fourier transforms.
"""

import numpy as np
import pytest

import dascore as dc
from dascore.transform.fft import dft, idft, rfft
from dascore.units import get_quantity
from dascore.utils.misc import suppress_warnings

F_0 = 2


@pytest.fixture(scope="session")
def sin_patch():
    """Get the sine wave patch, set units for testing."""
    patch = dc.get_example_patch("sin_wav", sample_rate=100, duration=3, frequency=F_0)
    out = patch.set_units("V", time="s", distance="m")
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
        time_units = get_quantity(sin_patch.get_coord("time").units)
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
        coord_to_drop = set(patch.coords.dim_to_coord_map["time"])
        coords_to_keep = set(patch.coords.coord_map) - coord_to_drop
        # do dft
        out = dft(patch, "time")
        # ensure kept coords are kept and dropped are dropped.
        new_coords = set(out.coords.coord_map)
        assert coord_to_drop.isdisjoint(new_coords)
        assert coords_to_keep.issubset(new_coords)

    def test_real_fft(self, sin_patch):
        """Ensure real fft works."""
        out = sin_patch.tran.dft("time", real=True)
        coord = out.get_coord("ft_time")
        freq_ax = out.dims.index("ft_time")
        assert coord.min() == 0
        ar = np.argmax(np.abs(out.data), axis=freq_ax)
        assert np.allclose(ar, ar[0])
        max_freq = np.abs(coord.data[ar[0]])
        assert np.isclose(max_freq, F_0, rtol=0.01)

    def test_all_dims(self, fft_sin_patch_all):
        """Ensure fft can be done on all axis."""
        patch = fft_sin_patch_all
        assert all((x.startswith("ft_") for x in patch.dims))

    def test_real_multiple_dims(self, sin_patch):
        """Ensure the real axis can be specified."""
        patch = sin_patch
        out = patch.tran.dft(dim=("distance", "time"), real="distance")
        assert all((x.startswith("ft_") for x in out.dims))
        real_coord = out.get_coord("ft_distance")
        assert real_coord.min() == 0


class TestInverseDiscreteFourierTransform:
    """Inverse DFT suite."""

    def test_invertible(self, sin_patch, ifft_sin_patch_time):
        """Ensure pre dft and idft(dft(patch)) are equal."""
        patch1 = sin_patch
        patch2 = ifft_sin_patch_time
        assert patch1.equals(patch2)


class TestRfft:
    """
    Tests for the real fourier transform.

    TODO: Remove this in a few versions
    """

    @pytest.fixture(scope="class")
    def rfft_patch(self, random_patch):
        """return the random patched transformed along time w/ rrft."""
        with suppress_warnings(DeprecationWarning):
            out = rfft(random_patch, dim="time")
        return out

    def test_dims(self, rfft_patch):
        """Ensure ft of original axis shows up in dimensions."""
        dims = rfft_patch.dims
        start_freq = [x.startswith("ft_") for x in dims]
        assert any(start_freq)

    def test_abs_rrft(self, rfft_patch):
        """Ensure abs works with rfft to get amplitude spectra."""
        out = rfft_patch.abs()
        assert np.allclose(out.data, np.abs(rfft_patch.data))

    def test_time_coord_units(self, random_patch, rfft_patch):
        """Ensure time label units have been correctly set."""
        units1 = random_patch.coords.coord_map["time"].units
        units2 = rfft_patch.coords.coord_map["ft_time"].units
        assert get_quantity(units1) == 1 / get_quantity(units2)

    def test_data_units(self, random_patch):
        """Ensure data units have been updated."""
        patch = random_patch.update_attrs(data_units="m/s")
        with suppress_warnings(DeprecationWarning):
            fft_patch = patch.tran.rfft("time")
        dunits1 = get_quantity(patch.attrs.data_units)
        dunits2 = get_quantity(fft_patch.attrs.data_units)
        assert dunits2 == dunits1 * get_quantity("second")
