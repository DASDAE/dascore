"""Tests for Fourier transforms."""

from __future__ import annotations

import numpy as np
import pytest

from dascore.transform.fft import rfft
from dascore.units import get_quantity
from dascore.utils.misc import suppress_warnings


class TestRfft:
    """Tests for the real fourier transform."""

    @pytest.fixture(scope="class")
    def rfft_patch(self, random_patch):
        """Return the random patched transformed along time w/ rrft."""
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
            fft_patch = patch.rfft("time")
        dunits1 = get_quantity(patch.attrs.data_units)
        dunits2 = get_quantity(fft_patch.attrs.data_units)
        assert dunits2 == dunits1 * get_quantity("second")
