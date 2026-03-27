"""Tests for Fourier transforms."""

from __future__ import annotations

import numpy as np

from dascore.units import get_quantity


class TestDftReal:
    """Tests for the real one-sided dft."""

    def test_dims(self, random_patch):
        """Ensure ft of original axis shows up in dimensions."""
        dims = random_patch.dft("time", real="time").dims
        start_freq = [x.startswith("ft_") for x in dims]
        assert any(start_freq)

    def test_abs_rrft(self, random_patch):
        """Ensure abs works with the real dft to get amplitude spectra."""
        dft_patch = random_patch.dft("time", real="time")
        out = dft_patch.abs()
        assert np.allclose(out.data, np.abs(dft_patch.data))

    def test_time_coord_units(self, random_patch):
        """Ensure time label units have been correctly set."""
        rfft_patch = random_patch.dft("time", real="time")
        units1 = random_patch.coords.coord_map["time"].units
        units2 = rfft_patch.coords.coord_map["ft_time"].units
        assert get_quantity(units1) == 1 / get_quantity(units2)

    def test_data_units(self, random_patch):
        """Ensure data units have been updated."""
        patch = random_patch.update_attrs(data_units="m/s")
        fft_patch = patch.dft("time", real="time")
        dunits1 = get_quantity(patch.attrs.data_units)
        dunits2 = get_quantity(fft_patch.attrs.data_units)
        assert dunits2 == dunits1 * get_quantity("second")
