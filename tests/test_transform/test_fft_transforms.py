"""
Tests for fft style transforms.
"""

import numpy as np
import pytest

from dascore.transform.fft import rfft


class TestRfft:
    """Tests for the real fourier transform."""

    @pytest.fixture(scope="class")
    def rfft_patch(self, random_patch):
        """return the random patched transformed along time w/ rrft."""
        out = rfft(random_patch, dim="time")
        return out

    def test_dims(self, rfft_patch):
        """Ensure frequency shows up in dimensions."""
        dims = rfft_patch.dims
        start_freq = [x.startswith("frequency") for x in dims]
        assert any(start_freq)

    def test_abs_rrft(self, rfft_patch):
        """Ensure abs works with rfft to get amplitude spectra."""
        out = rfft_patch.abs()
        assert np.allclose(out.data, np.abs(rfft_patch.data))
