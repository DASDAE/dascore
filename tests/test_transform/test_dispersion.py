"""Tests for Dispersion transforms."""
import numpy as np
import pytest

from dascore.transform import phase_shift
from dascore.utils.misc import suppress_warnings


class TestDispersion:
    """Tests for the dispersion module."""

    @pytest.fixture(scope="class")
    def dispersion_patch(self, random_patch):
        """Return the random patched transformed along time w/ rrft."""
        test_vels = np.linspace(1500, 5000, 351)
        with suppress_warnings(DeprecationWarning):
            out = phase_shift(random_patch, test_vels, direction="rtl", approxdf=2.0)
        return out

    def test_dispersion(self, dispersion_patch):
        """Check consistency of test_dispersion module."""
        dims = dispersion_patch.dims
        dimvel = [x.startswith("vel") for x in dims]
        dimfreq = [x.startswith("freq") for x in dims]

        assert any(dimvel)
        assert any(dimfreq)

        vels = dispersion_patch.coords.get_array("velocity")
        # assert velocity dimension
        freqs = dispersion_patch.coords.get_array("frequency")
        # assert frequency dimension

        assert np.array_equal(vels, np.linspace(1500, 5000, 351))
        # Check that the velocity output is correct

        assert freqs[1] - freqs[0] > 1.9 and freqs[1] - freqs[0] < 2.1
        # check that the approximate frequency resolution is obtained
