"""Tests for Dispersion transforms."""

import numpy as np
import pytest

import dascore as dc
from dascore import get_example_patch
from dascore.exceptions import ParameterError
from dascore.transform import dispersion_phase_shift
from dascore.utils.misc import suppress_warnings


class TestDispersion:
    """Tests for the dispersion module."""

    @pytest.fixture(scope="class")
    def dispersion_patch(self, random_patch):
        """Return the random patched transformed to frequency-velocity."""
        test_vels = np.linspace(1500, 5000, 351)
        with suppress_warnings(DeprecationWarning):
            out = dispersion_phase_shift(random_patch, test_vels, approx_resolution=2.0)
        return out

    def test_dispersion(self, dispersion_patch):
        """Check consistency of test_dispersion module."""
        # assert velocity dimension
        assert "velocity" in dispersion_patch.dims
        # assert frequency dimension
        assert "frequency" in dispersion_patch.dims
        vels = dispersion_patch.coords.get_array("velocity")
        freqs = dispersion_patch.coords.get_array("frequency")
        assert np.array_equal(vels, np.linspace(1500, 5000, 351))
        # Check that the velocity output is correct
        assert freqs[1] - freqs[0] > 1.9 and freqs[1] - freqs[0] < 2.1
        # check that the approximate frequency resolution is obtained

    def test_dispersion_no_resolution(self, random_patch):
        """Ensure dispersion calc works when no resolution nor limits are provided."""
        test_vels = np.linspace(1500, 5000, 50)
        # create a smaller patch so this runs quicker.
        patch = random_patch.select(
            time=(0, 50), distance=(0, 10), relative=True, samples=True
        )
        dispersive_patch = dispersion_phase_shift(patch, test_vels)
        assert isinstance(dispersive_patch, dc.Patch)
        assert "velocity" in dispersive_patch.dims
        assert "frequency" in dispersive_patch.dims

    def test_non_monotonic_velocities(self, random_patch):
        """Ensure non-monotonic velocities raise Parameter Error."""
        msg = "must be monotonically increasing"
        velocities = np.array([10, -2, 100, 42])
        with pytest.raises(ParameterError, match=msg):
            random_patch.dispersion_phase_shift(phase_velocities=velocities)

    def test_velocity_lt_0_raises(self, random_patch):
        """Ensure velocity values < 0 raise ParameterError."""
        msg = "Velocities must be positive"
        velocities = np.array([-1, 0, 1])
        with pytest.raises(ParameterError, match=msg):
            random_patch.dispersion_phase_shift(phase_velocities=velocities)

    def test_approx_resolution_gt_0(self, random_patch):
        """Ensure velocity values < 0 raise ParameterError."""
        msg = "Frequency resolution has to be positive"
        test_vels = np.linspace(1500, 5000, 10)
        with pytest.raises(ParameterError, match=msg):
            random_patch.dispersion_phase_shift(
                phase_velocities=test_vels,
                approx_resolution=-1,
            )

    def test_freq_range_gt_0(self):
        """Ensure negative Parameter Error."""
        msg = "Minimal and maximal frequencies have to be positive"
        velocities = np.linspace(1500, 5000, 50)
        patch = get_example_patch("dispersion_event")
        with pytest.raises(ParameterError, match=msg):
            patch.dispersion_phase_shift(
                phase_velocities=velocities,
                approx_resolution=1.0,
                approx_freq=[-10, 50],
            )
        with pytest.raises(ParameterError, match=msg):
            patch.dispersion_phase_shift(
                phase_velocities=velocities,
                approx_resolution=1.0,
                approx_freq=[10, -50],
            )

    def test_freq_range_raises(self, random_patch):
        """Ensure negative Parameter Error."""
        msg = "Maximal frequency needs to be larger than minimal frequency"
        velocities = np.linspace(1500, 5000, 50)
        with pytest.raises(ParameterError, match=msg):
            random_patch.dispersion_phase_shift(
                phase_velocities=velocities, approx_resolution=1.0, approx_freq=[23, 22]
            )

    def test_freq_range_nyquist(self, random_patch):
        """Ensure negative Parameter Error."""
        msg = "Frequency range cannot exceed Nyquist"
        velocities = np.linspace(1500, 5000, 50)
        with pytest.raises(ParameterError, match=msg):
            random_patch.dispersion_phase_shift(
                phase_velocities=velocities,
                approx_resolution=1.0,
                approx_freq=[5000, 8000],
            )

    def test_freq_range_yields_empty(self, random_patch):
        """Ensure negative Parameter Error."""
        msg = "Combination of frequency resolution and range is not an array"
        velocities = np.linspace(1500, 5000, 50)
        with pytest.raises(ParameterError, match=msg):
            random_patch.dispersion_phase_shift(
                phase_velocities=velocities,
                approx_resolution=2.0,
                approx_freq=[22.0, 22.1],
            )
