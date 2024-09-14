"""Tests for Dispersion transforms."""

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.transform.taup import _jit_taup_general, _jit_taup_uniform, tau_p
from dascore.utils.misc import suppress_warnings
from dascore.utils.time import to_timedelta64


def linear_slope_patch(nch, nt, vel, t0, dist=[]):
    """Returns the a patch with linear slope event"""
    # Create attributes, or metadata
    attrs = dict(category="DAS", id="linear_slope", data_units="um/(m * s)")

    dt = 1 / 1000

    if np.size(dist) != nch:
        distance_start = 0
        dx = 1.0
        distance_step = dx
        distance = distance_start + np.arange(nch) * distance_step
    else:
        distance = dist

    # Create coordinates, labels for each axis in the array.
    time_start = dc.to_datetime64("2017-09-18")
    time_step = to_timedelta64(dt)
    time = time_start + np.arange(nt) * time_step

    coords = dict(time=time, distance=distance)

    # Define dimensions (first label corresponds to data axis 0)
    dims = ("distance", "time")

    array = np.zeros(shape=(nch, nt))

    if vel > 0:
        for i in range(nch):
            t = int((t0 + distance[i] / vel) / dt)
            array[i, t] += 1.0
    else:
        for i in range(nch):
            t = int((t0 + np.abs((distance[-1] - distance[i]) / vel)) / dt)
            array[i, t] += 1.0

    pa = dc.Patch(data=array, coords=coords, attrs=attrs, dims=dims)
    return pa


class TestTauP:
    """Tests for the tau-p module."""

    @pytest.fixture(scope="class")
    def tau_p_patch(self, random_patch):
        """Returns the random patched transformed to tau-p."""
        pytest.importorskip("numba")
        test_vels = np.linspace(1500, 5000, 100)

        with suppress_warnings(DeprecationWarning):
            out = tau_p(random_patch, test_vels)
        return out

    def test_tau_p_consistency(self, tau_p_patch):
        """Checks consistency of tau_p module."""
        pytest.importorskip("numba")
        # assert time dimension
        assert "time" in tau_p_patch.dims
        # assert slowness dimension
        assert "slowness" in tau_p_patch.dims

        p_vals = tau_p_patch.coords.get_array("slowness")
        n_p_vals = len(p_vals)

        assert n_p_vals % 2 == 0
        n_p_vals = int(n_p_vals / 2)

        assert np.array_equal(
            p_vals[0:n_p_vals], -1.0 / np.flip(np.linspace(1500, 5000, 100))
        )
        assert np.array_equal(
            p_vals[n_p_vals : 2 * n_p_vals], 1.0 / np.linspace(1500, 5000, 100)
        )

    def test_negative_vel_raises(self, random_patch):
        """Ensures negative velocities raise an error"""
        msg = "Input velocities must be positive."
        velocities = np.array([1000, 1100, -200, 1000])
        with pytest.raises(ParameterError, match=msg):
            random_patch.tau_p(velocities=velocities)

    def test_non_monotonic_vel_raises(self, random_patch):
        """Ensures non monotonic velocities raise an error"""
        msg = "Input velocities must be monotonically increasing."
        velocities = np.array([1000, 1100, 300, 1200])
        with pytest.raises(ParameterError, match=msg):
            random_patch.tau_p(velocities=velocities)

        velocities = np.array([1000, 1100, 1100, 1200])
        with pytest.raises(ParameterError, match=msg):
            random_patch.tau_p(velocities=velocities)

    def test_slowness_vals(self):
        """Ensures correct slowness and tau values are computed"""
        pytest.importorskip("numba")
        test_vels = np.linspace(1000, 3000, 101)
        nch = 1000
        nt = 2000

        # positive slope
        vel = 1500
        t0 = 0.3
        linear_patch = linear_slope_patch(nch, nt, vel, t0)
        tau_p_patch = linear_patch.tau_p(test_vels)
        p_vals = tau_p_patch.get_coord("slowness")
        t_vals = tau_p_patch.get_coord("time")
        a = tau_p_patch.data
        (p_ind, t_ind) = np.unravel_index(a.argmax(), a.shape)
        assert np.abs(1.0 / p_vals[p_ind] - vel) < 20
        assert np.abs((t_vals[t_ind] - t_vals[0]) / np.timedelta64(1, "s") - t0) < 0.02

        # negative slope
        vel = -2000
        t0 = 0.5
        linear_patch = linear_slope_patch(nch, nt, vel, t0)
        tau_p_patch = linear_patch.tau_p(test_vels)
        p_vals = tau_p_patch.get_coord("slowness")
        t_vals = tau_p_patch.get_coord("time")
        a = tau_p_patch.data
        (p_ind, t_ind) = np.unravel_index(a.argmax(), a.shape)
        assert np.abs(1.0 / p_vals[p_ind] - vel) < 20
        assert np.abs((t_vals[t_ind] - t_vals[0]) / np.timedelta64(1, "s") - t0) < 0.02

        dist = np.zeros(nch)
        cumdist = 0
        for i in range(nch):
            dist[i] = cumdist
            cumdist = cumdist + 1.0 + 1.0 * (i % 2 == 0)

        # positive slope, non-equal distance
        vel = 1800
        t0 = 0.4
        linear_patch = linear_slope_patch(nch, nt, vel, t0, dist)
        tau_p_patch = linear_patch.tau_p(test_vels)
        p_vals = tau_p_patch.get_coord("slowness")
        t_vals = tau_p_patch.get_coord("time")
        a = tau_p_patch.data
        (p_ind, t_ind) = np.unravel_index(a.argmax(), a.shape)
        assert np.abs(1.0 / p_vals[p_ind] - vel) < 20
        assert np.abs((t_vals[t_ind] - t_vals[0]) / np.timedelta64(1, "s") - t0) < 0.02

        # negative slope, non-equal distance
        vel = -1700
        t0 = 0.1
        linear_patch = linear_slope_patch(nch, nt, vel, t0, dist)
        tau_p_patch = linear_patch.tau_p(test_vels)
        p_vals = tau_p_patch.get_coord("slowness")
        t_vals = tau_p_patch.get_coord("time")
        a = tau_p_patch.data
        (p_ind, t_ind) = np.unravel_index(a.argmax(), a.shape)
        assert np.abs(1.0 / p_vals[p_ind] - vel) < 20
        assert np.abs((t_vals[t_ind] - t_vals[0]) / np.timedelta64(1, "s") - t0) < 0.02

    def test_coverage(self):
        """
        A small test of the jit'ed functions in python mode for coverage reasons.
        """
        pytest.importorskip("numba")
        data = np.zeros((10, 10))
        dx = 1
        dt = 1
        p_vals = np.array([1])
        _jit_taup_uniform.func(data, dx, dt, p_vals)
        distance = np.arange(10) * dt
        _jit_taup_general.func(data, distance, dt, p_vals)
