"""Tests for correlate patch processing function."""

import numpy as np
import pytest

import dascore as dc
from dascore.units import m, s
from dascore.utils.time import to_float


class TestCorrelateInternal:
    """Tests case of intra-patch correlation function."""

    moveout_velocity = 100

    @pytest.fixture(scope="class")
    def corr_patch(self):
        """Create a patch of sin waves whose correlation can be easily checked."""
        patch = dc.get_example_patch(
            "sin_wav",
            sample_rate=100,
            frequency=range(10, 20),
            duration=5,
            channel_count=10,
        ).taper(time=0.5)
        # normalize energy so autocorrection is 1
        time_axis = patch.dims.index("time")
        data = patch.data
        norm = np.linalg.norm(data, axis=time_axis, keepdims=True)
        return patch.update(data=data / norm)

    @pytest.fixture(scope="session")
    def ricker_moveout_patch(self):
        """Create a patch with a ricker moveout."""
        patch = dc.get_example_patch(
            "ricker_moveout",
            velocity=self.moveout_velocity,
            duration=5,
        )
        return patch

    def test_basic_correlation(self, corr_patch):
        """Ensure correlation works along distance dim."""
        dist = corr_patch.get_coord("distance")[0]
        out = corr_patch.correlate(distance=dist)
        fft_ax = corr_patch.dims.index("time")
        fft_len = len(out.get_coord("lag_time"))
        argmax = np.argmax(out.data, axis=fft_ax)
        max_values = np.max(out.data, axis=fft_ax)
        # since the traces are all the same, the auto-correlation, or zero
        # lag should be maximized.
        assert np.all(np.isclose(argmax, fft_len // 2))
        # the max values should also be near 1 since the patch fixture
        # was normalized.
        assert np.allclose(max_values, 1.0, atol=0.01)
        # the sampling rates should not have changed.
        assert np.isclose(
            to_float(out.get_coord("lag_time").step),
            to_float(corr_patch.get_coord("time").step),
        )
        dstep1 = out.get_coord("distance").step
        dstep2 = corr_patch.get_coord("distance").step
        assert dstep1 == dstep2

    def test_correlation_with_lag(self, corr_patch):
        """Ensure correlation works with a lag specified."""
        lag = 1.9
        out = corr_patch.correlate(distance=0, samples=True, lag=lag)
        coord = out.get_coord("lag_time").values
        assert to_float(coord[0]) >= -lag
        assert to_float(coord[-1]) <= lag

    def test_time_lags(self, ricker_moveout_patch):
        """Ensure time lags are consistent with expected velocities."""
        corr = ricker_moveout_patch.correlate(distance=0)
        # get predicted lat times
        argmax = np.argmax(corr.data, axis=0)
        distances = corr.get_coord("distance").values
        lag_times = to_float(corr.get_coord("lag_time").values[argmax])
        # get calculated times, they should be close to lag times
        expected_times = distances / self.moveout_velocity
        assert np.allclose(lag_times, expected_times)

    def test_units(self, random_patch):
        """Ensure units can be passed as kwarg and lag params."""
        c_patch = random_patch.correlate(distance=10 * m, lag=2 * s)
        assert isinstance(c_patch, dc.Patch)

    def test_complex_patch(self, ricker_moveout_patch):
        """Ensure correlate works with a patch that has complex data."""
        data = ricker_moveout_patch.data
        rm_patch = ricker_moveout_patch.new(data=data + data * 1j)
        corr = rm_patch.correlate(distance=0)
        # get predicted lat times
        argmax = np.argmax(corr.data, axis=0)
        distances = corr.get_coord("distance").values
        lag_times = to_float(corr.get_coord("lag_time").values[argmax])
        # get calculated times, they should be close to lag times
        expected_times = distances / self.moveout_velocity
        assert np.allclose(lag_times, expected_times)
