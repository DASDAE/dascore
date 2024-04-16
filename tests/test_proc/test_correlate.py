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

    def test_correlation_freq_domain_patch(self, corr_patch):
        """
        Test correlation when the input patch is already in the frequency domain.
        """
        # perform FFT on the original patch to simulate frequency domain data
        fft_patch = corr_patch.dft("time")
        out = fft_patch.correlate(distance=0, samples=True)
        # check if the data is real, as a simple proxy for being in the time domain
        assert np.isrealobj(
            out.data
        ), "Expected the output data to be real, indicating time domain."
        # check if the shape of the output is the same as the original patch
        # (since no "lag" argument is used)
        assert (
            out.seconds == corr_patch.seconds
        ), "Expected size of the output time coordinate to match the original patch"
        assert (
            out.channel_count == corr_patch.channel_count
        ), "Expected size of the output distance coordinate to match the original patch"

    def test_correlation_ifft_false(self, corr_patch):
        """
        Ensure correlate function can return the result in the frequency domain
        when the ifft flag is set to Flase.
        """
        # return correlation in frequency domain
        correlated_patch_freq_domain = corr_patch.correlate(
            distance=2, samples=True, ifft=False
        )

        # check if the returned data is in the frequency domain
        assert np.iscomplexobj(
            correlated_patch_freq_domain.data
        ), "Expected the output to be complex, indicating freq. domain representation."
        # need to add asserts to test coords and attrs

    def test_correlation_with_step(self, corr_patch):
        """
        Ensure the correlation function properly skips rows/columns according to
        the step argument.
        """
        step_size = 2
        out = corr_patch.correlate(distance=0, samples=True, step_size=step_size)
        # Verify that only the correct indices have been considered in the output.
        # Since `samples=True` and `distance=0`, we expect the first row/column to
        # be used as the master channel.
        input_shape = corr_patch.data.shape
        dist_axis = corr_patch.dims.index("distance")
        expected_indices = range(0, input_shape[dist_axis], step_size)
        # check if the output data shape matches the expected number of correlations
        expected_shape = len(expected_indices)
        assert (
            out.data.shape[dist_axis] == expected_shape
        ), f"Expected shape {expected_shape}, but got {out.data.shape[dist_axis]}"
