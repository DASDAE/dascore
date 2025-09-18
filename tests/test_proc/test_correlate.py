"""Tests for correlate patch processing function."""

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import UnitError
from dascore.units import m
from dascore.utils.time import to_float


class TestCorrelateShift:
    """Tests for the correlation shift function."""

    @pytest.fixture(scope="class")
    def random_patch_odd(self):
        """Create a random patch with odd number of time samples."""
        patch = dc.get_example_patch("random_das", shape=(2, 11))
        return patch

    @pytest.fixture(scope="class")
    def random_patch_even(self):
        """Create a random patch with even number of time samples."""
        patch = dc.get_example_patch("random_das", shape=(2, 10))
        return patch

    def test_auto_correlation(self, random_dft_patch):
        """Perform auto correlation and undo shifting."""
        dft_conj = random_dft_patch.conj()
        dft_sq = random_dft_patch * dft_conj
        idft = dft_sq.idft()
        auto_patch = idft.correlate_shift(dim="time")
        assert np.allclose(np.imag(auto_patch.data), 0)
        assert "lag_time" in auto_patch.dims
        coord_array = auto_patch.get_array("lag_time")
        # ensure the max value happens at zero lag time.
        time_ax = auto_patch.get_axis("lag_time")
        argmax = np.argmax(random_dft_patch.data, axis=time_ax)
        assert np.all(coord_array[argmax] == dc.to_timedelta64(0))

    def test_auto_correlation_odd_coord(self, random_patch_odd):
        """Ensure correlate_shift works when dim's coord length is odd."""
        dft = random_patch_odd.dft(dim="time")
        dft_conj = dft.conj()
        dft_sq = dft * dft_conj
        idft = dft_sq.idft()
        assert isinstance(idft.correlate_shift(dim="time"), dc.Patch)

    def test_auto_correlation_even_coord(self, random_patch_even):
        """Ensure correlate_shift works when dim's coord length is even."""
        dft = random_patch_even.dft(dim="time")
        dft_conj = dft.conj()
        dft_sq = dft * dft_conj
        idft = dft_sq.idft()
        assert isinstance(idft.correlate_shift(dim="time"), dc.Patch)


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
        ).taper(time=0.05)
        # normalize energy so autocorrection is 1
        time_axis = patch.get_axis("time")
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
        fft_ax = corr_patch.get_axis("time")
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

    def test_transpose_independent(self, corr_patch):
        """The order of the dims shouldn't affect the result."""
        new_order = corr_patch.dims[::-1]
        patch1 = corr_patch
        patch2 = corr_patch.transpose(*new_order)
        corr1 = patch1.correlate(time=3, samples=True)
        corr2 = patch2.correlate(time=3, samples=True)
        assert corr1.transpose(*corr2.dims).equals(corr2)

    def test_time_lags(self, ricker_moveout_patch):
        """Ensure time lags are consistent with expected velocities."""
        corr = ricker_moveout_patch.correlate(distance=0)
        # get predicted lat times
        argmax = np.argmax(corr.data, axis=0)
        distances = corr.get_coord("distance").values
        lag_times = to_float(corr.get_coord("lag_time").values[argmax])
        # get calculated times, they should be close to lag times
        expected_times = distances / self.moveout_velocity
        assert np.allclose(lag_times.flatten(), expected_times)

    def test_units(self, random_patch):
        """Ensure units can be passed as kwarg params."""
        c_patch = random_patch.correlate(distance=10 * m)
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
        assert np.allclose(lag_times.flatten(), expected_times)

    def test_correlation_freq_domain_patch(self, corr_patch):
        """
        Test correlation when the input patch is already in the frequency
        domain.
        """
        # perform FFT on the original patch to simulate frequency domain data
        fft_patch = corr_patch.dft("time")
        out = fft_patch.correlate(distance=0, samples=True)
        # The patch should still be complex.
        assert np.issubdtype(out.data.dtype, np.complexfloating)
        # Check if the shape of the output is the same as the original patch
        # plus a dimension 1 for the source.
        assert out.shape == tuple([*corr_patch.shape, 1])

    def test_correlate_decimated_patch(self, corr_patch):
        """Ensure a decimated patch can be correlated."""
        out = corr_patch.decimate(distance=2, filter_type=None).correlate(
            distance=1, samples=True
        )
        assert isinstance(out, dc.Patch)

    def test_correlate_units_raises(self, corr_patch):
        """When the patch doesn't have units an error should raise."""
        patch = corr_patch.set_units(distance=None)
        with pytest.raises(UnitError):
            patch.correlate(distance=0 * m)

    def test_correlate_units(self, corr_patch):
        """When the patch has units it should work to specify them."""
        patch = corr_patch.set_units(distance="m")
        out1 = patch.correlate(distance=1 * m)
        assert isinstance(out1, dc.Patch)
        out2 = patch.correlate(distance=np.array([1, 2]) * m)
        assert isinstance(out2, dc.Patch)

    def test_lag_deprecated(self, corr_patch):
        """Ensure the lag parameter is deprecated."""
        with pytest.warns(DeprecationWarning):
            corr_patch.correlate(time=1, lag=10, samples=True)
