"""Tests for signal whiten."""
import numpy as np
import pytest

from dascore import get_example_patch
from dascore.exceptions import ParameterError
from dascore.units import Hz


class TestWhiten:
    """Tests for the whiten module."""

    @pytest.fixture(scope="class")
    def test_patch(self):
        """Return a shot-record used for testing patch."""
        test_patch = get_example_patch("dispersion_event")
        return test_patch.resample(time=(200 * Hz))

    def test_whiten(self, test_patch):
        """Check consistency of test_dispersion module."""
        # assert velocity dimension
        whitened_patch = test_patch.whiten(5, time=(10, 50))
        assert "distance" in whitened_patch.dims
        # assert time dimension
        assert "time" in whitened_patch.dims
        assert np.array_equal(
            test_patch.coords.get_array("time"), whitened_patch.coords.get_array("time")
        )

        assert np.array_equal(
            test_patch.coords.get_array("distance"),
            whitened_patch.coords.get_array("distance"),
        )

    def test_default_whiten_no_input(self, test_patch):
        """Ensure whiten can run without any input."""
        whitened_patch = test_patch.whiten()
        assert "distance" in whitened_patch.dims
        assert "time" in whitened_patch.dims
        assert np.array_equal(
            test_patch.coords.get_array("time"), whitened_patch.coords.get_array("time")
        )
        assert np.array_equal(
            test_patch.coords.get_array("distance"),
            whitened_patch.coords.get_array("distance"),
        )

    def test_default_whiten_no_smoothing_window(self, test_patch):
        """Ensure whiten can run without smoothing window size."""
        whitened_patch = test_patch.whiten(time=(5, 60))
        assert "distance" in whitened_patch.dims
        assert "time" in whitened_patch.dims
        assert np.array_equal(
            test_patch.coords.get_array("time"), whitened_patch.coords.get_array("time")
        )
        assert np.array_equal(
            test_patch.coords.get_array("distance"),
            whitened_patch.coords.get_array("distance"),
        )

    def test_smooth_window_params(self, test_patch):
        """Ensure incorrect values for smooth window raise ParameterError."""
        msg = "Frequency smoothing size must be positive"
        with pytest.raises(ParameterError, match=msg):
            test_patch.whiten(smooth_size=-1, time=[30, 60])

        msg = "Frequency smoothing size is larger than Nyquist"
        with pytest.raises(ParameterError, match=msg):
            test_patch.whiten(smooth_size=110, time=[30, 60])

    def test_default_whiten_no_freq_range(self, test_patch):
        """Ensure whiten can run without frequency range."""
        whitened_patch = test_patch.whiten(smooth_size=10)
        assert "distance" in whitened_patch.dims
        assert "time" in whitened_patch.dims
        assert np.array_equal(
            test_patch.coords.get_array("time"), whitened_patch.coords.get_array("time")
        )
        assert np.array_equal(
            test_patch.coords.get_array("distance"),
            whitened_patch.coords.get_array("distance"),
        )

    def test_edge_whiten(self, test_patch):
        """Ensure whiten can run with edge cases frequency range."""
        whitened_patch = test_patch.whiten(smooth_size=10, time=[0, 50])
        assert "distance" in whitened_patch.dims
        assert "time" in whitened_patch.dims
        assert np.array_equal(
            test_patch.coords.get_array("time"), whitened_patch.coords.get_array("time")
        )
        assert np.array_equal(
            test_patch.coords.get_array("distance"),
            whitened_patch.coords.get_array("distance"),
        )

        whitened_patch = test_patch.whiten(smooth_size=10, time=[50, 100])
        assert "distance" in whitened_patch.dims
        assert "time" in whitened_patch.dims
        assert np.array_equal(
            test_patch.coords.get_array("time"), whitened_patch.coords.get_array("time")
        )
        assert np.array_equal(
            test_patch.coords.get_array("distance"),
            whitened_patch.coords.get_array("distance"),
        )

    def test_short_windows_raises(self, test_patch):
        """Ensure too narrow frequency choices raise ParameterError."""
        msg = "Frequency range is too narrow"
        with pytest.raises(ParameterError, match=msg):
            test_patch.whiten(smooth_size=3, time=[10.02, 10.03])

        msg = "Frequency smoothing size is smaller than default frequency resolution"
        with pytest.raises(ParameterError, match=msg):
            test_patch.whiten(smooth_size=0.001, time=[10, 40])

    def test_longer_smooth_than_range_raises(self, test_patch):
        """Ensure smoothing window larger than
        frequency range raises ParameterError.
        """
        msg = "Frequency smoothing size is larger than frequency range"
        with pytest.raises(ParameterError, match=msg):
            test_patch.whiten(smooth_size=40, time=[10, 40])

    def test_taper_param_raises(self, test_patch):
        """Ensures wrong Tukey alpha parameter raises Paremeter Error."""
        msg = "Tukey alpha needs to be between 0 and 1"
        with pytest.raises(ParameterError, match=msg):
            test_patch.whiten(smooth_size=3, tukey_alpha=-0.1, time=[10, 50])
        with pytest.raises(ParameterError, match=msg):
            test_patch.whiten(smooth_size=3, tukey_alpha=1.1, time=[10, 50])

    def test_whiten_monochromatic_input(self):
        """Ensures correct behavior on monochromatic signal."""
        patch = get_example_patch("sin_wav", frequency=100, sample_rate=500)
        dft_pre = patch.dft("time", real=True)

        white_patch = patch.whiten(smooth_size=5, time=[80, 120])
        dft_post = white_patch.dft("time", real=True)

        # Approx. symmetry for range outside frequency
        ratio_noise = np.median(
            np.abs(dft_post.select(ft_time=(120, 160)).data)
        ) / np.median(np.abs(dft_post.select(ft_time=(40, 80)).data))
        assert 0.5 < ratio_noise < 2

        # Increasing peak-to-average value in smoothing window region, right side
        post_ratio = np.median(
            np.abs(dft_post.select(ft_time=(99, 101)).data)
        ) / np.median(np.abs(dft_post.select(ft_time=(101, 105)).data))
        pre_ratio = np.median(
            np.abs(dft_pre.select(ft_time=(99, 101)).data)
        ) / np.median(np.abs(dft_pre.select(ft_time=(101, 105)).data))
        assert post_ratio / pre_ratio < 0.5

        # Increasing peak-to-average value in smoothing window region, left side
        post_ratio = np.median(
            np.abs(dft_post.select(ft_time=(99, 101)).data)
        ) / np.median(np.abs(dft_post.select(ft_time=(95, 99)).data))
        pre_ratio = np.median(
            np.abs(dft_pre.select(ft_time=(99, 101)).data)
        ) / np.median(np.abs(dft_pre.select(ft_time=(95, 99)).data))
        assert post_ratio / pre_ratio < 0.5

        # Increasing peak-to-average value in frequency range, left side
        post_ratio = np.median(
            np.abs(dft_post.select(ft_time=(99, 101)).data)
        ) / np.median(np.abs(dft_post.select(ft_time=(105, 120)).data))
        pre_ratio = np.median(
            np.abs(dft_pre.select(ft_time=(99, 101)).data)
        ) / np.median(np.abs(dft_pre.select(ft_time=(105, 120)).data))
        assert post_ratio / pre_ratio < 0.1

        # Increasing peak-to-average value in frequency range, right side
        post_ratio = np.median(
            np.abs(dft_post.select(ft_time=(99, 101)).data)
        ) / np.median(np.abs(dft_post.select(ft_time=(80, 95)).data))
        pre_ratio = np.median(
            np.abs(dft_pre.select(ft_time=(99, 101)).data)
        ) / np.median(np.abs(dft_pre.select(ft_time=(80, 95)).data))
        assert post_ratio / pre_ratio < 0.1

    def test_whiten_along_distance(self, test_patch):
        """Ensure whitening runs along other axis."""
        whitened_patch = test_patch.whiten(distance=(0.01, 0.03))
        assert "distance" in whitened_patch.dims
        assert "time" in whitened_patch.dims
        assert np.array_equal(
            test_patch.coords.get_array("time"), whitened_patch.coords.get_array("time")
        )
        assert np.array_equal(
            test_patch.coords.get_array("distance"),
            whitened_patch.coords.get_array("distance"),
        )
