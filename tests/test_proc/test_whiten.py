"""Tests for signal whiten."""

import numpy as np
import pytest

import dascore as dc
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

    def test_bad_water_level_raises(self, test_patch):
        """Ensure bad water level values raise ParameterError."""
        msg = "water_level must be a float"

        with pytest.raises(ParameterError, match=msg):
            test_patch.whiten(water_level=[1, 2, 3], smooth_size=10)
        with pytest.raises(ParameterError, match=msg):
            test_patch.whiten(water_level=np.array([1, 2, 3]), smooth_size=10)
        with pytest.raises(ParameterError, match=msg):
            test_patch.whiten(water_level=-0.1, smooth_size=10)

    def test_whiten_monochromatic_input(self):
        """Ensures correct behavior on monochromatic signal."""
        patch = get_example_patch("sin_wav", frequency=100, sample_rate=500)
        dft_pre = patch.dft("time", real=True)

        import dascore as dc

        dc._bob = True

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
        whitened_patch = test_patch.whiten(distance=(0.001, 0.03))
        assert "distance" in whitened_patch.dims
        assert "time" in whitened_patch.dims
        assert np.array_equal(
            test_patch.coords.get_array("time"), whitened_patch.coords.get_array("time")
        )
        assert np.array_equal(
            test_patch.coords.get_array("distance"),
            whitened_patch.coords.get_array("distance"),
        )

    def test_whiten_dft_input(self, test_patch):
        """
        Ensure whiten function returns dft patch when dft patch is input.
        """
        dft = test_patch.dft("time", real=True)
        whitened_patch_freq_domain = dft.whiten(smooth_size=5, time=None)

        # check if the returned data is in the frequency domain
        assert np.iscomplexobj(
            whitened_patch_freq_domain.data
        ), "Expected the output to be complex, indicating freq. domain patch."

        assert "ft_time" in dft.coords.coord_map

    def test_whiten_df_all_parameters(self, test_patch):
        """Ensure whiten accepts all args in dft form."""
        fft_patch = test_patch.dft("time", real=True)

        whitened_patch_freq_domain = fft_patch.whiten(
            smooth_size=5, time=(30, 60), water_level=0.01
        )
        assert isinstance(whitened_patch_freq_domain, dc.Patch)

    def test_no_time_no_kwargs_raises(self, random_patch):
        """Ensure if no kwargs and patch doesn't have time an error is raised."""
        patch = random_patch.rename_coords(time="money")
        msg = "and patch has no time dimension"
        with pytest.raises(ParameterError, match=msg):
            patch.whiten()

    def test_bad_dim_name_in_kwargs_raises(self, random_patch):
        """Ensure a bad dimension name raises."""
        msg = "whiten but it is not in patch dimensions"
        with pytest.raises(ParameterError, match=msg):
            random_patch.whiten(bad_dim=(1, 10))

    def test_multiple_kwargs_raises(self, random_patch):
        """Ensure passing multiple kwargs raises error."""
        msg = "must specify a single patch dimension"
        with pytest.raises(ParameterError, match=msg):
            random_patch.whiten(time=None, distance=None)

    def test_helpful_error_message(self, random_patch):
        """Ensure helpful error message is used when a bad kwarg is passed."""
