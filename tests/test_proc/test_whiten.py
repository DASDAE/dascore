"""Tests for signal whiten."""

import warnings

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

import dascore as dc
from dascore import get_example_patch
from dascore.exceptions import ParameterError
from dascore.proc.whiten import Whiten
from dascore.units import Hz
from dascore.warnings import NumpyFallbackWarning


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
        assert np.iscomplexobj(whitened_patch_freq_domain.data), (
            "Expected the output to be complex, indicating freq. domain patch."
        )

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


class TestWhitenMetadata:
    """Whiten works out its result from metadata alone."""

    def test_single_precision(self, random_patch):
        """Single precision data come back single, as the transform keeps them."""
        patch = random_patch.new(data=np.asarray(random_patch.data, np.float32))
        out, _ = Whiten(smooth_size=5).get_metadata(patch.drop_data())
        assert out.shape == (300, 2000)
        assert out.dtype == patch.whiten(smooth_size=5).dtype == np.float32

    @pytest.mark.parametrize("band", [{}, {"time": (10, 40)}])
    @pytest.mark.parametrize("transformed", [False, True])
    @pytest.mark.parametrize("smooth_size", [None, 5])
    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float64, np.complex64, np.complex128]
    )
    def test_metadata_dtype_is_the_kernels(
        self, random_patch, dtype, transformed, smooth_size, band
    ):
        """Metadata state the dtype whiten's data come out in, band or not."""
        patch = random_patch.isel(distance=slice(0, 20), time=slice(0, 256))
        patch = patch.dft("time") if transformed else patch
        data = np.asarray(patch.data)
        # A real spectrum, such as amplitudes, keeps only the real part.
        data = data if np.dtype(dtype).kind == "c" else data.real
        patch = patch.new(data=data.astype(dtype))
        out, _ = Whiten(smooth_size=smooth_size, **band).get_metadata(patch.drop_data())
        whitened = patch.whiten(smooth_size=smooth_size, **band)
        assert out.dtype == whitened.dtype
        assert np.finfo(out.dtype).bits == np.finfo(dtype).bits

    def test_water_level_raises_the_floor(self, random_patch):
        """The smoothed spectrum is floored at the water level times its peak."""
        floored = random_patch.whiten(smooth_size=5, water_level=0.5)
        spectrum = random_patch.dft("time", real=True)
        axis = spectrum.get_axis("ft_time")
        window = spectrum.get_coord("ft_time").get_sample_count(5)
        amp = np.abs(spectrum.data)
        smooth = uniform_filter1d(amp, window, axis=axis, mode="wrap")
        smooth = np.maximum(smooth, 0.5 * smooth.max())
        flat = amp / smooth * np.exp(1j * np.angle(spectrum.data))
        expected = spectrum.new(data=flat).idft()
        assert np.allclose(floored.data, expected.data)


class TestWhitenOnDask:
    """Whiten runs on dask natively, never through numpy."""

    @pytest.mark.parametrize("transformed", [False, True])
    @pytest.mark.parametrize(
        "kwargs", [{}, {"smooth_size": 5, "water_level": 0.5, "time": (10, 40)}]
    )
    def test_native(self, random_patch, transformed, kwargs):
        """Dask data come back dask, with no fallback warning, and numpy's values."""
        da = pytest.importorskip("dask.array")
        patch = random_patch.dft("time", real=True) if transformed else random_patch
        lazy = patch.new(data=da.from_array(patch.data, chunks=(100, -1)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            out = lazy.whiten(**kwargs)
        assert isinstance(out.data, da.Array)
        assert np.allclose(out.data.compute(), patch.whiten(**kwargs).data)

    def test_chunked_along_frequency(self, random_patch):
        """Smoothing a spectrum chunked along frequency runs natively."""
        da = pytest.importorskip("dask.array")
        patch = random_patch.dft("time", real=True)
        lazy = patch.new(data=da.from_array(patch.data, chunks=(-1, 500)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            out = lazy.whiten(smooth_size=5)
        assert isinstance(out.data, da.Array)
        assert np.allclose(out.data.compute(), patch.whiten(smooth_size=5).data)
