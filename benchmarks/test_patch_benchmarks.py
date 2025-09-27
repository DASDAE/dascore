"""Benchmarks for patch functions using pytest-codspeed."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import dascore as dc
from dascore.utils.patch import get_start_stop_step


@pytest.fixture(scope="module")
def example_patch():
    """Get the example patch for benchmarks."""
    return dc.get_example_patch()


@pytest.fixture(scope="module")
def patch_uneven_time():
    """Get a patch with uneven time coord."""
    patch = dc.get_example_patch()
    time = patch.get_coord("time")
    rand = np.random.RandomState(39)
    new_vales = rand.random(len(time))
    return patch.update_coords(time=new_vales)


@pytest.fixture()
def cleanup_mpl():
    """Close all open matplotlib figures after test."""
    yield
    plt.close("all")


class TestProcessingBenchmarks:
    """Benchmarks for patch processing operations."""

    @pytest.fixture(scope="module")
    def interp_time(self, example_patch):
        """Get an array for interpolation."""
        # This is a fixture as to not affect the timing.
        patch = example_patch
        # upsample time
        start, stop, step = get_start_stop_step(patch, "time")
        step = dc.to_timedelta64(dc.to_float(step) / 2)
        return np.arange(start, stop, step)

    @pytest.mark.benchmark
    def test_pass_filter(self, example_patch):
        """Time the pass filter."""
        patch = example_patch
        patch.pass_filter(distance=(0.1, 0.2))
        patch.pass_filter(time=(10.2, None))
        patch.pass_filter(time=(None, 100.22))
        patch.pass_filter(time=(10, 100))

    @pytest.mark.benchmark
    def test_median_filter(self, example_patch):
        """Time the median filter."""
        patch = example_patch
        patch.median_filter(distance=5, time=5, samples=True)
        patch.median_filter(time=5, samples=True)

    @pytest.mark.benchmark
    def test_interpolate(self, example_patch, interp_time):
        """Time interpolate operations."""
        patch = example_patch
        patch.interpolate(time=interp_time)

    @pytest.mark.benchmark
    def test_decimate(self, example_patch):
        """Time decimation."""
        patch = example_patch
        patch.decimate(time=2)
        patch.decimate(time=10, filter_type="iir")
        patch.decimate(time=10, filter_type="fir")
        patch.decimate(time=10, filter_type=None)

    @pytest.mark.benchmark
    def test_select(self, example_patch):
        """Selecting on time/distance dimension"""
        patch = example_patch
        patch.select(distance=(100, 200))
        t1 = patch.attrs["time_min"] + np.timedelta64(1, "s")
        t2 = t1 + np.timedelta64(3, "s")
        patch.select(time=(None, t1))
        patch.select(time=(t1, None))
        patch.select(time=(t1, t2))

    @pytest.mark.benchmark
    def test_sobel_filter(self, example_patch):
        """Time the Sobel filter."""
        patch = example_patch
        patch.sobel_filter(dim="time")

    @pytest.mark.benchmark
    def test_standardize(self, example_patch):
        """Time standardization operation."""
        patch = example_patch
        patch.standardize(dim="time")

    @pytest.mark.benchmark
    def test_taper(self, example_patch):
        """Time tapering operations."""
        patch = example_patch
        patch.taper(time=0.1)

    @pytest.mark.benchmark
    def test_transpose(self, example_patch):
        """Time transpose operations."""
        patch = example_patch
        dims = patch.dims[::-1]
        patch.transpose(*dims)

    @pytest.mark.benchmark
    def test_roll(self, example_patch):
        """Time roll/shift operations."""
        patch = example_patch
        patch.roll(time=10, samples=True)

    @pytest.mark.benchmark
    def test_snap_coords(self, patch_uneven_time):
        """Time coordinate snapping."""
        patch = patch_uneven_time
        patch.snap_coords("time")

    @pytest.mark.benchmark
    def test_hampel_filter_non_approximate(self, example_patch):
        """Time the Hampel filter."""
        example_patch.hampel_filter(
            threshold=3.0, distance=3, time=5, samples=True, approximate=False
        )

    @pytest.mark.benchmark
    def test_hampel_filter(self, example_patch):
        """Time the Hampel filter."""
        example_patch.hampel_filter(
            threshold=3.0, distance=3, time=5, samples=True, approximate=True
        )

    @pytest.mark.benchmark
    def test_wiener_filter(self, example_patch):
        """Time the Wiener filter."""
        patch = example_patch
        patch.wiener_filter(time=3, samples=True)


class TestTransformBenchmarks:
    """Benchmarks for patch transform operations."""

    @pytest.fixture
    def dft_patch(self, example_patch):
        """Get DFT patch for benchmarks."""
        return example_patch.dft("time")

    @pytest.mark.benchmark
    def test_indefinite_integrate(self, example_patch):
        """Integrate along time axis."""
        example_patch.integrate(dim="time", definite=False)

    @pytest.mark.benchmark
    def test_definite_integrate(self, example_patch):
        """Integrate along time axis."""
        example_patch.integrate(dim="time", definite=True)

    @pytest.mark.benchmark
    def test_differentiate(self, example_patch):
        """Differentiate along time axis."""
        example_patch.differentiate(dim="time")

    @pytest.mark.benchmark
    def test_dft(self, example_patch):
        """The discrete fourier transform."""
        example_patch.dft(dim="time")

    @pytest.mark.benchmark
    def test_idft(self, dft_patch):
        """The inverse of the fourier transform."""
        dft_patch.idft()

    @pytest.mark.benchmark
    def test_stft(self, example_patch):
        """Time short time fourier transform transform."""
        patch = example_patch
        patch.stft(time=1, overlap=0.25)

    @pytest.mark.benchmark
    def test_hilbert(self, example_patch):
        """Time Hilbert transform."""
        patch = example_patch
        patch.hilbert(dim="time")

    @pytest.mark.benchmark
    def test_envelope(self, example_patch):
        """Time envelope calculation."""
        patch = example_patch
        patch.envelope(dim="time")


class TestVisualizationBenchmarks:
    """Benchmarks for patch visualization operations (or str repr)."""

    @pytest.mark.benchmark
    @pytest.mark.usefixtures("cleanup_mpl")
    def test_waterfall(self, example_patch):
        """Timing for waterfall patch."""
        example_patch.viz.waterfall()

    @pytest.mark.benchmark
    def test_str(self, example_patch):
        """Timing for getting str rep."""
        str(example_patch)

    @pytest.mark.benchmark
    def test_repr(self, example_patch):
        """Time representation generation."""
        repr(example_patch)

    @pytest.mark.usefixtures("cleanup_mpl")
    @pytest.mark.benchmark
    def test_wiggle(self, example_patch):
        """Time wiggle plot visualization."""
        patch = example_patch.select(distance=(0, 100))  # Subset for performance
        patch.viz.wiggle()


class TestAggregationBenchmarks:
    """Benchmarks for patch aggregation operations."""

    @pytest.mark.benchmark
    def test_mean(self, example_patch):
        """Time mean aggregation."""
        patch = example_patch
        patch.mean(dim="time")
        patch.mean(dim="distance")

    @pytest.mark.benchmark
    def test_max(self, example_patch):
        """Time max aggregation."""
        patch = example_patch
        patch.max(dim="time")
        patch.max(dim="distance")

    @pytest.mark.benchmark
    def test_min(self, example_patch):
        """Time min aggregation."""
        patch = example_patch
        patch.min(dim="time")
        patch.min(dim="distance")

    @pytest.mark.benchmark
    def test_std(self, example_patch):
        """Time standard deviation aggregation."""
        patch = example_patch
        patch.std(dim="time")
        patch.std(dim="distance")

    @pytest.mark.benchmark
    def test_sum(self, example_patch):
        """Time sum aggregation."""
        patch = example_patch
        patch.sum(dim="time")
        patch.sum(dim="distance")

    @pytest.mark.benchmark
    def test_median(self, example_patch):
        """Time median aggregation."""
        patch = example_patch
        patch.median(dim="time")
        patch.median(dim="distance")

    @pytest.mark.benchmark
    def test_first(self, example_patch):
        """Time first aggregation."""
        patch = example_patch
        patch.first(dim="time")

    @pytest.mark.benchmark
    def test_last(self, example_patch):
        """Time last aggregation."""
        patch = example_patch
        patch.last(dim="distance")


class TestRollingBenchmarks:
    """Benchmarks for rolling window operations."""

    @pytest.fixture(scope="module")
    def small_roller(self, example_patch):
        """Get a rolling object"""
        return example_patch.rolling(time=5, samples=True)

    @pytest.fixture(scope="module")
    def big_roller(self, example_patch):
        """Get a large rolling object"""
        patch = example_patch
        time = patch.get_coord("time")
        roll_time = dc.to_float(time.coord_range()) / 4
        return example_patch.rolling(time=roll_time)

    @pytest.mark.benchmark
    def test_rolling_small_roller_mean(self, small_roller):
        """Time rolling mean calculation for small roller."""
        small_roller.mean()

    @pytest.mark.benchmark
    def test_rolling_large_roller_mean(self, big_roller):
        """Time rolling mean calculation."""
        big_roller.mean()
