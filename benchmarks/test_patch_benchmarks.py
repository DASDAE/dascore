"""Benchmarks for patch functions using pytest-codspeed."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.utils.patch import get_start_stop_step


@pytest.fixture
def example_patch():
    """Get the example patch for benchmarks."""
    return dc.get_example_patch()


class TestProcessingBenchmarks:
    """Benchmarks for patch processing operations."""

    @pytest.mark.benchmark
    def test_pass_filter_performance(self, example_patch):
        """Time the pass filter."""
        patch = example_patch
        patch.pass_filter(distance=(0.1, 0.2))
        patch.pass_filter(time=(10.2, None))
        patch.pass_filter(time=(None, 100.22))
        patch.pass_filter(time=(10, 100))

    @pytest.mark.benchmark
    def test_median_filter_performance(self, example_patch):
        """Time the median filter."""
        patch = example_patch
        patch.median_filter(distance=5, time=5, samples=True)
        patch.median_filter(time=5, samples=True)

    @pytest.mark.benchmark
    def test_resample_performance(self, example_patch):
        """Time resample operations."""
        patch = example_patch
        # upsample time
        start, stop, step = get_start_stop_step(patch, "time")
        patch.interpolate(time=np.arange(start, stop, step / 2))
        # up sample distance
        start, stop, step = get_start_stop_step(patch, "distance")
        new_coord = np.arange(start, stop, step / 2.2)
        patch.interpolate(distance=new_coord)

    @pytest.mark.benchmark
    def test_decimate_performance(self, example_patch):
        """Timing decimate."""
        patch = example_patch
        patch.decimate(time=2)
        patch.decimate(time=10, filter_type="iir")
        patch.decimate(time=10, filter_type="fir")
        patch.decimate(time=10, filter_type=None)

    @pytest.mark.benchmark
    def test_select_performance(self, example_patch):
        """Timing select."""
        patch = example_patch
        patch.select(distance=(100, 200))
        t1 = patch.attrs["time_min"] + np.timedelta64(1, "s")
        t2 = t1 + np.timedelta64(3, "s")
        patch.select(time=(None, t1))
        patch.select(time=(t1, None))
        patch.select(time=(t1, t2))


class TestTransformBenchmarks:
    """Benchmarks for patch transform operations."""

    @pytest.fixture
    def dft_patch(self, example_patch):
        """Get DFT patch for benchmarks."""
        return example_patch.tran.dft("time")

    @pytest.mark.benchmark
    def test_indefinite_integrate_performance(self, example_patch):
        """Integrate along time axis."""
        example_patch.integrate(dim="time", definite=False)

    @pytest.mark.benchmark
    def test_definite_integrate_performance(self, example_patch):
        """Integrate along time axis."""
        example_patch.integrate(dim="time", definite=True)

    @pytest.mark.benchmark
    def test_differentiate_performance(self, example_patch):
        """Differentiate along time axis."""
        example_patch.differentiate(dim="time")

    @pytest.mark.benchmark
    def test_dft_performance(self, example_patch):
        """The discrete fourier transform."""
        example_patch.dft(dim="time")

    @pytest.mark.benchmark
    def test_idft_performance(self, dft_patch):
        """The inverse of the fourier transform."""
        dft_patch.idft()


class TestVisualizationBenchmarks:
    """Benchmarks for patch visualization operations."""

    @pytest.mark.benchmark
    def test_waterfall_performance(self, example_patch):
        """Timing for waterfall patch."""
        example_patch.viz.waterfall()
        # Clean up matplotlib figures
        import matplotlib.pyplot as plt

        plt.close("all")

    @pytest.mark.benchmark
    def test_str_performance(self, example_patch):
        """Timing for getting str rep."""
        str(example_patch)
