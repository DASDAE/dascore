"""
Benchmarks for patch functions.
"""

import numpy as np

import dascore as dc
from dascore.utils.patch import get_start_stop_step


class ProcessingSuite:
    """
    Suite for processing functions.
    """

    patch: dc.Patch

    def setup(self):
        """just load the default patch."""
        self.patch = dc.get_example_patch()

    def time_pass_filter(self):
        """Time the pass filter."""
        self.patch.pass_filter(distance=(0.1, 0.2))
        self.patch.pass_filter(time=(10.2, None))
        self.patch.pass_filter(time=(None, 100.22))
        self.patch.pass_filter(time=(10, 100))

    def time_median_filter(self):
        """Time the median filter"""
        self.patch.median_filter(kernel_size=(5, 3))
        self.patch.median_filter(kernel_size=5)

    def time_resample(self):
        """Time resample operations."""
        # upsample time
        start, stop, step = get_start_stop_step(self.patch, "time")
        self.patch.interpolate(time=np.arange(start, stop, step / 2))
        # up sample distance
        start, stop, step = get_start_stop_step(self.patch, "distance")
        new_coord = np.arange(start, stop, step / 2.2)
        self.patch.interpolate(distance=new_coord)

    def time_decimate(self):
        """Timing decimate."""
        self.patch.decimate(time=2)
        self.patch.decimate(time=10, filter_type="iir")
        self.patch.decimate(time=10, filter_type="fir")
        self.patch.decimate(time=10, filter_type=None)

    def time_select(self):
        """timing select."""
        self.patch.select(distance=(100, 200))
        t1 = self.patch.attrs["time_min"] + np.timedelta64(1, "s")
        t2 = t1 + np.timedelta64(3, "s")
        self.patch.select(time=(None, t1))
        self.patch.select(time=(t1, None))
        self.patch.select(time=(t1, t2))


class TransformSuite:
    """Timing for various transformations."""

    def setup(self):
        """just load the default patch."""
        self.patch = dc.get_example_patch()

    def time_indefinite_integrate(self):
        """integrate along time axis."""
        self.patch.tran.integrate(dim="time", defininte=False)

    def time_definite_integrate(self):
        """integrate along time axis."""
        self.patch.tran.integrate(dim="time", defininte=True)

    def time_differentiate(self):
        """differentiate along time axis"""
        self.patch.tran.differentiate(dim="time")


class VizSuite:
    """Timing for visualizations."""

    patch: dc.Patch

    def setup(self):
        """just load the default patch."""
        self.patch = dc.get_example_patch()

    def teardown(self):
        """just load the default patch."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def time_waterfall(self):
        """Timing for waterfall patch."""
        patch = self.patch
        patch.viz.waterfall()

    def time_str(self):
        """Timing for getting str rep."""
        str(self.patch)
