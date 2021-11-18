"""
Tests for decimation
"""

import numpy as np


class TestDecimate:
    """Ensure Patch can be decimated."""

    def test_decimate_no_lowpass(self, random_patch):
        """Simple decimation"""
        p1 = random_patch
        old_time = p1.coords["time"]
        old_dt = old_time[1:] - old_time[:-1]
        # apply decimation,
        pa2 = random_patch.proc.decimate(2, lowpass=False)
        new_time = pa2.coords["time"]
        new_dt = new_time[1:] - new_time[:-1]
        # ensure distance between time samples and shapes have changed
        len_ratio = np.round(len(old_dt) / len(new_dt))
        assert np.isclose(len_ratio, 2.0)
        dt_ratio = np.round(new_dt[0] / old_dt[0])
        assert np.isclose(dt_ratio, 2.0)

    def test_update_time_max(self, random_patch):
        """Ensure the time_max is updated after decimation."""
        out = random_patch.proc.decimate(10, lowpass=False)
        assert out.attrs["time_max"] == out.coords["time"].max()

    def test_update_delta_dim(self, random_patch):
        """
        Since decimate changes the spacing of dimension this should be updated.
        """
        dt1 = random_patch.attrs["d_time"]
        out = random_patch.proc.decimate(10, lowpass=False)
        assert out.attrs["d_time"] == dt1 * 10
