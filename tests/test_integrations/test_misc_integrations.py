"""
A collection of integration-type tests that don't seem to fit anywhere else.
"""

import numpy as np
import pytest

import dascore as dc
from dascore.utils.downloader import fetch


@pytest.fixture(scope="module")
def train_patch():
    """Get an example patch with trains in it."""
    out = dc.spool(fetch("UoU_lf_urban.hdf5"))[0]
    return out


class TestFilterInteractions:
    """Tests for various filters working together."""

    def test_filter_stft(self, random_patch):
        """Ensure a stft transformed patch can be filtered."""
        patch = random_patch
        stft_patch = patch.stft(time=0.1, overlap=None)
        out = stft_patch.pass_filter(time=(0.01, ...))
        assert isinstance(out, dc.Patch)
        assert out.dims == stft_patch.dims

    def test_slope_filter_stft(self, train_patch):
        """Tests for filtering velocity anomalies in different bands."""
        # First apply stft.
        stft_patch = train_patch.stft(
            time=3, overlap=0.5, taper_window=("tukey", 0.1), detrend=True
        )
        # Then sum power over 1 to 2 Hz
        banded_patch = (
            (stft_patch.abs() ** 2).select(ft_time=(1, 2)).sum("ft_time").squeeze()
        )
        # Then try to slope filter.
        mph = dc.get_quantity("miles/hour")
        filt = [10 * mph, 30 * mph, 100 * mph, 130 * mph]
        sf = banded_patch.slope_filter(filt)
        assert "slope_filter" in str(sf.attrs.history)
        assert sf.data.shape == banded_patch.data.shape
        assert sf.dims == banded_patch.dims
        assert "slope_filter" in str(sf.attrs.history)


class TestAlignCoordCorrelation:
    """
    Test that shifting can be undone when forward aligned before correlation
    and reversed after correlation.
    """

    velocity = 100  # m/s

    @pytest.fixture(scope="class")
    def ricker_patch(self):
        """Get a ricker patch with a shift coordinate."""
        patch = dc.get_example_patch(
            "ricker_moveout",
            velocity=self.velocity,
            duration=4,
            channel_count=10,
            peak_time=1.5,
        )

        shift = -patch.get_coord("distance").values / self.velocity
        return patch.update_coords(shift=("distance", shift))

    @pytest.fixture(scope="class")
    def corr_no_shift(self, ricker_patch):
        """Perform the correlation without a shift."""
        return ricker_patch.correlate(distance=0)

    @pytest.fixture(scope="class")
    def corr_shift(self, ricker_patch):
        """Perform the correlation without a shift, undo shift after."""
        out = (
            ricker_patch.align_to_coord(time="shift")
            .dropna("time")
            .correlate(distance=0)
            .align_to_coord(lag_time="shift", reverse=True)
            .dropna("lag_time")
        )
        return out

    def test_correlation_equivalent(self, corr_no_shift, corr_shift):
        """Ensure the correlation is equivalent before/after shift."""
        pa1 = corr_no_shift.select(lag_time=(-0.75, 1.5001))
        pa2 = corr_shift.select(lag_time=(-0.75, 1.5001))
        # The patches should now be the same size and have the same values.
        assert pa1.shape == pa2.shape
        assert np.allclose(pa1.data, pa2.data)
