"""
A collection of integration-type tests that don't seem to fit anywhere else.
"""

import pytest

import dascore as dc
from dascore.utils.downloader import fetch


class TestFilterInteractions:
    """Tests for various filters working together."""

    @pytest.fixture
    def train_patch(self):
        """Get an example patch with trains in it."""
        out = dc.spool(fetch("UoU_lf_urban.hdf5"))[0]
        return out

    def test_filter_stft(self, random_patch):
        """Ensure a stft transformed patch can be filtered."""
        patch = random_patch
        stft = patch.stft(time=0.1, overlap=None)
        out = stft.pass_filter(time=(0.01, ...))
        assert isinstance(out, dc.Patch)

    def test_slope_filter_stft(self, train_patch):
        """Tests for filtering velocity anomalies in different bands."""
        # First apply stft.
        stft = train_patch.stft(
            time=3, overlap=0.5, taper_window=("tukey", 0.1), detrend=True
        )
        # Then sum power over 1 to 2 Hz
        banded_patch = (stft.abs() ** 2).select(ft_time=(1, 2)).sum("ft_time").squeeze()
        # Then try to slope filter.
        mph = dc.get_quantity("miles/hour")
        filt = [10 * mph, 30 * mph, 100 * mph, 130 * mph]
        sf = banded_patch.slope_filter(filt)
        assert "slope_filter" in str(sf.attrs.history)
