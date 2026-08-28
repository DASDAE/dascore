"""Tests for plotting a spectrogram."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dascore.exceptions import ParameterError
from dascore.viz.spectrogram import _get_other_dim, _spectrogram_patch


def test_get_other_dim_valid():
    """Ensure _get_other_dim correctly returns the other dimension."""
    dims = ("time", "distance")
    assert _get_other_dim("time", dims) == "distance"
    assert _get_other_dim("distance", dims) == "time"


def test_get_other_dim_invalid():
    """Ensure _get_other_dim raises a ValueError."""
    dims = ("time", "distance")
    with pytest.raises(ValueError, match="not in patch's dimensions"):
        _get_other_dim("frequency", dims)


def test_get_other_dim_invalid_dim_type():
    """Ensure _get_other_dim raises a TypeError when dim is not a string."""
    dims = ("time", "distance")
    with pytest.raises(TypeError, match="Expected 'dim' to be a string"):
        _get_other_dim(("time",), dims)


class TestPlotSpectrogram:
    """Test for basic."""

    @pytest.fixture()
    def spectro_axis(self, random_patch):
        """Return the axis from the spectrogram function."""
        patch = random_patch.aggregate(dim="distance")
        return patch.viz.spectrogram()

    def test_axis_returned(self, random_patch):
        """Ensure a matplotlib axis is returned."""
        axis = random_patch.viz.spectrogram(dim="time")
        assert axis is not None
        assert isinstance(axis, plt.Axes)

    def test_invalid_dim(self, random_patch):
        """Ensure ValueError is raised for invalid dimensions."""
        with pytest.raises(ValueError, match="not in patch's dimensions"):
            random_patch.viz.spectrogram(dim="frequency")

    def test_aggr_time(self, random_patch):
        """Ensure aggr_domain=time works well."""
        axis = random_patch.viz.spectrogram(aggr_domain="time")
        assert isinstance(axis, plt.Axes)

    def test_aggr_frequency(self, random_patch):
        """Ensure aggr_domain=frequency works well."""
        axis = random_patch.viz.spectrogram(aggr_domain="frequency")
        assert isinstance(axis, plt.Axes)

    def test_invalid_aggr_domain(self, random_patch):
        """Ensure ValueError is raised for invalid aggr_domain."""
        with pytest.raises(ValueError, match=r"should be 'time' or 'frequency'."):
            random_patch.viz.spectrogram(aggr_domain="invalid")

    def test_invalid_patch_dims(self, random_patch):
        """Ensure ValueError is raised for patches with invalid dimensions."""
        patch_3d = random_patch.correlate(distance=[0, 1])
        with pytest.raises(
            ValueError, match="Can only make spectrogram of 1D or 2D patches"
        ):
            patch_3d.viz.spectrogram(dim="distance")

    def test_1d_patch(self, random_patch):
        """Ensure spectrogram works with 1D patch."""
        patch = random_patch.select(distance=0, samples=True).squeeze()
        axis = patch.viz.spectrogram(dim="time")
        assert isinstance(axis, plt.Axes)

    @pytest.mark.parametrize("aggr_domain", ["time", "frequency"])
    def test_length_one_other_dim(self, random_patch, aggr_domain):
        """A single channel still has a spectrogram; there is just no mean."""
        patch = random_patch.select(distance=(0, 1), samples=True)
        assert patch.ndim == 2, "the length one dimension should be kept"
        axis = patch.viz.spectrogram(dim="time", aggr_domain=aggr_domain)
        assert isinstance(axis, plt.Axes)

    def test_show(self, random_patch, shown):
        """Ensure show path is callable."""
        axis = random_patch.viz.spectrogram(dim="time", show=True)
        assert isinstance(axis, plt.Axes)
        axis = random_patch.viz.spectrogram(dim="time", show=True)
        assert isinstance(axis, plt.Axes)
        assert shown

    @staticmethod
    def _image_shape(axis):
        """Return the shape of the data backing the spectrogram image."""
        images = axis.get_images()
        if images:
            return images[0].get_array().shape
        return axis.collections[0].get_array().shape

    @pytest.mark.parametrize("aggr_domain", ["frequency", "time"])
    def test_window_reaches_stft_2d(self, random_patch, aggr_domain):
        """The window is stft's, and changes the picture. See #661."""
        default = random_patch.viz.spectrogram(dim="time", aggr_domain=aggr_domain)
        windowed = random_patch.viz.spectrogram(
            dim="time", aggr_domain=aggr_domain, time=64, samples=True
        )
        assert self._image_shape(default) != self._image_shape(windowed)

    def test_window_reaches_stft_1d(self, random_patch):
        """And for a single trace. See #661."""
        patch = random_patch.select(distance=0, samples=True).squeeze()
        default = patch.viz.spectrogram(dim="time")
        windowed = patch.viz.spectrogram(dim="time", time=64, samples=True)
        assert self._image_shape(default) != self._image_shape(windowed)

    def test_nfft_adds_frequency_rows(self, random_patch):
        """A longer FFT draws more frequency rows for the same window."""
        plain = random_patch.viz.spectrogram(time=64, samples=True)
        padded = random_patch.viz.spectrogram(time=64, samples=True, nfft=256)
        assert self._image_shape(padded)[0] > self._image_shape(plain)[0]

    def test_scipy_spelling_refused(self, random_patch):
        """Scipy's nperseg is not a dimension; the window is time=..."""
        with pytest.raises(ParameterError, match="give the window as time="):
            random_patch.viz.spectrogram(nperseg=64)

    @pytest.mark.parametrize("aggr_domain", ["frequency", "time"])
    def test_is_stft_squared(self, random_patch, aggr_domain):
        """What is drawn is |stft|² with the other dimension averaged."""
        power = _spectrogram_patch(
            random_patch, "time", aggr_domain, time=64, samples=True
        )
        if aggr_domain == "time":
            averaged = random_patch.aggregate("distance", method="mean")
            expected = averaged.stft(time=64, samples=True).abs() ** 2
        else:
            power_2d = random_patch.stft(time=64, samples=True).abs() ** 2
            expected = power_2d.aggregate("distance", method="mean")
        expected = expected.squeeze()
        assert np.allclose(power.data, expected.data)
