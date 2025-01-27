"""Tests for plotting a spectrogram."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from dascore.viz.spectrogram import _get_other_dim


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
        with pytest.raises(ValueError, match="should be 'time' or 'frequency'."):
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

    def test_show(self, random_patch, monkeypatch):
        """Ensure show path is callable."""
        monkeypatch.setattr(plt, "show", lambda: None)
        axis = random_patch.viz.spectrogram(dim="time", show=True)
        assert isinstance(axis, plt.Axes)
