"""Tests for plotting a spectrogram."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest


class TestPlotSpectrogram:
    """Test for basic."""

    @pytest.fixture()
    def spectro_axis(self, random_patch):
        """Return the axis from the spectrogram function."""
        patch = random_patch.aggregate(dim="distance")
        return patch.viz.spectrogram()

    def test_axis_returned(self, spectro_axis):
        """Ensure a matplotlib axis is returned."""
        assert spectro_axis is not None
        assert isinstance(spectro_axis, plt.Axes)

    def test_distance(self, random_patch):
        """Test for doing histogram in distance direction."""
        patch = random_patch.aggregate(dim="time")
        out = patch.viz.spectrogram()
        assert isinstance(out, plt.Axes)

    def test_raises_value_error(self, random_patch):
        """Ensure a ValueError was raised."""
        with pytest.raises(ValueError, match="requires 1D"):
            random_patch.viz.spectrogram()

    def test_1d_patch(self, random_patch):
        """Ensure spectrogram works with 1D patch."""
        out = random_patch.mean("time").squeeze()
        out.viz.spectrogram()

    def test_show(self, random_patch, monkeypatch):
        """Ensure show path is callable."""
        monkeypatch.setattr(plt, "show", lambda: None)
        mean_patch = random_patch.aggregate(dim="time")
        mean_patch.viz.spectrogram(show=True)
