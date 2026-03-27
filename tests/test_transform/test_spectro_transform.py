"""Compatibility coverage for spectrogram visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt


class TestSpectroTransform:
    """Tests for the remaining visualization path."""

    def test_viz_spectrogram_still_works(self, random_patch):
        """Visualization should still be able to compute a spectrogram."""
        ax = random_patch.viz.spectrogram(dim="time")
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "time"
        assert "ft_time" in ax.get_ylabel()
        assert not ax.get_title()
        assert ax.images or ax.collections

    def test_viz_spectrogram_1d_patch(self, random_patch):
        """The 1D visualization path should still render content."""
        patch = random_patch.select(distance=0, samples=True).squeeze()
        ax = patch.viz.spectrogram(dim="time")
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "time"
        assert "ft_time" in ax.get_ylabel()
        assert ax.images or ax.collections
