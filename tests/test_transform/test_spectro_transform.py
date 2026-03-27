"""Compatibility coverage for spectrogram visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt


class TestSpectroTransform:
    """Tests for the remaining visualization path."""

    def test_viz_spectrogram_still_works(self, random_patch):
        """Visualization should still be able to compute a spectrogram."""
        ax = random_patch.viz.spectrogram(dim="time")
        assert isinstance(ax, plt.Axes)
