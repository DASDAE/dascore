"""Tests for some doc examples which had problems at one point."""

from __future__ import annotations

import matplotlib.pyplot as plt

import dascore as dc


class TestQuickStart:
    """A few examples from the quickstart."""

    def test_filter_plot(self):
        """Test get, taper, filter, plot."""
        patch = (
            dc.get_example_patch("example_event_1")
            .taper(time=0.05)
            .pass_filter(time=(None, 300))
        )
        ax = patch.viz.waterfall(scale=0.2)
        assert isinstance(ax, plt.Axes)
