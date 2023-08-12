"""Tests for wiggle plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class TestWiggle:
    """Tests for wiggle plot."""

    def test_returns_axes(self, random_patch):
        """Call waterfall plot, return."""
        # modify patch to include line at start
        data = np.array(random_patch.data)
        data[:100, :100] = 2.0  # create an origin block for testing axis line up
        data[:100, -100:] = -2.0  #
        out = random_patch.new(data=data)
        ax = out.viz.wiggle()
        # check labels
        assert random_patch.dims[0].lower() in ax.get_ylabel().lower()
        assert random_patch.dims[1].lower() in ax.get_xlabel().lower()
        assert isinstance(ax, plt.Axes)

    def test_example(self):
        """Test the example from the docs."""
        import dascore as dc

        patch = dc.examples._sin_wave_patch(
            sample_rate=1000,
            frequency=[200, 10],
            channel_count=2,
        )
        _ = patch.viz.wiggle()
