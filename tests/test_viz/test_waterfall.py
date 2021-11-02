"""
Tests for waterfall plots.
"""
import matplotlib.pyplot as plt


class TestWaterfall:
    """Tests for waterfall plot"""

    def test_returns_figures(self, random_patch):
        """Call waterfall plot, return"""
        # modify patch to include line at start
        data = random_patch.data
        data[:100, :100] = 2.5  # create an origin block for testing axis line up
        out = random_patch.new(data=data)
        ax = out.viz.waterfall()
        assert isinstance(ax, plt.Axes)
