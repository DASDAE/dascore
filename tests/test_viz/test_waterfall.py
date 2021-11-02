"""
Tests for waterfall plots.
"""


class TestWaterfall:
    """Tests for waterfall plot"""

    def test_returns_figures(self, random_das_array):
        """Call waterfall plot, return"""
        out = random_das_array.viz.waterfall()
