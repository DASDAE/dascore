"""
Tests for waterfall plots.
"""
import matplotlib.pyplot as plt


class TestWaterfall:
    """Tests for waterfall plot"""

    def test_returns_axes(self, random_patch):
        """Call waterfall plot, return"""
        # modify patch to include line at start
        data = random_patch.data
        data[:100, :100] = 2.0  # create an origin block for testing axis line up
        data[:100, -100:] = -2.0  #
        out = random_patch.new(data=data)
        ax = out.viz.waterfall()
        # check labels
        assert random_patch.dims[0].capitalize() in ax.get_ylabel()
        assert random_patch.dims[1].capitalize() in ax.get_xlabel()
        assert isinstance(ax, plt.Axes)

    def test_colorbar_scale(self, random_patch):
        """Tests for the scaling parameter."""
        ax_scalar = random_patch.viz.waterfall(scale=0.2)
        assert ax_scalar is not None
        seq_scalar = random_patch.viz.waterfall(scale=[0.1, 0.3])
        assert seq_scalar is not None

    def test_colorbar_absolute_scale(self, random_patch):
        """Tests for absolute scaling of colorbar."""
        patch = random_patch.new(data=random_patch.data * 100 - 50)
        ax1 = patch.viz.waterfall(scale_type="absolute", scale=(-50, 50))
        assert ax1 is not None
        ax2 = patch.viz.waterfall(scale_type="absolute", scale=10)
        assert ax2 is not None
