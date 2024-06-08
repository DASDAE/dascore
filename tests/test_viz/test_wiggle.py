"""Tests for wiggle plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import dascore as dc


class TestWiggle:
    """Tests for wiggle plot."""

    @pytest.fixture()
    def small_patch(self, random_patch):
        """A small patch to cut back on plot time."""
        pa = random_patch.select(distance=(10, 15), samples=True)
        return pa

    def test_example(self):
        """Test the example from the docs."""
        patch = dc.examples.sin_wave_patch(
            sample_rate=1000,
            frequency=[200, 10],
            channel_count=2,
        )
        _ = patch.viz.wiggle()

    def test_returns_axes(self, random_patch):
        """Call waterfall plot, return."""
        data = np.array(random_patch.data)
        data[:100, :100] = 2.0  # create an origin block for testing axis line up
        data[:100, -100:] = -2.0  #
        out = random_patch.new(data=data)
        ax = out.viz.wiggle()
        # check labels
        assert random_patch.dims[0].lower() in ax.get_ylabel().lower()
        assert random_patch.dims[1].lower() in ax.get_xlabel().lower()
        assert isinstance(ax, plt.Axes)

    def test_shading(self, small_patch):
        """Ensure shading parameter works."""
        _ = small_patch.viz.wiggle(shade=True)

    def test_non_time_axis(self, random_patch):
        """Ensure another dimension works."""
        sub_patch = random_patch.select(time=(10, 20), samples=True)
        ax = sub_patch.viz.wiggle(dim="distance")
        assert "distance" in str(ax.get_xlabel())
        assert "time" in str(ax.get_ylabel())

    def test_show(self, random_patch, monkeypatch):
        """Ensure show path is callable."""
        monkeypatch.setattr(plt, "show", lambda: None)
        random_patch.viz.wiggle(show=True)
