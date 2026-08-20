"""Tests for wiggle plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError


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
        assert "Distance [m]" in str(ax.get_xlabel())
        assert "Time" in str(ax.get_ylabel())

    def test_show(self, random_patch, monkeypatch):
        """Ensure show path is callable."""
        monkeypatch.setattr(plt, "show", lambda: None)
        random_patch.viz.wiggle(show=True)

    def test_1d_patch(self, random_patch):
        """Test that wiggle works with 1D patches (issue #462)."""
        # Create a 1D patch by reducing one dimension
        patch_1d = random_patch.mean("distance", dim_reduce="squeeze")
        # This should work without raising an assertion error
        ax = patch_1d.viz.wiggle()
        assert isinstance(ax, plt.Axes)
        # The remaining dimension should be on the x-axis
        assert patch_1d.dims[0].lower() in ax.get_xlabel().lower()

    def test_1d_patch_show(self, random_patch, monkeypatch):
        """Test that show works with 1D patches (issue #462)."""
        monkeypatch.setattr(plt, "show", lambda: None)
        patch_1d = random_patch.mean("distance", dim_reduce="squeeze")
        patch_1d.viz.wiggle(show=True)

    def test_1d_data_label(self, random_patch):
        """The y axis of a single trace should show the data type and units."""
        patch = random_patch.update_attrs(data_type="velocity", data_units="m/s")
        patch_1d = patch.mean("distance", dim_reduce="squeeze")
        ylabel = patch_1d.viz.wiggle().get_ylabel()
        assert ylabel == "velocity [m / s]"
        # either one alone is used as is.
        type_only = patch_1d.update_attrs(data_units="")
        assert type_only.viz.wiggle().get_ylabel() == "velocity"
        units_only = patch_1d.update_attrs(data_type="")
        assert units_only.viz.wiggle().get_ylabel() == "m / s"
        # with neither data type nor units, fall back to a generic label.
        bare = patch_1d.update_attrs(data_type="", data_units="")
        assert bare.viz.wiggle().get_ylabel() == "amplitude"

    def test_1d_shade(self, random_patch):
        """Shading should fill the positive part of a single trace."""
        patch_1d = random_patch.mean("distance", dim_reduce="squeeze")
        # Center the trace so it has both signs.
        centered = patch_1d - np.mean(patch_1d.data)
        assert np.min(centered.data) < 0 < np.max(centered.data)
        ax = centered.viz.wiggle(shade=True)
        assert len(ax.collections) == 1
        # The fill is clipped to the zero line and never goes below it.
        verts = np.concatenate([x.vertices for x in ax.collections[0].get_paths()])
        assert np.all(verts[:, 1] >= 0)

    def test_1d_alpha(self, random_patch):
        """A single trace is opaque by default, but alpha can still be set."""
        patch_1d = random_patch.mean("distance", dim_reduce="squeeze")
        assert patch_1d.viz.wiggle().lines[0].get_alpha() == 1.0
        assert patch_1d.viz.wiggle(alpha=0.3).lines[0].get_alpha() == 0.3
        # 2D patches keep the old default so wiggles can overlap.
        assert random_patch.viz.wiggle().lines[0].get_alpha() == 0.2

    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_singleton_dim_squeezed(self, random_patch, dim):
        """
        A 2D patch with a length one dimension (e.g., a single OTDR trace
        stored as (time: 1, distance: N)) should plot as one line along the
        other dimension rather than one wiggle per sample.
        """
        patch = random_patch.select(**{dim: 0, "samples": True})
        assert len(patch.dims) == 2
        other_dim = next(iter(set(patch.dims) - {dim}))
        ax = patch.viz.wiggle()
        assert len(ax.lines) == 1
        assert other_dim in ax.get_xlabel().lower()
        assert len(ax.lines[0].get_xdata()) == len(patch.get_coord(other_dim))
        # The single trace path is used: opaque line, y axis not inverted.
        assert ax.lines[0].get_alpha() == 1.0
        assert not ax.yaxis_inverted()

    def test_3d_with_singleton_dim(self, range_patch_3d):
        """A 3D patch with one length one dim should plot as a 2D wiggle."""
        patch = range_patch_3d.select(time=0, samples=True)
        assert patch.ndim == 3
        ax = patch.viz.wiggle(dim="distance")
        assert len(ax.lines) == len(patch.get_coord("smell"))
        # But the default dim was squeezed away, so say so rather than
        # complain that "time" isn't a dimension of the (original) patch.
        with pytest.raises(ParameterError, match="after squeezing"):
            patch.viz.wiggle()

    def test_empty_dim(self, random_patch):
        """A patch with an empty dimension plots nothing but shouldn't raise."""
        patch = random_patch.select(distance=(1e9, None))
        assert len(patch.get_coord("distance")) == 0
        assert isinstance(patch.viz.wiggle(), plt.Axes)

    def test_no_figure_on_error(self, random_patch):
        """A rejected call shouldn't leave an empty figure behind."""
        patch = random_patch.select(time=0, distance=0, samples=True)
        plt.close("all")
        with pytest.raises(ParameterError):
            patch.viz.wiggle()
        assert not plt.get_fignums()

    def test_single_sample_raises(self, random_patch):
        """A patch with a single sample has nothing to plot."""
        patch = random_patch.select(time=0, distance=0, samples=True)
        with pytest.raises(ParameterError, match="single sample"):
            patch.viz.wiggle()

    def test_3d_patch_raises(self, range_patch_3d):
        """More than two non-trivial dimensions can't be wiggle plotted."""
        with pytest.raises(ParameterError, match="1D or 2D"):
            range_patch_3d.viz.wiggle()
