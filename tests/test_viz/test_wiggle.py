"""Tests for wiggle plots."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PolyCollection

import dascore as dc
from dascore.exceptions import ParameterError


def _get_traces(ax):
    """Get the segments of the wiggle line collection."""
    (lines,) = (x for x in ax.collections if isinstance(x, LineCollection))
    return lines.get_segments()


def _get_shade_vertices(ax):
    """Get the vertices of all shading polygons."""
    (poly,) = (x for x in ax.collections if isinstance(x, PolyCollection))
    return poly, np.concatenate([x.vertices for x in poly.get_paths()])


def _peak_is_above_offset(ax):
    """Return True if the first trace's peak is drawn above its offset line."""
    trace = _get_traces(ax)[0]
    x, low, high = trace[0, 0], trace[:, 1].min(), trace[:, 1].max()
    # Display coordinates grow upward on the screen, whatever the axis does.
    (_, low_y), (_, high_y) = ax.transData.transform([(x, low), (x, high)])
    return high_y > low_y


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

    def test_non_time_axis(self, random_patch):
        """Ensure another dimension works."""
        sub_patch = random_patch.select(time=(10, 20), samples=True)
        ax = sub_patch.viz.wiggle(dim="distance")
        assert "Distance [m]" in str(ax.get_xlabel())
        assert "Time" in str(ax.get_ylabel())
        # The y ticks label traces by their time, not a date axis of offsets.
        ax.figure.canvas.draw()
        labels = [x.get_text() for x in ax.get_yticklabels()]
        times = sub_patch.coords.get_array("time")
        assert labels[0] == str(times[0])

    def test_show(self, random_patch, shown):
        """Ensure show path is callable."""
        random_patch.viz.wiggle(show=True)
        assert shown

    def test_1d_patch(self, random_patch):
        """Test that wiggle works with 1D patches (issue #462)."""
        # Create a 1D patch by reducing one dimension
        patch_1d = random_patch.mean("distance", dim_reduce="squeeze")
        # This should work without raising an assertion error
        ax = patch_1d.viz.wiggle()
        assert isinstance(ax, plt.Axes)
        # The remaining dimension should be on the x-axis
        assert patch_1d.dims[0].lower() in ax.get_xlabel().lower()

    def test_1d_patch_show(self, random_patch, shown):
        """Test that show works with 1D patches (issue #462)."""
        patch_1d = random_patch.mean("distance", dim_reduce="squeeze")
        patch_1d.viz.wiggle(show=True)
        assert shown

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
        _, verts = _get_shade_vertices(ax)
        # The fill is clipped to the zero line and never goes below it.
        assert np.all(verts[:, 1] >= 0)

    def test_1d_alpha(self, random_patch):
        """A single trace is opaque by default, but alpha can still be set."""
        patch_1d = random_patch.mean("distance", dim_reduce="squeeze")
        assert patch_1d.viz.wiggle().lines[0].get_alpha() == 1.0
        assert patch_1d.viz.wiggle(alpha=0.3).lines[0].get_alpha() == 0.3
        # 2D patches keep the old default so wiggles can overlap.
        ax = random_patch.viz.wiggle()
        assert ax.collections[0].get_alpha() == 0.2

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
        assert len(_get_traces(ax)) == len(patch.get_coord("smell"))
        # But the default dim was squeezed away, so say so rather than
        # complain that "time" isn't a dimension of the (original) patch.
        with pytest.raises(ParameterError, match="after squeezing"):
            patch.viz.wiggle()

    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_empty_dim(self, random_patch, dim):
        """
        A patch with an empty dimension, whether or not it is the connected
        one, plots nothing but shouldn't raise.
        """
        coord = random_patch.get_coord(dim)
        patch = random_patch.select(**{dim: (coord.max() + coord.step, None)})
        assert len(patch.get_coord(dim)) == 0
        ax = patch.viz.wiggle()
        assert isinstance(ax, plt.Axes)
        assert not ax.lines and not ax.collections

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

    def test_one_collection_per_patch(self, random_patch):
        """All traces share one line collection and the view fits them."""
        ax = random_patch.viz.wiggle()
        assert not ax.lines
        traces = _get_traces(ax)
        assert len(traces) == len(random_patch.get_coord("distance"))
        # Each trace is plotted against the (numeric) time coordinate.
        time = mdates.date2num(random_patch.coords.get_array("time"))
        assert np.allclose(traces[0][:, 0], time)
        lo, hi = sorted(ax.get_xlim())
        assert lo <= time.min() and time.max() <= hi

    def test_1d_datetime_axis(self, random_patch):
        """A single trace against time keeps a date axis."""
        patch_1d = random_patch.mean("distance", dim_reduce="squeeze")
        ax = patch_1d.viz.wiggle()
        time = mdates.date2num(patch_1d.coords.get_array("time"))
        assert np.allclose(ax.lines[0].get_xdata(), time)
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)

    @pytest.mark.parametrize("n_traces", [2, 10, 11, 300])
    def test_tick_count(self, random_patch, n_traces):
        """
        No more than ten ticks, each on a trace's offset and labeled with
        that trace's coordinate, matching the old locator_params selection.
        """
        patch = random_patch.select(distance=(0, n_traces), samples=True)
        n_traces = len(patch.get_coord("distance"))
        ax = patch.viz.wiggle()
        ticks = ax.get_yticks()
        assert 0 < len(ticks) <= 10
        step = max(int(0.99 + n_traces / 10), 1)
        distance = patch.coords.get_array("distance")
        labels = [x.get_text() for x in ax.get_yticklabels()]
        assert labels == [str(x) for x in distance[::step]]
        # Each trace is data + offset, so recover the offsets of labeled traces.
        traces = _get_traces(ax)
        shown = range(0, n_traces, step)
        offsets = [traces[i][0, 1] - patch.data[i, 0] for i in shown]
        assert np.allclose(ticks, offsets)

    def test_2d_shade(self, small_patch):
        """
        Shading makes one polygon per trace which covers only the part of the
        trace above its offset.
        """
        # Center the traces so each has both signs.
        patch = small_patch - np.mean(small_patch.data)
        ax = patch.viz.wiggle(shade=True)
        poly, _ = _get_shade_vertices(ax)
        traces = _get_traces(ax)
        offsets = [x[0, 1] - patch.data[i, 0] for i, x in enumerate(traces)]
        paths = poly.get_paths()
        assert len(paths) == len(offsets)
        for path, offset in zip(paths, offsets, strict=True):
            assert np.all(path.vertices[:, 1] >= offset - 1e-9)
            # Some of the polygon is on the offset (clipped) and some above.
            on_offset = np.isclose(path.vertices[:, 1], offset)
            assert on_offset.any() and (~on_offset).any()

    def test_shade_crossings(self):
        """The shading should stop where the trace crosses its offset."""
        data = np.array([[1.0, -1.0, 1.0, 1.0]]).T  # one trace, 4 samples
        patch = dc.Patch(
            data=data,
            coords={"time": np.arange(4.0), "distance": [0.0]},
            dims=("time", "distance"),
        )
        ax = patch.viz.wiggle(shade=True)
        _, verts = _get_shade_vertices(ax)
        x, y = verts[:, 0], verts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        # Two triangles of area 0.25 (crossings at x=0.5, 1.5) and a 1x1 box.
        assert np.isclose(area, 1.5)

    def test_nan_data(self, small_patch):
        """NaNs in the data shouldn't break plotting or shading."""
        data = np.array(small_patch.data)
        data[2, 10:20] = np.nan
        data[3, :] = np.nan  # an entirely missing trace
        patch = small_patch.new(data=data)
        ax = patch.viz.wiggle(shade=True)
        traces = _get_traces(ax)
        assert len(traces) == len(patch.get_coord("distance"))
        assert np.all(np.isfinite(ax.get_ylim()))
        _, verts = _get_shade_vertices(ax)
        assert np.all(np.isfinite(verts))

    def test_all_nan_data(self, small_patch):
        """A patch with no finite data should still produce a plot."""
        patch = small_patch.new(data=np.full(small_patch.shape, np.nan))
        ax = patch.viz.wiggle(shade=True)
        assert np.all(np.isfinite(ax.get_yticks()))
        assert np.all(np.isfinite(ax.get_ylim()))

    def test_1d_shade_includes_baseline(self, random_patch):
        """Shading an all-positive trace must keep the zero line in view."""
        patch_1d = random_patch.mean("distance", dim_reduce="squeeze")
        patch = patch_1d + 10  # everything well above zero
        ax = patch.viz.wiggle(shade=True)
        assert min(ax.get_ylim()) <= 0


class TestWiggleOrientation:
    """Tests for which direction the wiggle plot's y axis runs."""

    @pytest.fixture()
    def small_patch(self, random_patch):
        """A small patch to cut back on plot time."""
        return random_patch.select(distance=(10, 15), samples=True).select(
            time=(0, 20), samples=True
        )

    def test_distance_traces_not_inverted(self, small_patch):
        """Traces stacked along distance leave distance increasing upward."""
        ax = small_patch.viz.wiggle()
        assert not ax.yaxis_inverted()

    def test_time_traces_inverted(self, small_patch):
        """Traces stacked along time keep the time increases downward convention."""
        ax = small_patch.viz.wiggle(dim="distance")
        assert ax.yaxis_inverted()

    def test_positive_data_deflects_up(self, small_patch):
        """An all positive patch, like an envelope, must deflect up."""
        patch = small_patch.envelope("time")
        assert np.all(patch.data >= 0)
        assert _peak_is_above_offset(patch.viz.wiggle())

    def test_time_traces_deflect_with_their_axis(self, small_patch):
        """
        Stacking traces along time inverts the axis they are offset along,
        so their amplitudes point down with it. Documented, not desirable;
        a time axis running upward would be the greater surprise.
        """
        patch = small_patch.envelope("time")
        assert not _peak_is_above_offset(patch.viz.wiggle(dim="distance"))

    def test_inversion_is_idempotent(self, small_patch):
        """Drawing on an already time-down axis must not flip it back."""
        _, ax = plt.subplots(1)
        for _ in range(2):
            small_patch.viz.wiggle(dim="distance", ax=ax)
            assert ax.yaxis_inverted()

    @pytest.mark.parametrize("trace_dim", ["distance", "time"])
    def test_agrees_with_waterfall(self, small_patch, trace_dim):
        """Both plots invert the y axis under the same conditions."""
        connected_dim = next(iter(set(small_patch.dims) - {trace_dim}))
        wiggle_ax = small_patch.viz.wiggle(dim=connected_dim)
        # Waterfall puts the patch's first dimension on the y axis.
        water_ax = small_patch.transpose(trace_dim, connected_dim).viz.waterfall()
        assert wiggle_ax.yaxis_inverted() == water_ax.yaxis_inverted()
