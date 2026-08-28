"""Tests for waterfall plots."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

import dascore as dc
from dascore.examples import inventory_patch_pair
from dascore.exceptions import ParameterError
from dascore.units import get_quantity_str, percent
from dascore.utils.misc import suppress_warnings
from dascore.utils.time import is_datetime64, to_timedelta64
from dascore.viz._labels import BAR_GID, MAX_LABELS, MAX_RUNS, SEAM_GID
from dascore.viz._lanes import string_colors


def check_label_units(patch, ax):
    """Ensure patch label units match axis."""
    axis_dict = {0: "yaxis", 1: "xaxis"}
    dims = patch.dims
    # Check coordinate names
    for coord_name in dims:
        coord = patch.coords.coord_map[coord_name]
        if is_datetime64(coord[0]):
            continue  # just skip datetimes for now.
        index = dims.index(coord_name)
        axis = getattr(ax, axis_dict[index])
        label_text = axis.get_label().get_text().lower()
        assert str(coord.units.units) in label_text
        assert coord_name in label_text
    # check colorbar labels
    cax = ax.images[-1].colorbar
    yaxis_label = cax.ax.yaxis.label.get_text()
    assert str(patch.attrs.data_units.units) in yaxis_label
    assert str(patch.attrs.data_type) in yaxis_label


@pytest.fixture(scope="session")
def patch_random_start(event_patch_1):
    """Get a patch with a random, odd, starttime."""
    random_starttime = dc.to_datetime64("2020-01-02T02:12:11.02232")
    attrs = dict(event_patch_1.attrs)
    coords = {i: v for i, v in event_patch_1.coords.items()}
    time = coords["time"].values - coords["time"].min()
    coords["time"] = time + random_starttime
    patch = event_patch_1.update(attrs=attrs, coords=coords)
    return patch


@pytest.fixture(scope="session")
def patch_two_times(random_patch):
    """Create a patch with two time dims."""
    dist = random_patch.coords.get_array("distance")
    pa = random_patch.update_coords(distance=dc.to_datetime64(dist))
    return pa


class TestWaterfall:
    """Tests for waterfall plot."""

    @pytest.fixture()
    def timedelta_patch(self, random_patch):
        """Make a patch with one dimension dtype of timedelta64."""
        old_coord = random_patch.get_coord("time")
        new_time = to_timedelta64(np.arange(len(old_coord)))
        return random_patch.update_coords(time=new_time)

    @pytest.fixture()
    def distance_gap_patch(self, random_patch):
        """Create a patch with one large gap in its distance coordinate."""
        coord = random_patch.get_coord("distance")
        values = np.asarray(coord).copy()
        split = len(values) // 2
        values[split:] += coord.step * 10
        return random_patch.update_coords(distance=values), split

    @pytest.fixture()
    def time_gap_patch(self, random_patch):
        """Create a patch with one large gap in its time coordinate."""
        coord = random_patch.get_coord("time")
        values = np.asarray(coord).copy()
        split = len(values) // 2
        values[split:] += coord.step * 10
        return random_patch.update_coords(time=values), split

    def test_even_coordinates_use_image(self, random_patch):
        """Evenly sampled coordinates retain the fast image renderer."""
        ax = random_patch.viz.waterfall(cbar=False)
        assert isinstance(ax.images[0], AxesImage)
        assert not any(isinstance(x, QuadMesh) for x in ax.collections)

    def test_irregular_timedelta_coordinates_use_mesh(self, timedelta_patch):
        """Irregular timedelta coordinates are converted to seconds for meshes."""
        values = np.asarray(timedelta_patch.get_coord("time")).copy()
        split = len(values) // 2
        values[split:] += np.timedelta64(10, "s")
        patch = timedelta_patch.update_coords(time=values)
        ax = patch.viz.waterfall(cbar=False)
        mesh = ax.collections[0]
        assert isinstance(mesh, QuadMesh)
        assert np.all(np.isfinite(mesh.get_coordinates()))

    def test_singleton_irregular_coordinate_uses_mesh(self, random_patch):
        """A singleton irregular coordinate receives finite cell edges."""
        patch = random_patch.select(distance=0, samples=True)
        distance = np.asarray(patch.get_coord("distance"))
        patch = patch.update_coords(distance=distance)
        assert not patch.get_coord("distance").evenly_sampled
        with pytest.warns(UserWarning, match="Singleton coordinate"):
            ax = patch.viz.waterfall(cbar=False)
        mesh = ax.collections[0]
        assert isinstance(mesh, QuadMesh)
        assert mesh.get_coordinates().shape[:2] == tuple(x + 1 for x in patch.shape)

    def test_nonmonotonic_coordinate_uses_image(self, random_patch):
        """Nonmonotonic coordinates retain the image-rendering fallback."""
        distance = np.asarray(random_patch.get_coord("distance")).copy()
        distance[[1, 2]] = distance[[2, 1]]
        patch = random_patch.update_coords(distance=distance)
        ax = patch.viz.waterfall(cbar=False)
        assert isinstance(ax.images[0], AxesImage)
        assert not any(isinstance(x, QuadMesh) for x in ax.collections)

    def test_gap_uses_masked_mesh(self, distance_gap_patch):
        """A gap color adds one masked mesh band."""
        patch, split = distance_gap_patch
        ax = patch.viz.waterfall(gap_color="white", cbar=False)
        mesh = ax.collections[0]
        array = mesh.get_array()
        mask = np.ma.getmaskarray(array)
        assert isinstance(mesh, QuadMesh)
        assert not ax.images
        assert array.shape == (patch.shape[0] + 1, patch.shape[1])
        assert np.all(mask[split, :])
        assert mesh.get_coordinates().shape[:2] == (
            patch.shape[0] + 2,
            patch.shape[1] + 1,
        )
        assert np.allclose(mesh.cmap.get_bad(), [1, 1, 1, 1])

    def test_gap_mesh_colorbar_and_scale(self, distance_gap_patch):
        """Mesh plots retain waterfall colorbar and scaling behavior."""
        patch, _ = distance_gap_patch
        ax = patch.viz.waterfall(
            scale=(-1, 1), scale_type="absolute", gap_color="white"
        )
        mesh = ax.collections[0]
        assert mesh.colorbar is not None
        assert mesh.get_clim() == (-1, 1)

    def test_bridge_gap_doesnt_expand_data(self, distance_gap_patch):
        """The default extends cells across a gap without adding a band."""
        patch, _ = distance_gap_patch
        ax = patch.viz.waterfall(cbar=False)
        mesh = ax.collections[0]
        assert mesh.get_array().shape == patch.shape
        assert mesh.get_coordinates().shape[:2] == tuple(x + 1 for x in patch.shape)

    def test_gap_color(self, distance_gap_patch):
        """A specified gap color is assigned to masked mesh cells."""
        patch, _ = distance_gap_patch
        ax = patch.viz.waterfall(gap_color="white", cbar=False)
        assert np.allclose(ax.collections[0].cmap.get_bad(), [1, 1, 1, 1])

    def test_gaps_in_both_axes(self, distance_gap_patch):
        """Gap bands can be inserted along both dimensions."""
        patch, distance_split = distance_gap_patch
        time = patch.get_coord("time")
        values = np.asarray(time).copy()
        time_split = len(values) // 2
        values[time_split:] += time.step * 10
        patch = patch.update_coords(time=values)
        ax = patch.viz.waterfall(gap_color="white", cbar=False)
        mesh_array = ax.collections[0].get_array()
        mask = np.ma.getmaskarray(mesh_array)
        assert mesh_array.shape == tuple(x + 1 for x in patch.shape)
        assert np.all(mask[distance_split, :])
        assert np.all(mask[:, time_split])

    def test_time_gap_keeps_regular_ticks(self, time_gap_patch):
        """Time-axis ticks remain monotonic and evenly spaced across gaps."""
        patch, split = time_gap_patch
        ax = patch.viz.waterfall(gap_color="white", cbar=False)
        mask = np.ma.getmaskarray(ax.collections[0].get_array())
        ax.get_figure().canvas.draw()
        tick_diffs = np.diff(ax.get_xticks())
        assert np.all(mask[:, split])
        assert np.all(tick_diffs > 0)
        assert np.allclose(tick_diffs, tick_diffs[0])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"gap_factor": 1}, "gap_factor"),
            ({"gap_factor": np.inf}, "gap_factor"),
        ],
    )
    def test_bad_gap_options_raise(self, random_patch, kwargs, match):
        """Invalid gap display options raise an informative error."""
        with pytest.raises(ParameterError, match=match):
            random_patch.viz.waterfall(**kwargs)

    def test_returns_axes(self, random_patch):
        """Call waterfall plot, return."""
        # modify patch to include line at start
        data = np.array(random_patch.data)
        data[:100, :100] = 2.0  # create an origin block for testing axis line up
        data[:100, -100:] = -2.0  #
        out = random_patch.new(data=data)
        ax = out.viz.waterfall()

        # check labels
        assert random_patch.dims[0] in ax.get_ylabel().lower()
        assert random_patch.dims[1] in ax.get_xlabel().lower()
        assert isinstance(ax, plt.Axes)

    def test_y_axis_inverted_only_when_time_like(self, random_patch):
        """Time on the y axis increases downward; other dimensions do not."""
        ax = random_patch.viz.waterfall()  # distance on the y axis
        assert not ax.yaxis_inverted()
        ax = random_patch.transpose("time", "distance").viz.waterfall()
        assert ax.yaxis_inverted()

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

    def test_doc_intro_example(self, event_patch_1):
        """Simple test to ensure the doc examples can be run."""
        patch = event_patch_1.pass_filter(time=(None, 300))
        _ = patch.viz.waterfall(scale=0.04)
        _ = patch.transpose("distance", "time").viz.waterfall(scale=0.04)

    def test_time_axis_label_int_overflow(self, random_patch):
        """Make sure the time axis labels are correct (windows compatibility)."""
        ax = random_patch.viz.waterfall()
        name = ["y", "x"][random_patch.get_axis("time")]
        # Get the piece of the label corresponding to the starttime
        # WE can just grab the offset text.
        sub_ax = getattr(ax, f"{name}axis")
        plt.tight_layout()  # need to call this to get offset to show up.
        offset_str = sub_ax.get_major_formatter().get_offset()
        min_time = random_patch.coords.get_array("time").min()
        assert str(min_time).startswith(offset_str)

    def test_no_colorbar(self, random_patch):
        """Ensure the colorbar can be disabled."""
        ax = random_patch.viz.waterfall(cbar=False)
        assert ax.images[-1].colorbar is None

    def test_units(self, random_patch):
        """Test that units show up in labels."""
        # standard units
        pa = random_patch.set_units("m/s")
        ax = pa.viz.waterfall()
        check_label_units(pa, ax)
        # weird units
        new = pa.set_units(
            "furlongs/fortnight",
            distance="feet",
        )
        ax = new.viz.waterfall()
        check_label_units(new, ax)

    def test_time_no_units(self, patch_two_times):
        """time-like dims shouldn't show units in label."""
        pa = patch_two_times
        dims = pa.dims
        ax = pa.viz.waterfall()
        assert ax.get_xlabel().casefold() == dims[1].casefold()
        assert ax.get_ylabel().casefold() == dims[0].casefold()

    def test_patch_with_data_type(self, random_patch):
        """Ensure a patch with data_type titles the colorbar."""
        patch = random_patch.update_attrs(
            data_type="strain rate",
            data_units="1/s",
        )
        ax = patch.viz.waterfall()
        check_label_units(patch, ax)

    def test_timedelta_axis(self, timedelta_patch):
        """Ensure plot works when one axis has timedelta dtype. See #309."""
        # if this doesnt raise it probably works ;)
        ax = timedelta_patch.viz.waterfall()
        assert ax is not None

    def test_show(self, random_patch, shown):
        """Ensure show path is callable."""
        random_patch.viz.waterfall(show=True)
        assert shown

    def test_log(self, random_patch):
        """Ensure log is callable."""
        ax = random_patch.viz.waterfall(log=True)

        # Retrieve the colorbar label
        cb = ax.get_figure().get_axes()[-1]
        cb_label = cb.get_ylabel()

        # Retrieve the expected data type and data units
        data_type = str(random_patch.attrs["data_type"])
        data_units = get_quantity_str(random_patch.attrs.data_units) or ""
        expected_dunits = f" [{data_units}]" if data_units else ""

        # Construct the expected label
        expected_label = f"{data_type}{expected_dunits} - log_10"

        # Check if the colorbar label matches the expected label
        assert cb_label == expected_label, (
            f"Expected '{expected_label}', but got '{cb_label}'"
        )

    def test_incomplete_time_coord(self):
        """Test waterfall plot with incomplete time coordinates (issue #534)."""
        # Create a patch with aggregated time dimension (all NaN coordinates)
        spool = dc.get_example_spool()
        sub = spool.chunk(time=2, overlap=1)
        aggs = dc.spool([x.max("time") for x in sub]).concatenate(time=None)[0]
        # This should not raise an error
        ax = aggs.viz.waterfall()
        assert ax is not None
        assert isinstance(ax, plt.Axes)

    def test_bad_relative_scale_raises(self, random_patch):
        """Ensure malformed relative scales raise ParameterError."""
        msg = "Relative scale"
        # Negative value in scale.
        with pytest.raises(ParameterError, match=msg):
            random_patch.viz.waterfall(scale=(-0.1, 0.9), scale_type="relative")
        # Reversed order.
        with pytest.raises(ParameterError, match=msg):
            random_patch.viz.waterfall(scale=(0.9, 0.1), scale_type="relative")
        # More than two values
        with pytest.raises(ParameterError, match=msg):
            random_patch.viz.waterfall(scale=(0.1, 0.2, 0.9), scale_type="relative")

    def test_long_absolute_scale_raises(self, random_patch):
        """Three absolute values are limits for nothing, so say so."""
        with pytest.raises(ParameterError, match="scale must be"):
            random_patch.viz.waterfall(scale=(0.1, 0.2, 0.9), scale_type="absolute")

    def test_invalid_scale_type_raises(self, random_patch):
        """Ensure invalid scale_type values raise ParameterError."""
        msg = "scale_type must be one of"
        # Invalid string
        with pytest.raises(ParameterError, match=msg):
            random_patch.viz.waterfall(scale_type="invalid")
        # Typo
        with pytest.raises(ParameterError, match=msg):
            random_patch.viz.waterfall(scale_type="relatve")
        # Case sensitivity
        with pytest.raises(ParameterError, match=msg):
            random_patch.viz.waterfall(scale_type="Relative")

    def test_non_2d_patch_raises(self, random_patch):
        """Ensure non-2D patches raise ParameterError."""
        msg = "Can only make waterfall plot of 2D Patch"
        # Test with 1D patch - select single value and squeeze
        patch_1d = random_patch.select(distance=0, samples=True).squeeze()
        assert len(patch_1d.dims) == 1
        with pytest.raises(ParameterError, match=msg):
            patch_1d.viz.waterfall()

    def test_constant_patch(self, random_patch):
        """Ensure the plotting works on constant value patches."""
        data = np.ones(random_patch.shape)
        patch = random_patch.update(data=data)
        ax = patch.viz.waterfall()
        assert isinstance(ax, plt.Axes)

    def test_all_nan_patch(self, random_patch):
        """All-NaN data should plot without leaking scaling warnings."""
        data = np.full(random_patch.shape, np.nan)
        patch = random_patch.update(data=data)
        with suppress_warnings(action="error"):
            ax = patch.viz.waterfall()
        assert isinstance(ax, plt.Axes)

    def test_constant_data_with_relative_scale(self, random_patch):
        """Ensure constant data works with non-zero relative scale."""
        data = np.ones(random_patch.shape) * 42.0
        patch = random_patch.update(data=data)
        # Should not raise, epsilon handling should prevent degenerate limits
        ax = patch.viz.waterfall(scale=0.5, scale_type="relative")
        assert isinstance(ax, plt.Axes)
        # Verify colorbar limits are not identical
        clim = ax.images[0].get_clim()
        assert clim[0] != clim[1], "Colorbar limits should not be identical"

    def test_negative_relative_scale_raises(self, random_patch):
        """A negative relative scale would put the limits in the wrong order."""
        with pytest.raises(ParameterError, match="greater than 0"):
            random_patch.viz.waterfall(scale=-0.5, scale_type="relative")

    def test_scale_zero_raises(self, random_patch):
        """Ensure scale=0 with relative scaling raises ParameterError."""
        msg = "Relative scale value of 0"
        with pytest.raises(ParameterError, match=msg):
            random_patch.viz.waterfall(scale=0, scale_type="relative")
        # Also test with constant data to ensure same behavior
        data = np.ones(random_patch.shape)
        patch = random_patch.update(data=data)
        with pytest.raises(ParameterError, match=msg):
            patch.viz.waterfall(scale=0, scale_type="relative")

    def test_percent_scale(self, random_patch):
        """Ensure the percent unit works with scale."""
        ax = random_patch.viz.waterfall(scale=10 * percent, scale_type="absolute")
        assert ax is not None

    def test_invalid_scale_disables_colorbar_extend(self, random_patch):
        """Ensure invalid scales fall back to extend='neither'."""
        ax = random_patch.viz.waterfall(scale=(np.nan, 1), show=False)
        cbar = ax.images[-1].colorbar
        assert cbar.extend == "neither"

    def test_log_with_zero_data_has_finite_clim(self, random_patch):
        """Ensure that color limits work with in log-scale with zeros"""
        patch = random_patch.update(data=np.zeros(random_patch.shape))
        ax = patch.viz.waterfall(log=True)
        assert np.all(np.isfinite(ax.images[0].get_clim()))

    def test_default_colormap_uses_data_type(self, random_patch):
        """Ensure colormap is chosen automatically."""
        patch = random_patch.update_attrs(data_type="velocity")
        ax = patch.viz.waterfall()
        assert ax.images[0].cmap is not None


def _legend(ax):
    """Return the legend naming the labels, wherever it was placed."""
    figure = ax.get_figure()
    return figure.legends[0] if figure.legends else ax.get_legend()


def _legend_labels(ax):
    """Return the text of every entry in that legend."""
    legend = _legend(ax)
    return [] if legend is None else [x.get_text() for x in legend.get_texts()]


def _bars(ax, axis):
    """Return (low, high, spine, color) for every bar drawn."""
    out = []
    for line in ax.lines:
        if not str(line.get_gid()).startswith(BAR_GID):
            continue
        across = line.get_xdata() if axis == "y" else line.get_ydata()
        along = line.get_ydata() if axis == "y" else line.get_xdata()
        out.append(
            (float(along[0]), float(along[1]), float(across[0]), line.get_color())
        )
    return out


def _on_canvas(line):
    """Where a line actually lands, in display pixels."""
    points = np.column_stack([line.get_xdata(), line.get_ydata()])
    return line.get_transform().transform(points)


def _changes(ax, axis):
    """Return where a hairline was drawn across the image."""
    getter = "get_xdata" if axis == "x" else "get_ydata"
    return sorted(
        float(getattr(x, getter)()[0])
        for x in ax.lines
        if str(x.get_gid()).startswith(SEAM_GID)
    )


def _spans(ax, axis):
    """Return the stretch each label covers, once however many spines."""
    return sorted({(x[0], x[1]) for x in _bars(ax, axis)})


def _span_color(ax, axis, low, high):
    """Return the one color every bar over that stretch is drawn in."""
    colors = {
        x[3]
        for x in _bars(ax, axis)
        if x[0] == pytest.approx(low) and x[1] == pytest.approx(high)
    }
    assert len(colors) == 1, f"expected one color, got {len(colors)}"
    return colors.pop()


class TestLabelCoord:
    """Tests for marking the stretches a label coordinate covers."""

    # Deliberately not size // 2, the one index where the extent's cell
    # edge and the midpoint between two coordinate values coincide.
    split_fraction = 3

    @pytest.fixture(scope="class")
    def zone_patch(self, random_patch):
        """A patch whose distance is parted into two named zones."""
        size = random_patch.coords.coord_size("distance")
        split = size // self.split_fraction
        zones = np.where(np.arange(size) < split, "north", "south")
        return random_patch.update_coords(zone=("distance", zones))

    @pytest.fixture(scope="class")
    def unsorted_zone_patch(self, random_patch):
        """A patch whose first label is not the first alphabetically.

        Codes run in order of appearance and colors are keyed by sorted
        name, so a fixture where the two agree cannot tell them apart.
        """
        size = random_patch.coords.coord_size("distance")
        split = size // self.split_fraction
        zones = np.where(np.arange(size) < split, "south", "north")
        return random_patch.update_coords(zone=("distance", zones))

    @pytest.fixture(scope="class")
    def phase_patch(self, random_patch):
        """A patch whose time is parted into two named phases."""
        size = random_patch.coords.coord_size("time")
        split = size // self.split_fraction
        phases = np.where(np.arange(size) < split, "early", "late")
        return random_patch.update_coords(phase=("time", phases))

    @pytest.fixture()
    def zone_gap_patch(self, random_patch):
        """A patch with a distance gap, parted into two named zones."""
        coord = random_patch.get_coord("distance")
        values = np.asarray(coord).copy()
        split = len(values) // self.split_fraction
        values[split:] += coord.step * 10
        zones = np.where(np.arange(len(values)) < split, "near", "far")
        patch = random_patch.update_coords(distance=values, zone=("distance", zones))
        return patch, split

    @pytest.fixture(scope="class")
    def enriched_patch(self):
        """The inventory example patch, carrying the labels its path states."""
        patch, inventory = inventory_patch_pair()
        return patch.enrich(inventory)

    def _edge(self, patch, dim, index):
        """Where the image put the edge between two samples of a dimension."""
        coord = patch.get_coord(dim)
        low, high = np.asarray(coord).min(), np.asarray(coord).max()
        if is_datetime64(coord.dtype):
            low, high = (mdates.date2num(dc.to_datetime64(x)) for x in (low, high))
        return float(low) + (float(high) - float(low)) * index / len(coord)

    def test_each_label_gets_a_bar_on_both_spines(self, zone_patch):
        """Every stretch is marked on the two spines it runs between."""
        ax = zone_patch.viz.waterfall(label_coord="zone")
        bars = _bars(ax, "y")
        assert len(bars) == 4
        # The hairline is a separate artist and is not counted as a bar.
        assert len(ax.lines) == len(bars) + len(_changes(ax, "y"))
        # Two stretches, each drawn on the near and the far spine.
        assert sorted(x[2] for x in bars) == [0.0, 0.0, 1.0, 1.0]
        assert len(_spans(ax, "y")) == 2

    def test_bars_cover_the_stretch_each_label_holds(self, zone_patch):
        """A bar runs from where its label starts to where it stops."""
        ax = zone_patch.viz.waterfall(label_coord="zone")
        size = zone_patch.coords.coord_size("distance")
        split = size // self.split_fraction
        edges = [self._edge(zone_patch, "distance", x) for x in (0, split, size)]
        assert _spans(ax, "y") == [
            (pytest.approx(edges[0]), pytest.approx(edges[1])),
            (pytest.approx(edges[1]), pytest.approx(edges[2])),
        ]
        colors = string_colors(["north", "south"])
        assert _span_color(ax, "y", edges[0], edges[1]) == colors["north"]
        assert _span_color(ax, "y", edges[1], edges[2]) == colors["south"]

    def test_a_hairline_joins_the_bars_at_each_change(self, zone_patch):
        """Where the label changes, a faint line crosses the image."""
        ax = zone_patch.viz.waterfall(label_coord="zone")
        size = zone_patch.coords.coord_size("distance")
        split = size // self.split_fraction
        # One change only: the two outer ends are the patch's own edges.
        assert _changes(ax, "y") == [
            pytest.approx(self._edge(zone_patch, "distance", split))
        ]
        hair = next(x for x in ax.lines if str(x.get_gid()).startswith(SEAM_GID))
        # Faint enough to locate a boundary without competing with data.
        assert hair.get_linewidth() < 1.0
        assert hair.get_alpha() < 0.7

    def test_a_hairline_marks_the_edge_of_an_absent_stretch(self, random_patch):
        """A label meeting a stretch stating nothing is still a change."""
        size = random_patch.coords.coord_size("distance")
        values = np.full(size, "", dtype="<U5")
        low, high = size // 3, 2 * size // 3
        values[low:high] = "mid"
        patch = random_patch.update_coords(tag=("distance", values))
        ax = patch.viz.waterfall(label_coord="tag")
        assert _changes(ax, "y") == [
            pytest.approx(self._edge(patch, "distance", low)),
            pytest.approx(self._edge(patch, "distance", high)),
        ]

    def test_bars_land_on_the_spines_on_the_canvas(self, zone_patch):
        """The bars reach the axes edges, not merely claim to.

        The data handed to matplotlib is the same whichever way the
        blended transform is built, so only where a bar lands can tell
        the two apart.
        """
        ax = zone_patch.viz.waterfall(label_coord="zone")
        ax.get_figure().canvas.draw()
        box = ax.get_window_extent()
        for line in [x for x in ax.lines if str(x.get_gid()).startswith(BAR_GID)]:
            points = _on_canvas(line)
            across, along = points[:, 0], points[:, 1]
            # Upright on the page, and on one of the two upright edges.
            assert across[0] == pytest.approx(across[1])
            assert min(abs(across[0] - box.x0), abs(across[0] - box.x1)) < 1.0
            # Running along part of the axes, neither collapsed nor loose.
            assert box.y0 - 1 <= min(along)
            assert max(along) <= box.y1 + 1
            assert max(along) - min(along) > 1.0

    def test_a_label_keeps_its_color_whatever_its_order(self, unsorted_zone_patch):
        """Colors follow the label, not the order it first appears in."""
        ax = unsorted_zone_patch.viz.waterfall(label_coord="zone")
        colors = string_colors(["north", "south"])
        first, second = _spans(ax, "y")
        # South is stated first here, and is still south's color.
        assert _span_color(ax, "y", *first) == colors["south"]
        assert _span_color(ax, "y", *second) == colors["north"]
        assert _legend_labels(ax) == ["south", "north"]

    def test_bar_ends_on_the_image_cell_edge(self, zone_patch):
        """The bar ends on the cell edge, not the coordinate midpoint."""
        ax = zone_patch.viz.waterfall(label_coord="zone")
        coord = np.asarray(zone_patch.get_coord("distance"))
        split = len(coord) // self.split_fraction
        midpoint = (coord[split - 1] + coord[split]) / 2
        drawn = _spans(ax, "y")[0][1]
        assert drawn == pytest.approx(self._edge(zone_patch, "distance", split))
        # The two genuinely differ here, so the test can tell them apart.
        assert drawn != pytest.approx(midpoint)

    def test_label_on_time_runs_along_the_other_spines(self, phase_patch):
        """A coordinate over time is marked on the bottom and top."""
        ax = phase_patch.viz.waterfall(label_coord="phase")
        size = phase_patch.coords.coord_size("time")
        split = size // self.split_fraction
        edges = [self._edge(phase_patch, "time", x) for x in (0, split, size)]
        assert _spans(ax, "x") == [
            (pytest.approx(edges[0]), pytest.approx(edges[1])),
            (pytest.approx(edges[1]), pytest.approx(edges[2])),
        ]
        assert sorted(x[2] for x in _bars(ax, "x")) == [0.0, 0.0, 1.0, 1.0]

    @pytest.mark.parametrize("gap_color", [None, "gray"])
    def test_bar_ends_on_the_mesh_cell_edge(self, zone_gap_patch, gap_color):
        """A bar ends where its own last cell ends, not across a gap.

        With a gap band opened the two zones no longer share an edge, and
        a bar drawn to where the next one starts would state that the
        near zone covers ground the mesh shows no data for.
        """
        patch, split = zone_gap_patch
        ax = patch.viz.waterfall(label_coord="zone", gap_color=gap_color)
        assert isinstance(ax.collections[0], QuadMesh)
        values = np.asarray(patch.get_coord("distance"))
        step = np.median(np.abs(np.diff(values)))
        near_end, far_start = _spans(ax, "y")[0][1], _spans(ax, "y")[1][0]
        if gap_color is None:
            # The cells bridge the gap, so they meet halfway across it.
            middle = (values[split - 1] + values[split]) / 2
            assert near_end == pytest.approx(middle)
            assert far_start == pytest.approx(middle)
        else:
            # A band was opened, so each zone stops at its own edge and
            # the band between them is claimed by neither.
            assert near_end == pytest.approx(values[split - 1] + step / 2)
            assert far_start == pytest.approx(values[split] - step / 2)
            assert near_end < far_start

    def test_legend_names_the_labels(self, zone_patch):
        """The legend names each label and is titled by the coordinate."""
        ax = zone_patch.viz.waterfall(label_coord="zone")
        assert _legend_labels(ax) == ["north", "south"]
        assert _legend(ax).get_title().get_text() == "zone"

    def test_legend_is_drawn_inside_the_figure(self, zone_patch):
        """The names fit on the page, not past its right edge."""
        ax = zone_patch.viz.waterfall(label_coord="zone")
        figure = ax.get_figure()
        figure.canvas.draw()
        assert _legend(ax).get_window_extent().x1 <= figure.bbox.x1

    def test_legend_clears_the_colorbars_own_labels(self, zone_patch):
        """The names sit past the bar's ticks, not the bar's rectangle.

        A colorbar carries its ticks and its name outside the rectangle
        it reports as its position, so measuring the rectangle alone
        would lay the legend over them.
        """
        ax = zone_patch.viz.waterfall(label_coord="zone")
        figure = ax.get_figure()
        figure.canvas.draw()
        drawn = max(x.get_tightbbox().x1 for x in figure.axes)
        assert _legend(ax).get_window_extent().x0 >= drawn

    def test_colorbar_still_matches_the_image(self, zone_patch):
        """Making room for the legend moves the image and its bar together.

        The two hang off one gridspec; repositioning either alone parts
        them, which is why the room is taken from the figure margin.
        """
        plain = zone_patch.viz.waterfall()
        named = zone_patch.viz.waterfall(label_coord="zone")
        # Room really was taken, or the alignment below proves nothing.
        assert named.get_position().width < plain.get_position().width
        figure = named.get_figure()
        cax = next(x for x in figure.axes if x is not named)
        assert cax.get_position().x0 > named.get_position().x1
        assert cax.get_position().y0 == pytest.approx(named.get_position().y0)
        assert cax.get_position().height == pytest.approx(named.get_position().height)

    def test_each_label_keeps_its_own_color(self, zone_patch):
        """A label is drawn in the color the palette gives that label."""
        ax = zone_patch.viz.waterfall(label_coord="zone")
        expected = string_colors(["north", "south"])
        handles = _legend(ax).legend_handles
        drawn = dict(zip(_legend_labels(ax), [x.get_color() for x in handles]))
        assert drawn == {"north": expected["north"], "south": expected["south"]}

    def test_membership_covers_only_what_it_holds(self, enriched_patch):
        """A boolean coordinate marks its True stretch and names itself."""
        ax = enriched_patch.viz.waterfall(label_coord="noisy")
        assert _legend_labels(ax) == ["noisy"]
        # A membership entry already carries the name, so no title repeats it.
        assert _legend(ax).get_title().get_text() == ""
        held = np.flatnonzero(enriched_patch.coords.get_array("noisy"))
        expected = (
            self._edge(enriched_patch, "distance", held[0]),
            self._edge(enriched_patch, "distance", held[-1] + 1),
        )
        assert _spans(ax, "y") == [
            (pytest.approx(expected[0]), pytest.approx(expected[1]))
        ]

    def test_all_true_membership_covers_the_patch(self, random_patch):
        """A group covering everything is marked over the whole spine."""
        size = random_patch.coords.coord_size("distance")
        patch = random_patch.update_coords(tag=("distance", np.full(size, True)))
        ax = patch.viz.waterfall(label_coord="tag")
        assert _legend_labels(ax) == ["tag"]
        whole = (self._edge(patch, "distance", 0), self._edge(patch, "distance", size))
        assert _spans(ax, "y") == [(pytest.approx(whole[0]), pytest.approx(whole[1]))]

    def test_enriched_labels_are_drawn(self, enriched_patch):
        """A label group an inventory projected on a patch reaches the plot."""
        ax = enriched_patch.viz.waterfall(label_coord="zone")
        assert _legend_labels(ax) == ["north", "south"]

    def test_absent_stretches_leave_bare_spine(self, random_patch):
        """Where a coordinate states nothing, no bar is drawn."""
        size = random_patch.coords.coord_size("distance")
        values = np.full(size, "", dtype="<U5")
        low, high = size // 3, 2 * size // 3
        values[low:high] = "mid"
        patch = random_patch.update_coords(tag=("distance", values))
        ax = patch.viz.waterfall(label_coord="tag")
        assert _legend_labels(ax) == ["mid"]
        # One stretch only: the two unstated ones each side get nothing.
        expected = (
            self._edge(patch, "distance", low),
            self._edge(patch, "distance", high),
        )
        assert _spans(ax, "y") == [
            (pytest.approx(expected[0]), pytest.approx(expected[1]))
        ]

    def test_recurring_label_gets_one_entry(self, random_patch):
        """A label stated in two places is named once and marked twice."""
        size = random_patch.coords.coord_size("distance")
        quarter = size // 4
        values = np.where((np.arange(size) // quarter) % 2, "odd", "even")
        patch = random_patch.update_coords(tag=("distance", values))
        ax = patch.viz.waterfall(label_coord="tag")
        assert _legend_labels(ax) == ["even", "odd"]
        # Four stretches, each on two spines.
        assert len(_spans(ax, "y")) == 4
        assert len(_bars(ax, "y")) == 8
        colors = string_colors(["even", "odd"])
        first = _spans(ax, "y")[0]
        assert _span_color(ax, "y", *first) == colors["even"]

    def test_datetime_labels_read_as_times(self, random_patch):
        """A nanosecond datetime label names a time, not a count of them."""
        size = random_patch.coords.coord_size("distance")
        start = dc.to_datetime64("2020-01-01")
        later = start + np.timedelta64(1, "s")
        values = np.where(np.arange(size) < size // 3, start, later)
        patch = random_patch.update_coords(tag=("distance", values))
        ax = patch.viz.waterfall(label_coord="tag")
        assert all(x.startswith("2020-01-01") for x in _legend_labels(ax))

    def test_nulls_beside_booleans_state_membership(self, random_patch):
        """A boolean group carrying nulls is still a membership group."""
        size = random_patch.coords.coord_size("distance")
        values = np.array([True] * size, dtype=object)
        values[: size // 3] = None
        values[2 * size // 3 :] = False
        patch = random_patch.update_coords(tag=("distance", values))
        ax = patch.viz.waterfall(label_coord="tag")
        assert _legend_labels(ax) == ["tag"]
        assert len(_spans(ax, "y")) == 1

    def test_null_strings_do_not_raise(self, random_patch):
        """A string group carrying nulls plots rather than failing on them."""
        size = random_patch.coords.coord_size("distance")
        values = np.array(["a"] * size, dtype=object)
        values[size // 3 : 2 * size // 3] = pd.NA
        patch = random_patch.update_coords(tag=("distance", values))
        ax = patch.viz.waterfall(label_coord="tag")
        assert _legend_labels(ax) == ["a"]
        assert len(_spans(ax, "y")) == 2

    def test_caller_axes_keeps_its_legend_inside(self, zone_patch):
        """A figure this call did not build keeps the room it had."""
        figure, (left, right) = plt.subplots(1, 2)
        before = right.get_position().bounds
        ax = zone_patch.viz.waterfall(label_coord="zone", ax=left)
        figure.canvas.draw()
        # The neighbour did not move, and the names sit over the image.
        assert right.get_position().bounds == before
        assert not figure.legends
        assert ax.get_legend().get_window_extent().x1 <= ax.get_window_extent().x1

    def test_a_callers_own_legend_survives(self, zone_patch):
        """Naming the labels does not drop what the caller had named."""
        _, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="expected")
        ax.legend(loc="lower left")
        zone_patch.viz.waterfall(label_coord="zone", ax=ax)
        drawn = [x for x in ax.artists if isinstance(x, Legend)] + [ax.get_legend()]
        named = {y.get_text() for x in drawn if x for y in x.get_texts()}
        assert {"expected", "north", "south"} <= named

    def test_many_labels_spill_into_columns(self, random_patch):
        """A legend too tall for the page is set in as many columns as fit."""
        size = random_patch.coords.coord_size("distance")
        values = np.asarray([f"group{x * MAX_LABELS // size}" for x in range(size)])
        patch = random_patch.update_coords(tag=("distance", values))
        with plt.rc_context({"figure.figsize": (6.0, 1.6)}):
            ax = patch.viz.waterfall(label_coord="tag")
        legend = _legend(ax)
        assert len(_legend_labels(ax)) == MAX_LABELS
        assert legend._ncols > 1
        assert legend.get_window_extent().height <= ax.get_figure().bbox.height

    def test_the_image_is_put_back_after_measuring(self, zone_patch):
        """Hiding the image to measure the legend does not leave it hidden.

        Seating the legend lays the figure out several times, and the
        image is hidden across those passes so it is not rastered each
        time. It has to come back, whatever happened in between.
        """
        ax = zone_patch.viz.waterfall(label_coord="zone")
        assert ax.images
        assert all(x.get_visible() for x in ax.images)
        assert all(x.get_visible() for x in ax.collections)

    def test_every_artist_carries_its_own_id(self, random_patch):
        """No two artists share a gid, which is written out as an SVG id."""
        size = random_patch.coords.coord_size("distance")
        quarter = size // 4
        values = np.where((np.arange(size) // quarter) % 2, "odd", "even")
        patch = random_patch.update_coords(tag=("distance", values))
        ax = patch.viz.waterfall(label_coord="tag")
        gids = [x.get_gid() for x in ax.lines]
        assert len(gids) == len(set(gids))
        assert sum(str(x).startswith(BAR_GID) for x in gids) == 8
        assert sum(str(x).startswith(SEAM_GID) for x in gids) == 3

    def test_ids_are_unique_across_the_whole_figure(self, zone_patch):
        """An id names one element of the document, not one of a call.

        A figure may carry a labelled plot on every axes, and one axes
        may be drawn on twice; numbering within the call would repeat.
        """
        _, (left, right) = plt.subplots(1, 2)
        for ax in (left, right, left):
            zone_patch.viz.waterfall(label_coord="zone", ax=ax, cbar=False)
        gids = [x.get_gid() for ax in (left, right) for x in ax.lines]
        assert len(gids) == len(set(gids))

    @pytest.mark.parametrize("occupied", [(0, 2), (1,), ()])
    def test_ids_skip_numbers_already_in_use(self, zone_patch, occupied):
        """A number already on the figure is passed over, not reused.

        Counting artists would land on an occupied number wherever the
        ones in use are not a run from zero -- which is what removing an
        artist and drawing again leaves behind.
        """
        _, ax = plt.subplots()
        for number in occupied:
            ax.add_line(Line2D([0, 1], [0, 1], gid=f"{BAR_GID}-0-{number}"))
        zone_patch.viz.waterfall(label_coord="zone", ax=ax, cbar=False)
        gids = [x.get_gid() for x in ax.lines]
        assert len(gids) == len(set(gids))

    def test_ids_survive_an_artist_being_removed(self, zone_patch):
        """Drawing again after a removal does not reuse the freed number."""
        _, ax = plt.subplots()
        zone_patch.viz.waterfall(label_coord="zone", ax=ax, cbar=False)
        bars = [x for x in ax.lines if str(x.get_gid()).startswith(BAR_GID)]
        bars[0].remove()
        zone_patch.viz.waterfall(label_coord="zone", ax=ax, cbar=False)
        gids = [x.get_gid() for x in ax.lines]
        assert len(gids) == len(set(gids))

    def test_ids_are_the_same_every_build(self, zone_patch):
        """The same figure carries the same ids however often it is built."""
        builds = []
        for _ in range(2):
            _, ax = plt.subplots()
            zone_patch.viz.waterfall(label_coord="zone", ax=ax, cbar=False)
            builds.append([x.get_gid() for x in ax.lines])
        assert builds[0] == builds[1]

    def test_no_label_coord_draws_nothing(self, zone_patch):
        """The default leaves the plot as it was."""
        ax = zone_patch.viz.waterfall()
        assert not ax.lines
        assert _legend(ax) is None

    @pytest.mark.parametrize(
        ("name", "match"),
        [
            ("bad_name", "not a coordinate"),
            ("distance", "is a dimension"),
        ],
    )
    def test_bad_name_raises(self, zone_patch, name, match):
        """A name which states no label coordinate is refused."""
        before = len(plt.get_fignums())
        with pytest.raises(ParameterError, match=match):
            zone_patch.viz.waterfall(label_coord=name)
        # Refused before an axes exists, so no figure is left behind.
        assert len(plt.get_fignums()) == before

    def test_multi_dim_coord_raises(self, random_patch):
        """A coordinate over both dimensions names no axis to be drawn on."""
        patch = random_patch.update_coords(
            grid=(("distance", "time"), np.zeros(random_patch.shape))
        )
        with pytest.raises(ParameterError, match="names no one axis"):
            patch.viz.waterfall(label_coord="grid")

    @pytest.mark.parametrize(
        ("values", "match"),
        [
            (np.full(300, "", dtype="<U4"), "states no labels"),
            (np.full(300, np.nan), "states no labels"),
            (np.full(300, False), "states no labels"),
            (np.arange(300).astype(float), "distinct labels"),
            (np.arange(300) % 2 == 0, "changes value"),
        ],
        ids=["blank", "nan", "false", "many-labels", "many-runs"],
    )
    def test_refusal_leaves_no_figure(self, random_patch, values, match):
        """What a coordinate states is judged before anything is drawn.

        Refusing after the image and colorbar exist would leave an owned
        figure open, and a caller's axes half drawn on.
        """
        size = random_patch.coords.coord_size("distance")
        patch = random_patch.update_coords(tag=("distance", values[:size]))
        plt.close("all")
        with pytest.raises(ParameterError, match=match):
            patch.viz.waterfall(label_coord="tag")
        assert not plt.get_fignums()

    def test_values_printing_alike_are_told_apart(self, random_patch):
        """Two values which print the same are named, and colored, apart."""
        size = random_patch.coords.coord_size("distance")
        values = np.array([1] * (size // 2) + ["1"] * (size - size // 2), dtype=object)
        patch = random_patch.update_coords(tag=("distance", values))
        ax = patch.viz.waterfall(label_coord="tag")
        assert _legend_labels(ax) == ["1 (int)", "1 (str)"]
        assert len({x[3] for x in _bars(ax, "y")}) == 2

    def test_close_numbers_are_not_qualified(self, random_patch):
        """A name which is already unique is left as it is."""
        size = random_patch.coords.coord_size("distance")
        values = np.where(np.arange(size) < size // 3, 1.0, 1.0000001)
        patch = random_patch.update_coords(tag=("distance", values))
        ax = patch.viz.waterfall(label_coord="tag")
        assert _legend_labels(ax) == ["1.0", "1.0000001"]

    def test_constrained_layout_reserves_the_room(self, zone_patch):
        """A figure whose engine holds the margins is asked, not overruled.

        Constrained layout ignores subplots_adjust, so taking the room
        that way would leave the names hanging off the page.
        """
        with plt.rc_context({"figure.constrained_layout.use": True}):
            ax = zone_patch.viz.waterfall(label_coord="zone")
            figure = ax.get_figure()
            figure.canvas.draw()
            assert figure.get_layout_engine() is not None
            assert _legend(ax).get_window_extent().x1 <= figure.bbox.x1

    def test_tight_layout_still_reserves_the_room(self, zone_patch):
        """An engine which is not constrained is stood down, not obeyed.

        Only constrained layout keeps room for a legend outside the axes;
        any other engine recomputes the margins at draw time and would
        undo the room made for it.
        """
        with plt.rc_context({"figure.autolayout": True}):
            ax = zone_patch.viz.waterfall(label_coord="zone")
            figure = ax.get_figure()
            figure.canvas.draw()
            drawn = max(x.get_tightbbox().x1 for x in figure.axes)
            box = _legend(ax).get_window_extent()
            assert box.x0 >= drawn
            assert box.x1 <= figure.bbox.x1

    @pytest.mark.parametrize("over", [False, True])
    def test_label_ceiling_is_inclusive(self, random_patch, over):
        """Exactly MAX_LABELS is allowed; one more is refused."""
        size = random_patch.coords.coord_size("distance")
        count = MAX_LABELS + int(over)
        values = np.asarray([f"g{x * count // size}" for x in range(size)])
        assert len(set(values)) == count
        patch = random_patch.update_coords(tag=("distance", values))
        if over:
            with pytest.raises(ParameterError, match="distinct labels"):
                patch.viz.waterfall(label_coord="tag")
        else:
            assert len(_legend_labels(patch.viz.waterfall(label_coord="tag")))

    @pytest.mark.parametrize("over", [False, True])
    def test_change_ceiling_is_inclusive(self, random_patch, over):
        """Exactly MAX_RUNS changes is allowed; one more is refused."""
        size = random_patch.coords.coord_size("distance")
        changes = MAX_RUNS + int(over)
        # Alternating over the head gives one change per element there;
        # the tail repeats the last of them and adds none.
        head = np.arange(changes + 1) % 2 == 0
        values = np.full(size, head[-1], dtype=bool)
        values[: changes + 1] = head
        assert int(np.count_nonzero(np.diff(values))) == changes
        patch = random_patch.update_coords(tag=("distance", values))
        if over:
            with pytest.raises(ParameterError, match="changes value"):
                patch.viz.waterfall(label_coord="tag")
        else:
            assert patch.viz.waterfall(label_coord="tag") is not None

    def test_bars_stay_inside_their_axes_after_a_zoom(self, zone_patch):
        """A bar keeps to its own axes when the limits change.

        Bars are placed in data coordinates along their dimension, so an
        unclipped one paints across the whole figure once a zoom moves
        the limits under it -- into a neighbour which was never given a
        label_coord.
        """
        figure, (top, bottom) = plt.subplots(2, 1)
        zone_patch.viz.waterfall(label_coord="zone", ax=top, cbar=False)
        zone_patch.viz.waterfall(ax=bottom, cbar=False)
        size = zone_patch.coords.coord_size("distance")
        top.set_ylim(size // 3, size // 2)
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3] / 255.0
        below = int(pixels.shape[0] - top.get_window_extent().y0) + 3
        north = np.array(string_colors(["north", "south"])["north"][:3])
        strays = np.abs(pixels[below:] - north).sum(axis=2) < 0.05
        assert not strays.any()

    def test_names_too_wide_to_seat_do_not_crash(self, random_patch):
        """Names wanting more room than there is leave the picture alone.

        Narrowing once per pass without a floor drove the right margin
        past the left one, which matplotlib refuses outright.
        """
        size = random_patch.coords.coord_size("distance")
        wide = "x" * 80
        values = np.where(np.arange(size) < size // 2, wide + "_a", wide + "_b")
        patch = random_patch.update_coords(tag=("distance", values))
        ax = patch.viz.waterfall(label_coord="tag")
        assert ax.get_position().width > 0
