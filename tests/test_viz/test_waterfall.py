"""Tests for waterfall plots."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.image import AxesImage

import dascore as dc
from dascore.examples import inventory_patch_pair
from dascore.exceptions import ParameterError
from dascore.units import get_quantity_str, percent
from dascore.utils.time import is_datetime64, to_timedelta64
from dascore.viz._labels import MAX_LABELS
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
        across = line.get_xdata() if axis == "y" else line.get_ydata()
        along = line.get_ydata() if axis == "y" else line.get_xdata()
        out.append(
            (float(along[0]), float(along[1]), float(across[0]), line.get_color())
        )
    return out


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
        """The bar ends where the mesh parted the two zones."""
        patch, split = zone_gap_patch
        ax = patch.viz.waterfall(label_coord="zone", gap_color=gap_color)
        assert isinstance(ax.collections[0], QuadMesh)
        values = np.asarray(patch.get_coord("distance"))
        if gap_color is None:
            # The cells bridge the gap, so they meet halfway across it.
            expected = (values[split - 1] + values[split]) / 2
        else:
            # A band was opened, so the far zone starts at its own edge.
            expected = values[split] - np.median(np.abs(np.diff(values))) / 2
        assert _spans(ax, "y")[0][1] == pytest.approx(expected)

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
        """Making room for the legend moves the image and its bar together."""
        ax = zone_patch.viz.waterfall(label_coord="zone")
        figure = ax.get_figure()
        cax = next(x for x in figure.axes if x is not ax)
        assert cax.get_position().y0 == pytest.approx(ax.get_position().y0)
        assert cax.get_position().height == pytest.approx(ax.get_position().height)

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

    def test_close_numbers_keep_distinct_names(self, random_patch):
        """Two values which round to one name are still named apart."""
        size = random_patch.coords.coord_size("distance")
        values = np.where(np.arange(size) < size // 3, 1.0, 1.0000001)
        patch = random_patch.update_coords(tag=("distance", values))
        ax = patch.viz.waterfall(label_coord="tag")
        assert len(set(_legend_labels(ax))) == 2

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
        "values",
        [
            np.full(300, "", dtype="<U4"),
            np.full(300, np.nan),
            np.full(300, False),
        ],
        ids=["blank", "nan", "false"],
    )
    def test_no_labels_raises(self, random_patch, values):
        """A coordinate stating nothing throughout has nothing to draw."""
        size = random_patch.coords.coord_size("distance")
        patch = random_patch.update_coords(tag=("distance", values[:size]))
        with pytest.raises(ParameterError, match="states no labels"):
            patch.viz.waterfall(label_coord="tag")

    def test_too_many_labels_raises(self, random_patch):
        """A coordinate too varied to name is a quantity, not a set of labels."""
        size = random_patch.coords.coord_size("distance")
        patch = random_patch.update_coords(
            tag=("distance", np.arange(size).astype(float))
        )
        with pytest.raises(ParameterError, match="distinct labels"):
            patch.viz.waterfall(label_coord="tag")

    def test_too_many_runs_raises(self, random_patch):
        """A coordinate changing every sample states no stretches."""
        size = random_patch.coords.coord_size("distance")
        patch = random_patch.update_coords(tag=("distance", np.arange(size) % 2 == 0))
        with pytest.raises(ParameterError, match="changes value"):
            patch.viz.waterfall(label_coord="tag")
