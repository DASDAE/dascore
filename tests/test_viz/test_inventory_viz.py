"""Tests for the plots an inventory draws of itself."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PatchCollection

import dascore as dc
from dascore.core import inventory as inv
from dascore.exceptions import ParameterError
from dascore.viz import VizInventoryNameSpace
from dascore.viz.inventory import (
    COMPONENT_COLORS,
    _distance_window,
    _legend_names,
    map_path,
    path,
    timeline,
)


def _lanes(ax):
    """The lane names an axes shows, top to bottom."""
    return [x.get_text() for x in ax.get_yticklabels()]


def _boxes(ax):
    """The patch collections on an axes."""
    return [x for x in ax.collections if isinstance(x, PatchCollection)]


def _bar_label(ax) -> str:
    """The label of the colorbar beside an axes, however it is laid out."""
    bar = ax.get_figure().axes[-1]
    return bar.get_ylabel() or bar.get_xlabel()


def _legend_labels(ax):
    """The legend entries, or an empty list with no legend."""
    legend = ax.get_legend()
    return [] if legend is None else [x.get_text() for x in legend.get_texts()]


def _main_path(epoch: int) -> inv.OpticalPath:
    """One epoch of the surveyed path; the second is the repaired fiber."""
    run = 400.0 if epoch == 1 else 402.0
    times = {"end_time": "2026-07-01"} if epoch == 1 else {"start_time": "2026-07-01"}
    return inv.OpticalPath(
        name="main",
        location_code="00",
        optical_components=(
            inv.FiberSegment(name="lead", optical_length=100.0),
            inv.Connector(name="patch"),
            inv.FiberSegment(name="run", optical_length=run),
            inv.Terminator(name="end"),
        ),
        geometry=(
            # Two surveyed runs with an unsurveyed gap from 300 to 350.
            inv.Geometry(
                name="west",
                distance=(100.0, 300.0),
                coordinates={"x": (0.0, 200.0), "y": (0.0, 0.0), "z": (0.0, -1.0)},
            ),
            inv.Geometry(
                name="east",
                distance=(350.0, 500.0),
                coordinates={"x": (250.0, 400.0), "y": (0.0, 5.0), "z": (-1.0, 0.0)},
            ),
            # Columns of those same stretches, so they share the runs' names.
            inv.Geometry(
                name="west",
                distance=(100.0, 300.0),
                coordinates={"chainage": (0.0, 200.0), "depth": (0.5, 1.5)},
                units={"chainage": "m"},
            ),
            inv.Geometry(
                name="east",
                distance=(350.0, 500.0),
                coordinates={"chainage": (250.0, 400.0)},
                units={"chainage": "m"},
            ),
        ),
        coupling=(
            inv.CouplingCondition(
                start_distance=100.0, end_distance=300.0, coupling_type="trench"
            ),
            inv.CouplingCondition(
                start_distance=350.0, end_distance=500.0, coupling_type="conduit"
            ),
        ),
        labels=(
            inv.OpticalPathLabel(
                start_distance=100.0, end_distance=200.0, group="zone", value="north"
            ),
            inv.OpticalPathLabel(
                start_distance=200.0, end_distance=400.0, group="zone", value="south"
            ),
            # A label group states membership by stating no value.
            inv.OpticalPathLabel(
                start_distance=150.0, end_distance=300.0, group="noisy"
            ),
            inv.OpticalPathLabel(
                start_distance=300.0, end_distance=400.0, group="noisy"
            ),
            inv.OpticalPathLabel(
                start_distance=100.0, end_distance=200.0, group="count", value=0
            ),
            inv.OpticalPathLabel(
                start_distance=200.0, end_distance=300.0, group="count", value=2.5
            ),
        ),
        **times,
    )


def build_site_inventory() -> inv.Inventory:
    """An inventory with two path epochs, a bare spur, and varied acquisitions."""
    spur = inv.OpticalPath(
        name="spur",
        location_code="01",
        optical_components=(inv.FiberSegment(name="spur", optical_length=50.0),),
    )
    common = dict(data_category="DAS", sample_rate=100.0, gauge_length=10.0)
    acquisitions = (
        inv.Acquisition(
            code="RAW",
            location_code="00",
            start_time="2026-06-01",
            end_time="2026-06-15",
            data_type="strain_rate",
            spatial_interval=1.0,
            interrogator=inv.Interrogator(manufacturer="Fake", model="FI-1"),
            distance_map=inv.DistanceMap(channel=(0.0, 300.0), distance=(100.0, 400.0)),
            **common,
        ),
        inv.Acquisition(
            code="RAW",
            location_code="00",
            start_time="2026-07-01",
            spatial_interval=1.0,
            interrogator=inv.Interrogator(serial_number="sn-9"),
            # One point states an origin but no extent, so it draws as a tick.
            distance_map=inv.DistanceMap(channel=(0.0,), distance=(100.0,)),
            **common,
        ),
        inv.Acquisition(code="AUX", location_code="01", interrogator="int-1", **common),
        inv.Acquisition(code="NIL", location_code="02", **common),
    )
    array = inv.FiberArray(
        code="L1",
        acquisitions=acquisitions,
        optical_paths=(_main_path(1), _main_path(2), spur),
    )
    return inv.Inventory(
        coordinate_reference_system=inv.CoordinateReferenceSystem(
            authority="",
            code="",
            name="site grid",
            coordinate_labels=("x", "y", "z"),
            units=("meter", "meter", "meter"),
        ),
        resources=[inv.Interrogator(resource_id="int-1")],
        networks=(inv.Network(code="DAS", fiber_arrays=(array,)),),
    ).check()


def build_labeled_inventory(values: int, crs, lines: int = 1) -> inv.Inventory:
    """One path of one label group, stating this many distinct values."""
    labels = tuple(
        inv.OpticalPathLabel(
            start_distance=float(x * 10),
            end_distance=float(x * 10 + 8),
            group="hole",
            value="\n".join([f"H{x % values:02d}"] * lines),
        )
        for x in range(24)
    )
    return inv.Inventory(
        coordinate_reference_system=crs,
        networks=(
            inv.Network(
                code="DAS",
                fiber_arrays=(
                    inv.FiberArray(
                        code="L2",
                        optical_paths=(
                            inv.OpticalPath(
                                name="holes",
                                location_code="00",
                                optical_components=(
                                    inv.FiberSegment(name="run", optical_length=300.0),
                                ),
                                labels=labels,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).check()


@pytest.fixture(scope="module")
def site():
    """The inventory most tests draw."""
    return build_site_inventory()


@pytest.fixture(scope="module")
def tunnel():
    """The tunnel example, which has real epochs and a surveyed coil gap."""
    return dc.get_example_inventory("tunnel")


class TestNamespace:
    """The plots hang off inventory.viz."""

    def test_registered(self, tunnel):
        """Inventory.viz is the viz namespace, with the three verbs."""
        assert isinstance(tunnel.viz, VizInventoryNameSpace)
        assert tunnel.viz.path.__name__ == "path"
        assert tunnel.viz.map.__name__ == "map_path"
        assert tunnel.viz.timeline.__name__ == "timeline"

    def test_declared_as_an_entry_point(self):
        """An install must carry the namespace, not just an import of it.

        Importing dascore.viz registers the namespace as a side effect, so
        every other test here would pass with the entry point deleted.
        """
        text = (Path(dc.__file__).parent.parent / "pyproject.toml").read_text()
        block = text.split('[project.entry-points."dascore.inventory_namespace"]')[1]
        block = block.split("[")[0]
        assert 'viz = "dascore.viz:VizInventoryNameSpace"' in block

    def test_namespace_call(self, tunnel):
        """Calling through the namespace passes the inventory."""
        ax = tunnel.viz.timeline()
        assert len(_lanes(ax)) == 2


class TestSelectPath:
    """Naming the path a plot is of."""

    def test_no_paths(self):
        """An inventory without paths has nothing to plot."""
        empty = inv.Inventory(
            networks=(
                inv.Network(code="DAS", fiber_arrays=(inv.FiberArray(code="A"),)),
            )
        )
        with pytest.raises(ParameterError, match="holds no optical paths"):
            path(empty)

    def test_ambiguous(self, site):
        """Several candidates demand a choice, and are listed."""
        with pytest.raises(ParameterError, match="holds 3 optical paths") as info:
            path(site)
        assert "DAS.L1.00 (from the beginning)" in str(info.value)
        assert "DAS.L1.00 (from 2026-07-01)" in str(info.value)

    def test_address_and_time(self, site):
        """An address plus a time picks one epoch."""
        ax = path(site, "DAS.L1.00", time="2026-08-01")
        assert ax.get_title("left").endswith("from 2026-07-01")
        ax = path(site, "DAS.L1.00", time="2026-06-10")
        assert ax.get_title("left").endswith("from the beginning")

    def test_name(self, site):
        """A path's name works where it is unique."""
        ax = path(site, "spur")
        assert ax.get_title("left").startswith("DAS.L1.01")

    def test_unknown_name(self, site):
        """An unknown name lists the addresses."""
        with pytest.raises(ParameterError, match="No optical path matches 'nope'"):
            path(site, "nope")

    def test_object(self, site):
        """The path object itself is accepted, and a foreign one refused."""
        spur = site.networks[0].fiber_arrays[0].optical_paths[2]
        assert path(site, spur).get_title("left").startswith("DAS.L1.01")
        foreign = spur.model_copy()
        with pytest.raises(ParameterError, match="not part of this inventory"):
            path(site, foreign)

    def test_epochs_need_a_time(self, site):
        """An address names a path, so it cannot pick among its epochs."""
        with pytest.raises(ParameterError, match="has 2 epochs") as info:
            path(site, "DAS.L1.00")
        assert "Pass a time" in str(info.value)

    def test_ambiguous_acquisition_key(self, site):
        """A key naming two acquisition epochs asks for a time, in our terms."""
        with pytest.raises(ParameterError, match="does not resolve") as info:
            path(site, acquisition_key="DAS.L1.00.RAW")
        assert "2 acquisitions" in str(info.value)
        assert "Pass a time" in str(info.value)

    def test_unknown_acquisition_key_keeps_its_error(self, site):
        """A key which no time can fix reports what actually went wrong."""
        with pytest.raises(ParameterError, match="does not resolve") as info:
            path(site, acquisition_key="DAS.L1.00.NOPE")
        # The count resolve reported, not a story about epochs.
        assert "0 acquisitions" in str(info.value)

    def test_containers_decide_which_epoch(self):
        """A path stating no time is effective when its containers are."""

        def build(code, **times):
            one = inv.OpticalPath(
                name="main",
                location_code="00",
                optical_components=(inv.FiberSegment(name="f", optical_length=100.0),),
            )
            array = inv.FiberArray(code="L1", optical_paths=(one,), **times)
            return inv.Network(code=code, fiber_arrays=(array,), **times)

        inventory = inv.Inventory(
            networks=(
                build("AA", start_time="2020-01-01", end_time="2021-01-01"),
                build("BB", start_time="2021-01-01"),
            )
        ).check()
        # Neither path states a bound, so only their containers can tell
        # them apart; asking the path alone would call both effective.
        assert path(inventory, time="2020-06-01").get_title("left").startswith("AA")
        assert path(inventory, time="2022-06-01").get_title("left").startswith("BB")
        lanes = timeline(inventory, kind="optical_path")
        assert lanes.get_xlabel() == "Time"
        boxes = _boxes(lanes)[0].get_paths()
        assert len(boxes) == 1

    def test_no_path_effective_then(self):
        """A time nothing is effective at says so, rather than listing paths."""
        one = inv.OpticalPath(
            name="main",
            location_code="00",
            start_time="2020-01-01",
            end_time="2021-01-01",
            optical_components=(inv.FiberSegment(name="f", optical_length=100.0),),
        )
        array = inv.FiberArray(code="L1", optical_paths=(one,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="N", fiber_arrays=(array,)),)
        ).check()
        with pytest.raises(ParameterError, match="is effective at"):
            path(inventory, time="1999-01-01")

    def test_acquisition_key(self, site):
        """An acquisition key resolves through the inventory."""
        ax = path(site, acquisition_key="DAS.L1.00.RAW", time="2026-06-10")
        assert ax.get_title("left").startswith("DAS.L1.00")

    def test_acquisition_key_without_path(self, site):
        """An acquisition on a location with no path cannot be drawn."""
        with pytest.raises(ParameterError, match="resolves to no optical path"):
            path(site, acquisition_key="DAS.L1.02.NIL")


class TestPath:
    """The tracks along one path."""

    def test_all_tracks(self, site):
        """Every track becomes a lane, channels first."""
        ax = path(site, "DAS.L1.00", time="2026-06-10")
        assert _lanes(ax) == [
            "channels (RAW)",
            "components",
            "coupling",
            "zone",
            "noisy",
            "count",
        ]
        assert ax.get_xlabel() == "Optical distance [m]"
        # Components take their fixed colors, so the legend names the types.
        assert "FiberSegment" in _legend_labels(ax)

    def test_a_narrow_figure_keeps_its_legend_on_the_page(self, site):
        """A figure too small to seat a legend must not be made nonsense of.

        The lanes keep some of the figure whatever the legend needs, and
        neither they nor it leave the canvas.
        """
        crowded = build_labeled_inventory(24, site.coordinate_reference_system)
        ax = path(crowded, "DAS.L2.00", figsize=(3.0, 2.0))
        figure = ax.get_figure()
        figure.draw_without_rendering()
        lanes = ax.get_window_extent()
        assert lanes.height > 0 and lanes.y0 >= 0
        assert lanes.y1 <= figure.bbox.height

    def test_a_value_on_two_lines_is_kept_room_for_both(self, site):
        """A legend entry of two lines stands as tall as two of one.

        Counting entries and not lines would keep too little room, and
        the legend would go below into space nobody reserved.
        """
        crs = site.coordinate_reference_system
        flat = path(build_labeled_inventory(12, crs), "DAS.L2.00")
        short = flat.get_figure().get_size_inches()[1]
        plt.close("all")
        tall = path(build_labeled_inventory(12, crs, lines=2), "DAS.L2.00")
        figure = tall.get_figure()
        assert figure.get_size_inches()[1] > short
        figure.draw_without_rendering()
        box = figure.legends[0].get_window_extent(figure.canvas.get_renderer())
        assert box.y0 >= 0 and box.y1 <= tall.get_window_extent().y0

    def test_room_is_kept_for_a_legend_which_names_many_values(self, site):
        """A path naming more values than it has lanes needs a taller figure.

        The legend goes below the lanes there, and without the rows it
        takes the lanes give up the room instead.
        """
        crs = site.coordinate_reference_system
        few = path(build_labeled_inventory(2, crs), "DAS.L2.00")
        few.get_figure().draw_without_rendering()
        short, lanes = few.get_figure().get_size_inches()[1], _lanes(few)
        room = few.get_window_extent().height / few.get_figure().dpi
        plt.close("all")
        many = path(build_labeled_inventory(24, crs), "DAS.L2.00")
        figure = many.get_figure()
        figure.draw_without_rendering()
        # Same one lane either way, so only the legend can move the height.
        assert _lanes(many) == lanes == ["components", "hole"]
        assert figure.get_size_inches()[1] > short
        # The room is kept for the legend, not taken from the lanes: with
        # no allowance at all these lanes lose a third of their height.
        assert many.get_window_extent().height / figure.dpi >= room * 0.9
        box = figure.legends[0].get_window_extent(figure.canvas.get_renderer())
        assert box.y0 >= 0 and box.y1 <= figure.bbox.height

    def test_a_mapping_names_its_numbers_and_only_those(self, site):
        """A number is a colorbar until a mapping gives it a swatch.

        A mapping which names some of them names only those, which is
        what the legend beside them will show.
        """
        frame = pd.DataFrame(
            {"lane": ["count"] * 3, "value": [0, 1, 2], "start": 0.0, "end": 1.0}
        )
        assert _legend_names(frame, None) == []
        assert _legend_names(frame, "red") == []
        assert _legend_names(frame, {0: "red", 1: "blue", 2: "green"}) == [
            "0",
            "1",
            "2",
        ]
        assert _legend_names(frame, {"count": {0: "red"}}) == ["0"]

    def test_a_flat_mapping_names_only_what_it_holds(self, site):
        """Room is kept for the swatches drawn, not for every value."""
        crowded = build_labeled_inventory(24, site.coordinate_reference_system)
        frame = pd.DataFrame(
            {"lane": ["hole"] * 3, "value": ["a", "b", "c"], "start": 0.0, "end": 1.0}
        )
        assert _legend_names(frame, {"a": "red"}) == ["a"]
        # And the figure is no taller for the values it does not name.
        one = path(crowded, "DAS.L2.00", color={"H00": "red"})
        assert one.get_figure().get_size_inches()[1] < 3.0

    def test_one_color_for_every_lane_needs_no_legend_room(self, site):
        """A figure which names no value has no legend to keep room for."""
        crowded = build_labeled_inventory(24, site.coordinate_reference_system)
        named = path(crowded, "DAS.L2.00")
        tall = named.get_figure().get_size_inches()[1]
        plt.close("all")
        plain = path(crowded, "DAS.L2.00", color="red")
        assert plain.get_figure().legends == []
        assert plain.get_figure().get_size_inches()[1] < tall

    def test_tracks_selected_in_order(self, site):
        """tracks= picks lanes and orders them."""
        ax = path(site, "DAS.L1.00", time="2026-06-10", tracks=("zone", "coupling"))
        assert _lanes(ax) == ["zone", "coupling"]
        ax = path(site, "DAS.L1.00", time="2026-06-10", tracks="channels")
        assert _lanes(ax) == ["channels (RAW)"]

    def test_unknown_track(self, site):
        """A track which is not a track nor a label group is refused."""
        with pytest.raises(ParameterError, match="'nope' is not a track"):
            path(site, "DAS.L1.00", time="2026-06-10", tracks="nope")

    def test_tracks_with_nothing(self, site):
        """Asking for a lane the path has no rows for is an error."""
        with pytest.raises(ParameterError, match="nothing to draw for tracks"):
            path(site, "spur", tracks="channels")

    def test_acquisition_not_effective(self, site):
        """An acquisition which overlaps the path but not the time is left out."""
        ax = path(site, "DAS.L1.00", time="2026-06-20")
        assert not any(x.startswith("channels") for x in _lanes(ax))

    def test_point_distance_map(self, site):
        """A single-point distance map draws as a tick, not a guessed span."""
        ax = path(site, "DAS.L1.00", time="2026-08-01", tracks="channels")
        assert ax.lines
        assert ax.lines[0].get_xdata()[0] == 100.0

    def test_columns(self, site):
        """Named columns get panels under the lanes, labelled with units."""
        ax = path(site, "DAS.L1.00", time="2026-06-10", columns=("chainage", "depth"))
        panels = ax.get_figure().axes[1:]
        assert [x.get_ylabel() for x in panels] == ["chainage [m]", "depth"]
        assert panels[-1].get_xlabel() == "Optical distance [m]"
        assert ax.get_xlabel() == ""
        # Depth is stated from 100 to 300 m only, so the line breaks outside.
        xs, ys = panels[1].lines[0].get_data()
        inside = (xs > 100) & (xs < 300)
        assert np.isfinite(ys[inside]).all() and np.isnan(ys[~inside]).all()

    def test_column_string(self, site):
        """A single column name is accepted without a tuple."""
        ax = path(site, "DAS.L1.00", time="2026-06-10", columns="chainage")
        assert len(ax.get_figure().axes) == 2

    def test_position_column_refused(self, site):
        """A CRS axis is drawn by the map, not as a panel."""
        with pytest.raises(ParameterError, match="position axis of the CRS"):
            path(site, "DAS.L1.00", time="2026-06-10", columns="x")

    def test_unknown_column(self, site):
        """A column the path does not state is refused."""
        with pytest.raises(ParameterError, match="no geometry column named 'azimuth'"):
            path(site, "DAS.L1.00", time="2026-06-10", columns="azimuth")

    def test_ax_with_columns_refused(self, site):
        """Panels need their own figure, so ax and columns conflict."""
        _, ax = plt.subplots()
        with pytest.raises(ParameterError, match="builds the figure"):
            path(site, "DAS.L1.00", time="2026-06-10", columns="chainage", ax=ax)

    def test_ax_without_columns(self, site):
        """Lanes alone draw onto an axes a caller provides."""
        _, ax = plt.subplots()
        out = path(site, "DAS.L1.00", time="2026-06-10", ax=ax)
        assert out is ax
        assert ax.get_xlabel() == "Optical distance [m]"

    def test_distance_window(self, site):
        """distance=(low, high) sets the window; None runs to the end."""
        ax = path(site, "DAS.L1.00", time="2026-06-10", distance=(200, 300))
        low, high = ax.get_xlim()
        assert low < 200 and high > 300 and high < 320
        ax = path(site, "DAS.L1.00", time="2026-06-10", distance=(400, None))
        assert ax.get_xlim()[1] > 500

    @pytest.mark.parametrize(
        "asked, match",
        [
            (5, "must be a .low, high. pair"),
            ((10, 5), "must be increasing"),
            ((900, 1000), "clips everything away"),
        ],
    )
    def test_bad_window(self, site, asked, match):
        """A window which is not a window is explained."""
        with pytest.raises(ParameterError, match=match):
            path(site, "DAS.L1.00", time="2026-06-10", distance=asked)

    def test_window_ellipsis(self):
        """An Ellipsis means the same as None at either end."""
        assert _distance_window((..., 10), (0.0, 20.0)) == (0.0, 10.0)
        assert _distance_window((5, ...), (0.0, 20.0)) == (5.0, 20.0)

    def test_zero_length_path(self):
        """A path whose components state no length has no axis."""
        stub = inv.OpticalPath(
            name="stub", location_code="09", optical_components=(inv.Connector(),)
        )
        array = inv.FiberArray(code="A", optical_paths=(stub,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="N", fiber_arrays=(array,)),)
        )
        with pytest.raises(ParameterError, match="has no length"):
            path(inventory)

    def test_color_override_and_figsize(self, site, shown):
        """color= reaches the renderer, figsize the figure, show plt.show."""
        ax = path(
            site,
            "DAS.L1.00",
            time="2026-06-10",
            tracks="coupling",
            color="black",
            figsize=(4, 3),
            show=True,
        )
        assert shown
        assert tuple(ax.get_figure().get_size_inches()) == (4.0, 3.0)
        assert np.allclose(_boxes(ax)[0].get_facecolors()[0][:3], [0, 0, 0])

    def test_components_keep_their_own_colors(self, site):
        """The component vocabulary is closed, so its colors are pinned."""
        ax = path(site, "DAS.L1.00", time="2026-06-10", tracks="components")
        colors = _boxes(ax)[0].get_facecolors()
        expected = plt.matplotlib.colors.to_rgba(COMPONENT_COLORS["FiberSegment"])
        assert np.allclose(colors[0], expected)

    def test_tracks_do_not_move_the_palette(self, site):
        """Drawing some tracks colors them as drawing all of them does."""
        every = path(site, "DAS.L1.00", time="2026-06-10")
        full = {
            x.get_text(): tuple(np.round(y.get_facecolor(), 5))
            for x, y in zip(
                every.get_legend().get_texts(),
                every.get_legend().legend_handles,
                strict=True,
            )
        }
        plt.close("all")
        some = path(site, "DAS.L1.00", time="2026-06-10", tracks=("zone",))
        part = {
            x.get_text(): tuple(np.round(y.get_facecolor(), 5))
            for x, y in zip(
                some.get_legend().get_texts(),
                some.get_legend().legend_handles,
                strict=True,
            )
        }
        shared = set(full) & set(part)
        assert shared
        for name in shared:
            assert full[name] == part[name], f"{name} changed color with tracks="

    def test_a_refusal_leaves_no_figure(self, site):
        """A window which clips everything away builds no figure to leak."""
        plt.close("all")
        before = plt.get_fignums()
        with pytest.raises(ParameterError):
            path(site, "DAS.L1.00", time="2026-06-10", distance=(5000, 6000))
        assert plt.get_fignums() == before

    def test_columns_stay_aligned_with_the_lanes(self):
        """A colorbar must not steal width from the lanes alone."""
        readings = tuple(
            inv.OpticalPathLabel(
                start_distance=100.0 + 10 * index,
                end_distance=110.0 + 10 * index,
                group="reading",
                value=float(index),
            )
            # Enough distinct numbers to earn a colorbar rather than labels.
            for index in range(9)
        )
        one = inv.OpticalPath(
            name="main",
            location_code="00",
            optical_components=(inv.FiberSegment(name="f", optical_length=300.0),),
            geometry=(
                inv.Geometry(
                    name="run",
                    distance=(100.0, 300.0),
                    coordinates={"chainage": (0.0, 200.0)},
                    units={"chainage": "m"},
                ),
            ),
            labels=readings,
        )
        array = inv.FiberArray(code="L1", optical_paths=(one,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="N", fiber_arrays=(array,)),)
        ).check()
        ax = path(inventory, columns="chainage")
        figure = ax.get_figure()
        figure.draw_without_rendering()
        assert len(figure.axes) == 3, "no colorbar was drawn, so nothing is tested"
        panel = figure.axes[1]
        assert ax.get_position().x1 == pytest.approx(panel.get_position().x1, abs=1e-6)

    def test_tunnel_epochs(self, tunnel):
        """The tunnel's repair splits its path; both epochs draw."""
        before = path(tunnel, time="2024-07-01", distance=(1495, 1780))
        after = path(tunnel, time="2024-10-01", distance=(1495, 1780))
        assert before.get_title("left") != after.get_title("left")
        assert _lanes(before) == _lanes(after)


class TestMap:
    """Where the fiber is."""

    def test_default_axes(self, site):
        """The first two CRS axes are the plan view; all paths draw."""
        ax = map_path(site)
        assert ax.get_xlabel() == "x [meter]"
        assert ax.get_ylabel() == "y [meter]"
        lines = [x for x in ax.collections if isinstance(x, LineCollection)]
        # Two epochs of the main path; the spur places nothing.
        assert len(lines) == 2
        assert ax.get_aspect() == 1.0

    def test_gap_breaks_polyline(self, site):
        """Unsurveyed fiber is a break in the line, never a bridge."""
        ax = map_path(site, "DAS.L1.00", time="2026-06-10", x="x", y="z")
        segments = next(
            x for x in ax.collections if isinstance(x, LineCollection)
        ).get_segments()
        xs = np.concatenate([s[:, 0] for s in segments])
        assert not ((xs > 201.0) & (xs < 249.0)).any()
        # A bridge is one long segment with no interior point, so looking
        # only at where samples fell would not see it.
        crossing = [
            s for s in segments if s[:, 0].min() < 205.0 < 245.0 < s[:, 0].max()
        ]
        assert not crossing, "a segment spans fiber nobody placed"

    def test_a_short_gap_is_still_a_gap(self):
        """A gap narrower than the sample spacing still breaks the line."""
        one = inv.OpticalPath(
            name="long",
            location_code="00",
            optical_components=(inv.FiberSegment(name="f", optical_length=100_000.0),),
            geometry=(
                inv.Geometry(
                    name="west",
                    distance=(0.0, 50_000.0),
                    coordinates={"x": (0.0, 500.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                ),
                inv.Geometry(
                    name="east",
                    distance=(50_010.0, 100_000.0),
                    coordinates={
                        "x": (600.0, 1000.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                    },
                ),
            ),
        )
        array = inv.FiberArray(code="L1", optical_paths=(one,))
        inventory = inv.Inventory(
            coordinate_reference_system=inv.CoordinateReferenceSystem(
                authority="",
                code="",
                name="grid",
                coordinate_labels=("x", "y", "z"),
                units=("meter", "meter", "meter"),
            ),
            networks=(inv.Network(code="N", fiber_arrays=(array,)),),
        ).check()
        # 10 m of gap on a 100 km path: a 1000-point grid steps over it.
        ax = map_path(inventory, x="x", y="y")
        segments = next(
            x for x in ax.collections if isinstance(x, LineCollection)
        ).get_segments()
        crossing = [
            s for s in segments if s[:, 0].min() < 505.0 < 595.0 < s[:, 0].max()
        ]
        assert not crossing

    def test_time_filters(self, site):
        """A time keeps only the epochs valid then."""
        ax = map_path(site, time="2026-08-01")
        lines = [x for x in ax.collections if isinstance(x, LineCollection)]
        assert len(lines) == 1

    def test_same_axis(self, site):
        """X and y must differ."""
        with pytest.raises(ParameterError, match="both 'x'"):
            map_path(site, x="x", y="x")

    def test_not_an_axis(self, site):
        """A non-axis column is refused with a pointer at path()."""
        with pytest.raises(ParameterError, match="is not an axis"):
            map_path(site, x="chainage", y="y")

    def test_nothing_placed(self, site):
        """A path with no geometry cannot be mapped."""
        with pytest.raises(ParameterError, match="places itself in the CRS"):
            map_path(site, "spur")

    def test_color_distance_colorbar(self, site):
        """The default coloring earns a distance colorbar."""
        ax = map_path(site, "DAS.L1.00", time="2026-06-10")
        assert "Optical distance" in _bar_label(ax)

    def test_scale_covers_what_the_view_shows(self, tunnel):
        """Fiber a projection collapses to a point spends no colormap."""
        plan = map_path(tunnel, time="2024-07-01")
        flat = next(x for x in plan.collections if isinstance(x, LineCollection))
        plt.close("all")
        section = map_path(tunnel, x="x", y="z", time="2024-07-01")
        deep = next(x for x in section.collections if isinstance(x, LineCollection))
        # Seen from above the boreholes are points, so the trench gets the
        # whole scale; side-on they are 20 m of visible fiber and count.
        assert flat.norm.vmax < deep.norm.vmax
        drawn = np.asarray(deep.get_array())
        assert flat.norm.vmax < drawn.max()

    def test_a_view_which_shows_no_length(self):
        """Where every segment collapses, the scale still spans the values."""
        one = inv.OpticalPath(
            name="hole",
            location_code="00",
            optical_components=(inv.FiberSegment(name="f", optical_length=40.0),),
            geometry=(
                inv.Geometry(
                    name="down",
                    distance=(0.0, 40.0),
                    # Straight down: nothing to see in plan view at all.
                    coordinates={"x": (5.0, 5.0), "y": (2.0, 2.0), "z": (0.0, -40.0)},
                ),
            ),
        )
        array = inv.FiberArray(code="L1", optical_paths=(one,))
        inventory = inv.Inventory(
            coordinate_reference_system=inv.CoordinateReferenceSystem(
                authority="",
                code="",
                name="grid",
                coordinate_labels=("x", "y", "z"),
                units=("meter", "meter", "meter"),
            ),
            networks=(inv.Network(code="N", fiber_arrays=(array,)),),
        ).check()
        ax = map_path(inventory)
        line = next(x for x in ax.collections if isinstance(x, LineCollection))
        assert line.norm.vmax > line.norm.vmin

    def test_discrete_values_get_a_stepped_scale(self, tunnel):
        """Three boreholes are three categories, not a ramp through 1.5."""
        ax = map_path(tunnel, x="x", y="z", color="borehole", time="2024-07-01")
        bar = ax.get_figure().axes[-1]
        ticks = [x for x in bar.get_yticks() if x] or list(bar.get_xticks())
        assert [round(float(x), 3) for x in ticks] == [1.0, 2.0, 3.0]

    def test_one_number_is_not_a_scale(self):
        """A column stating one value everywhere still draws."""

        def build(location, value, group="reading"):
            return inv.OpticalPath(
                name=f"p{location}",
                location_code=location,
                optical_components=(inv.FiberSegment(name="f", optical_length=200.0),),
                geometry=(
                    inv.Geometry(
                        name="run",
                        distance=(0.0, 200.0),
                        coordinates={
                            "x": (0.0, 100.0),
                            "y": (float(location), float(location)),
                            "z": (0.0, 0.0),
                        },
                    ),
                ),
                labels=(
                    (
                        inv.OpticalPathLabel(
                            start_distance=0.0,
                            end_distance=200.0,
                            group=group,
                            value=value,
                        ),
                    )
                    if value is not None
                    else ()
                ),
            )

        def wrap(*paths):
            array = inv.FiberArray(code="L1", optical_paths=paths)
            return inv.Inventory(
                coordinate_reference_system=inv.CoordinateReferenceSystem(
                    authority="",
                    code="",
                    name="grid",
                    coordinate_labels=("x", "y", "z"),
                    units=("meter", "meter", "meter"),
                ),
                networks=(inv.Network(code="N", fiber_arrays=(array,)),),
            ).check()

        ax = map_path(wrap(build("01", 7.0)), color="reading")
        line = next(x for x in ax.collections if isinstance(x, LineCollection))
        assert line.norm.vmax > line.norm.vmin

        # One path states the number, the other says nothing under that
        # name, so the drawn pieces are a mixture of scaled and unscaled.
        plt.close("all")
        ax = map_path(wrap(build("01", 7.0), build("02", None)), color="reading")
        assert "n/a" in _legend_labels(ax)
        assert len([x for x in ax.collections if isinstance(x, LineCollection)]) == 2

    def test_color_column(self, site):
        """A geometry column colors continuously, labelled by its name."""
        ax = map_path(site, "DAS.L1.00", time="2026-06-10", color="chainage")
        assert _bar_label(ax) == "chainage"

    def test_color_label_group(self, site):
        """A string label group gives a legend, with unplaced fiber named."""
        ax = map_path(site, "DAS.L1.00", time="2026-06-10", color="zone")
        labels = _legend_labels(ax)
        assert labels[:2] == ["north", "south"]
        assert "n/a" in labels
        assert ax.get_legend().get_title().get_text() == "zone"

    def test_color_a_membership_group(self, site):
        """A group everything belongs to is named by the group, not by None."""
        ax = map_path(site, "DAS.L1.00", time="2026-06-10", color="noisy")
        labels = _legend_labels(ax)
        assert "noisy" in labels
        assert "None" not in labels

    def test_color_numeric_group(self, site):
        """A numeric label group colors continuously."""
        ax = map_path(site, "DAS.L1.00", time="2026-06-10", color="count")
        assert _bar_label(ax) == "count"

    def test_color_coupling(self, site):
        """Coupling types color the fiber."""
        ax = map_path(site, "DAS.L1.00", time="2026-06-10", color="coupling")
        assert _legend_labels(ax)[:2] == ["trench", "conduit"]

    def test_unstated_numeric_is_drawn(self, site):
        """Fiber whose color value is unstated is drawn grey, not made invisible."""
        ax = map_path(site, "DAS.L1.00", time="2026-06-10", color="count")
        collection = next(x for x in ax.collections if isinstance(x, LineCollection))
        # The cable is placed from 350 m on, but states no count there, so
        # those segments are masked and take the colormap's "bad" color.
        assert np.ma.getmaskarray(collection.get_array()).any()
        bad = collection.get_cmap().get_bad()
        assert bad[3] == pytest.approx(1.0), "unstated fiber would be invisible"
        assert "n/a" in _legend_labels(ax)

    def test_one_palette_for_every_path(self):
        """A value is one color across the paths of one figure."""

        def build(location, values):
            return inv.OpticalPath(
                name=f"p{location}",
                location_code=location,
                optical_components=(inv.FiberSegment(name="f", optical_length=200.0),),
                geometry=(
                    inv.Geometry(
                        name="run",
                        distance=(0.0, 200.0),
                        coordinates={
                            "x": (0.0, 100.0),
                            "y": (float(location), float(location)),
                            "z": (0.0, 0.0),
                        },
                    ),
                ),
                labels=tuple(
                    inv.OpticalPathLabel(
                        start_distance=100.0 * index,
                        end_distance=100.0 * (index + 1),
                        group="zone",
                        value=value,
                    )
                    for index, value in enumerate(values)
                ),
            )

        # The two paths state the same two values in opposite order, so a
        # palette built per path would give each value two colors.
        array = inv.FiberArray(
            code="L1",
            optical_paths=(
                build("01", ("north", "south")),
                build("02", ("south", "north")),
            ),
        )
        inventory = inv.Inventory(
            coordinate_reference_system=inv.CoordinateReferenceSystem(
                authority="",
                code="",
                name="grid",
                coordinate_labels=("x", "y", "z"),
                units=("meter", "meter", "meter"),
            ),
            networks=(inv.Network(code="N", fiber_arrays=(array,)),),
        ).check()
        ax = map_path(inventory, color="zone")
        lines = [x for x in ax.collections if isinstance(x, LineCollection)]
        assert len(lines) == 2
        first, second = (x.get_colors() for x in lines)
        # Path 01 begins in north and ends in south; path 02 is the other
        # way round. A palette built per path would give both first
        # segments color zero, so it is the crossed pairs which tell.
        assert np.allclose(first[0], second[-1]), "north has two colors"
        assert np.allclose(second[0], first[-1]), "south has two colors"
        assert not np.allclose(first[0], second[0])
        assert [x.get_text() for x in ax.get_legend().get_texts()] == ["north", "south"]

    def test_shared_color_scale_across_paths(self):
        """One numeric scale spans every path, not one scale each."""

        def build(location, value):
            return inv.OpticalPath(
                name=f"p{location}",
                location_code=location,
                optical_components=(inv.FiberSegment(name="f", optical_length=200.0),),
                geometry=(
                    inv.Geometry(
                        name="run",
                        distance=(0.0, 200.0),
                        coordinates={
                            "x": (0.0, 100.0),
                            "y": (float(location), float(location)),
                            "z": (0.0, 0.0),
                        },
                    ),
                ),
                labels=(
                    inv.OpticalPathLabel(
                        start_distance=0.0,
                        end_distance=100.0,
                        group="reading",
                        value=value,
                    ),
                    inv.OpticalPathLabel(
                        start_distance=100.0,
                        end_distance=200.0,
                        group="reading",
                        value=value + 1.0,
                    ),
                ),
            )

        # One path states 0-1, the other 100-101. Normalized per path they
        # would take identical colors despite stating different numbers.
        array = inv.FiberArray(
            code="L1", optical_paths=(build("01", 0.0), build("02", 100.0))
        )
        inventory = inv.Inventory(
            coordinate_reference_system=inv.CoordinateReferenceSystem(
                authority="",
                code="",
                name="grid",
                coordinate_labels=("x", "y", "z"),
                units=("meter", "meter", "meter"),
            ),
            networks=(inv.Network(code="N", fiber_arrays=(array,)),),
        ).check()
        ax = map_path(inventory, color="reading")
        lines = [
            x
            for x in ax.collections
            if isinstance(x, LineCollection) and x.get_array() is not None
        ]
        assert len(lines) == 2
        # One scale object, spanning what both paths state.
        assert lines[0].norm is lines[1].norm
        norm = lines[0].norm
        assert norm.vmin <= 0.0 and norm.vmax >= 101.0
        assert norm(0.0) != norm(100.0)

    def test_a_path_without_the_color_is_unstated(self):
        """A placed path saying nothing under that name is drawn, not fatal."""

        def build(location, labels=()):
            return inv.OpticalPath(
                name=f"p{location}",
                location_code=location,
                optical_components=(inv.FiberSegment(name="f", optical_length=200.0),),
                geometry=(
                    inv.Geometry(
                        name="run",
                        distance=(0.0, 200.0),
                        coordinates={
                            "x": (0.0, 100.0),
                            "y": (float(location), float(location)),
                            "z": (0.0, 0.0),
                        },
                    ),
                ),
                labels=labels,
            )

        zoned = build(
            "01",
            (
                inv.OpticalPathLabel(
                    start_distance=0.0, end_distance=200.0, group="zone", value="north"
                ),
            ),
        )
        array = inv.FiberArray(code="L1", optical_paths=(zoned, build("02")))
        inventory = inv.Inventory(
            coordinate_reference_system=inv.CoordinateReferenceSystem(
                authority="",
                code="",
                name="grid",
                coordinate_labels=("x", "y", "z"),
                units=("meter", "meter", "meter"),
            ),
            networks=(inv.Network(code="N", fiber_arrays=(array,)),),
        ).check()
        ax = map_path(inventory, color="zone")
        assert len([x for x in ax.collections if isinstance(x, LineCollection)]) == 2
        assert _legend_labels(ax) == ["north", "n/a"]

    def test_map_needs_a_path_effective_then(self):
        """A time no path is effective at draws nothing, and says why."""
        one = inv.OpticalPath(
            name="main",
            location_code="00",
            start_time="2020-01-01",
            end_time="2021-01-01",
            optical_components=(inv.FiberSegment(name="f", optical_length=100.0),),
            geometry=(
                inv.Geometry(
                    name="run",
                    distance=(0.0, 100.0),
                    coordinates={"x": (0.0, 1.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                ),
            ),
        )
        array = inv.FiberArray(code="L1", optical_paths=(one,))
        inventory = inv.Inventory(
            coordinate_reference_system=inv.CoordinateReferenceSystem(
                authority="",
                code="",
                name="grid",
                coordinate_labels=("x", "y", "z"),
                units=("meter", "meter", "meter"),
            ),
            networks=(inv.Network(code="N", fiber_arrays=(array,)),),
        ).check()
        with pytest.raises(ParameterError, match="is effective at"):
            map_path(inventory, time="1999-01-01")

    def test_unknown_color_lists_what_the_inventory_states(self, site):
        """A name no path states is refused, and the message says what is."""
        with pytest.raises(ParameterError, match="names neither") as info:
            map_path(site, color="nope")
        assert "zone" in str(info.value)

    def test_color_unknown(self, site):
        """An unknown coloring lists what would work."""
        with pytest.raises(ParameterError, match="names neither"):
            map_path(site, "DAS.L1.00", time="2026-06-10", color="nope")

    def test_legend_off_and_ax(self, site):
        """legend=False draws none; a given ax keeps its figure size."""
        figure, ax = plt.subplots(figsize=(3, 3))
        out = map_path(site, "DAS.L1.00", time="2026-06-10", ax=ax, legend=False)
        assert out is ax
        assert len(figure.axes) == 1
        assert tuple(figure.get_size_inches()) == (3.0, 3.0)

    def test_explicit_aspect(self, site):
        """aspect= is honored."""
        ax = map_path(site, aspect=2.0)
        assert ax.get_aspect() == 2.0

    def test_geographic_aspect(self):
        """Degrees are not metres, so a geographic map is not forced equal."""
        ax = map_path(dc.get_example_inventory("random_das"))
        assert ax.get_aspect() == "auto"
        assert "degree" in ax.get_xlabel()

    def test_tunnel_coil(self, tunnel):
        """The tunnel's slack coil is a gap in its section view."""
        ax = map_path(tunnel, x="x", y="z", color="section", time="2024-07-01")
        assert "borehole" in _legend_labels(ax)

    def test_show(self, site, shown):
        """Show calls plt.show."""
        map_path(site, show=True)
        assert shown


class TestTimeline:
    """When each part was valid."""

    def test_lanes_and_interrogators(self, site):
        """Paths and acquisitions each get a lane; colors name interrogators."""
        ax = timeline(site)
        lanes = _lanes(ax)
        assert lanes[:2] == ["DAS.L1.00 [path]", "DAS.L1.01 [path]"]
        assert "DAS.L1.00.RAW" in lanes and "DAS.L1.02.NIL" in lanes
        labels = _legend_labels(ax)
        for expected in ("optical path", "Fake FI-1", "sn-9", "interrogator"):
            assert expected in labels
        assert "no interrogator" in labels
        assert ax.get_xlabel() == "Time"

    def test_kind(self, site):
        """kind= keeps only acquisitions or only paths."""
        assert all("[path]" in x for x in _lanes(timeline(site, kind="optical_path")))
        assert not any(
            "[path]" in x for x in _lanes(timeline(site, kind="acquisition"))
        )

    def test_color_data_type(self, site):
        """Data type coloring names the unstated ones."""
        labels = _legend_labels(timeline(site, color="data_type"))
        assert "strain_rate" in labels and "unstated" in labels

    def test_color_kind(self, site):
        """Kind coloring has two entries."""
        labels = _legend_labels(timeline(site, color="kind"))
        assert sorted(labels) == ["acquisition", "optical path"]

    @pytest.mark.parametrize("bad", [dict(kind="nope"), dict(color="nope")])
    def test_bad_options(self, site, bad):
        """Unknown kind or color is refused."""
        with pytest.raises(ParameterError, match="nope"):
            timeline(site, **bad)

    def test_time_window(self, site):
        """time=(start, end) sets the axis and leaves out epochs beyond it."""
        ax = timeline(site, time=("2026-06-01", "2026-06-30"))
        low, high = ax.get_xlim()
        assert high - low == pytest.approx(29.0)
        # The repaired path and its acquisition start in July.
        assert len(_boxes(ax)[0].get_paths()) == 1
        assert "DAS.L1.00.RAW" in _lanes(ax)

    def test_open_epochs_are_hatched(self, site):
        """An epoch stating no bound runs off that side of the axis."""
        ax = timeline(site, kind="acquisition")
        hatched = [x for x in _boxes(ax) if x.get_hatch()]
        assert hatched, "an unbounded epoch drew no open edge"

    def test_half_open_window(self, site):
        """One end of the window may be left to the data."""
        ax = timeline(site, time=("2026-06-20", None))
        low, high = ax.get_xlim()
        assert low < high
        ax = timeline(site, time=(None, "2026-06-20"))
        assert ax.get_xlim()[0] < ax.get_xlim()[1]

    @pytest.mark.parametrize(
        "bad, match",
        [
            (("2026-07-01", "2026-06-01"), "must be increasing"),
            ("nope", "must be a .start, end. pair"),
            (("not a time", None), "not a time"),
        ],
    )
    def test_bad_time_window(self, site, bad, match):
        """A window which is not a window is refused before anything is drawn."""
        plt.close("all")
        with pytest.raises(ParameterError, match=match):
            timeline(site, time=bad)
        assert plt.get_fignums() == []

    def test_window_on_an_inventory_stating_no_time(self):
        """A window still works where the epochs state nothing themselves."""
        undated = dc.get_example_inventory("random_das")
        low, high = timeline(undated, time=("2026-01-01", None)).get_xlim()
        assert high > low
        low, high = timeline(undated, time=(None, "2026-01-01")).get_xlim()
        assert high > low

    def test_a_bound_which_is_not_a_time(self, site):
        """A bound which parses to nothing is refused like any other."""
        with pytest.raises(ParameterError, match="not a time"):
            timeline(site, time=(float("nan"), None))

    def test_time_window_empty(self):
        """A window nothing falls in is an error, not a blank figure."""
        acquisition = inv.Acquisition(
            code="RAW",
            location_code="00",
            start_time="2026-06-01",
            end_time="2026-06-15",
            data_category="DAS",
            sample_rate=1.0,
            gauge_length=1.0,
        )
        array = inv.FiberArray(code="A", acquisitions=(acquisition,))
        bounded = inv.Inventory(
            networks=(inv.Network(code="N", fiber_arrays=(array,)),)
        )
        with pytest.raises(ParameterError, match="falls within time"):
            timeline(bounded, time=("2020-01-01", "2020-02-01"))
        # And a window after it, since the epoch states both of its bounds.
        with pytest.raises(ParameterError, match="falls within time"):
            timeline(bounded, time=("2030-01-01", "2030-02-01"))

    def test_window_excludes_touching_epoch(self):
        """An epoch which ends where the window starts does not overlap it."""
        acquisition = inv.Acquisition(
            code="RAW",
            location_code="00",
            start_time="2026-06-01",
            end_time="2026-06-15",
            data_category="DAS",
            sample_rate=1.0,
            gauge_length=1.0,
        )
        array = inv.FiberArray(code="A", acquisitions=(acquisition,))
        bounded = inv.Inventory(
            networks=(inv.Network(code="N", fiber_arrays=(array,)),)
        )
        with pytest.raises(ParameterError, match="falls within time"):
            timeline(bounded, time=("2026-06-15", "2026-07-01"))
        with pytest.raises(ParameterError, match="falls within time"):
            timeline(bounded, time=("2026-05-01", "2026-06-01"))

    def test_no_epochs(self):
        """An inventory whose epochs state no time still draws, and says so."""
        ax = timeline(dc.get_example_inventory("random_das"))
        assert "states one" in ax.get_xlabel()
        assert list(ax.get_xticks()) == []

    def test_nothing_to_draw(self):
        """No acquisitions and no paths is an error."""
        empty = inv.Inventory(
            networks=(
                inv.Network(code="DAS", fiber_arrays=(inv.FiberArray(code="A"),)),
            )
        )
        with pytest.raises(ParameterError, match="nothing with a time epoch"):
            timeline(empty)

    def test_ax_and_show(self, site, shown):
        """A given ax is drawn on; show calls plt.show."""
        _, ax = plt.subplots()
        assert timeline(site, ax=ax, show=True) is ax
        assert shown

    def test_tunnel_repair(self, tunnel):
        """The tunnel's path lane holds two epochs split at the repair."""
        ax = timeline(tunnel, kind="optical_path")
        boxes = _boxes(ax)[0]
        assert len(boxes.get_paths()) == 2
