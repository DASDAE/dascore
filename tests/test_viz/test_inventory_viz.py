"""Tests for the plots an inventory draws of itself."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PatchCollection

import dascore as dc
from dascore.core import inventory as inv
from dascore.exceptions import ParameterError
from dascore.viz import VizInventoryNameSpace
from dascore.viz.inventory import _distance_window, map_path, path, timeline


def _lanes(ax):
    """The lane names an axes shows, top to bottom."""
    return [x.get_text() for x in ax.get_yticklabels()]


def _boxes(ax):
    """The patch collections on an axes."""
    return [x for x in ax.collections if isinstance(x, PatchCollection)]


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
            inv.OpticalPathLabel(
                start_distance=150.0, end_distance=300.0, group="noisy", value=True
            ),
            inv.OpticalPathLabel(
                start_distance=300.0, end_distance=400.0, group="noisy", value=False
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
        assert ax.get_title().endswith("from 2026-07-01")
        ax = path(site, "DAS.L1.00", time="2026-06-10")
        assert ax.get_title().endswith("from the beginning")

    def test_name(self, site):
        """A path's name works where it is unique."""
        ax = path(site, "spur")
        assert ax.get_title().startswith("DAS.L1.01")

    def test_unknown_name(self, site):
        """An unknown name lists the addresses."""
        with pytest.raises(ParameterError, match="No optical path matches 'nope'"):
            path(site, "nope")

    def test_object(self, site):
        """The path object itself is accepted, and a foreign one refused."""
        spur = site.networks[0].fiber_arrays[0].optical_paths[2]
        assert path(site, spur).get_title().startswith("DAS.L1.01")
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
        with pytest.raises(ParameterError, match="names more than one"):
            path(site, acquisition_key="DAS.L1.00.RAW")

    def test_acquisition_key(self, site):
        """An acquisition key resolves through the inventory."""
        ax = path(site, acquisition_key="DAS.L1.00.RAW", time="2026-06-10")
        assert ax.get_title().startswith("DAS.L1.00")

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

    def test_color_override_and_figsize(self, site, monkeypatch):
        """color= reaches the renderer, figsize the figure, show plt.show."""
        called = []
        monkeypatch.setattr(plt, "show", lambda: called.append(True))
        ax = path(
            site,
            "DAS.L1.00",
            time="2026-06-10",
            tracks="coupling",
            color="black",
            figsize=(4, 3),
            show=True,
        )
        assert called
        assert tuple(ax.get_figure().get_size_inches()) == (4.0, 3.0)
        assert np.allclose(_boxes(ax)[0].get_facecolors()[0][:3], [0, 0, 0])

    def test_tunnel_epochs(self, tunnel):
        """The tunnel's repair splits its path; both epochs draw."""
        before = path(tunnel, time="2024-07-01", distance=(1495, 1780))
        after = path(tunnel, time="2024-10-01", distance=(1495, 1780))
        assert before.get_title() != after.get_title()
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
        bar = ax.get_figure().axes[-1]
        assert "Optical distance" in bar.get_ylabel()

    def test_color_column(self, site):
        """A geometry column colors continuously, labelled by its name."""
        ax = map_path(site, "DAS.L1.00", time="2026-06-10", color="chainage")
        assert ax.get_figure().axes[-1].get_ylabel() == "chainage"

    def test_color_label_group(self, site):
        """A string label group gives a legend, with unplaced fiber named."""
        ax = map_path(site, "DAS.L1.00", time="2026-06-10", color="zone")
        labels = _legend_labels(ax)
        assert labels[:2] == ["north", "south"]
        assert "not stated" in labels
        assert ax.get_legend().get_title().get_text() == "zone"

    def test_color_numeric_group(self, site):
        """A numeric label group colors continuously."""
        ax = map_path(site, "DAS.L1.00", time="2026-06-10", color="count")
        assert ax.get_figure().axes[-1].get_ylabel() == "count"

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
        assert "not stated" in _legend_labels(ax)

    def test_one_palette_for_every_path(self, site):
        """A value is one color across the paths of one figure."""
        ax = map_path(site, color="zone")
        lines = [x for x in ax.collections if isinstance(x, LineCollection)]
        assert len(lines) == 2
        first, second = (x.get_colors() for x in lines)
        # Both epochs state north then south, so their colors must agree.
        assert np.allclose(first[0], second[0])
        assert len(set(map(tuple, np.vstack([first, second])))) == 3

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

    def test_show(self, site, monkeypatch):
        """Show calls plt.show."""
        called = []
        monkeypatch.setattr(plt, "show", lambda: called.append(True))
        map_path(site, show=True)
        assert called


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

    def test_ax_and_show(self, site, monkeypatch):
        """A given ax is drawn on; show calls plt.show."""
        called = []
        monkeypatch.setattr(plt, "show", lambda: called.append(True))
        _, ax = plt.subplots()
        assert timeline(site, ax=ax, show=True) is ax
        assert called

    def test_tunnel_repair(self, tunnel):
        """The tunnel's path lane holds two epochs split at the repair."""
        ax = timeline(tunnel, kind="optical_path")
        boxes = _boxes(ax)[0]
        assert len(boxes.get_paths()) == 2
