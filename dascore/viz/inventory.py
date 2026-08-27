"""Visualizations of an inventory: its path, its layout, and its epochs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch as PatchArtist

from dascore.exceptions import InvalidInventoryError, ParameterError
from dascore.utils.intervals import interval_masks, normalize_value, value_kind
from dascore.utils.plotting import _format_time_axis, _get_ax

from . import _lanes
from ._lanes import UNCOVERED_COLOR, _default_label, plot_lanes

if TYPE_CHECKING:
    from dascore.constants import timeable_types
    from dascore.core.inventory import Inventory, OpticalPath

# Components are a closed set, so their colors can be too.
# Okabe-Ito, so the components stay apart from each other under color
# blindness and off the hue grid the label groups are drawn from.
COMPONENT_COLORS = {
    "FiberSegment": "#0072B2",
    "Splice": "#E69F00",
    "Connector": "#009E73",
    "Terminator": "#CC79A7",
}


def _iter_paths(inventory):
    """Yield every optical path with the address which names it."""
    for network in inventory.networks:
        for array in network.fiber_arrays:
            for path in array.optical_paths:
                address = f"{network.code}.{array.code}.{path.location_code}"
                yield address, network, array, path


def _effective_epoch(*models):
    """The epoch a model is really valid over, clipped by its containers."""
    start, end = pd.NaT, pd.NaT
    for model in models:
        low, high = model.time_min, model.time_max
        if not pd.isnull(low):
            start = low if pd.isnull(start) else max(start, low)
        if not pd.isnull(high):
            end = high if pd.isnull(end) else min(end, high)
    return start, end


def _effective_at(time, *models) -> bool:
    """Whether a time falls inside every one of these epochs.

    A child which states no bound defers to its container, so asking the
    path alone would call it valid whenever the path is, which is every
    time it states nothing.
    """
    if time is None:
        return True
    return all(x.is_effective_at(time) for x in models)


def _sample_distances(path, low: float, high: float, count: int) -> np.ndarray:
    """Distances to read a path's geometry at, between low and high.

    A uniform grid can step straight over an unsurveyed stretch shorter
    than its spacing, and the picture would then bridge fiber nobody
    placed. Every gap contributes a sample, so every gap is seen.
    """
    spans = sorted(
        (float(min(x.distance)), float(max(x.distance))) for x in path.geometry
    )
    covered: list[list[float]] = []
    for start, end in spans:
        if covered and start <= covered[-1][1]:
            covered[-1][1] = max(covered[-1][1], end)
        else:
            covered.append([start, end])
    holes = [
        0.5 * (covered[index][1] + covered[index + 1][0])
        for index in range(len(covered) - 1)
    ]
    inside = [x for x in holes if low < x < high]
    grid = np.linspace(low, high, count)
    if not inside:
        return grid
    return np.unique(np.concatenate([grid, np.asarray(inside, dtype=float)]))


def _epoch_label(path) -> str:
    """Name a path epoch by when it starts, for a chart title."""
    if pd.isnull(path.time_min):
        return "from the beginning"
    return f"from {str(path.time_min)[:10]}"


def _select_path(inventory, optical_path=None, acquisition_key=None, time=None):
    """Return the (address, array, path) a caller means, or explain."""
    found = list(_iter_paths(inventory))
    if not found:
        msg = "This inventory holds no optical paths, so there is nothing to plot."
        raise ParameterError(msg)
    if acquisition_key is not None:
        try:
            context = inventory.resolve(acquisition_key, time)
        except InvalidInventoryError as error:
            # Keep what resolve said; a key can fail for reasons no time
            # fixes, and claiming ambiguity would hide them.
            hint = (
                ""
                if time is not None
                else " Pass a time as well, if the key names several epochs."
            )
            msg = (
                f"Acquisition key {acquisition_key!r} does not resolve to one "
                f"acquisition: {error}{hint}"
            )
            raise ParameterError(msg) from error
        if context.optical_path is None:
            msg = (
                f"Acquisition key {acquisition_key!r} resolves to no optical "
                "path, so there is nothing to draw against optical distance."
            )
            raise ParameterError(msg)
        for address, _, array, path in found:
            if path is context.optical_path:
                return address, array, path
    if optical_path is not None and not isinstance(optical_path, str):
        for address, _, array, path in found:
            if path is optical_path:
                return address, array, path
        msg = "That optical path is not part of this inventory."
        raise ParameterError(msg)
    candidates = found
    if time is not None:
        candidates = [x for x in found if _effective_at(time, x[1], x[2], x[3])]
        if not candidates:
            stated = sorted({f"{x[0]} ({_epoch_label(x[3])})" for x in found})
            msg = (
                f"No optical path is effective at {time}. The paths are: "
                + ", ".join(stated)
                + "."
            )
            raise ParameterError(msg)
    if optical_path is not None:
        matched = [x for x in candidates if x[0] == optical_path]
        if not matched:
            matched = [x for x in candidates if x[3].name == optical_path]
        if not matched:
            names = sorted({x[0] for x in candidates})
            msg = f"No optical path matches {optical_path!r}. The paths are: {names}."
            raise ParameterError(msg)
        candidates = matched
    if len(candidates) == 1:
        address, _, array, path = candidates[0]
        return address, array, path
    names = sorted({f"{x[0]} ({_epoch_label(x[3])})" for x in candidates})
    if len({x[0] for x in candidates}) == 1:
        # One address, several epochs of it: only a time tells them apart.
        msg = (
            f"Optical path {candidates[0][0]!r} has {len(candidates)} epochs, "
            "so which one to plot must be stated. Pass a time, since an "
            "address names the path rather than one epoch of it. The epochs "
            "are: " + ", ".join(names) + "."
        )
        raise ParameterError(msg)
    msg = (
        f"This inventory holds {len(candidates)} optical paths, so which one "
        "to plot must be stated. Pass optical_path=<address>, "
        "acquisition_key=<key>, or a time. The paths are: " + ", ".join(names) + "."
    )
    raise ParameterError(msg)


def _path_acquisitions(array, path, time=None):
    """The acquisitions which interrogate a path while it is valid."""
    out = []
    for acquisition in array.acquisitions:
        if acquisition.location_code != path.location_code:
            continue
        if not acquisition.overlaps(path):
            continue
        if time is not None and not acquisition.is_effective_at(time):
            continue
        out.append(acquisition)
    return out


def _track_frame(path, acquisitions) -> pd.DataFrame:
    """Flatten a path's tracks into one frame of intervals."""
    rows = []
    for acquisition in acquisitions:
        dist_map = acquisition.distance_map
        if dist_map is None:
            continue
        distances = dist_map.distance
        rows.append(
            {
                "lane": f"acquisition ({acquisition.code})",
                "start": float(distances[0]),
                "end": float(distances[-1]),
                "value": acquisition.code,
                "label": acquisition.code,
            }
        )
    for component in path.optical_components:
        rows.append(
            {
                "lane": "components",
                "start": component.distance_min,
                "end": component.distance_max,
                "value": type(component).__name__,
                "label": component.name or type(component).__name__,
            }
        )
    for coupling in path.coupling:
        rows.append(
            {
                "lane": "coupling",
                "start": coupling.distance_min,
                "end": coupling.distance_max,
                "value": coupling.coupling_type,
                "label": coupling.coupling_type,
            }
        )
    for item in path.labels:
        rows.append(
            {
                "lane": item.group,
                "start": item.distance_min,
                "end": item.distance_max,
                "value": item.value,
                # The renderer's own rule for what a value reads as.
                "label": _default_label(item.value),
            }
        )
    return pd.DataFrame(rows)


TRACKS = ("acquisition", "components", "coupling")


def _column_panels(path, columns, crs) -> list[str]:
    """Decide which geometry columns get their own line panel."""
    if not columns:
        return []
    axes = tuple(crs.coordinate_labels)
    stated = [x for x in path.geometry_columns() if x not in axes]
    wanted = [columns] if isinstance(columns, str) else list(columns)
    for name in wanted:
        if name in axes:
            msg = (
                f"{name!r} is a position axis of the CRS, which map() draws; "
                f"path() draws the columns along the fiber, here {tuple(stated)}."
            )
            raise ParameterError(msg)
        if name not in stated:
            msg = (
                f"This optical path states no geometry column named {name!r}; "
                f"it states {tuple(stated)}."
            )
            raise ParameterError(msg)
    return wanted


def _column_units(path, name) -> str:
    """The units a geometry column is stated in, if any."""
    for segment in path.geometry:
        if name in segment.units:
            return segment.units[name]
    return ""


def _select_tracks(frame, tracks, path):
    """Keep only the lanes a caller asked for, in the order asked."""
    if tracks is None:
        return frame
    groups = tuple(dict.fromkeys(x.group for x in path.labels))
    wanted = [tracks] if isinstance(tracks, str) else list(tracks)
    keep = []
    for name in wanted:
        if name == "acquisition":
            keep.extend(
                x for x in dict.fromkeys(frame["lane"]) if x.startswith("acquisition (")
            )
        elif name in TRACKS or name in groups:
            keep.append(name)
        else:
            msg = (
                f"{name!r} is not a track of this optical path; the tracks are "
                f"{TRACKS} and the label groups are {groups}."
            )
            raise ParameterError(msg)
    out = frame[frame["lane"].isin(keep)]
    if out.empty:
        msg = f"This optical path has nothing to draw for tracks={tracks!r}."
        raise ParameterError(msg)
    order = {lane: index for index, lane in enumerate(keep)}
    return out.sort_values("lane", key=lambda col: col.map(order), kind="stable")


def _distance_window(asked, span):
    """Resolve a (min, max) distance selection against a path's span."""
    if asked is None:
        return span
    try:
        low, high = asked
    except (TypeError, ValueError):
        msg = f"distance={asked!r} must be a (min, max) pair."
        raise ParameterError(msg) from None
    low = span[0] if low is None or low is ... else float(low)
    high = span[1] if high is None or high is ... else float(high)
    if high <= low:
        msg = f"distance={asked!r} must be increasing."
        raise ParameterError(msg)
    if high <= span[0] or low >= span[1]:
        msg = (
            f"distance={asked!r} lies outside the path's span {span}, so it "
            "clips everything away."
        )
        raise ParameterError(msg)
    return (low, high)


def path(
    inventory: Inventory,
    optical_path: str | OpticalPath | None = None,
    *,
    acquisition_key: str | None = None,
    time: timeable_types | None = None,
    distance: tuple | None = None,
    tracks: str | Sequence[str] | None = None,
    columns: str | Sequence[str] | None = None,
    n_samples: int = 1000,
    color: str | Mapping | None = None,
    max_labels: int = 200,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot what lies along one optical path, against optical distance.

    Every track the path describes becomes a lane: the stretch each
    acquisition places channels on, the optical components which give it
    its length, how it is coupled to the ground, and one lane per label
    group. A geometry column such as chainage or depth can be drawn as a
    line panel beneath, sharing the distance axis; it breaks wherever the
    path states no value rather than bridging the gap. Where the fiber
    physically is belongs to map().

    Parameters
    ----------
    inventory
        The inventory holding the path.
    optical_path
        The path to draw, as an ``network.array.location`` address, a
        path name, or the object. Optional when the choice is not
        ambiguous.
    acquisition_key
        Resolve the path from an acquisition key instead.
    time
        The instant to resolve at, which is how one epoch of a repaired
        path is chosen.
    distance
        The optical distances to draw between, as (min, max). Either
        end may be None, or ..., to run to the path's own bound. A long
        lead-in otherwise crushes the instrumented part into a corner.
    tracks
        Which lanes to draw, in order: any of "acquisition", "components",
        "coupling", and the path's label group names. None draws all.
    columns
        Geometry columns to draw as line panels beneath the lanes. The
        CRS's position axes are refused, since they belong on a map.
    n_samples
        How finely the columns are sampled.
    color
        Passed to the lane renderer to override its colors.
    max_labels
        Draw no lane text at all past this many intervals. Below that
        count, a label too wide for its box is turned on its side, and
        dropped only if it does not fit that way either.
    ax
        An Axes to draw the lanes on; one is created, a lane tall per
        track, when None. Pass one to say how large the plot is. Column
        panels need their own figure, so passing this and naming columns
        is refused; size that figure with
        `path(...).get_figure().set_size_inches(width, height)`, which
        lays it out again at the size asked for.
    show
        Whether to call plt.show.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.viz.inventory import path
    >>>
    >>> inventory = dc.get_example_inventory("tunnel")
    >>> _ = path(inventory, time="2024-07-01", distance=(1495, 1780))
    >>> _ = path(inventory, time="2024-07-01", tracks=("coupling", "section"))
    """
    address, array, chosen = _select_path(
        inventory, optical_path, acquisition_key, time
    )
    crs = inventory.coordinate_reference_system
    columns = _column_panels(chosen, columns, crs)
    if ax is not None and columns:
        msg = (
            "path draws its columns in their own panels, so it builds the "
            "figure and cannot add them to the axes passed as ax. Pass ax "
            "without columns, or leave ax unset."
        )
        raise ParameterError(msg)
    if chosen.optical_length <= 0:
        msg = (
            f"Optical path {address!r} has no length, since its components "
            "state none, so there is no distance axis to draw."
        )
        raise ParameterError(msg)
    limits = _distance_window(distance, (chosen.distance_min, chosen.distance_max))
    frame = _track_frame(chosen, _path_acquisitions(array, chosen, time))
    # The palette is the path's, not this figure's, so drawing some of the
    # tracks colors them as drawing all of them does.
    vocabulary = list(frame.loc[frame["lane"] != "components", "value"])
    frame = _select_tracks(frame, tracks, chosen)
    lanes = list(dict.fromkeys(frame["lane"]))
    if ax is None:
        # A legend which goes below the lanes needs height kept for it;
        # without that the lanes give up the room instead and every bar is
        # squeezed into a sliver. There is no figure to measure yet, so
        # both the standing height of one column and the rows it breaks
        # into are estimated from the labels; guessing low gives back less
        # room than intended, which is the harmless direction.
        named = _legend_names(frame, color)
        width = 10.0
        lane_height = 1.2 + 0.42 * len(lanes)
        # A legend which would stand nearly as tall as the lanes it names
        # reads better under them, and short of that it belongs at their
        # side. Deciding here rather than leaving it to the renderer is
        # what keeps the two from disagreeing: room kept below would
        # otherwise be room enough to sit beside, and go unused.
        column = _lanes.legend_column_points(named) / 72.0
        legend_rows = (
            0
            if column <= 0.8 * lane_height
            else _lanes.estimate_legend_rows(named, 72.0 * width)
        )
        # Capped: a figure taller than a page is not more readable.
        height = min(
            lane_height + 1.1 * len(columns) + 0.3 * legend_rows,
            14.0,
        )
        figure, all_axes = plt.subplots(
            1 + len(columns),
            1,
            figsize=(width, height),
            sharex=True,
            height_ratios=[max(2.0, 0.5 * len(lanes))] + [1] * len(columns),
            squeeze=False,
            layout="constrained",
        )
        all_axes = all_axes[:, 0]
        ax, panels = all_axes[0], all_axes[1:]
    else:
        figure, panels, legend_rows = None, [], 0
    pad = 0.02 * (limits[1] - limits[0])
    plot_lanes(
        frame,
        ax=ax,
        lane="lane",
        value="value",
        label="label",
        lanes=lanes,
        color=_lane_colors(color),
        vocabulary=vocabulary,
        max_labels=max_labels,
        x_limits=(limits[0] - pad, limits[1] + pad),
        x_label="" if len(panels) else "Optical distance [m]",
        colorbar_axes=[ax, *panels] if len(panels) else None,
        manage_figure=figure is not None,
        # Room was kept below for a legend, so that is where it goes;
        # letting it choose again would find the room and sit beside it.
        legend="below" if legend_rows else True,
    )
    named = [address, *([chosen.name] if chosen.name else []), _epoch_label(chosen)]
    ax.set_title(" · ".join(named), loc="left", fontsize="medium")
    distances = _sample_distances(chosen, limits[0], limits[1], n_samples)
    for index, (panel, name) in enumerate(zip(panels, columns, strict=True)):
        values = chosen.column_at(name, distances)
        panel.plot(distances, values, color=plt.get_cmap(_lanes.LANE_CMAP)(index % 10))
        units = _column_units(chosen, name)
        panel.set_ylabel(f"{name} [{units}]" if units else name)
        panel.grid(color="0.9", linewidth=0.5)
        panel.set_axisbelow(True)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
    if figure is not None and len(panels):
        panels[-1].set_xlabel("Optical distance [m]")
        figure.align_ylabels()
    if show:
        plt.show()
    return ax


def _time_window(asked):
    """Resolve a (min, max) time selection to matplotlib dates."""
    if asked is None:
        return None, None
    try:
        low, high = asked
    except (TypeError, ValueError):
        msg = f"time={asked!r} must be a (min, max) pair."
        raise ParameterError(msg) from None

    def one(value):
        if value is None or value is ...:
            return None
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError) as error:
            msg = f"time={asked!r} states a bound which is not a time."
            raise ParameterError(msg) from error
        if pd.isnull(stamp):
            msg = f"time={asked!r} states a bound which is not a time."
            raise ParameterError(msg)
        return mdates.date2num(stamp.to_pydatetime())

    low, high = one(low), one(high)
    if low is not None and high is not None and high <= low:
        msg = f"time={asked!r} must be increasing."
        raise ParameterError(msg)
    return low, high


def _legend_names(frame, color) -> list[str]:
    """What a legend of these lanes would name, in the order it names it.

    Only what earns a swatch counts. A single color for the whole figure
    earns no legend at all; a lane which states no value earns one swatch
    named for the lane; and a lane of numbers reads from its colorbar, or
    from the numbers printed in its boxes where there are few enough of
    them, so it names none. A mapping is the exception: it gives swatches
    to the values it holds, numbers included, and to no others.
    """
    if isinstance(color, str):
        return []
    flat, keyed = {}, {}
    if isinstance(color, Mapping):
        for name, entry in color.items():
            # Keyed by lane it holds a mapping of values; keyed by value
            # the key is the value itself.
            if isinstance(entry, Mapping):
                keyed[name] = entry
            else:
                flat[name] = entry
    out = []
    for lane, rows in frame.groupby("lane", sort=False):
        values = list(dict.fromkeys(rows["value"]))
        mapping = keyed.get(lane) or flat
        if mapping:
            out.extend(str(x) for x in values if x in mapping)
        elif all(pd.isnull(x) for x in values):
            out.append(str(lane))
        else:
            out.extend(str(x) for x in values if isinstance(x, str) and x)
    return list(dict.fromkeys(out))


def _lane_colors(color):
    """Pin the tracks whose vocabulary is closed, honoring an override."""
    if color is not None:
        return color
    return {"components": COMPONENT_COLORS}


def map_path(
    inventory: Inventory,
    optical_path: str | OpticalPath | None = None,
    *,
    acquisition_key: str | None = None,
    time: timeable_types | None = None,
    x: str | None = None,
    y: str | None = None,
    color: str = "distance",
    n_samples: int = 1000,
    cmap: str = "viridis",
    linewidth: float = 2.5,
    aspect: str | float | None = None,
    ax: plt.Axes | None = None,
    legend: bool = True,
    show: bool = False,
) -> plt.Axes:
    """
    Plot where an inventory's fiber physically goes.

    The polyline is the path's geometry read through the inventory's
    coordinate reference system. Stretches which state no position are
    left out rather than bridged, so a slack coil or an unsurveyed run
    reads as the gap it is.

    With no path named this draws every path which places itself, since
    a map of one cable in an inventory of several is a strange default.

    Parameters
    ----------
    inventory
        The inventory to draw.
    optical_path
        A path address, name, or object; None draws all of them.
    acquisition_key
        Resolve one path from an acquisition key instead.
    time
        The instant to resolve at.
    x, y
        The CRS axes to draw, defaulting to the first two the CRS
        declares. A borehole needs that default overridden: a hole runs
        straight down, so a plan view collapses it to a point.
    color
        "distance", a geometry column, a label group, or "coupling".
    n_samples
        How finely the path is sampled.
    cmap
        Colormap for a continuous coloring.
    linewidth
        Width of the drawn fiber.
    aspect
        Axes aspect; None picks equal when both axes share units.
    ax
        An Axes to draw on.
    legend
        Whether to draw the legend or colorbar.
    show
        Whether to call plt.show.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.viz.inventory import map_path
    >>>
    >>> inventory = dc.get_example_inventory("tunnel")
    >>> _ = map_path(inventory, x="x", y="z", color="section")
    """
    crs = inventory.coordinate_reference_system
    labels = list(crs.coordinate_labels)
    x = x if x is not None else labels[0]
    y = y if y is not None else (labels[1] if len(labels) > 1 else labels[0])
    if x == y:
        msg = f"x and y are both {x!r}; a map needs two different axes."
        raise ParameterError(msg)
    for name in (x, y):
        try:
            crs.axis_index(name)
        except Exception as error:  # the CRS explains itself better than we can
            msg = (
                f"{name!r} is not an axis of this inventory's CRS, whose axes "
                f"are {tuple(labels)}. A column which is not an axis is "
                "drawn by path(), not on a map."
            )
            raise ParameterError(msg) from error
    if optical_path is None and acquisition_key is None:
        chosen = [
            (a, arr, p)
            for a, net, arr, p in _iter_paths(inventory)
            if _effective_at(time, net, arr, p)
        ]
        if not chosen:
            msg = f"No optical path in this inventory is effective at {time}."
            raise ParameterError(msg)
    else:
        chosen = [_select_path(inventory, optical_path, acquisition_key, time)]
    own_figure = ax is None
    ax = _get_ax(ax)
    x_axis, y_axis = crs.axis_index(x), crs.axis_index(y)
    handles: dict = {}
    palette: dict = {}
    if color != "distance":
        # Checked over every path drawn: one path saying nothing about a
        # group is fiber with no value, which the map already draws.
        stated = {"coupling"}
        for _, _, one in chosen:
            stated |= set(one.geometry_columns())
            stated |= {x.group for x in one.labels}
        if color not in stated:
            msg = (
                f"color={color!r} names neither optical distance, a geometry "
                f"column, a label group, nor 'coupling'. This inventory "
                f"states {tuple(sorted(stated))}."
            )
            raise ParameterError(msg)
    # Two passes: every path is measured before any is drawn, so that one
    # color scale spans them all rather than the last one drawn winning.
    pieces = []
    for address, _, one in chosen:
        distances = _sample_distances(
            one, one.distance_min, one.distance_max, n_samples
        )
        coords = one.coordinates_at(distances, crs)
        points = np.column_stack([coords[:, x_axis], coords[:, y_axis]])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        mid = 0.5 * (distances[:-1] + distances[1:])
        # A segment touching an unplaced sample is not fiber we can draw.
        good = ~np.isnan(segments).any(axis=(1, 2))
        if not good.any():
            continue
        values, colors = _segment_colors(one, color, mid[good], crs, handles, palette)
        pieces.append((segments[good], values, colors))
    drawn = len(pieces)
    scalar = None
    # A value nothing states is not fiber nothing placed. Left as NaN it
    # would map to a transparent color and the cable would simply vanish.
    unstated = any(
        values is not None and bool(np.isnan(values).any()) for _, values, _ in pieces
    )
    if unstated:
        handles.setdefault("n/a", PatchArtist(facecolor=UNPLACED, label="n/a"))
    if drawn:
        finite = _shown_values(pieces)
        stated = [v[np.isfinite(v)] for _, v, _ in pieces if v is not None]
        stated = [v for v in stated if len(v)]
        norm, scale, ticks, beyond = None, None, None, "neither"
        if finite:
            # One scale for every path, stepped where the values are a
            # handful of numbered categories rather than a quantity.
            scale, norm, ticks = _lanes.numeric_scale(np.concatenate(finite), cmap)
            scale = scale.with_extremes(bad=UNPLACED)
            whole = np.concatenate(stated)
            under = bool(whole.min() < norm.vmin)
            over = bool(whole.max() > norm.vmax)
            beyond = (
                ("both" if under else "max")
                if over
                else ("min" if under else "neither")
            )
        for segments, values, colors in pieces:
            collection = LineCollection(
                list(segments),
                linewidths=linewidth,
                colors=colors,
                cmap=scale if values is not None else None,
                norm=norm if values is not None else None,
                capstyle="round",
            )
            if values is not None:
                collection.set_array(values)
                scalar = collection
            ax.add_collection(collection)
        ax.autoscale_view()
    if not drawn:
        msg = (
            "No optical path in this inventory places itself in the CRS, so "
            "there is no layout to draw. A path is placed by a geometry "
            f"segment stating the CRS's axes {tuple(labels)}."
        )
        raise ParameterError(msg)
    ax.set_xlabel(_axis_label(crs, x))
    ax.set_ylabel(_axis_label(crs, y))
    if aspect is None:
        same = crs.units[x_axis] == crs.units[y_axis]
        aspect = "equal" if same and "degree" not in crs.units[x_axis] else "auto"
    ax.set_aspect(aspect)
    shrink = 1.0
    if aspect == "equal" and own_figure:
        # An equal aspect on a long thin cable draws a short strip in a
        # tall figure, so give the figure the data's shape and the
        # colorbar the strip's height rather than the figure's.
        low_x, high_x = ax.get_xlim()
        low_y, high_y = ax.get_ylim()
        figure = ax.get_figure()
        ratio = abs(high_y - low_y) / (abs(high_x - low_x) or 1.0)
        drawn = figure.get_figwidth() * ratio
        figure.set_figheight(float(np.clip(drawn + 1.2, 1.6, 9.0)))
        shrink = float(np.clip(drawn / figure.get_figheight(), 0.25, 1.0))
    ax.grid(color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    if legend and scalar is not None:
        # A tall label beside a short strip of axes clips; lay the bar out
        # the way the data is laid out instead.
        flat = shrink < 0.45
        bar = ax.get_figure().colorbar(
            scalar,
            ax=ax,
            location="bottom" if flat else "right",
            fraction=0.12 if flat else 0.05,
            pad=0.25 if flat else 0.02,
            aspect=45 if flat else 20,
            shrink=1.0 if flat else shrink,
            # An arrow where fiber is drawn past the end of the scale.
            extend=beyond,
        )
        bar.set_label("Optical distance [m]" if color == "distance" else color)
        if ticks is not None:
            # A stepped scale reads at its steps; half a borehole is not one.
            bar.set_ticks(list(ticks))
    if legend and handles:
        # A colorbar already occupies the strip beside the axes.
        offset = 1.12 if scalar is not None and shrink >= 0.45 else 1.01
        ax.legend(
            handles=list(handles.values()),
            loc="upper left",
            bbox_to_anchor=(offset, 1.0),
            frameon=False,
            fontsize="small",
            # The colorbar beside it already carries the name.
            title=None if scalar is not None else color,
        )
    if show:
        plt.show()
    return ax


def _shown_values(pieces) -> list:
    """The colored values of segments this projection actually shows.

    A borehole seen from above is a point: it is drawn, but it displays
    no length, and letting it into the scale spends most of the colormap
    on fiber the reader cannot see.
    """
    stated = [v[np.isfinite(v)] for _, v, _ in pieces if v is not None]
    stated = [v for v in stated if len(v)]
    if not stated:
        return []
    corners = np.concatenate([x.reshape(-1, 2) for x, _, _ in pieces])
    floor = float(max(np.ptp(corners[:, 0]), np.ptp(corners[:, 1]))) * 1e-3
    shown = []
    for segments, values, _ in pieces:
        if values is None:
            continue
        steps = segments[:, 1] - segments[:, 0]
        drawn = np.hypot(steps[:, 0], steps[:, 1])
        keep = values[np.isfinite(values) & (drawn > floor)]
        if len(keep):
            shown.append(keep)
    # Every segment collapsed, so the projection shows no lengths at all.
    return shown or stated


def _axis_label(crs, name) -> str:
    """Label a map axis with the CRS's name for it and its units."""
    index = crs.axis_index(name)
    units = crs.units[index] if index < len(crs.units) else ""
    return f"{name} [{units}]" if units else str(name)


UNPLACED = mcolors.to_rgba(UNCOVERED_COLOR)


def _segment_colors(one, color, mid, crs, handles, palette):
    """Return (values, colors) for one path's segments; one of them is None."""
    if color == "distance":
        return mid, None
    if color in one.geometry_columns():
        return one.column_at(color, mid), None
    if color == "coupling":
        items = list(one.coupling)
        keys = [x.coupling_type for x in items]
    else:
        items = [x for x in one.labels if x.group == color]
        keys = [x.value for x in items]
        if not items:
            # This path states nothing under that name; another one does.
            handles.setdefault("n/a", PatchArtist(facecolor=UNPLACED, label="n/a"))
            return None, [UNPLACED] * len(mid)
    masks = interval_masks(mid, [x.interval for x in items])
    kinds = {
        value_kind(normalize_value(k)) for k in keys if not _lanes._is_membership(k)
    }
    if not kinds:
        # Every row states membership, so the group itself is the value
        # and belonging to it is the only thing there is to color.
        colors = [UNPLACED] * len(mid)
        base = plt.get_cmap(_lanes.STRING_CMAP)(_lanes.WHEEL_ORDER[0])
        for mask in masks:
            for position in np.flatnonzero(mask):
                colors[position] = base
        handles.setdefault(color, PatchArtist(facecolor=base, label=color))
        if any(c is UNPLACED for c in colors):
            handles.setdefault("n/a", PatchArtist(facecolor=UNPLACED, label="n/a"))
        return None, colors
    if kinds == {"numeric"}:
        values = np.full(len(mid), np.nan)
        for item, mask in zip(items, masks, strict=True):
            values[mask] = float(normalize_value(item.value))
        return values, None
    # The palette is the figure's, not this path's, so one value is one
    # color however many paths are drawn and whatever order they state it.
    wheel = plt.get_cmap(_lanes.STRING_CMAP)
    order = _lanes.WHEEL_ORDER
    for key in dict.fromkeys(map(str, keys)):
        palette.setdefault(key, wheel(order[len(palette) % len(order)]))
    seen = palette
    colors = [UNPLACED] * len(mid)
    for key, mask in zip(keys, masks, strict=True):
        placed = np.flatnonzero(mask)
        if not len(placed):
            # Stating a value over fiber which has no position places
            # nothing, so it earns no entry in the legend.
            continue
        for position in placed:
            colors[position] = seen[str(key)]
        handles.setdefault(
            str(key), PatchArtist(facecolor=seen[str(key)], label=str(key))
        )
    if any(c is UNPLACED for c in colors):
        handles.setdefault("n/a", PatchArtist(facecolor=UNPLACED, label="n/a"))
    return None, colors


def timeline(
    inventory: Inventory,
    *,
    kind: str = "both",
    color: str = "interrogator",
    time: tuple | None = None,
    ax: plt.Axes | None = None,
    legend: bool = True,
    show: bool = False,
) -> plt.Axes:
    """
    Plot when each part of an inventory was valid.

    One lane per acquisition and per optical path lineage, drawn against
    time. An epoch which states no start or no end is unbounded rather
    than missing, and is drawn running off that side of the axis.

    Parameters
    ----------
    inventory
        The inventory to draw.
    kind
        "both", "acquisition", or "optical_path".
    color
        "interrogator", "data_type", or "kind".
    time
        The times to draw between, as (min, max). Either end may be
        None, or ..., to run to what the epochs themselves state.
    ax
        An Axes to draw on.
    legend
        Whether to draw the legend.
    show
        Whether to call plt.show.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.viz.inventory import timeline
    >>>
    >>> inventory = dc.get_example_inventory("tunnel")
    >>> _ = timeline(inventory)
    """
    if kind not in {"both", "acquisition", "optical_path"}:
        msg = (
            f"kind={kind!r} is not a timeline selection; the options are "
            "('both', 'acquisition', 'optical_path')."
        )
        raise ParameterError(msg)
    if color not in {"interrogator", "data_type", "kind"}:
        msg = (
            f"color={color!r} is not a timeline coloring; the options are "
            "('interrogator', 'data_type', 'kind')."
        )
        raise ParameterError(msg)
    rows = []
    for network in inventory.networks:
        for array in network.fiber_arrays:
            if kind in {"both", "optical_path"}:
                for one in array.optical_paths:
                    start, end = _effective_epoch(network, array, one)
                    rows.append(
                        {
                            "lane": f"{network.code}.{array.code}."
                            f"{one.location_code} [path]",
                            "start": start,
                            "end": end,
                            "value": "optical path",
                            "label": one.name,
                        }
                    )
            if kind in {"both", "acquisition"}:
                for acquisition in array.acquisitions:
                    start, end = _effective_epoch(network, array, acquisition)
                    rows.append(
                        {
                            "lane": f"{network.code}.{array.code}."
                            f"{acquisition.location_code}.{acquisition.code}",
                            "start": start,
                            "end": end,
                            "value": _acquisition_color_value(
                                inventory, acquisition, color
                            ),
                            "label": "",
                        }
                    )
    if not rows:
        msg = (
            "This inventory holds nothing with a time epoch, so there is no "
            "timeline to draw."
        )
        raise ParameterError(msg)
    frame = pd.DataFrame(rows)
    known = pd.concat([frame["start"], frame["end"]]).dropna()
    asked_low, asked_high = _time_window(time)
    if ax is None:
        lanes = len(dict.fromkeys(frame["lane"]))
        _, ax = plt.subplots(
            1, figsize=(9.0, min(1.0 + 0.55 * lanes, 14.0)), layout="constrained"
        )

    if asked_low is not None or asked_high is not None:
        # A month is an arbitrary width, and only reached where one end is
        # asked for and nothing states a time to take the other from.
        month, stated = (
            30.0,
            [
                mdates.date2num(pd.Timestamp(x).to_pydatetime())
                for x in (known.min(), known.max())
            ]
            if len(known)
            else [None, None],
        )
        low = asked_low if asked_low is not None else stated[0]
        high = asked_high if asked_high is not None else stated[1]
        if low is None:
            low = high - month
        if high is None or high <= low:
            high = low + month
        dated = True
    elif len(known):
        low = mdates.date2num(pd.Timestamp(known.min()).to_pydatetime())
        high = mdates.date2num(pd.Timestamp(known.max()).to_pydatetime())
        pad = (high - low) * 0.05 or 30.0
        low, high = low - pad, high + pad
        dated = True
    else:
        # Nothing states a time, which is legal and common. Drawing bars on
        # a fabricated axis would invite the lengths to be read as facts.
        low, high, dated = 0.0, 1.0, False
    frame["open_start"] = frame["start"].isna()
    frame["open_end"] = frame["end"].isna()
    frame["start"] = [
        low if pd.isnull(x) else mdates.date2num(pd.Timestamp(x).to_pydatetime())
        for x in frame["start"]
    ]
    frame["end"] = [
        high if pd.isnull(x) else mdates.date2num(pd.Timestamp(x).to_pydatetime())
        for x in frame["end"]
    ]
    # An epoch outside the window is left out rather than clipped to a
    # sliver at the edge, which would read as an epoch which ended there.
    frame = frame[(frame["start"] < high) & (frame["end"] > low)]
    if frame.empty:
        msg = f"No epoch in this inventory falls within time={time!r}."
        raise ParameterError(msg)
    plot_lanes(
        frame,
        ax=ax,
        lane="lane",
        value="value",
        label="label",
        color=None,
        x_limits=(low, high),
        legend=legend,
    )
    if dated:
        _format_time_axis(ax, "time", "x")
        ax.set_xlabel("Time")
    else:
        ax.set_xticks([])
        ax.set_xlabel("time (no epoch in this inventory states one)")
    if show:
        plt.show()
    return ax


def _acquisition_color_value(inventory, acquisition, color) -> str:
    """The string an acquisition is colored by."""
    if color == "kind":
        return "acquisition"
    if color == "data_type":
        return acquisition.data_type or "unstated"
    interrogator = acquisition.interrogator
    if isinstance(interrogator, str):
        interrogator = inventory.get_resource(interrogator)
    if interrogator is None:
        return "no interrogator"
    name = f"{interrogator.manufacturer} {interrogator.model}".strip()
    return name or interrogator.serial_number or "interrogator"
