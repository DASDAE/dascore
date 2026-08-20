"""Visualizations of an inventory: its path, its layout, and its epochs."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch as PatchArtist

from dascore.exceptions import ParameterError
from dascore.utils.intervals import interval_masks, normalize_value, value_kind
from dascore.utils.plotting import _format_time_axis, _get_ax, _get_cmap

from ._lanes import LANE_CMAP, plot_lanes

# Components are a closed set, so their colors can be too.
COMPONENT_COLORS = {
    "FiberSegment": "#4c72b0",
    "Splice": "#dd8452",
    "Connector": "#55a868",
    "Terminator": "#c44e52",
}


def _iter_paths(inventory):
    """Yield every optical path with the address which names it."""
    for network in inventory.networks:
        for array in network.fiber_arrays:
            for path in array.optical_paths:
                address = f"{network.code}.{array.code}.{path.location_code}"
                yield address, network, array, path


def _epoch_label(path) -> str:
    """Name a path epoch by when it starts, for a chart title."""
    if pd.isnull(path.start_time):
        return "from the beginning"
    return f"from {str(path.start_time)[:10]}"


def _select_path(inventory, optical_path=None, acquisition_key=None, time=None):
    """Return the (address, array, path) a caller means, or explain."""
    found = list(_iter_paths(inventory))
    if not found:
        msg = "This inventory holds no optical paths, so there is nothing to plot."
        raise ParameterError(msg)
    if acquisition_key is not None:
        context = inventory.resolve(acquisition_key, time)
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
        candidates = [x for x in found if x[3].is_effective_at(time)]
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
                "lane": f"channels ({acquisition.code})",
                "start": float(distances[0]),
                "end": float(distances[-1]),
                "value": acquisition.code,
                "label": acquisition.code,
            }
        )
    for component, (low, high) in zip(
        path.optical_components, path.component_intervals(), strict=True
    ):
        rows.append(
            {
                "lane": "components",
                "start": low,
                "end": high,
                "value": type(component).__name__,
                "label": component.name or type(component).__name__,
            }
        )
    for coupling in path.coupling:
        rows.append(
            {
                "lane": "coupling",
                "start": coupling.start_distance,
                "end": coupling.end_distance,
                "value": coupling.coupling_type,
                "label": coupling.coupling_type,
            }
        )
    for item in path.labels:
        # A boolean states only membership, which the lane name already
        # says; anything else is worth reading off the box.
        value = item.value
        text = (
            ""
            if isinstance(value, bool)
            else f"{value:g}"
            if isinstance(value, float)
            else str(value)
        )
        rows.append(
            {
                "lane": item.group,
                "start": item.start_distance,
                "end": item.end_distance,
                "value": value,
                "label": text,
            }
        )
    return pd.DataFrame(rows)


TRACKS = ("channels", "components", "coupling")


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
        if name == "channels":
            keep.extend(
                x for x in dict.fromkeys(frame["lane"]) if x.startswith("channels")
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
    """Resolve a (low, high) distance selection against a path's span."""
    if asked is None:
        return span
    try:
        low, high = asked
    except (TypeError, ValueError):
        msg = f"distance={asked!r} must be a (low, high) pair."
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
    inventory,
    optical_path=None,
    *,
    acquisition_key: str | None = None,
    time=None,
    distance: tuple | None = None,
    tracks: str | Sequence[str] | None = None,
    columns: str | Sequence[str] | None = None,
    n_samples: int = 1000,
    color=None,
    max_labels: int = 200,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot what lies along one optical path, against optical distance.

    Every track the path describes becomes a lane: the channels each
    acquisition places on it, the optical components which give it its
    length, how it is coupled to the ground, and one lane per label
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
        The optical distances to draw between, as (low, high); either
        may be None to run to the path's end. A long lead-in otherwise
        crushes the instrumented part of a path into a corner.
    tracks
        Which lanes to draw, in order: any of "channels", "components",
        "coupling", and the path's label group names. None draws all.
    columns
        Geometry columns to draw as line panels beneath the lanes. The
        CRS's position axes are refused, since they belong on a map.
    n_samples
        How finely the columns are sampled.
    color
        Passed to the lane renderer to override its colors.
    max_labels
        Draw no lane text at all past this many intervals.
    ax
        An Axes to draw the lanes on. Column panels need their own
        figure, so passing this and naming columns is refused.
    figsize
        Size of the figure built when ax is None.
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
    frame = _track_frame(chosen, _path_acquisitions(array, chosen, time))
    frame = _select_tracks(frame, tracks, chosen)
    lanes = list(dict.fromkeys(frame["lane"]))
    if ax is None:
        height = 1.2 + 0.42 * len(lanes) + 1.1 * len(columns)
        figure, all_axes = plt.subplots(
            1 + len(columns),
            1,
            figsize=figsize or (10.0, height),
            sharex=True,
            height_ratios=[max(2.0, 0.5 * len(lanes))] + [1] * len(columns),
            squeeze=False,
        )
        all_axes = all_axes[:, 0]
        ax, panels = all_axes[0], all_axes[1:]
    else:
        figure, panels = None, []
    limits = _distance_window(distance, (chosen.start_distance, chosen.end_distance))
    pad = 0.02 * (limits[1] - limits[0])
    plot_lanes(
        frame,
        ax=ax,
        lane="lane",
        value="value",
        label="label",
        lanes=lanes,
        color=_lane_colors(color),
        max_labels=max_labels,
        x_limits=(limits[0] - pad, limits[1] + pad),
        x_label="" if len(panels) else "Optical distance [m]",
    )
    ax.set_title(f"{address}  ·  {chosen.name or 'path'}  ·  {_epoch_label(chosen)}")
    distances = np.linspace(limits[0], limits[1], n_samples)
    for index, (panel, name) in enumerate(zip(panels, columns, strict=True)):
        values = chosen.column_at(name, distances)
        panel.plot(distances, values, color=plt.get_cmap(LANE_CMAP)(index % 10))
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


def _lane_colors(color):
    """Pin the tracks whose vocabulary is closed, honoring an override."""
    if color is not None:
        return color
    return {"components": COMPONENT_COLORS}


def map_path(
    inventory,
    optical_path=None,
    *,
    acquisition_key: str | None = None,
    time=None,
    x: str | None = None,
    y: str | None = None,
    color: str = "distance",
    n_samples: int = 1000,
    cmap: str = "cividis_r",
    linewidth: float = 2.5,
    aspect=None,
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
        The CRS axes to draw. They default to the first two the CRS
        declares, which a borehole may need overriding: a hole runs
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
        chosen = [(a, arr, p) for a, _, arr, p in _iter_paths(inventory)]
        if time is not None:
            chosen = [x_ for x_ in chosen if x_[2].is_effective_at(time)]
    else:
        chosen = [_select_path(inventory, optical_path, acquisition_key, time)]
    own_figure = ax is None
    ax = _get_ax(ax)
    x_axis, y_axis = crs.axis_index(x), crs.axis_index(y)
    handles: dict = {}
    # Two passes: every path is measured before any is drawn, so that one
    # color scale spans them all rather than the last one drawn winning.
    pieces = []
    for index, (address, _, one) in enumerate(chosen):
        distances = np.linspace(one.start_distance, one.end_distance, n_samples)
        coords = one.coordinates_at(distances, crs)
        points = np.column_stack([coords[:, x_axis], coords[:, y_axis]])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        mid = 0.5 * (distances[:-1] + distances[1:])
        # A segment touching an unplaced sample is not fiber we can draw.
        good = ~np.isnan(segments).any(axis=(1, 2))
        if not good.any():
            continue
        values, colors = _segment_colors(one, color, mid[good], crs, index, handles)
        pieces.append((segments[good], values, colors))
    drawn = len(pieces)
    scalar = None
    if drawn:
        finite = [v[np.isfinite(v)] for _, v, _ in pieces if v is not None]
        finite = [v for v in finite if len(v)]
        norm = None
        if finite:
            low = float(min(v.min() for v in finite))
            high = float(max(v.max() for v in finite))
            norm = plt.Normalize(low, high if high > low else low + 1.0)
        for segments, values, colors in pieces:
            collection = LineCollection(
                list(segments),
                linewidths=linewidth,
                colors=colors,
                cmap=_get_cmap(cmap) if values is not None else None,
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
        figure.set_figheight(float(np.clip(drawn + 1.2, 2.4, 9.0)))
        shrink = float(np.clip(drawn / figure.get_figheight(), 0.25, 1.0))
    ax.grid(color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    if legend and scalar is not None:
        bar = ax.get_figure().colorbar(
            scalar, ax=ax, fraction=0.05, pad=0.02, shrink=shrink
        )
        bar.set_label("Optical distance [m]" if color == "distance" else color)
    elif legend and handles:
        ax.legend(
            handles=list(handles.values()),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize="small",
            title=color,
        )
    if show:
        plt.show()
    return ax


def _axis_label(crs, name) -> str:
    """Label a map axis with the CRS's name for it and its units."""
    index = crs.axis_index(name)
    units = crs.units[index] if index < len(crs.units) else ""
    return f"{name} [{units}]" if units else str(name)


UNPLACED = (0.8, 0.8, 0.8, 1.0)


def _segment_colors(one, color, mid, crs, index, handles):
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
            groups = tuple(dict.fromkeys(x.group for x in one.labels))
            msg = (
                f"color={color!r} names neither optical distance, a geometry "
                f"column ({one.geometry_columns()}), a label group ({groups}), "
                "nor 'coupling'."
            )
            raise ParameterError(msg)
    masks = interval_masks(mid, [x.interval for x in items])
    kinds = {value_kind(normalize_value(k)) for k in keys}
    if kinds == {"numeric"}:
        values = np.full(len(mid), np.nan)
        for item, mask in zip(items, masks, strict=True):
            values[mask] = float(normalize_value(item.value))
        return values, None
    palette = plt.get_cmap("tab20")
    seen = {k: palette(i % 20) for i, k in enumerate(dict.fromkeys(map(str, keys)))}
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
        handles.setdefault(
            "not stated", PatchArtist(facecolor=UNPLACED, label="not stated")
        )
    return None, colors


def timeline(
    inventory,
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
        The times to draw between, as (start, end).
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
                    rows.append(
                        {
                            "lane": f"{network.code}.{array.code}."
                            f"{one.location_code} [path]",
                            "start": one.start_time,
                            "end": one.end_time,
                            "value": "optical path",
                            "label": one.name,
                        }
                    )
            if kind in {"both", "acquisition"}:
                for acquisition in array.acquisitions:
                    rows.append(
                        {
                            "lane": f"{network.code}.{array.code}."
                            f"{acquisition.location_code}.{acquisition.code}",
                            "start": acquisition.start_time,
                            "end": acquisition.end_time,
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
    if ax is None:
        lanes = len(dict.fromkeys(frame["lane"]))
        _, ax = plt.subplots(1, figsize=(9.0, 1.0 + 0.55 * lanes))

    if time is not None:
        low, high = (mdates.date2num(pd.Timestamp(x).to_pydatetime()) for x in time)
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
    frame = frame[(frame["start"] <= high) & (frame["end"] >= low)]
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
