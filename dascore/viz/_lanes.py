"""
A general renderer for intervals laid out in horizontal lanes.

The inventory draws its tracks with this, a spool can draw what it
covers, and an annotation set is the same shape over a patch dimension.
So the input is a dataframe of intervals rather than any one of those
objects, and the columns it reads are named by the caller.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch as PatchArtist
from matplotlib.patches import Rectangle

from dascore.exceptions import ParameterError
from dascore.utils.intervals import normalize_value, value_kind
from dascore.utils.plotting import _format_time_axis, _get_ax, _get_cmap

# Palettes are module level so that two figures of one inventory agree.
STRING_CMAP = "tab20"
# tab20 runs dark, light, dark, light, so consecutive categories come out
# as two shades of one hue and read as one variable. Take the dark half
# first, and skip its two greys, which the uncovered colors already use.
WHEEL_ORDER = (0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 3, 5, 7, 9, 11, 13, 17, 19)
LANE_CMAP = "tab10"
NUMERIC_CMAP = "viridis"
UNCOVERED_COLOR = "0.7"

# The fraction of the x axis hatched where a bar runs off the end of it.
_OPEN_FRACTION = 0.02
_MAX_SUB_ROWS = 8
# Past this many distinct numbers a lane earns a colorbar rather than
# relying on the value printed in each box.
_MAX_DISCRETE = 6


def _as_numeric(values):
    """Return values as floats, converting datetimes to matplotlib dates."""
    array = np.asarray(values)
    if _is_dated(values):
        # Losing nanosecond precision is fine; this is a picture.
        stamps = pd.DatetimeIndex(array.ravel())
        if stamps.tz is not None:
            stamps = stamps.tz_convert("UTC").tz_localize(None)
        return mdates.date2num(stamps.to_numpy()).reshape(array.shape)
    return array.astype(float)


def _is_membership(value) -> bool:
    """Whether a row states membership of its lane rather than a value.

    A frame carries that as no value at all, which pandas spells None,
    NaN or NA depending on what else the column holds.
    """
    return bool(pd.isna(value))


def _default_label(value) -> str:
    """Text for a value which was not given a label of its own."""
    if isinstance(value, str):
        return value
    if _is_membership(value):
        return ""
    # A number states itself; a membership lane is named by its lane instead.
    return f"{value:g}" if isinstance(value, float) else str(value)


def _is_dated(values) -> bool:
    """Whether a column of interval bounds states times rather than numbers."""
    array = np.asarray(values)
    return bool(
        pd.api.types.is_datetime64_any_dtype(values)
        or (
            array.dtype == object
            and len(array)
            and isinstance(array.flat[0], datetime.datetime | np.datetime64)
        )
    )


def _read_frame(intervals, start, end, lane, value, label):
    """Pull the named columns out into a frame, and say if it is dated."""
    if not isinstance(intervals, pd.DataFrame):
        intervals = pd.DataFrame(intervals)
    missing = [x for x in (start, end) if x not in intervals.columns]
    if missing:
        msg = (
            f"An interval frame needs the columns {sorted(missing)}; this one "
            f"has {list(intervals.columns)}. Name the columns holding the "
            "interval bounds with the start and end arguments."
        )
        raise ParameterError(msg)
    for name, kind in ((lane, "lane"), (value, "value"), (label, "label")):
        if name is not None and name not in intervals.columns:
            msg = (
                f"{kind}={name!r} is not a column of this frame, which has "
                f"{list(intervals.columns)}."
            )
            raise ParameterError(msg)
    out = pd.DataFrame(index=intervals.index)
    out["start"] = _as_numeric(intervals[start])
    out["end"] = _as_numeric(intervals[end])
    out["lane"] = intervals[lane].astype(str) if lane else ""
    if value:
        # a missing value (None, or the NaN a frame of mixed lanes spells
        # it as) states membership; it is not a number
        values = intervals[value].astype(object)
        out["value"] = values.where(values.notna(), None)
    else:
        out["value"] = None
    if label:
        out["label"] = intervals[label].astype(str)
    elif value:
        out["label"] = [_default_label(x) for x in out["value"].tolist()]
    else:
        out["label"] = ""
    for flag in ("open_start", "open_end"):
        col = intervals[flag] if flag in intervals.columns else False
        out[flag] = np.asarray(col, dtype=bool) if flag in intervals.columns else False
    dated = _is_dated(intervals[start]) or _is_dated(intervals[end])
    return out, dated


def _lane_kind(values) -> str:
    """Return the one value kind a lane states, refusing a mixture."""
    kinds = {value_kind(normalize_value(x)) for x in values if not _is_membership(x)}
    kinds.discard(None)
    kinds.discard("membership")
    if not kinds:
        return "none"
    if len(kinds) > 1:
        return "mixed"
    return kinds.pop()


def _pack_rows(frame) -> np.ndarray:
    """Assign each interval a sub-row so overlapping ones do not collide."""
    order = np.argsort(frame["start"].to_numpy(), kind="stable")
    rows = np.zeros(len(frame), dtype=int)
    ends: list[float] = []
    starts = frame["start"].to_numpy()
    stops = frame["end"].to_numpy()
    for index in order:
        for row, last in enumerate(ends):
            if starts[index] >= last:
                rows[index] = row
                ends[row] = stops[index]
                break
        else:
            rows[index] = len(ends)
            ends.append(stops[index])
    return np.minimum(rows, _MAX_SUB_ROWS - 1)


def _string_colors(frame, vocabulary=None, cmap_name=STRING_CMAP) -> dict:
    """Map every string value to a stable color.

    The vocabulary widens the palette beyond what this frame holds, so a
    figure of part of a subject colors it as a figure of all of it does.
    """
    seen = list(frame["value"].tolist()) + list(vocabulary or [])
    values = sorted({x for x in seen if isinstance(x, str) and x != ""})
    cmap = plt.get_cmap(cmap_name)
    return {
        value: cmap(WHEEL_ORDER[index % len(WHEEL_ORDER)])
        for index, value in enumerate(values)
    }


def numeric_scale(values, cmap_name=NUMERIC_CMAP):
    """Return (cmap, norm, ticks) for a column of numbers.

    A handful of distinct values is a set of categories which happen to
    be numbered, so it gets one color each and a stepped bar reading at
    the values themselves. Anything more is a quantity, and ramps.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    base = _get_cmap(cmap_name)
    unique = np.unique(finite)
    if len(unique) < 2:
        low = float(unique[0]) if len(unique) else 0.0
        return base, plt.Normalize(low, low + 1.0), None
    if len(unique) <= _MAX_DISCRETE:
        picks = np.linspace(0.12, 0.9, len(unique))
        listed = ListedColormap([base(x) for x in picks])
        middles = (unique[:-1] + unique[1:]) / 2
        edges = np.concatenate(
            [
                [unique[0] - (middles[0] - unique[0])],
                middles,
                [unique[-1] + (unique[-1] - middles[-1])],
            ]
        )
        return listed, BoundaryNorm(edges, listed.N), unique
    return base, plt.Normalize(float(unique.min()), float(unique.max())), None


def _resolve_colors(rows, kind, lane_index, string_map, color):
    """Return one color per row, and a legend/colorbar description."""
    if isinstance(color, Mapping) and any(
        isinstance(x, Mapping) for x in color.values()
    ):
        # Keyed by lane. A lane the mapping does not name takes the
        # default treatment rather than being matched against lane names.
        color = color.get(rows["lane"].iloc[0])
    if isinstance(color, Mapping):
        # A mapping is keyed by the value, and membership has one key
        # however the column's dtype spelled it.
        keys = [None if _is_membership(x) else x for x in rows["value"]]
        colors = [color.get(x, UNCOVERED_COLOR) for x in keys]
        used = {x: color[x] for x in keys if x in color}
        return colors, ("legend", used)
    if isinstance(color, str) and kind != "numeric":
        return [color] * len(rows), None
    if kind == "string":
        colors = [string_map.get(x, UNCOVERED_COLOR) for x in rows["value"]]
        return colors, (
            "legend",
            {x: string_map[x] for x in rows["value"] if x in string_map},
        )
    if kind == "numeric":
        values = np.asarray(
            [
                np.nan if _is_membership(x) else float(normalize_value(x))
                for x in rows["value"]
            ],
            dtype=float,
        )
        if isinstance(color, str):
            try:
                cmap = _get_cmap(color)
            except (ValueError, KeyError):
                # A color name, not a colormap: one color for the lane, as
                # a lane of any other kind would take it.
                return [color] * len(rows), None
        else:
            cmap = _get_cmap(NUMERIC_CMAP)
        if len(np.unique(values[np.isfinite(values)])) < 2:
            # One value is not a scale, so it gets a color and its number
            # rather than a colorbar reading from it to a value nothing has.
            # A row which states none is still not that value.
            return [UNCOVERED_COLOR if np.isnan(x) else cmap(0.5) for x in values], None
        cmap, norm, ticks = numeric_scale(values, getattr(cmap, "name", NUMERIC_CMAP))
        # A value nothing states maps to a transparent color unless the
        # colormap is told otherwise, and the box would simply vanish.
        cmap = cmap.with_extremes(bad=UNCOVERED_COLOR)
        colors = [cmap(norm(x)) for x in values]
        if ticks is not None:
            # Few enough to be read off the boxes they are printed in.
            return colors, None
        # Each numeric lane is its own scale, so each earns its own bar;
        # one bar for two lanes would read from a scale only one of them has.
        return colors, ("colorbar", (rows["lane"].iloc[0], cmap, norm))
    # An unvalued lane states membership: every row takes the one color, so
    # the lane reads as one variable.
    base = plt.get_cmap(LANE_CMAP)(lane_index % 10)
    return [base] * len(rows), ("legend", {rows["lane"].iloc[0]: base})


def _draw_open_edges(ax, rows, y_low, height, colors, span):
    """Hatch the outer sliver of any bar which runs off the axis."""
    marks = []
    width = span * _OPEN_FRACTION
    for (_, row), color in zip(rows.iterrows(), colors, strict=True):
        for flag, edge in (("open_start", row["start"]), ("open_end", row["end"])):
            if not row[flag]:
                continue
            left = edge if flag == "open_start" else edge - width
            marks.append((Rectangle((left, y_low), width, height), color))
    if not marks:
        return
    patches = PatchCollection(
        [x for x, _ in marks],
        facecolors=[c for _, c in marks],
        hatch="///",
        edgecolor="white",
        linewidth=0,
        zorder=3,
    )
    ax.add_collection(patches)


def _fit_labels(ax, placements, max_labels):
    """Draw the labels which fit in their box, and drop the rest."""
    if len(placements) > max_labels:
        return
    figure = ax.get_figure()
    # Lay the figure out before measuring: a label is compared against its
    # box in pixels, and both move when the axes does.
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    transform = ax.transData
    for text, x_mid, y_mid, width in placements:
        if not text:
            continue
        artist = ax.text(
            x_mid,
            y_mid,
            text,
            ha="center",
            va="center",
            fontsize=plt.rcParams["font.size"] * 0.8,
            zorder=4,
            clip_on=True,
            # A dark fill would otherwise swallow the text sitting on it.
            path_effects=[pe.withStroke(linewidth=1.3, foreground="white")],
        )
        left = transform.transform((x_mid - width / 2, y_mid))[0]
        right = transform.transform((x_mid + width / 2, y_mid))[0]
        if artist.get_window_extent(renderer).width > (right - left):
            artist.remove()


def plot_lanes(
    intervals,
    ax: plt.Axes | None = None,
    *,
    start: str = "start",
    end: str = "end",
    lane: str | None = None,
    value: str | None = None,
    label: str | None = None,
    lanes: Sequence[str] | None = None,
    color=None,
    vocabulary: Sequence | None = None,
    pack: bool = True,
    legend: bool | str = "auto",
    max_labels: int = 200,
    x_limits: tuple | None = None,
    x_label: str = "",
    lane_height: float = 0.8,
    colorbar_axes: Sequence[plt.Axes] | None = None,
    show: bool = False,
) -> plt.Axes:
    """
    Draw a frame of intervals as horizontal lanes.

    Parameters
    ----------
    intervals
        A dataframe with one row per interval.
    ax
        A matplotlib Axes; one is created when None.
    start, end
        Columns holding the interval bounds. They may be numbers or
        datetimes, and equal bounds make the row a point marker.
    lane
        Column naming the lane a row belongs to; None puts every row in
        one unnamed lane.
    value
        Column deciding each row's color. Strings are categorical,
        numbers continuous, and a row with no value states membership
        of the lane (true and false are not values; an interval outside
        the lane has no row).
    label
        Column holding the text drawn in each box. Values supply it by
        default: text as itself, a number as its digits, and a row which
        states no value nothing, since its lane already names it.
    lanes
        The lanes to draw, in order. Names with no rows are kept as empty
        lanes, so two figures of different subjects still line up.
    color
        A color for every row, a mapping of value to color, or a mapping
        of lane name to such a mapping. A lane whose values are numbers
        reads a color string as the name of a colormap, or as a color
        where it names no colormap.
    vocabulary
        Values to reserve colors for beyond those this frame holds, so a
        figure of part of a subject colors it as a figure of all of it.
    pack
        Whether overlapping intervals are packed into sub-rows.
    legend
        Whether to draw a legend and any colorbars. False, or "off",
        draws neither; anything else draws what the colors earn.
    max_labels
        Draw no text at all past this many intervals.
    x_limits
        Limits for the x axis, in data units.
    x_label
        Label for the x axis.
    lane_height
        Fraction of a lane's row filled by its bars.
    colorbar_axes
        The axes a colorbar takes its room from; the drawn axes alone by
        default. Pass every axes of a shared-x figure, or the others keep
        a width this one gives up.
    show
        Whether to call plt.show.

    Examples
    --------
    >>> import pandas as pd
    >>> from dascore.viz._lanes import plot_lanes
    >>>
    >>> frame = pd.DataFrame(
    ...     {
    ...         "group": ["zone", "zone", "noisy"],
    ...         "start": [0.0, 10.0, 5.0],
    ...         "end": [10.0, 20.0, 15.0],
    ...         "value": ["north", "south", None],
    ...     }
    ... )
    >>> _ = plot_lanes(frame, lane="group", value="value")
    """
    frame, dated = _read_frame(intervals, start, end, lane, value, label)
    if not len(frame):
        msg = "The interval frame holds no rows, so there is nothing to draw."
        raise ParameterError(msg)
    backwards = frame["end"] < frame["start"]
    if backwards.any():
        row = frame[backwards].iloc[0]
        msg = (
            f"Interval ({row['start']}, {row['end']}) in lane "
            f"{row['lane']!r} ends before it starts."
        )
        raise ParameterError(msg)
    ax = _get_ax(ax)
    order = list(dict.fromkeys(frame["lane"])) if lanes is None else list(lanes)
    if lanes is not None and len(set(order)) != len(order):
        msg = f"lanes names a lane twice; each lane is drawn once. Got {lanes}."
        raise ParameterError(msg)
    # A lane given its own mapping is colored from that, so its values
    # must not also spend slots in the palette the other lanes draw from.
    pinned = set()
    if isinstance(color, Mapping):
        pinned = {k for k, v in color.items() if isinstance(v, Mapping)}
    unpinned = frame[~frame["lane"].isin(pinned)] if pinned else frame
    string_map = _string_colors(unpinned, vocabulary)
    # Fix the x limits before any text, since a label is measured in pixels.
    if x_limits is None:
        low = float(np.nanmin(frame["start"]))
        high = float(np.nanmax(frame["end"]))
        pad = (high - low) * 0.02 or 0.5
        x_limits = (low - pad, high + pad)
    else:
        x_limits = tuple(float(x) for x in _as_numeric(np.asarray(x_limits)))
    ax.set_xlim(*x_limits)
    span = x_limits[1] - x_limits[0]

    legend_entries: dict = {}
    colorbars: list[tuple] = []
    placements: list[tuple] = []
    for index, name in enumerate(order):
        rows = frame[frame["lane"] == name]
        y_centre = -index
        if not len(rows):
            continue
        kind = _lane_kind(rows["value"])
        if kind == "mixed":
            msg = (
                f"Lane {name!r} mixes value kinds, or values with rows of "
                "none, so it has no one color scheme. A group states one "
                "variable: a value in every row, or none at all."
            )
            raise ParameterError(msg)
        sub_rows = _pack_rows(rows) if pack else np.zeros(len(rows), dtype=int)
        n_sub = int(sub_rows.max()) + 1
        height = lane_height / n_sub
        colors, described = _resolve_colors(rows, kind, index, string_map, color)
        if described and described[0] == "legend":
            legend_entries.update(described[1])
        elif described and described[0] == "colorbar":
            colorbars.append(described[1])
        boxes, box_colors, points, point_colors = [], [], [], []
        for (_, row), row_color, sub in zip(
            rows.iterrows(), colors, sub_rows, strict=True
        ):
            low = y_centre - lane_height / 2 + sub * height
            width = row["end"] - row["start"]
            if width <= 0:
                # A point marker covers nothing but still documents a place.
                points.append((row["start"], low, height))
                point_colors.append(row_color)
                continue
            boxes.append(Rectangle((row["start"], low), width, height))
            box_colors.append(row_color)
            placements.append(
                (row["label"], row["start"] + width / 2, low + height / 2, width)
            )
        if boxes:
            ax.add_collection(
                PatchCollection(
                    boxes,
                    facecolors=box_colors,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=2,
                )
            )
        for (x, low, tall), point_color in zip(points, point_colors, strict=True):
            ax.plot(
                [x, x],
                [low, low + tall],
                color=point_color,
                linewidth=2.0,
                zorder=3,
                solid_capstyle="butt",
            )
            ax.plot([x], [low + tall], marker="v", markersize=5, color=point_color)
        _draw_open_edges(
            ax, rows, y_centre - lane_height / 2, lane_height, colors, span
        )

    ax.set_yticks(-np.arange(len(order)), [str(x) for x in order])
    ax.set_ylim(-(len(order) - 1) - lane_height, lane_height)
    if x_label:
        ax.set_xlabel(x_label)
    ax.grid(axis="x", color="0.85", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    # The left spine is hidden, so its tick marks are dashes after a name.
    ax.tick_params(axis="y", length=0, pad=4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if dated:
        _format_time_axis(ax, x_label or "time", "x")
    _fit_labels(ax, placements, max_labels)
    if legend and legend != "off":
        for name, cmap, norm in colorbars:
            bar = ax.get_figure().colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=list(colorbar_axes) if colorbar_axes else ax,
                fraction=0.05,
                pad=0.02,
            )
            bar.set_label(name)
    if legend and legend_entries and legend != "off":
        handles = [
            PatchArtist(facecolor=color, label=name)
            for name, color in legend_entries.items()
        ]
        # A colorbar already occupies the strip beside the axes.
        offset = 1.01 + 0.17 * len(colorbars)
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(offset, 1.0),
            frameon=False,
            fontsize="small",
        )
    if show:
        plt.show()
    return ax
