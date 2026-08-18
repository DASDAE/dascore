"""
A general renderer for intervals laid out in horizontal lanes.

The inventory draws its tracks with this, a spool can draw what it covers
and where its gaps are, and an annotation set is the same shape over a
patch dimension. So the input is a dataframe of intervals rather than any
one of those objects, and the columns it reads are named by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch as PatchArtist
from matplotlib.patches import Rectangle

from dascore.exceptions import ParameterError
from dascore.utils.intervals import normalize_value, value_kind
from dascore.utils.misc import suppress_warnings
from dascore.utils.plotting import _get_ax, _get_cmap

# Palettes are module level so that two figures of one inventory agree.
STRING_CMAP = "tab20"
LANE_CMAP = "tab10"
NUMERIC_CMAP = "viridis"
GAP_COLOR = "0.88"
UNCOVERED_COLOR = "0.7"

# The fraction of a bar overdrawn with hatching where it runs off the axis.
_OPEN_FRACTION = 0.02
_MAX_SUB_ROWS = 8
# Past this many distinct numbers a lane earns a colorbar rather than
# relying on the value printed in each box.
_MAX_DISCRETE = 6


def _as_numeric(values):
    """Return values as floats, converting datetimes to matplotlib dates."""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64) or isinstance(
        getattr(array, "dtype", None), pd.DatetimeTZDtype
    ):
        # Losing nanosecond precision is fine; this is a picture.
        with suppress_warnings(UserWarning):
            stamps = pd.to_datetime(pd.Series(array.ravel()))
            return mdates.date2num(stamps.dt.to_pydatetime()).reshape(array.shape)
    return array.astype(float)


def _default_label(value) -> str:
    """Text for a value which was not given a label of its own."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None:
        return ""
    # A number states itself; a boolean group is named by its lane instead.
    return f"{value:g}" if isinstance(value, float) else str(value)


def _read_frame(intervals, start, end, lane, value, label):
    """Pull the named columns out into a frame with canonical names."""
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
    out["start"] = _as_numeric(intervals[start].to_numpy())
    out["end"] = _as_numeric(intervals[end].to_numpy())
    out["lane"] = intervals[lane].astype(str) if lane else ""
    out["value"] = intervals[value] if value else None
    if label:
        out["label"] = intervals[label].astype(str)
    elif value:
        out["label"] = [_default_label(x) for x in intervals[value].tolist()]
    else:
        out["label"] = ""
    for flag in ("open_start", "open_end"):
        col = intervals[flag] if flag in intervals.columns else False
        out[flag] = np.asarray(col, dtype=bool) if flag in intervals.columns else False
    return out


def _lane_kind(values) -> str:
    """Return the one value kind a lane states, refusing a mixture."""
    kinds = {value_kind(normalize_value(x)) for x in values if x is not None}
    kinds.discard(None)
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


def _string_colors(frame, cmap_name=STRING_CMAP) -> dict:
    """Map every string value in the frame to a stable color."""
    values = sorted(
        {x for x in frame["value"].tolist() if isinstance(x, str) and x != ""}
    )
    cmap = plt.get_cmap(cmap_name)
    return {value: cmap(index % cmap.N) for index, value in enumerate(values)}


def _resolve_colors(rows, kind, lane_index, string_map, color):
    """Return one color per row, and a legend/colorbar description."""
    if isinstance(color, Mapping) and any(
        isinstance(x, Mapping) for x in color.values()
    ):
        # Keyed by lane. A lane the mapping does not name takes the
        # default treatment rather than being matched against lane names.
        color = color.get(rows["lane"].iloc[0])
    if isinstance(color, Mapping):
        colors = [color.get(x, UNCOVERED_COLOR) for x in rows["value"]]
        used = {x: color[x] for x in rows["value"] if x in color}
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
            [float(normalize_value(x)) for x in rows["value"]], dtype=float
        )
        cmap = _get_cmap(color if isinstance(color, str) else NUMERIC_CMAP)
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        if high <= low:
            # One value is not a scale, so it gets a color and its number
            # rather than a colorbar reading from it to a value nothing has.
            return [cmap(0.5)] * len(rows), None
        norm = plt.Normalize(low, high)
        colors = [cmap(norm(x)) for x in values]
        if len(set(values.tolist())) <= _MAX_DISCRETE:
            return colors, None
        return colors, ("colorbar", (cmap, norm))
    # Boolean and unvalued lanes take one color, so the lane reads as one
    # variable; a False interval is drawn faintly rather than dropped.
    base = plt.get_cmap(LANE_CMAP)(lane_index % 10)
    colors = [
        base if normalize_value(x) is not False else (*base[:3], 0.25)
        for x in rows["value"]
    ]
    return colors, ("legend", {rows["lane"].iloc[0]: base})


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
    ax.get_figure().canvas.draw_idle()
    transform = ax.transData
    for text, x_mid, y_mid, width in placements:
        if not text:
            continue
        # Measure the box in pixels; a label wider than its box is noise.
        left = transform.transform((x_mid - width / 2, y_mid))[0]
        right = transform.transform((x_mid + width / 2, y_mid))[0]
        needed = len(text) * plt.rcParams["font.size"] * 0.6
        if (right - left) < needed:
            continue
        ax.text(
            x_mid,
            y_mid,
            text,
            ha="center",
            va="center",
            fontsize=plt.rcParams["font.size"] * 0.8,
            zorder=4,
            clip_on=True,
        )


def _gap_rows(rows, limits):
    """Return the intervals a lane does not cover, inside limits."""
    spans = sorted(
        (a, b) for a, b in zip(rows["start"], rows["end"], strict=True) if b > a
    )
    merged: list[list[float]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    low, high = limits
    out, position = [], low
    for lo, hi in merged:
        if lo > position:
            out.append((position, min(lo, high)))
        position = max(position, hi)
    if position < high:
        out.append((position, high))
    return [x for x in out if x[1] > x[0]]


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
    gaps: bool = False,
    pack: bool = True,
    legend: bool | str = "auto",
    max_labels: int = 200,
    x_limits: tuple | None = None,
    x_label: str = "",
    lane_height: float = 0.8,
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
        numbers continuous, and booleans state membership of the lane.
    label
        Column holding the text drawn in each box; defaults to the value
        where the value is text.
    lanes
        The lanes to draw, in order. Names with no rows are kept as empty
        lanes, so two figures of different subjects still line up.
    color
        A color for every row, a mapping of value to color, or a mapping
        of lane name to either of those.
    gaps
        Whether to also draw what each lane does not cover.
    pack
        Whether overlapping intervals are packed into sub-rows.
    legend
        Whether to draw a legend or colorbar. "auto" draws one when the
        colors mean something beyond the lane they are in.
    max_labels
        Draw no text at all past this many intervals.
    x_limits
        Limits for the x axis, in data units.
    x_label
        Label for the x axis.
    lane_height
        Fraction of a lane's row filled by its bars.
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
    ...         "value": ["north", "south", True],
    ...     }
    ... )
    >>> _ = plot_lanes(frame, lane="group", value="value")
    """
    frame = _read_frame(intervals, start, end, lane, value, label)
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
    string_map = _string_colors(frame)
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
    colorbar: tuple | None = None
    placements: list[tuple] = []
    for index, name in enumerate(order):
        rows = frame[frame["lane"] == name]
        y_centre = -index
        if not len(rows):
            continue
        kind = _lane_kind(rows["value"])
        if kind == "mixed":
            msg = (
                f"Lane {name!r} mixes value kinds, so it has no one color "
                "scheme. A group states one variable; split the kinds into "
                "separate lanes."
            )
            raise ParameterError(msg)
        sub_rows = _pack_rows(rows) if pack else np.zeros(len(rows), dtype=int)
        n_sub = int(sub_rows.max()) + 1
        height = lane_height / n_sub
        colors, described = _resolve_colors(rows, kind, index, string_map, color)
        if described and described[0] == "legend":
            legend_entries.update(described[1])
        elif described and described[0] == "colorbar":
            colorbar = described[1]
        if gaps:
            gap_spans = _gap_rows(rows, x_limits)
            if gap_spans:
                ax.add_collection(
                    PatchCollection(
                        [
                            Rectangle(
                                (lo, y_centre - lane_height / 2),
                                hi - lo,
                                lane_height,
                            )
                            for lo, hi in gap_spans
                        ],
                        facecolors=GAP_COLOR,
                        edgecolor="none",
                        zorder=1,
                    )
                )
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
                linewidth=1.5,
                zorder=3,
                solid_capstyle="butt",
            )
            ax.plot([x], [low + tall], marker="v", markersize=4, color=point_color)
        _draw_open_edges(
            ax, rows, y_centre - lane_height / 2, lane_height, colors, span
        )

    ax.set_yticks(-np.arange(len(order)), [str(x) for x in order])
    ax.set_ylim(-(len(order) - 1) - lane_height, lane_height)
    if x_label:
        ax.set_xlabel(x_label)
    ax.grid(axis="x", color="0.9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    _fit_labels(ax, placements, max_labels)
    if legend and colorbar is not None:
        cmap, norm = colorbar
        ax.get_figure().colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            fraction=0.05,
            pad=0.02,
        )
    if legend and legend_entries and legend != "off":
        handles = [
            PatchArtist(facecolor=color, label=name)
            for name, color in legend_entries.items()
        ]
        if gaps:
            handles.append(PatchArtist(facecolor=GAP_COLOR, label="not covered"))
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize="small",
        )
    if show:
        plt.show()
    return ax


def lane_gaps(intervals, *, start="start", end="end", lane=None, limits=None):
    """
    Return what a frame of intervals does not cover, lane by lane.

    This is the derivation behind ``plot_lanes(..., gaps=True)``, kept
    separate because "where are the holes" is worth asking without a
    figure attached to the answer.

    Examples
    --------
    >>> import pandas as pd
    >>> from dascore.viz._lanes import lane_gaps
    >>>
    >>> frame = pd.DataFrame({"start": [0.0, 20.0], "end": [10.0, 30.0]})
    >>> lane_gaps(frame)[["start", "end"]].to_numpy().tolist()
    [[10.0, 20.0]]
    """
    frame = _read_frame(intervals, start, end, lane, None, None)
    out = []
    for name in dict.fromkeys(frame["lane"]):
        rows = frame[frame["lane"] == name]
        span = limits or (rows["start"].min(), rows["end"].max())
        for low, high in _gap_rows(rows, tuple(float(x) for x in span)):
            out.append({"lane": name, "start": low, "end": high})
    return pd.DataFrame(out, columns=["lane", "start", "end"])
