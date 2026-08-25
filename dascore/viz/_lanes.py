"""
A general renderer for intervals laid out in horizontal lanes.

The inventory draws its tracks with this, a spool can draw what it
covers, and an annotation set is the same shape over a patch dimension.
So the input is a dataframe of intervals rather than any one of those
objects, and the columns it reads are named by the caller.
"""

from __future__ import annotations

import colorsys
import datetime
from collections.abc import Mapping, Sequence

import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba_array
from matplotlib.font_manager import FontProperties
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.patches import Patch as PatchArtist
from matplotlib.patches import Rectangle
from matplotlib.textpath import text_to_path

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

# One box is parted from the next by a stroke of this width, in points.
SEPARATOR_COLOR = "white"
SEPARATOR_WIDTH = 0.5
# How many separators wide a box must be before it can afford to carry
# one. Two leaves a box at least as much of itself as it gives away.
_SEPARATOR_ROOM = 2.0


# The wheel holds every color tab20 offers which is not a grey, so a
# palette past it can only repeat itself. Values then walk the hue circle:
# stepping by the golden ratio keeps neighbors in the sorted order apart,
# and alternating shade separates two which still land on a similar hue.
_GOLDEN_STEP = 0.6180339887498949

# Points per inch: the unit both a label and its box are measured in, so
# the same figure keeps the same labels whatever dpi it is drawn at.
_MEASURED_DPI = 72.0

# What matplotlib sets successive lines of one label apart by, as a
# multiple of the font size.
_LINE_SPACING = 1.2

# What matplotlib makes of the fontsize="small" the legends ask for.
_SMALL_SCALE = 0.833

# A legend entry is its label plus a swatch and the gaps around it, which
# come to about this many times the text height.
_SWATCH_WIDTH = 3.0

# What matplotlib sets legend rows apart by, as a multiple of the size of
# the text in them.
_LEGEND_PITCH = 1.6

# Clearance a label needs inside its box, in points. Without it a label
# the exact width of its box touches the one in the next box, and the two
# read as one word.
_LABEL_PAD = 3.0

# The fraction of the x axis hatched where a bar runs off the end of it.
_OPEN_FRACTION = 0.02
_MAX_SUB_ROWS = 8
# Past this many distinct numbers a lane earns a colorbar rather than
# relying on the value printed in each box.
_MAX_DISCRETE = 6


class _SeparatedBoxes(PatchCollection):
    """
    Boxes parted by a stroke which never outgrows the box it borders.

    The separator is a fixed width while a box is however wide the axis
    makes it, so a box narrower than the stroke is painted out by its
    own edge: it reads as the background rather than as itself. A short
    gap between two long runs is exactly that box, and drawing it white
    says there is no gap. So a box without the room for a separator is
    stroked in its own color instead, which reads as itself down to the
    pixel, and takes the separator back up once a zoom gives it room.
    """

    def __init__(self, patches, *, facecolors, bounds, **kwargs):
        super().__init__(
            patches,
            facecolors=facecolors,
            linewidth=SEPARATOR_WIDTH,
            **kwargs,
        )
        self._bounds = np.asarray(bounds, dtype=float)

    def draw(self, renderer):
        # How much room a box has is settled by the axis it is drawn on
        # and by what the renderer makes of a point, so it is answered
        # again at every draw rather than once.
        stale = self.stale
        colors = self._edge_colors(renderer)
        self.set_edgecolor(colors)  # ty: ignore[invalid-argument-type]
        # Choosing a color is how this collection draws itself rather
        # than a change made to it, and a blit which drew it on its own
        # would otherwise be left with a figure asking to be drawn again.
        self.stale = stale
        super().draw(renderer)

    def _edge_colors(self, renderer) -> np.ndarray:
        """The separator where a box has room for it, its own color where not."""
        flat = np.column_stack([self._bounds.ravel(), np.zeros(self._bounds.size)])
        drawn = self.get_transform().transform(flat)[:, 0]
        axes = self.axes
        if axes is not None:
            # What a box has room for is what it shows. A run reaching
            # off the axis is as wide as the part of it drawn, and the
            # rest is room it does not have here.
            drawn = np.clip(drawn, axes.bbox.x0, axes.bbox.x1)
        widths = np.abs(np.diff(drawn.reshape(self._bounds.shape), axis=1)).ravel()
        # The renderer says what a point comes to; not every backend
        # reads it as the figure's dpi over seventy-two.
        stroke = renderer.points_to_pixels(SEPARATOR_WIDTH)
        edges = np.tile(to_rgba_array(SEPARATOR_COLOR), (len(widths), 1))
        cramped = widths < _SEPARATOR_ROOM * stroke
        # Read now rather than kept, so a box recolored after it was
        # built is edged in the color it states now. One color may stand
        # for every box, as matplotlib lets it, so the colors cycle.
        faces = self.get_facecolor()
        if len(faces):
            edges[cramped] = faces[np.nonzero(cramped)[0] % len(faces)]
        return edges


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


# Shades the hue circle is walked at. Two values far enough apart in the
# walk come back to nearly the same hue, so they are told apart by shade
# instead; a prime number of them keeps that from lining up with the walk.
_SHADES = ((0.62, 0.72), (0.38, 0.92), (0.85, 0.55), (0.50, 0.98), (0.72, 0.85))


def _wide_colors(values) -> dict:
    """One distinct color per value, past what the wheel can hold.

    No two are ever the same, since the walk never lands twice on one
    hue, but past fifty or so they stop being easy to tell apart. A
    legend that long is asking more of color than color can carry.
    """
    out = {}
    for index, value in enumerate(values):
        hue = (index * _GOLDEN_STEP) % 1.0
        # Held near tab20's own saturation so the two schemes sit together
        # in a figure whose other lanes are still colored from the wheel.
        saturation, brightness = _SHADES[index % len(_SHADES)]
        out[value] = (*colorsys.hsv_to_rgb(hue, saturation, brightness), 1.0)
    return out


def string_colors(values, vocabulary=None, cmap_name=STRING_CMAP) -> dict:
    """
    Map every string value to a stable color.

    The vocabulary widens the palette beyond what these values hold, so a
    figure of part of a subject colors it as a figure of all of it does.
    Adding a value to the vocabulary itself moves the colors of the ones
    which sort after it, and pushing the count past the wheel moves all
    of them.

    Two figures of one subject share this, so a label group drawn as a
    lane and the same group drawn over a patch agree on their colors.

    Parameters
    ----------
    values
        The values to color. Anything which is not a non-empty string is
        skipped, since it states no category to color.
    vocabulary
        Further values to reserve colors for.
    cmap_name
        The categorical colormap the wheel is drawn from.

    Examples
    --------
    >>> from dascore.viz._lanes import string_colors
    >>> colors = string_colors(["south", "north"])
    >>> sorted(colors)
    ['north', 'south']
    """
    seen = list(values) + list(vocabulary or [])
    values = sorted({x for x in seen if isinstance(x, str) and x != ""})
    if len(values) > len(WHEEL_ORDER):
        # Cycling the wheel here would give two values one color, and a
        # legend which says one swatch means two things is worse than none.
        return _wide_colors(values)
    cmap = plt.get_cmap(cmap_name)
    return {
        value: cmap(WHEEL_ORDER[index % len(WHEEL_ORDER)])
        for index, value in enumerate(values)
    }


def _string_colors(frame, vocabulary=None, cmap_name=STRING_CMAP) -> dict:
    """Map every string value of an interval frame to a stable color."""
    return string_colors(frame["value"].tolist(), vocabulary, cmap_name)


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


def _box_points(transform, scale, x_mid, y_mid, width, height):
    """The size of one box, in points, however the axes is scaled."""
    low = transform.transform((x_mid - width / 2, y_mid - height / 2))
    high = transform.transform((x_mid + width / 2, y_mid + height / 2))
    return abs(high[0] - low[0]) * scale, abs(high[1] - low[1]) * scale


def _text_points(text: str, size: float) -> tuple[float, float]:
    """The room a label takes, in points, at any resolution.

    A renderer rounds each glyph to whole pixels, so the same text comes
    out a tenth wider at 50 dpi than at 300, and measuring what it drew
    would let the resolution decide which labels a figure keeps. These
    are the font's own metrics, which every resolution shares.
    """
    prop = FontProperties(size=size)
    # However the text artist will read this string, it is measured the
    # same way, or the two disagree about how much room it takes.
    parse = plt.rcParams["text.parse_math"] and cbook.is_math_text(text)
    ismath = "TeX" if plt.rcParams["text.usetex"] else parse
    # Matplotlib lays a newline out as another line; the metrics do not.
    lines = text.split("\n")
    measured = [
        text_to_path.get_text_width_height_descent(x, prop, ismath) for x in lines
    ]
    width = max(x[0] for x in measured)
    height = max(x[1] for x in measured) + (len(lines) - 1) * size * _LINE_SPACING
    return width, height


def _fit_labels(ax, placements, max_labels):
    """Draw each label the way it fits its box, and drop what cannot.

    Horizontal reads best, so it is tried first. A lane of many short
    stretches gives every box far less width than its text needs;
    turning the text on its side keeps those labels, which fitting
    horizontally alone would drop and leave readable only from the
    legend.
    """
    if len(placements) > max_labels:
        return
    figure = ax.get_figure()
    # Lay the figure out before measuring: a label is compared against its
    # box, and the box moves when the axes does.
    figure.draw_without_rendering()
    transform = ax.transData
    scale = _MEASURED_DPI / figure.dpi

    size = plt.rcParams["font.size"] * 0.8
    for text, x_mid, y_mid, width, height in placements:
        if not text:
            continue
        box = _box_points(transform, scale, x_mid, y_mid, width, height)
        room = (box[0] - _LABEL_PAD, box[1] - _LABEL_PAD)
        taken = _text_points(text, size)
        for rotation in (0, 90):
            # Turning the text swaps which way it has to fit.
            if rotation:
                taken = taken[::-1]
            if taken[0] > room[0] or taken[1] > room[1]:
                continue
            ax.text(
                x_mid,
                y_mid,
                text,
                ha="center",
                va="center",
                rotation=rotation,
                fontsize=size,
                zorder=4,
                clip_on=True,
                # A dark fill would otherwise swallow the text sitting on it.
                path_effects=[pe.withStroke(linewidth=1.3, foreground="white")],
            )
            break


def _label_lines(labels: Sequence) -> list[int]:
    """How many lines each of these labels is written on."""
    return [str(x).count("\n") + 1 for x in labels]


def legend_column_points(labels: Sequence) -> float:
    """How tall one column naming these would stand.

    plot_lanes measures the legend it draws. A caller sizing a figure
    before there is a figure to measure has only this.
    """
    pitch = plt.rcParams["font.size"] * _SMALL_SCALE * _LEGEND_PITCH
    return sum(_label_lines(labels)) * pitch


def estimate_legend_rows(labels: Sequence, width_points: float) -> int:
    """How many rows a legend naming these would take, laid out this wide.

    Also an estimate; see legend_column_points.
    """
    labels = [str(x) for x in labels]
    if not labels:
        return 0
    size = plt.rcParams["font.size"] * _SMALL_SCALE
    widest = max(_text_points(x, size)[0] for x in labels)
    columns = max(1, int(width_points // (widest + _SWATCH_WIDTH * size)))
    # Counted in single lines, since that is what a caller keeping room
    # for them counts in; a row is as tall as its tallest entry.
    return -(-len(labels) // columns) * max(_label_lines(labels))


def _legend_below(figure, ax, handles, owned):
    """Lay a legend out under the lanes, in as many columns as fit.

    How wide matplotlib draws a column is not worth predicting, so the
    widest layout is drawn and narrowed until it is inside the room it
    has. Narrowing further only makes it taller, so a legend still too
    wide in one column is as close as column count can get.
    """
    # Asking a figure for room outside the axes moves every other axes on
    # it, so it is only ever asked of a figure this call built. Any other
    # belongs to its caller, and the legend takes the room of the one
    # axes it was handed.
    outside = owned and isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine)
    room = figure.bbox.width if outside else ax.get_window_extent().width
    columns = len(handles)
    while True:
        if outside:
            legend = figure.legend(
                handles=handles,
                loc="outside lower center",
                ncol=columns,
                frameon=False,
                fontsize="small",
            )
        else:
            legend = ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                borderaxespad=0.0,
                ncol=columns,
                frameon=False,
                fontsize="small",
            )
        figure.draw_without_rendering()
        box = legend.get_window_extent()
        if columns == 1 or box.width <= room:
            break
        legend.remove()
        # Overshooting by a lot is common, so step to what did fit.
        columns = max(1, min(columns - 1, int(columns * room / box.width)))
    if outside:
        return legend
    # The legend hangs off the foot of the axes, so the axes rises by what
    # the legend took and the two together cover what the axes did. Giving
    # up more than half would leave less of the lanes than of the legend
    # naming them, and a legend taller than that is one no axes this size
    # can seat; it is drawn where it falls rather than pushing the lanes
    # off the page to make room.
    position = ax.get_position()
    # Undo what an earlier call took, so drawing twice into one axes does
    # not shrink it twice.
    given = getattr(ax, "_dascore_legend_room", 0.0)
    y_low, height = position.y0 - given, position.height + given
    taken = min(box.height / figure.bbox.height, height / 2)
    ax.set_position((position.x0, y_low + taken, position.width, height - taken))
    ax._dascore_legend_room = taken
    return legend


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
    manage_figure: bool = False,
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
        states no value nothing, since its lane already names it. Text
        too wide for its box is turned on its side, and dropped only
        when it does not fit that way either.
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
        draws neither. "below" puts the legend under the lanes, in
        columns, which is for a caller who sized the figure for it there.
        Anything else draws what the colors earn, beside the lanes where
        a column of them is shorter than the axes and below when not.
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
    manage_figure
        Whether a legend too tall to sit beside the lanes may take its
        room from the figure rather than from this axes. True only for a
        caller which built the figure, since taking room from a figure
        moves every other axes on it. Implied when ax is None.
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
    owned = manage_figure or ax is None
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
        boxes, box_colors, box_bounds = [], [], []
        points, point_colors = [], []
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
            box_bounds.append((row["start"], row["end"]))
            placements.append(
                (
                    row["label"],
                    row["start"] + width / 2,
                    low + height / 2,
                    width,
                    height,
                )
            )
        if boxes:
            # Widest first, so a box which can be covered is drawn over
            # the ones which would cover it. A separator reaches past
            # the box it borders, and the narrower the neighbour the
            # more of it a separator drawn later takes.
            widest = sorted(
                range(len(boxes)),
                key=lambda x: box_bounds[x][1] - box_bounds[x][0],
                reverse=True,
            )
            ax.add_collection(
                _SeparatedBoxes(
                    [boxes[x] for x in widest],
                    facecolors=[box_colors[x] for x in widest],
                    bounds=[box_bounds[x] for x in widest],
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
        figure = ax.get_figure()
        below = legend == "below"
        if not below:
            # A colorbar already occupies the strip beside the axes.
            offset = 1.01 + 0.17 * len(colorbars)
            beside = ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(offset, 1.0),
                frameon=False,
                fontsize="small",
            )
            figure.draw_without_rendering()
            # One column beside the lanes is the natural home, but a
            # figure can name more values than its axes is tall and the
            # column then runs off the bottom of the page. Drawn and
            # measured rather than predicted: how tall matplotlib sets
            # its rows is its own affair.
            below = beside.get_window_extent().height > ax.get_window_extent().height
            if below:
                beside.remove()
        if below:
            _legend_below(figure, ax, handles, owned)
    # Fit the labels last: the legend and the colorbars have taken their
    # room by now, so a label is measured against the box it lands in.
    _fit_labels(ax, placements, max_labels)
    if show:
        plt.show()
    return ax
