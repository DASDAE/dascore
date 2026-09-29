"""Draw an annotation set over the dimensions it is stated in."""

from __future__ import annotations

import itertools
import string
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import TABLEAU_COLORS, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from dascore.core.annotations import AnnotationSet, Group, Path, Region
from dascore.exceptions import ParameterError
from dascore.utils.plotting import _format_time_axis, _get_ax, _get_plot_values

# Spans, boxes and polygons sit over data, so their faces are see-through.
_FILL_ALPHA = 0.25
# Keys only lines and markers take; spans, boxes and polygons skip them.
_LINE_KEYS = {"marker", "markersize", "markeredgecolor", "markerfacecolor"}
_LINE_KEYS |= {"path_effects"}
# Ink for a feature with no color value; lines and markers get a white halo.
_INK = "black"


def _axes(dims, x, y):
    """Resolve the plotted dimensions, waterfall's by default."""
    if bad := [d for d in (x, y) if d is not None and d not in dims]:
        msg = f"{bad[0]!r} is not a dimension of the set; it has {list(dims)}."
        raise ParameterError(msg)
    if len(dims) == 1 and x is None and y is not None:
        return None, y  # across the y axis, e.g. time over a transposed waterfall
    if len(dims) <= 2:
        x = x or (dims[-1] if y != dims[-1] else dims[0])
        y = y or next((d for d in dims if d != x), None)
    if x is None or (y is None and len(dims) > 1):
        msg = f"A set stated in {list(dims)} needs x and y named."
        raise ParameterError(msg)
    if x == y:
        msg = f"x and y are both {x!r}; name two dimensions."
        raise ParameterError(msg)
    return x, y


def _column(annotations: AnnotationSet, name: str):
    """Return a lookup of a column's value by annotation row or feature id."""
    frame, features = annotations.annotations, annotations.features.set_index("id")
    if name not in frame.columns and name not in features.columns:
        msg = f"{name!r} is a column of neither the annotations nor the features."
        raise ParameterError(msg)
    ids = frame["feature_id"].fillna("")
    own = features[name] if name in features.columns else pd.Series(dtype=object)
    cells = ids.map(own)  # a blank row cell reads its feature's, and back
    rows = frame[name].where(frame[name].notna(), cells) if name in frame else cells

    def value(where):
        if not isinstance(where, str):
            out = rows[where]
        elif pd.isna(out := own.get(where)):
            members = rows[ids == where]
            out = members.iloc[0] if len(members) else None
        return None if pd.isna(out) else out

    return value


def _draw_region(ax, bounds, x, y, style: dict[str, Any]):
    """Draw one row, a marker, line, span, segment or box; return its artist."""
    xs, ys = (_get_plot_values(bounds[d]) if d in bounds else None for d in (x, y))
    fill = _fill_style(style)
    if xs is not None and ys is None:
        if xs[0] == xs[1]:
            return ax.axvline(xs[0], **style)
        else:
            return ax.axvspan(*xs, **fill)
    elif ys is not None and xs is None:
        if ys[0] == ys[1]:
            return ax.axhline(ys[0], **style)
        else:
            return ax.axhspan(*ys, **fill)
    elif xs is not None and ys is not None:
        x_point, y_point = xs[0] == xs[1], ys[0] == ys[1]
        if x_point and y_point:
            marker = {"marker": "o", "linestyle": "none", **style}
            return ax.plot(xs[:1], ys[:1], **marker)[0]
        elif x_point or y_point:
            return ax.plot(xs, ys, **style)[0]
        else:
            corner, size = (xs[0], ys[0]), (xs[1] - xs[0], ys[1] - ys[0])
            return ax.add_patch(Rectangle(corner, *size, **fill))


def _ring(ring, x, y, outer: bool) -> MplPath:
    """A closed ring, wound counterclockwise if outer and clockwise if a hole."""
    xy = np.column_stack([_get_plot_values(ring[d]) for d in (x, y)])
    area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    xy = xy if (area > 0) == outer else xy[::-1]
    return MplPath(np.vstack([xy, xy[:1]]), closed=True)


def _fill_style(style: dict[str, Any]) -> dict[str, Any]:
    """A filled artist's style: see-through face, solid edge, no marker keys or halo."""
    kept = {k: v for k, v in style.items() if k not in _LINE_KEYS}
    color = kept.pop("color")
    edge = {"facecolor": to_rgba(color, _FILL_ALPHA), "edgecolor": color}
    return {**edge, "linewidth": 1.2, **kept}


def plot(
    annotations: AnnotationSet,
    ax: plt.Axes | None = None,
    x: str | None = None,
    y: str | None = None,
    color: str | Sequence[float] | None = None,
    label: str | None = None,
    palette: Mapping | None = None,
    **style,
) -> plt.Axes:
    """
    Draw every feature of the set on an axis, e.g. over a waterfall.

    Parameters
    ----------
    annotations
        The set to draw.
    ax
        The axis to draw on; one is made if None.
    x, y
        Dimensions for the axes; default ``dims[-1]`` and ``dims[0]``, as
        `waterfall` puts them. A one-dimensional set draws along ``x``, or
        along ``y`` when only ``y`` is named.
        A set of more dimensions names both.
    color
        A matplotlib color, or a column of either table whose values are
        colored from ``palette`` first, then the property cycle, with a
        legend. A row blank on both tables is black; a feature blank on both
        reads its first member. By default everything is black.
    label
        A column whose values label the artists in a legend; a blank value
        is left out of it. Without it or a color column, a legend names the
        kinds drawn (picks, regions, paths, polygons) when there are two or
        more.
    palette
        Colors for values of the ``color`` column, e.g. ``{"P": "tab:red"}``;
        values not in it take cycle colors it does not already use.
    **style
        Passed to every artist, e.g. ``alpha`` or ``linewidth``, overriding
        the defaults; marker keys and ``path_effects`` reach only lines and
        markers.

    Notes
    -----
    A row is a marker where it states a value on both axes, a line or span
    where it states one axis, a segment where it states a value and a range,
    and a box for two ranges; a row stating neither axis is not drawn. A
    group draws its members, a path a line per part, and a polygon a filled
    patch per part with its holes left empty. A path or polygon
    not drawn in both axes is skipped. Lines and markers have a white halo,
    and filled artists a solid edge, to stand out over a waterfall.

    Examples
    --------
    >>> import pandas as pd
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> time = patch.get_coord("time").min()
    >>> picks = pd.DataFrame({"time": [time, time], "phase": ["P", "S"]})
    >>> picks["distance"] = [100.0, 200.0]
    >>> ann = dc.AnnotationSet.from_patch(patch, picks)
    >>> ax = patch.viz.waterfall()
    >>> ax = ann.viz.plot(ax=ax, color="phase")
    >>> ax = ann.viz.plot(color="phase", palette={"P": "tab:red", "S": "tab:cyan"})
    """
    style = normalize_kwargs(style, Line2D)
    x, y = _axes(annotations.dims, x, y)
    ax = _get_ax(ax)
    frame = annotations.annotations
    ids = frame["feature_id"].fillna("")
    by_color = isinstance(color, str) and (
        color in frame.columns or color in annotations.features.columns
    )
    color_of = _column(annotations, color) if by_color else None
    label_of = _column(annotations, label) if label is not None else color_of
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color")
    cycle = cycle or list(TABLEAU_COLORS.values())
    used = {to_rgba(c) for c in (palette or {}).values()}
    fresh = itertools.cycle([c for c in cycle if to_rgba(c) not in used] or cycle)
    colors, seen, drawn = defaultdict(fresh.__next__, palette or {}), set(), {}

    def styled(where):
        """The style of an artist for a row index or a feature id."""
        ink = _INK if color is None or by_color else color
        out: dict[str, Any] = {"color": ink, **style}
        width = float(out.get("linewidth") or plt.rcParams["lines.linewidth"])
        halo = pe.withStroke(linewidth=width + 2, foreground="white")
        out.setdefault("path_effects", [halo])
        if color_of is not None and (value := color_of(where)) is not None:
            out["color"] = colors[value]
        text = None if label_of is None else label_of(where)
        if text is not None and str(text) not in seen:
            seen.add(str(text))
            out["label"] = str(text)
        return out

    lone = iter(frame.index[ids == ""])
    for feature in annotations:
        geometry = feature.geometry
        if isinstance(geometry, Region | Group):
            lonely = isinstance(geometry, Region)
            rows = [next(lone)] if lonely else frame.index[ids == feature.id]
            regions = [geometry] if lonely else geometry.regions
            for where, region in zip(rows, regions, strict=True):
                if x in region.bounds or y in region.bounds:
                    art = _draw_region(ax, region.bounds, x, y, styled(int(where)))
                    drawn.setdefault(
                        "regions" if isinstance(art, Patch) else "picks", art
                    )
        elif isinstance(geometry, Path):
            for part in geometry.vertices:
                if x in part and y in part:
                    xy = (_get_plot_values(part[d]) for d in (x, y))
                    drawn.setdefault("paths", ax.plot(*xy, **styled(feature.id))[0])
        else:
            for part in (p for p in geometry.vertices if p[0].keys() >= {x, y}):
                rings = [
                    _ring(ring, x, y, number == 0) for number, ring in enumerate(part)
                ]
                path = MplPath.make_compound_path(*rings)
                patch = PathPatch(path, **_fill_style(styled(feature.id)))
                drawn.setdefault("polygons", ax.add_patch(patch))
    ax.autoscale_view()
    kinds = annotations.bounds().dtypes
    for dim, axis in ((x, "x"), (y, "y")):
        if dim is None:
            continue
        kind = kinds[f"{dim}_min"].kind
        if kind == "M":
            _format_time_axis(ax, dim, axis)
        elif not getattr(ax, f"get_{axis}label")():
            getattr(ax, f"set_{axis}label")(string.capwords(dim))
        # Time runs down the y axis, as waterfall draws it.
        if axis == "y" and kind in "Mm" and not ax.yaxis_inverted():
            ax.invert_yaxis()
    if label_of is None and len(drawn) > 1:
        for kind, art in drawn.items():
            art.set_label(kind)
            seen.add(kind)
    if seen:
        ax.legend()
    return ax
