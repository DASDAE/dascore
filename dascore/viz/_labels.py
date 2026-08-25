"""
Drawing where a label coordinate starts and stops over a patch dimension.

An inventory's label groups arrive on a patch as ordinary coordinates over
one dimension: a string one per channel, a boolean one stating membership,
or a number. Each states a stretch of the dimension rather than a value at
a point, so what a figure owes it is a line where it changes and a name in
a legend, not a color per sample.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from dascore.exceptions import ParameterError
from dascore.utils.gaps import get_gap_edges
from dascore.viz._lanes import (
    _as_numeric,
    _default_label,
    _legend_below,
    string_colors,
)

# Past this many a legend stops naming and starts listing, and a
# coordinate this varied reads as a quantity rather than a set of labels.
MAX_LABELS = 20

# Two colors share one boundary line, so each draws half of the dashes.
_DASH = 5.0

# Over an image and over a mesh alike.
_LINE_KWARGS = {"linewidth": 1.5, "zorder": 3}

# How much of the axes width a colorbar beside it takes, so a legend put
# further out clears it. The same estimate the lanes use.
_COLORBAR_WIDTH = 0.17


class LabelPlan(NamedTuple):
    """Which coordinate a figure draws as labels, and where it goes."""

    name: str
    dim: str
    axis: str


def label_plan(patch, name: str, dims_r: tuple[str, ...]) -> LabelPlan:
    """Return the coordinate to draw as labels, and the axis it lands on."""
    if name in patch.dims:
        msg = (
            f"label_coord={name!r} is a dimension of this patch. A label "
            "states one value over a stretch of a dimension, so it must be "
            "a coordinate associated with a dimension rather than the "
            "dimension itself."
        )
        raise ParameterError(msg)
    coord_dims = patch.coords.dim_map.get(name)
    if coord_dims is None:
        others = sorted(set(patch.coords.coord_map) - set(patch.dims))
        msg = (
            f"label_coord={name!r} is not a coordinate of this patch, whose "
            f"non-dimensional coordinates are {others}."
        )
        raise ParameterError(msg)
    if len(coord_dims) != 1 or coord_dims[0] not in dims_r:
        msg = (
            f"The {name!r} coordinate spans {list(coord_dims)}, so it names "
            "no one axis to be drawn on. A label coordinate belongs to "
            f"exactly one of the plotted dimensions, {list(dims_r)}."
        )
        raise ParameterError(msg)
    dim = coord_dims[0]
    # dims_r is (x, y); imshow and pcolormesh both put the last dim on x.
    return LabelPlan(name, dim, "x" if dim == dims_r[0] else "y")


def image_cell_edges(extents, dims_r: tuple[str, ...], dim: str, size: int):
    """Cell edges of one dimension of an image, in axis units.

    imshow lays its cells evenly across the extent it was handed, so the
    edges are read back from that extent. Reading them from the
    coordinate instead would put a line where the image has no edge
    whenever the two disagree, which is exactly when imshow was chosen
    over a mesh.
    """
    low, high = extents[:2] if dim == dims_r[0] else extents[2:]
    return low + (high - low) * np.arange(size + 1) / size


def mesh_cell_edges(values, gap_color, gap_factor):
    """Cell edges of one dimension of a mesh, in axis units.

    The very edges the mesh was drawn from: a gap it opened puts two
    edges where there was one, and moves every cell after it along.
    """
    edges, gap_mask = get_gap_edges(
        values, gap_factor if gap_color is not None else None
    )
    offsets = np.cumsum(np.concatenate([[0], gap_mask, [False]]))
    return _as_numeric(edges)[np.arange(len(offsets)) + offsets]


def _label_codes(values, name: str) -> tuple[np.ndarray, list[str]]:
    """Return a code per sample and the labels they index.

    A code of -1 is a stretch stating no label, which a boolean
    coordinate spells False, a string one "" and a numeric one NaN.
    """
    if values.dtype == np.dtype(bool):
        # Membership: the coordinate names itself, and False is not a
        # second category but the absence of the one it names.
        return np.where(values, 0, -1), [str(name)]
    missing = pd.isna(values)
    if values.dtype.kind in {"U", "S", "O"}:
        missing = missing | (values == "")
    codes, uniques = pd.factorize(np.where(missing, None, values))
    return codes, [_default_label(x) for x in uniques]


def _label_runs(values, name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return where the label changes, the codes, and the labels."""
    codes, labels = _label_codes(values, name)
    if not len(labels):
        msg = (
            f"The {name!r} coordinate states no labels, so there is nothing "
            "to draw; every one of its values is absent."
        )
        raise ParameterError(msg)
    if len(labels) > MAX_LABELS:
        msg = (
            f"The {name!r} coordinate states {len(labels)} distinct labels, "
            f"more than the {MAX_LABELS} a legend can name. A coordinate "
            "this varied is a quantity rather than a set of labels."
        )
        raise ParameterError(msg)
    # Never 0 and never len(codes), so the axes' own spines are left to
    # draw the two boundaries which sit on them.
    starts = np.flatnonzero(np.diff(codes) != 0) + 1
    return starts, codes, labels


def _draw_lines(ax, axis: str, edges, starts, codes, labels, colors) -> None:
    """Draw a line wherever the label changes, colored by what it parts."""
    line = ax.axvline if axis == "x" else ax.axhline
    for index in starts:
        position = edges[index]
        before, after = codes[index - 1], codes[index]
        stated = [x for x in (before, after) if x >= 0]
        if len(stated) == 1:
            # One side states nothing, so the boundary belongs wholly to
            # the other and is drawn solid in its color.
            line(position, color=colors[labels[stated[0]]], **_LINE_KWARGS)
            continue
        # A boundary parts two labels and belongs to neither, so it is
        # drawn twice, each color taking the dashes the other leaves.
        for offset, code in ((0.0, before), (_DASH, after)):
            line(
                position,
                color=colors[labels[code]],
                linestyle=(offset, (_DASH, _DASH)),
                **_LINE_KWARGS,
            )


def _add_legend(ax, name, labels, colors, membership, colorbars, owned) -> None:
    """Name the labels beside the axes, clear of any colorbar."""
    handles = [
        Line2D([], [], color=colors[x], linewidth=2.0, label=x) for x in labels
    ]
    figure = ax.get_figure()
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        # A colorbar already occupies the strip beside the axes.
        bbox_to_anchor=(1.01 + _COLORBAR_WIDTH * colorbars, 1.0),
        frameon=False,
        fontsize="small",
        # Membership names itself in its one entry, and a title would
        # then say the coordinate's name twice.
        title=None if membership else str(name),
        title_fontsize="small",
    )
    figure.draw_without_rendering()
    # A column beside the axes is the natural home, but a patch can state
    # more labels than its axes is tall and the column then runs off the
    # page. Drawn and measured rather than predicted, as the lanes do.
    if legend.get_window_extent().height > ax.get_window_extent().height:
        legend.remove()
        _legend_below(figure, ax, handles, owned)


def draw_labels(
    ax,
    plan: LabelPlan,
    values,
    edges,
    *,
    colorbars: int = 0,
    owned: bool = False,
) -> None:
    """
    Mark the limits of every label a coordinate states, and name them.

    Parameters
    ----------
    ax
        The axes the data was drawn on.
    plan
        The coordinate to draw and the axis its dimension was drawn on.
    values
        The coordinate's values, one per sample along its dimension.
    edges
        Cell edges along that dimension, in axis units, one more than
        there are samples.
    colorbars
        How many colorbars sit between the axes and the legend.
    owned
        Whether a legend too tall to sit beside the axes may take its
        room from the figure rather than from the axes.
    """
    values = np.asarray(values)
    starts, codes, labels = _label_runs(values, plan.name)
    membership = values.dtype == np.dtype(bool)
    colors = string_colors(labels)
    _draw_lines(ax, plan.axis, edges, starts, codes, labels, colors)
    _add_legend(ax, plan.name, labels, colors, membership, colorbars, owned)
