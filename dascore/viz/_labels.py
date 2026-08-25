"""
Drawing where a label coordinate starts and stops over a patch dimension.

An inventory's label groups arrive on a patch as ordinary coordinates over
one dimension: a string naming each sample, a boolean stating membership,
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
from dascore.viz._lanes import _as_numeric, _default_label, string_colors

# Past this many a legend stops naming and starts listing.
MAX_LABELS = 20

# Past this many boundaries the lines are closer together than the data
# behind them, and the figure reads as hatching rather than as limits.
MAX_BOUNDARIES = 200

# Two colors share one boundary line, so each draws half of the dashes.
_DASH = 5.0

# zorder 3 puts the line over an image and over a mesh alike.
_LINE_KWARGS = {"linewidth": 1.5, "zorder": 3}

# The gap kept between the legend and what it sits beside, and between
# the legend and the edge of the page, as a fraction of the figure width.
_LEGEND_PAD = 0.02


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
    return np.linspace(low, high, size + 1)


def mesh_cell_edges(edges, gap_mask):
    """Cell edges of one dimension of a mesh, in axis units.

    The very edges the mesh was drawn from, indexed by sample: a gap it
    opened puts two edges where there was one, so every cell after a gap
    sits one place further along.
    """
    offsets = np.cumsum(np.concatenate([[0], gap_mask, [False]]))
    return _as_numeric(edges)[np.arange(len(offsets)) + offsets]


def _stated_mask(values) -> np.ndarray:
    """Which samples state a value at all.

    How absence is spelled follows the dtype a coordinate can hold: a
    string array has no null, so it is the empty string there, and NaN or
    NA in anything which can carry one.
    """
    missing = np.asarray(pd.isna(values), dtype=bool)
    if values.dtype.kind in {"U", "S", "O"}:
        blank = np.zeros(values.shape, dtype=bool)
        # Only where a value is present: NA answers a comparison with NA.
        np.equal(values, "", out=blank, where=~missing)
        missing = missing | blank
    return ~missing


def _is_membership(stated) -> bool:
    """Whether the values a coordinate states are all true or false.

    A boolean array is the plain case; an object array of booleans is
    what a coordinate carrying nulls beside them comes to.
    """
    if stated.dtype == np.dtype(bool):
        return True
    return stated.dtype.kind == "O" and all(
        isinstance(x, bool | np.bool_) for x in stated
    )


def _label_codes(values, name: str) -> tuple[np.ndarray, list[str], bool]:
    """Return a code per sample and the labels those codes index.

    A code of -1 is a sample stating no label: False in a membership
    coordinate, "" in a string one, NaN in a number.
    """
    held = _stated_mask(values)
    stated = values[held]
    if not stated.size:
        return np.full(values.shape, -1, dtype=int), [], False
    if _is_membership(stated):
        # The coordinate names itself, and False is not a second category
        # but the absence of the one it names.
        codes = np.full(values.shape, -1, dtype=int)
        codes[held] = np.where(stated.astype(bool), 0, -1)
        return codes, [str(name)], True
    codes = np.full(values.shape, -1, dtype=int)
    # Factorized over the stated values alone, which keeps the
    # coordinate's own dtype: widening to hold a null would render a
    # nanosecond datetime as its count of nanoseconds.
    inner, uniques = pd.factorize(stated)
    codes[held] = inner
    labels = [_default_label(x) for x in uniques]
    if len(set(labels)) != len(labels):
        # Two values rounding to one name would share a color and a
        # legend entry, and the figure would call them the same thing.
        labels = [str(x) for x in uniques]
    return codes, labels, False


def _label_runs(values, name: str):
    """Return where the label changes, the codes, the labels, and the kind."""
    codes, labels, membership = _label_codes(values, name)
    if not np.any(codes >= 0):
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
    if len(starts) > MAX_BOUNDARIES:
        msg = (
            f"The {name!r} coordinate changes value {len(starts)} times, "
            f"more than the {MAX_BOUNDARIES} boundaries a figure can show. "
            "It states a value per sample rather than a stretch each."
        )
        raise ParameterError(msg)
    return starts, codes, labels, membership


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


def _legend_beside(figure, ax, handles, title):
    """Name the labels past everything else, in room made for them.

    Nothing reserves room outside an axes, so a legend anchored there is
    drawn off the page. The figure's right margin is pulled in instead,
    which moves the image and its colorbar together: the two hang off one
    gridspec, and repositioning either alone would part them.
    """
    kwargs = {
        "handles": handles,
        "loc": "upper left",
        "frameon": False,
        "fontsize": "small",
        "title": title,
        "title_fontsize": "small",
    }
    # Drawn once to be measured; how tall matplotlib sets its rows and how
    # wide it sets a column are its own affair rather than ours to predict.
    legend = figure.legend(**kwargs)
    figure.draw_without_rendering()
    box = legend.get_window_extent()
    tall = box.height / figure.bbox.height
    # A column taller than the page is spilled into as many as it takes.
    columns = max(1, int(np.ceil(tall)))
    if columns > 1:
        legend.remove()
        legend = figure.legend(ncol=columns, **kwargs)
        figure.draw_without_rendering()
        box = legend.get_window_extent()
    wide = box.width / figure.bbox.width
    legend.remove()
    right = figure.subplotpars.right
    # Half the figure is as much as the names may take; a legend needing
    # more would be larger than the picture it names.
    room = min(wide + 2 * _LEGEND_PAD, right / 2)
    figure.subplots_adjust(right=right - room)
    figure.draw_without_rendering()
    # Whatever ended up furthest right is what the legend sits beside,
    # so a colorbar is cleared without guessing how wide one is.
    edge = max(x.get_position().x1 for x in figure.axes)
    return figure.legend(
        ncol=columns,
        bbox_to_anchor=(edge + _LEGEND_PAD, ax.get_position().y1),
        bbox_transform=figure.transFigure,
        **kwargs,
    )


def _add_legend(ax, name, labels, colors, membership, owned):
    """Name the labels, beside the figure where there is room to make."""
    handles = [Line2D([], [], color=colors[x], linewidth=2.0, label=x) for x in labels]
    # Membership names itself in its one entry, and a title would then
    # say the coordinate's name twice.
    title = None if membership else str(name)
    if owned:
        return _legend_beside(ax.get_figure(), ax, handles, title)
    # The figure belongs to the caller, and making room in it would move
    # every other axes on it, so the legend takes room from the one axes
    # it was handed rather than being drawn over its neighbour.
    return ax.legend(
        handles=handles,
        loc="upper right",
        fontsize="small",
        title=title,
        title_fontsize="small",
        framealpha=0.8,
    )


def draw_labels(
    ax,
    plan: LabelPlan,
    values,
    edges,
    *,
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
    owned
        Whether the figure is this call's to make room in. False puts the
        legend inside the axes, since taking room from someone else's
        figure would move every other axes on it.
    """
    values = np.asarray(values)
    starts, codes, labels, membership = _label_runs(values, plan.name)
    colors = string_colors(labels)
    _draw_lines(ax, plan.axis, edges, starts, codes, labels, colors)
    _add_legend(ax, plan.name, labels, colors, membership, owned)
