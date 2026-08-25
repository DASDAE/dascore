"""
Drawing where a label coordinate starts and stops over a patch dimension.

An inventory's label groups arrive on a patch as ordinary coordinates over
one dimension: a string naming each sample, a boolean stating membership,
or a number. Each states a stretch of the dimension rather than a value at
a point, so what a figure owes it is that stretch marked off and named:
a bar along the axis, rather than a color per sample.

The bars sit on the spines rather than over the image. A wash over the
data would have to be strong enough to see, and a wash that strong has
recolored the data under it -- which on a diverging colormap is the
measurement itself.
"""

from __future__ import annotations

from contextlib import contextmanager
from itertools import pairwise
from typing import NamedTuple

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from dascore.exceptions import ParameterError
from dascore.viz._lanes import _as_numeric, _default_label, string_colors

# Past this many a legend stops naming and starts listing.
MAX_LABELS = 20

# Past this many changes the bars are narrower than the hairlines
# parting them, and the spine reads as hatching rather than as a set of
# ranges.
MAX_RUNS = 200

# How thick a bar is drawn, in points. Half of it falls outside the axes,
# so the spine it sits on is covered rather than merely traced.
_BAR_WIDTH = 7.0

# A hairline joining the two bars, so a change can be traced through the
# image rather than only read off its edges. Faint on purpose: it locates
# a boundary and does not compete with the data. The white stroke beneath
# is what keeps it visible over a busy image, where a grey line of this
# weight disappears into the noise.
_SEAM_KWARGS = {"color": "0.1", "linewidth": 0.7, "alpha": 0.55, "zorder": 4}
_SEAM_HALO_WIDTH = 2.0
_SEAM_HALO_ALPHA = 0.5

# What each artist is, for a caller reading a figure back. Each carries
# the prefix, the axes it sits on and its own number, so no two on one
# figure share an id.
BAR_GID = "dascore-label-bar"
SEAM_GID = "dascore-label-change"

# The gap kept between the legend and what it sits beside, and between
# the legend and the edge of the page, as a fraction of the figure width.
_LEGEND_PAD = 0.02

# How many times the figure is narrowed and measured again. Narrowing
# moves what the legend sits beside, so one pass is a guess and a handful
# settles it. Stopping short leaves the names further right than they
# want to be, which is a worse picture rather than a broken one.
_FITTING_PASSES = 4

# The share of the figure width the plotting area keeps whatever the
# legend asks for -- the image and its colorbar together. Names wanting
# more than the rest are ones no figure this size can seat beside the
# picture, and they run off the edge rather than squeezing the picture
# they name out of existence.
_MIN_AXES_WIDTH = 0.25


class LabelPlan(NamedTuple):
    """Which coordinate a figure draws as labels, and where it goes."""

    name: str
    dim: str
    axis: str


class LabelRuns(NamedTuple):
    """Where a label coordinate changes, and what it states between.

    Worked out before anything is drawn, so a coordinate this module
    refuses leaves no half-made figure behind.
    """

    starts: np.ndarray
    codes: np.ndarray
    labels: list[str]
    membership: bool


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
    """Where each cell of an image starts and ends, in axis units.

    imshow lays its cells evenly across the extent it was handed, so the
    edges are read back from that extent. Reading them from the
    coordinate instead would put a line where the image has no edge
    whenever the two disagree, which is exactly when imshow was chosen
    over a mesh.
    """
    low, high = extents[:2] if dim == dims_r[0] else extents[2:]
    edges = np.linspace(low, high, size + 1)
    return edges[:-1], edges[1:]


def mesh_cell_edges(edges, gap_mask):
    """Where each cell of a mesh starts and ends, in axis units.

    The very edges the mesh was drawn from, indexed by sample. A gap it
    opened puts two edges where there was one, so every cell after a gap
    sits one place further along -- and a cell bordering a gap ends where
    the gap begins, rather than where the cell beyond it starts.
    """
    edges = _as_numeric(edges)
    before = np.cumsum(np.concatenate([[0], gap_mask]))
    index = np.arange(len(before)) + before
    return edges[index], edges[index + 1]


def _stated_mask(values) -> np.ndarray:
    """Which samples state a value at all.

    How absence is spelled follows the dtype a coordinate can hold: a
    string array has no null, so it is the empty string there, and NaN or
    NA in anything which can carry one.
    """
    missing = np.asarray(pd.isna(values), dtype=bool)
    if values.dtype.kind in {"U", "S", "O"}:
        blank = np.zeros(values.shape, dtype=bool)
        # A bytes array holds bytes, and numpy will not compare the two.
        empty = b"" if values.dtype.kind == "S" else ""
        # Only where a value is present: NA answers a comparison with NA.
        np.equal(values, empty, out=blank, where=~missing)
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
    return codes, _disambiguate(labels, uniques), False


def _disambiguate(labels: list[str], values) -> list[str]:
    """Qualify by type any name which two different values still share.

    An object coordinate may hold the number 1 beside the string "1",
    which print alike however many digits are kept. Sharing the name
    would share the color too, and the legend would say one swatch
    means two things.
    """
    seen: dict[str, int] = {}
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
    return [
        f"{label} ({type(value).__name__})" if seen[label] > 1 else label
        for label, value in zip(labels, values, strict=True)
    ]


def label_runs(values, name: str) -> LabelRuns:
    """Return where the label changes, what it states, and of which kind."""
    values = np.asarray(values)
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
    if len(starts) > MAX_RUNS:
        msg = (
            f"The {name!r} coordinate changes value {len(starts)} times, "
            f"more than the {MAX_RUNS} changes a figure can show. It "
            "states a value per sample rather than a stretch each."
        )
        raise ParameterError(msg)
    return LabelRuns(starts, codes, labels, membership)


def _artist_ids(ax, prefix: str):
    """Number artists so no two on the figure share a gid.

    Matplotlib writes a gid out as an SVG id, which has to name one
    element of the whole document. Numbering within the call is not
    enough: a figure may carry a labelled plot on every axes, and one
    axes may be drawn on twice. Counted rather than made up, so the same
    figure carries the same ids however often it is built.
    """
    figure = ax.get_figure()
    panel = figure.axes.index(ax) if ax in figure.axes else 0
    already = sum(1 for x in ax.lines if str(x.get_gid()).startswith(prefix))
    return lambda index: f"{prefix}-{panel}-{already + index}"


def _draw_bars(ax, axis: str, edges, starts, codes, labels, colors) -> None:
    """Draw a bar along both spines over the stretch each label covers.

    A stretch stating nothing leaves bare spine, so what a group covers
    and what it merely passes over are read off the same edge.
    """
    if axis == "y":
        # One coordinate puts the bar on a spine and the other runs it
        # along the stretch; which is which is what the axis decides.
        transform = blended_transform_factory(ax.transAxes, ax.transData)
    else:
        transform = blended_transform_factory(ax.transData, ax.transAxes)
    starts_at, ends_at = edges
    identify = _artist_ids(ax, BAR_GID)
    bounds = np.concatenate([[0], starts, [len(codes)]]).astype(int)
    drawn = 0
    for low, high in pairwise(bounds):
        code = codes[low]
        if code < 0:
            continue
        # Ends where its own last cell ends, which is where a gap band
        # begins rather than where the cell beyond one starts.
        span = (starts_at[low], ends_at[high - 1])
        for spine in (0.0, 1.0):
            along = ((spine, spine), span) if axis == "y" else (span, (spine, spine))
            ax.plot(
                *along,
                transform=transform,
                color=colors[labels[code]],
                linewidth=_BAR_WIDTH,
                solid_capstyle="butt",
                # Clipped, or a bar keeps its data coordinates when the
                # limits change and paints across the rest of the figure.
                # The half of its width outside the axes goes with that.
                clip_on=True,
                zorder=5,
                gid=identify(drawn),
            )
            drawn += 1


@contextmanager
def _measuring(ax):
    """Hide what is costly to draw while the legend is being measured.

    Seating the legend means laying the figure out several times over,
    and each pass would otherwise raster the image again -- four fifths
    of the time a labelled waterfall takes on a large patch. Nothing the
    measurements read is drawn from it: the tick labels, the axes' own
    box and the legend are all laid out the same either way.
    """
    hidden = [x for x in (*ax.images, *ax.collections) if x.get_visible()]
    for artist in hidden:
        artist.set_visible(False)
    try:
        yield
    finally:
        for artist in hidden:
            artist.set_visible(True)


def _right_edge(figure) -> float:
    """How far right anything already drawn on the figure reaches."""
    boxes = [x.get_tightbbox() for x in figure.axes]
    return max(x.x1 for x in boxes if x is not None) / figure.bbox.width


def _draw_changes(ax, axis: str, edges, starts) -> None:
    """Join the two bars with a hairline wherever the label changes."""
    starts_at, _ = edges
    identify = _artist_ids(ax, SEAM_GID)
    line = ax.axvline if axis == "x" else ax.axhline
    for drawn, index in enumerate(starts):
        line(
            starts_at[index],
            path_effects=[
                pe.withStroke(
                    linewidth=_SEAM_HALO_WIDTH,
                    foreground="white",
                    alpha=_SEAM_HALO_ALPHA,
                )
            ],
            gid=identify(drawn),
            **_SEAM_KWARGS,
        )


def _measure(figure, ax, kwargs) -> int:
    """Make room for the legend, and say how many columns it wants.

    Drawn rather than predicted: how tall matplotlib sets its rows and
    how wide it sets a column are its own affair. The figure is narrowed
    and asked again after each move, since a colorbar carries its ticks
    and its own name outside the rectangle it reports as its position,
    and narrowing can relabel the ticks it carries.
    """
    legend = figure.legend(**kwargs)
    figure.draw_without_rendering()
    box = legend.get_window_extent()
    # A column taller than the page is spilled into as many as it takes.
    columns = max(1, int(np.ceil(box.height / figure.bbox.height)))
    if columns > 1:
        legend.remove()
        legend = figure.legend(ncol=columns, **kwargs)
        figure.draw_without_rendering()
        box = legend.get_window_extent()
    wide = box.width / figure.bbox.width
    legend.remove()
    for _ in range(_FITTING_PASSES):
        over = _right_edge(figure) + 2 * _LEGEND_PAD + wide - 1.0
        if over <= 0:
            break
        pars = figure.subplotpars
        floor = pars.left + _MIN_AXES_WIDTH
        if pars.right <= floor:
            break
        figure.subplots_adjust(right=max(floor, pars.right - over))
        figure.draw_without_rendering()
    return columns


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
        # The room is measured here rather than left to matplotlib, so
        # its own padding would put the legend past what was measured.
        "borderaxespad": 0.0,
    }
    engine = figure.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        # This engine keeps room for a legend placed outside the axes, so
        # it is asked for the room rather than overruled. No other engine
        # does, which is why the test is for this one and not for any.
        return figure.legend(
            **{**kwargs, "loc": "outside right upper", "borderaxespad": None}
        )
    if engine is not None:
        # Any other engine recomputes the margins when the figure is
        # drawn, undoing the room made below. This figure is the call's
        # own and is being laid out here explicitly.
        figure.set_layout_engine("none")
    with _measuring(ax):
        columns = _measure(figure, ax, kwargs)
    return figure.legend(
        ncol=columns,
        bbox_to_anchor=(_right_edge(figure) + _LEGEND_PAD, ax.get_position().y1),
        bbox_transform=figure.transFigure,
        **kwargs,
    )


def _add_legend(ax, name, labels, colors, membership, owned):
    """Name the labels, beside the figure where there is room to make."""
    handles = [Line2D([], [], color=colors[x], linewidth=3.0, label=x) for x in labels]
    # Membership names itself in its one entry, and a title would then
    # say the coordinate's name twice.
    title = None if membership else str(name)
    if owned:
        return _legend_beside(ax.get_figure(), ax, handles, title)
    # The figure belongs to the caller, and making room in it would move
    # every other axes on it, so the legend takes room from the one axes
    # it was handed rather than being drawn over its neighbour.
    previous = ax.get_legend()
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        fontsize="small",
        title=title,
        title_fontsize="small",
        framealpha=0.8,
    )
    if previous is not None:
        # An axes holds one legend, so asking for ours drops whatever the
        # caller had named there; it goes back as a plain artist.
        ax.add_artist(previous)
    return legend


def draw_labels(
    ax,
    plan: LabelPlan,
    runs: LabelRuns,
    edges,
    *,
    owned: bool = False,
) -> None:
    """
    Mark the stretch every label a coordinate states covers, and name them.

    Parameters
    ----------
    ax
        The axes the data was drawn on.
    plan
        The coordinate to draw and the axis its dimension was drawn on.
    runs
        Where the coordinate changes and what it states, from
        [`label_runs`](`dascore.viz._labels.label_runs`).
    edges
        Where each cell along that dimension starts and ends, in axis
        units, as a pair of arrays one sample long each.
    owned
        Whether the figure is this call's to make room in. False puts the
        legend inside the axes, since taking room from someone else's
        figure would move every other axes on it.
    """
    starts, codes, labels, membership = runs
    colors = string_colors(labels)
    _draw_bars(ax, plan.axis, edges, starts, codes, labels, colors)
    _draw_changes(ax, plan.axis, edges, starts)
    _add_legend(ax, plan.name, labels, colors, membership, owned)
