"""Visualizations of a spool: what it covers, and where it does not."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dascore.exceptions import ParameterError
from dascore.utils.plotting import _format_time_axis
from dascore.utils.time import to_datetime64

from ._lanes import plot_lanes

# Data and its absence are a closed pair, so their colors are pinned
# rather than drawn from the palette other values share.
COVERAGE_COLORS = {"data": "#0072B2", "gap": "#D55E00"}

# The report's own columns; anything else a row carries names its group.
_REPORT_COLUMNS = frozenset(
    {"span", "gap_total", "gap_size", "covered", "coverage", "group_id"}
)

# Steps a duration is worth reading in, largest first.
_UNITS = (
    ("d", 86_400.0),
    ("h", 3_600.0),
    ("m", 60.0),
    ("s", 1.0),
    ("ms", 1e-3),
    ("µs", 1e-6),
)


def _human_duration(value) -> str:
    """Say how long something lasted, in the largest unit which fits."""
    seconds = (
        float(pd.Timedelta(value).total_seconds()) if _is_time(value) else float(value)
    )
    if not np.isfinite(seconds) or seconds == 0:
        return ""
    size = abs(seconds)
    for name, scale in _UNITS:
        if size >= scale:
            return f"{size / scale:.1f} {name}".replace(".0 ", " ")
    return f"{size:.3g} s"


def _is_time(value) -> bool:
    """Whether a value states a time rather than a number."""
    return isinstance(value, pd.Timedelta | np.timedelta64)


def _read_selection(kwargs) -> tuple[str, tuple | None]:
    """Take the dimension to measure, and any window, from the kwargs."""
    if not kwargs:
        return "time", None
    if len(kwargs) > 1:
        msg = (
            f"A spool is drawn along one dimension; {sorted(kwargs)} names "
            f"{len(kwargs)}. Call once for each."
        )
        raise ParameterError(msg)
    ((dim, window),) = kwargs.items()
    if window is None or window is ...:
        return dim, None
    try:
        low, high = window
    except (TypeError, ValueError):
        msg = f"{dim}={window!r} must be a (start, end) pair, or ... for all of it."
        raise ParameterError(msg) from None
    return dim, (low, high)


def _lane_names(report: pd.DataFrame) -> list[str]:
    """Name each group by what tells it apart, and how complete it is."""
    stated = [
        x
        for x in report.columns
        if x not in _REPORT_COLUMNS and not x.startswith("_") and "_" not in x[-4:]
    ]
    telling = [x for x in stated if report[x].astype(str).nunique() > 1]
    described = []
    for _, row in report.iterrows():
        parts = [str(row[x]) for x in telling if pd.notnull(row[x])]
        described.append(" · ".join(parts))
    # Two groups can state the same attributes and still be two groups —
    # sampling rate and coordinate structure part them without being
    # shown — so where a description is shared its ordinal tells them
    # apart. A lane which named two groups would silently draw one.
    shared = {x for x in described if described.count(x) > 1}
    names = []
    for description, (_, row) in zip(described, report.iterrows(), strict=True):
        name = description
        if not name:
            name = f"group {row['group_id']}"
        elif description in shared:
            name = f"{description} ({row['group_id']})"
        names.append(f"{name}  {_percent(row['coverage'])}")
    return names


def _percent(value: float) -> str:
    """Say a fraction as a percentage, without rounding a hole away."""
    for places in range(4):
        text = f"{value:.{places}%}"
        # 100% is a claim about the whole span, so only a whole span earns it.
        if value >= 1.0 or float(text.rstrip("%")) < 100.0:
            return text
    return "<100%"


def _tile(report: pd.DataFrame, gaps: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Cut each group's span into the runs it holds and the holes it does not."""
    low, high = f"{dim}_min", f"{dim}_max"
    names = _lane_names(report)
    rows = []
    for name, (_, group) in zip(names, report.iterrows(), strict=True):
        holes = gaps[gaps["group_id"] == group["group_id"]].sort_values(low)
        edge = group[low]
        for _, hole in holes.iterrows():
            if hole[low] > edge:
                rows.append((name, edge, hole[low], "data", ""))
            rows.append(
                (name, hole[low], hole[high], "gap", _human_duration(hole["gap_size"]))
            )
            edge = hole[high]
        if edge <= group[high]:
            rows.append((name, edge, group[high], "data", ""))
    return pd.DataFrame(rows, columns=["lane", "start", "end", "kind", "label"])


def _window_bounds(window, frame, dated: bool):
    """Resolve a window against the data, in the units the data states."""
    if window is None:
        return None
    low, high = window
    edges = []
    for value, fallback in ((low, frame["start"].min()), (high, frame["end"].max())):
        if value is None or value is ...:
            edges.append(fallback)
            continue
        edges.append(to_datetime64(value) if dated else float(value))
    if edges[1] <= edges[0]:
        msg = f"The window {window!r} must be increasing."
        raise ParameterError(msg)
    return tuple(edges)


def coverage(
    spool,
    *,
    tolerance: float = 1.5,
    group: str | Sequence[str] | None = None,
    color=None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    **kwargs,
) -> plt.Axes:
    """
    Plot what a spool covers along a dimension, and where it does not.

    One lane per group of patches which could combine, drawn as the runs
    the group holds and the holes between them. The holes are the ones
    [`get_gaps`](`dascore.core.spool.Spool.get_gaps`) reports, so a gap
    drawn here is exactly a boundary
    [`chunk`](`dascore.Spool.chunk`) would refuse to close, and the
    percentage on each lane is that group's
    [`get_coverage`](`dascore.core.spool.Spool.get_coverage`).

    Parameters
    ----------
    spool
        The spool to measure.
    tolerance
        How many samples patches may be spaced and still count as
        contiguous. Same meaning as chunk's `tolerance`.
    group
        Attributes which separate patches into unrelated groups; a gap
        is never reported between two groups. Defaults to the config
        option `patch_kind_attrs`.
    color
        A mapping of "data" and "gap" to colors, overriding the default.
    ax
        A matplotlib Axes; one is created when None.
    figsize
        Size of the figure built when ax is None.
    show
        Whether to call plt.show.
    **kwargs
        The dimension to measure along, optionally with the window to
        draw: `time=("2020-01-01", None)`. `time=...` states the
        dimension and asks for all of it. Defaults to the whole of time.

    Notes
    -----
    Coverage is measured between patches, from the envelopes the index
    records, so a hole *inside* a patch is not visible here. A lane
    reading 100% says "nothing chunk would refuse to merge", not
    "nothing missing".

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.viz.spool import coverage
    >>>
    >>> spool = dc.get_example_spool("diverse_das")
    >>> _ = coverage(spool)
    >>> _ = coverage(spool, time=("2020-01-03", "2020-01-04"))
    """
    dim, window = _read_selection(kwargs)
    # get_coverage refuses a dimension the spool does not state, and
    # names the ones it does, so its message is better than any here.
    report = spool.get_coverage(dim, tolerance=tolerance, group=group)
    gaps = spool.get_gaps(dim, tolerance=tolerance, group=group)
    frame = _tile(report, gaps, dim)
    lanes = list(dict.fromkeys(frame["lane"]))
    if ax is None:
        height = min(1.2 + 0.45 * len(lanes), 14.0)
        _, ax = plt.subplots(1, figsize=figsize or (10.0, height), layout="constrained")
    dated = pd.api.types.is_datetime64_any_dtype(frame["start"])
    plot_lanes(
        frame,
        ax=ax,
        lane="lane",
        value="kind",
        label="label",
        lanes=lanes,
        color=color or COVERAGE_COLORS,
        x_limits=_window_bounds(window, frame, dated),
        x_label="" if dated else dim,
    )
    if dated:
        _format_time_axis(ax, dim, "x")
    if show:
        plt.show()
    return ax
