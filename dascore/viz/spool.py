"""Visualizations of a spool: what it covers, and where it does not."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from dascore.exceptions import ParameterError
from dascore.units import Quantity
from dascore.utils.chunk_plan import _REPORT_COLUMNS
from dascore.utils.display import (
    _SECONDS_IN_DAY,
    ACQUISITION_ATTR,
    group_names,
    human_duration,
    percent,
)
from dascore.utils.plotting import _format_time_axis, _get_cmap

from ._lanes import plot_lanes

# Data and its absence are a closed pair, so their colors are pinned
# rather than drawn from the palette other values share.
COVERAGE_COLORS = {"data": "#0072B2", "gap": "#D55E00"}
# A day the spool says nothing about is not a day it covers by nothing.
_UNSTATED_COLOR = "0.85"

# Each calendar method reads one number -- the seconds of the day the
# spool covers -- a different way, so each gets its own scale.
_CALENDAR_CMAPS = {"percent": "RdYlGn", "gap": "RdYlGn_r", "count": "YlGn"}
_CALENDAR_LABELS = {
    "percent": "Availability [%]",
    "gap": "Missing from the day",
    "count": "Patches overlapping the day",
}
_CALENDAR_METHODS = tuple(_CALENDAR_CMAPS)
# The durations a missing-time colorbar is worth reading in. They are
# named by the same formatter the gap labels use, so a colorbar and a
# bar of the coverage plot say a length the same way.
_GAP_TICKS = (1.0, 60.0, 600.0, 3_600.0, 6 * 3_600.0, _SECONDS_IN_DAY)

# A dimension states itself in these. They describe the extent a lane
# is drawn over, so naming the lane with them would repeat the axis.
_ENVELOPE_SUFFIXES = ("min", "max", "step", "units")


def _lane_names(report: pd.DataFrame, dim: str) -> list[str]:
    """Name each group by what tells it apart, and how complete it is."""
    # Named for the measured dimension rather than by suffix: an
    # attribute may legitimately be called `site_min`, and it names its
    # group as any other attribute does.
    envelope = {f"{dim}_{x}" for x in _ENVELOPE_SUFFIXES}
    names = group_names(
        report,
        ignore=set(_REPORT_COLUMNS) | envelope,
        ordinals=report["group_id"],
        fallback=ACQUISITION_ATTR,
    )
    return [
        f"{name}  {percent(coverage)}"
        for name, coverage in zip(names, report["coverage"], strict=True)
    ]


def _new_ax(ax, height: float):
    """Build the figure a plot of this height needs, unless given one."""
    if ax is not None:
        return ax
    _, ax = plt.subplots(1, figsize=(10.0, min(height, 14.0)), layout="constrained")
    return ax


def _runs(report: pd.DataFrame, gaps: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Cut each group's span into the contiguous runs it holds."""
    low, high = f"{dim}_min", f"{dim}_max"
    rows = []
    for _, group in report.iterrows():
        holes = gaps[gaps["group_id"] == group["group_id"]].sort_values(low)
        edge = group[low]
        for _, hole in holes.iterrows():
            # A run of one sample begins where the gap after it begins,
            # so equal bounds are a run -- plot_lanes draws it as a point
            # marker -- rather than a run which is not there.
            if hole[low] >= edge:
                rows.append((group["group_id"], edge, hole[low]))
            edge = hole[high]
        if edge <= group[high]:
            rows.append((group["group_id"], edge, group[high]))
    return pd.DataFrame(rows, columns=["group_id", "start", "end"])


def _gap_label(size, units: str, dated: bool) -> str:
    """Say how wide a gap is, in what its own dimension is measured in."""
    if dated:
        return human_duration(size)
    value = float(size)
    if not np.isfinite(value) or value == 0:
        return ""
    return f"{value:g} {units}".strip()


def _tile(report: pd.DataFrame, gaps: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Lay each group's runs and holes out as the lane which draws them."""
    low, high = f"{dim}_min", f"{dim}_max"
    names = dict(zip(report["group_id"], _lane_names(report, dim), strict=True))
    # Only time is measured in seconds. Any other dimension states what
    # it is measured in, and a gap along it is labelled with that.
    dated = pd.api.types.is_timedelta64_dtype(gaps["gap_size"])
    units = gaps.get(f"{dim}_units", pd.Series(dtype=object))
    rows = [
        (names[group_id], start, end, "data", "")
        for group_id, start, end in _runs(report, gaps, dim).itertuples(index=False)
    ]
    rows += [
        (
            names[hole["group_id"]],
            hole[low],
            hole[high],
            "gap",
            _gap_label(hole["gap_size"], str(units.get(index, "") or ""), dated),
        )
        for index, hole in gaps.iterrows()
    ]
    return pd.DataFrame(rows, columns=["lane", "start", "end", "kind", "label"])


def coverage(
    spool,
    dim: str = "time",
    *,
    tolerance: float | Quantity | np.timedelta64 = 1.5,
    group: str | Sequence[str] | None = None,
    color=None,
    ax: plt.Axes | None = None,
    show: bool = False,
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
    dim
        The dimension to measure along.
    tolerance
        How many samples patches may be spaced and still count as
        contiguous, or a quantity or timedelta stating that limit as a
        distance (eg `1 * s`). Same meaning as chunk's `tolerance`.
    group
        Attributes which separate patches into unrelated groups; a gap
        is never reported between two groups. Defaults to the config
        option `patch_kind_attrs`.
    color
        A mapping of "data" and "gap" to colors, overriding the default.
    ax
        A matplotlib Axes; one is created, a lane tall per group, when
        None. Pass one to say how large the plot is.
    show
        Whether to call plt.show.

    Notes
    -----
    The whole of what the spool holds is drawn. A part of it is a
    smaller spool: select it first, as
    `spool.select(time=(start, end)).viz.coverage()`, and the lanes and
    their percentages are of that window rather than of a window drawn
    over percentages of everything. To keep the whole picture and look
    at part of it instead -- where a lane running off the edge says the
    data does not stop there -- set the limits on the axes this returns.
    Labels are laid out for the extent drawn, so a zoom made that way
    shows fewer of them than it has room for.

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
    >>> _ = coverage(spool, "distance")
    >>> _ = coverage(spool.select(time=("2020-01-03", "2020-01-04")))
    """
    # get_coverage refuses a dimension the spool does not state, and
    # names the ones it does, so its message is better than any here.
    report = spool.get_coverage(dim, tolerance=tolerance, group=group)
    gaps = spool.get_gaps(dim, tolerance=tolerance, group=group)
    frame = _tile(report, gaps, dim)
    lanes = list(dict.fromkeys(frame["lane"]))
    ax = _new_ax(ax, 1.2 + 0.45 * len(lanes))
    dated = pd.api.types.is_datetime64_any_dtype(frame["start"])
    plot_lanes(
        frame,
        ax=ax,
        lane="lane",
        value="kind",
        label="label",
        lanes=lanes,
        color=color or COVERAGE_COLORS,
        x_label="" if dated else dim,
    )
    if dated:
        _format_time_axis(ax, dim, "x")
    if show:
        plt.show()
    return ax


def _extended(runs: pd.DataFrame, report: pd.DataFrame, dim: str) -> np.ndarray:
    """
    Return where each run stops covering time, not where its last sample is.

    An envelope ends on a sample, so a run of a whole day is one step
    short of the day. That is the convention `get_coverage`'s `span`
    keeps, and the one a calendar must not: a day of unbroken data has
    to read as a full day.
    """
    steps = dict(zip(report["group_id"], report[f"{dim}_step"], strict=True))
    zero = np.timedelta64(0, "ns")
    # A descending coordinate signs its step; a run is a step longer
    # than its envelope whichever way its samples are ordered.
    extra = [
        zero if pd.isnull(steps[x]) else abs(np.timedelta64(steps[x]))
        for x in runs["group_id"]
    ]
    return runs["end"].to_numpy() + np.array(extra, dtype="timedelta64[ns]")


def _union(starts: np.ndarray, ends: np.ndarray) -> list[tuple]:
    """
    Merge intervals so no moment is counted twice.

    Two acquisitions may run at once, and a day they both cover is one
    day covered, not two. Availability is a measure of the union.
    """
    merged: list[list] = []
    for start, end in sorted(zip(starts, ends, strict=True)):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(x) for x in merged]


def _day_seconds(intervals, days: np.ndarray) -> np.ndarray:
    """Return how many seconds of each day the intervals cover."""
    out = np.zeros(len(days))
    one_day = np.timedelta64(1, "D")
    second = np.timedelta64(1, "s")
    zero = np.timedelta64(0, "ns")
    for start, end in intervals:
        # Only the days an interval touches are measured against it; a
        # long archive has far more days than any one interval spans.
        first = max(int(np.searchsorted(days, start, "right")) - 1, 0)
        last = min(int(np.searchsorted(days, end, "right")) - 1, len(days) - 1)
        touched = days[first : last + 1]
        low = np.maximum(start, touched)
        high = np.minimum(end, touched + one_day)
        out[first : last + 1] += np.maximum(high - low, zero) / second
    return out


def _day_counts(contents: pd.DataFrame, days: np.ndarray, dim: str) -> np.ndarray:
    """Return how many patches overlap each day."""
    one_day = np.timedelta64(1, "D")
    starts = contents[f"{dim}_min"].to_numpy()
    ends = contents[f"{dim}_max"].to_numpy()
    return np.array(
        [((starts < day + one_day) & (ends >= day)).sum() for day in days],
        dtype=float,
    )


def _calendar_days(runs: pd.DataFrame, ends: np.ndarray) -> pd.DatetimeIndex:
    """Return every day the calendar shows, first and last included."""
    # A run covers up to its extended end without reaching it, so a run
    # stopping at midnight ends on the day before, and one reaching half
    # an hour past it earns the day it reaches into.
    first = runs["start"].min()
    last = ends.max() - np.timedelta64(1, "ns")
    # The last day is a day the spool has data in, so the calendar shows
    # it. An exclusive end would leave a one-day spool with no cells.
    return pd.date_range(
        pd.Timestamp(first).floor("D"), pd.Timestamp(last).floor("D"), freq="D"
    )


def _calendar_cells(days: pd.DatetimeIndex, values: np.ndarray):
    """
    Lay one value per day out as one row per month, one column per day.

    Returns the matrix and the name of each of its rows. Cells no date
    lands in -- a thirty-first of February -- stay NaN.
    """
    dates = days.to_numpy().astype("datetime64[D]")
    months = dates.astype("datetime64[M]")
    rows = (months - months[0]).astype(int)
    # Truncating to a month gives its first day, so the days since it
    # are the column, and every month starts in the first one.
    columns = (dates - months).astype(int)
    matrix = np.full((int(rows[-1]) + 1, 31), np.nan)
    matrix[rows, columns] = values
    labels = [
        pd.Timestamp(months[0] + np.timedelta64(x, "M")).strftime("%Y-%b")
        for x in range(matrix.shape[0])
    ]
    return matrix, labels


def _calendar_axes(matrix: np.ndarray, labels, ax) -> None:
    """Put the day and month names in the middle of their cells."""
    columns = np.arange(31)
    rows = np.arange(matrix.shape[0])
    for axis, ticks, names in (
        (ax.xaxis, columns, [str(x + 1) for x in columns]),
        (ax.yaxis, rows, labels),
    ):
        # Cell edges carry the grid, and their midpoints the names.
        axis.set_major_locator(ticker.FixedLocator(list(range(len(ticks) + 1))))
        axis.set_major_formatter(ticker.NullFormatter())
        axis.set_minor_locator(ticker.FixedLocator([x + 0.5 for x in ticks]))
        axis.set_minor_formatter(ticker.FixedFormatter(names))
    ax.tick_params(axis="both", which="minor", length=0)
    ax.set_xlabel("Day of month")
    ax.yaxis.set_inverted(True)


def calendar(
    spool,
    *,
    method: str = "percent",
    tolerance: float | Quantity | np.timedelta64 = 1.5,
    group: str | Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot how much data a spool holds on each day, as a calendar.

    One row per month and one column per day of month, colored by what
    `method` asks for. The runs a day is measured against are the ones
    [`get_gaps`](`dascore.core.spool.Spool.get_gaps`) leaves between its
    gaps, so a day drawn as full is a day
    [`chunk`](`dascore.Spool.chunk`) would merge end to end.

    Parameters
    ----------
    spool
        The spool to measure.
    method
        What each day says.

        - "percent": how much of the day holds data, as a percentage.
        - "gap": how much of the day does not, in seconds, log scaled.
        - "count": how many patches overlap the day.
    tolerance
        How many samples patches may be spaced and still count as
        contiguous, or a quantity or timedelta stating that limit as a
        distance (eg `1 * s`). Same meaning as chunk's `tolerance`.
    group
        Attributes which separate patches into unrelated groups. Defaults
        to the config option `patch_kind_attrs`. It decides which
        boundaries are gaps; a day is the union of every group's data,
        so regrouping moves a day's total only where it moves a gap.
    ax
        A matplotlib Axes; one is created, a row tall per month, when
        None. Pass one to say how large the plot is.
    show
        Whether to call plt.show.

    Notes
    -----
    Groups which run at once cover one day between them, not two, so a
    day is the measure of what they cover together and never exceeds
    100%. A day the spool says nothing about is drawn grey; one it
    covers by nothing at all is drawn as empty, which is a different
    claim.

    `group` is not a way to pick one of them: select the patches first,
    as `spool.select(tag="temperature").viz.calendar()`, and the
    calendar is of those alone. Days are chosen the same way — the
    calendar runs from the first day the spool holds to the last, so
    `spool.select(time=(start, end)).viz.calendar()` draws that season
    and measures it. It draws the days that selection kept: an outage
    at either end of the window falls outside the calendar rather than
    filling it, and a window with no data in it at all is a spool of
    nothing, which has no calendar. Draw the wider spool to see an
    outage measured.

    A run covers one sample past the last one it states, taken from the
    step its group reports. Patches within `sampling_group_tolerance` of
    each other share a group and so share that step, which rounds a
    day's total by at most one sample.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.viz.spool import calendar
    >>>
    >>> spool = dc.get_example_spool("sparse_dss")
    >>> _ = calendar(spool)
    >>> _ = calendar(spool, method="gap")
    """
    if method not in _CALENDAR_METHODS:
        msg = (
            f"method={method!r} is not a calendar measure; the options are "
            f"{_CALENDAR_METHODS}."
        )
        raise ParameterError(msg)
    dim = "time"
    report = spool.get_coverage(dim, tolerance=tolerance, group=group)
    gaps = spool.get_gaps(dim, tolerance=tolerance, group=group)
    if len(report) and not pd.api.types.is_datetime64_any_dtype(report[f"{dim}_min"]):
        msg = (
            "A calendar puts data on dates, and this spool states time "
            "relative to something rather than as a date. Draw it with "
            "Spool.viz.coverage, which measures a dimension as it is given."
        )
        raise ParameterError(msg)
    runs = _runs(report, gaps, dim)
    if not len(runs):
        msg = (
            "This spool holds no time to draw a calendar of. A window "
            "which keeps no patches leaves a spool of nothing, which is "
            "not the same as a spool of days covered by nothing: widen "
            "the selection, or draw the spool it was made from."
        )
        raise ParameterError(msg)
    ends = _extended(runs, report, dim)
    days = _calendar_days(runs, ends)
    stamps = days.to_numpy()
    if method == "count":
        values = _day_counts(spool.get_contents(), stamps, dim)
    else:
        covered = _day_seconds(_union(runs["start"].to_numpy(), ends), stamps)
        values = (
            covered / _SECONDS_IN_DAY * 100
            if method == "percent"
            else _SECONDS_IN_DAY - covered
        )
    matrix, labels = _calendar_cells(days, values)
    ax = _new_ax(ax, 1.5 + 0.5 * len(labels))
    ax.set_facecolor(_UNSTATED_COLOR)
    cmap = _get_cmap(_CALENDAR_CMAPS[method]).copy()
    cmap.set_bad(_UNSTATED_COLOR)
    kwargs: dict = {}
    if method == "gap":
        # A missing second is worth seeing beside a missing day, so the
        # scale is logarithmic. It clips rather than leaving zero out of
        # the scale: a day with nothing missing is the end of the range,
        # not a day the calendar knows nothing about.
        kwargs["norm"] = LogNorm(vmin=1.0, vmax=_SECONDS_IN_DAY, clip=True)
    elif method == "percent":
        kwargs["vmin"], kwargs["vmax"] = 0.0, 100.0
    mesh = ax.pcolormesh(np.ma.masked_invalid(matrix), cmap=cmap, **kwargs)
    ax.grid(True, color="silver", lw=0.3)
    _calendar_axes(matrix, labels, ax)
    figure = ax.get_figure()
    assert figure is not None, "an axes always belongs to a figure"
    bar = figure.colorbar(mesh, ax=ax)
    bar.ax.set_ylabel(_CALENDAR_LABELS[method])
    if method == "count":
        # Half a patch overlapped no day.
        bar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if method == "gap":
        bar.set_ticks(list(_GAP_TICKS))
        bar.set_ticklabels([human_duration(x) for x in _GAP_TICKS])
        bar.minorticks_off()
    if show:
        plt.show()
    return ax
