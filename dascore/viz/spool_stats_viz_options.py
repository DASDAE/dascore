"""Utilities for visualizing DASCore spool timing statistics.

The main entry point is :func:`viz_spool`.  It accepts either a DASCore spool
object or a pandas DataFrame with ``time_min`` and ``time_max`` columns.

Examples
--------
Plot everything::

    viz_spool(spool)

Plot only one panel::

    viz_spool(spool, plots="gap")

Plot selected panels::

    viz_spool(spool, plots=["gap", "gap_hist"])

Use a predefined layout::

    viz_spool(spool, layout="histograms")
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from rich import print as rich_print

OutlierMethod = Literal["mode", "mean", "median"]
PlotName = Literal["gap", "duration", "gap_hist", "duration_hist"]
LayoutName = Literal["all", "timeseries", "histograms", "summary"]

_VALID_METHODS: tuple[str, ...] = ("mode", "mean", "median")
_VALID_PLOTS: tuple[str, ...] = ("gap", "duration", "gap_hist", "duration_hist")
_LAYOUTS: dict[str, tuple[str, ...]] = {
    "all": _VALID_PLOTS,
    "timeseries": ("gap", "duration"),
    "histograms": ("gap_hist", "duration_hist"),
    "summary": ("gap", "duration", "gap_hist", "duration_hist"),
}

_TICK_MAP = [
    (0.000001, "1 us"),
    (0.00001, "10 us"),
    (0.0001, "100 us"),
    (0.001, "1 ms"),
    (0.01, "10 ms"),
    (0.1, "100 ms"),
    (1, "1 sec"),
    (10, "10 sec"),
    (60, "1 min"),
    (10 * 60, "10 min"),
    (3600, "1 hour"),
    (6 * 3600, "6 hours"),
    (86400, "1 day"),
    (7 * 86400, "1 week"),
    (30 * 86400, "1 month"),
]
_TICKS = np.asarray([x[0] for x in _TICK_MAP])
_TICK_LABELS = np.asarray([x[1] for x in _TICK_MAP])


@dataclass(frozen=True)
class SpoolTimingStats:
    """Computed arrays used by the spool timing plots."""

    filestart: np.ndarray
    fileend: np.ndarray
    duration: np.ndarray
    gap: np.ndarray
    outlier_index: np.ndarray


def _as_dataframe(spool) -> pd.DataFrame:
    """Return spool contents as a DataFrame."""
    if hasattr(spool, "get_contents"):
        df = spool.get_contents()
    elif isinstance(spool, pd.DataFrame):
        df = spool
    else:
        raise TypeError(
            f"Expected a DASCore spool or DataFrame, got {type(spool).__name__}."
        )

    required = {"time_min", "time_max"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}.")

    if len(df) < 2:
        raise ValueError("At least two files/rows are required to compute gaps.")

    return df.sort_values("time_min")


def _find_outliers(
    gap: np.ndarray,
    duration: np.ndarray,
    method: OutlierMethod = "median",
    tolerance_percent: float = 20,
) -> np.ndarray:
    """Identify large file gaps relative to a duration reference value."""
    method = method.lower()
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown outlier detection method {method!r}. "
            f"Allowed values are {list(_VALID_METHODS)}."
        )

    if method == "mode":
        pass  # TODO: implement mode-based outlier detection.  See
        # reference_value = stats.mode(duration, keepdims=False).mode
    elif method == "mean":
        reference_value = np.mean(duration)
    else:  # method == "median"
        reference_value = np.median(duration)

    threshold = reference_value * (1 + tolerance_percent / 100)
    return np.where(gap > threshold)[0]


def _compute_timing_stats(
    df: pd.DataFrame,
    method: OutlierMethod,
    tolerance_percent: float,
) -> SpoolTimingStats:
    """Compute file starts, file ends, durations, gaps, and gap outliers."""
    filestart = df["time_min"].to_numpy()
    fileend = df["time_max"].to_numpy()

    duration = (fileend - filestart) / np.timedelta64(1, "s")
    gap = (filestart[1:] - fileend[:-1]) / np.timedelta64(1, "s")
    outlier_index = _find_outliers(gap, duration, method, tolerance_percent)

    return SpoolTimingStats(
        filestart=filestart,
        fileend=fileend,
        duration=duration,
        gap=gap,
        outlier_index=outlier_index,
    )


def _normalize_plots(
    plots: str | Iterable[str] | None,
    layout: LayoutName | None,
) -> tuple[str, ...]:
    """Normalize user-provided plot/layout options into a tuple of plot names."""
    if plots is None:
        layout = "all" if layout is None else layout.lower()
        if layout not in _LAYOUTS:
            raise ValueError(
                f"Unknown layout {layout!r}. Allowed values are {sorted(_LAYOUTS)}."
            )
        requested = _LAYOUTS[layout]
    elif isinstance(plots, str):
        requested = (plots,)
    else:
        requested = tuple(plots)

    unknown = sorted(set(requested) - set(_VALID_PLOTS))
    if unknown:
        raise ValueError(
            f"Unknown plot(s): {unknown}. Allowed values are {list(_VALID_PLOTS)}."
        )

    if not requested:
        raise ValueError("At least one plot must be requested.")

    # Preserve order while removing duplicates.
    return tuple(dict.fromkeys(requested))


def _positive_limits(
    values: np.ndarray, *, lower_decades: int = 0
) -> tuple[float, float]:
    """Return log-scale limits for positive values."""
    positive = np.asarray(values)[np.asarray(values) > 0]
    if len(positive) == 0:
        raise ValueError(
            "Cannot make a log-scale plot because no positive values exist."
        )

    minlim = 10 ** (np.floor(np.log10(positive.min())) - lower_decades)
    maxlim = 10 ** (np.ceil(np.log10(positive.max())) + 1)
    return float(minlim), float(maxlim)


def _set_time_ticks(ax, minlim: float, maxlim: float) -> None:
    """Apply human-readable time ticks to a log-scale axis."""
    use = (minlim <= _TICKS) & (_TICKS <= maxlim)
    ax.set_xticks(_TICKS[use], labels=_TICK_LABELS[use])
    ax.tick_params(axis="x", labelrotation=90)


def _plot_gap(ax, stats_: SpoolTimingStats, annotate_gaps: bool = True) -> None:
    """Plot file gaps through time."""
    ax.semilogy(stats_.filestart[:-1], stats_.gap, ".")

    if len(stats_.outlier_index):
        ax.semilogy(
            stats_.filestart[stats_.outlier_index],
            stats_.gap[stats_.outlier_index],
            "r.",
        )

    if 0 < len(stats_.outlier_index) < 50 and annotate_gaps:
        for i in stats_.outlier_index:
            text = "  " + str(stats_.filestart[i])[:19]
            ax.text(
                stats_.filestart[i],
                stats_.gap[i],
                text,
                rotation=45,
                ha="left",
                va="center",
                fontsize=8,
                rotation_mode="anchor",
            )

    positive_gap = stats_.gap[stats_.gap > 0]
    if len(positive_gap):
        minlim, maxlim = _positive_limits(positive_gap)
        use = (minlim <= _TICKS) & (_TICKS <= maxlim)
        ax.set_yticks(_TICKS[use], labels=_TICK_LABELS[use])

    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(True)
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel("Gap between files")
    ax.set_title("File gaps")


def _plot_duration(ax, stats_: SpoolTimingStats, annotate_gaps: bool = True) -> None:
    """Plot file durations through time."""
    del annotate_gaps  # Keep a shared helper signature.

    ax.semilogy(stats_.filestart, stats_.duration, ".")
    minlim, maxlim = _positive_limits(stats_.duration)
    use = (minlim <= _TICKS) & (_TICKS <= maxlim)
    ax.set_yticks(_TICKS[use], labels=_TICK_LABELS[use])

    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(True)
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel("File duration")
    ax.set_title("File durations")


def _plot_gap_hist(ax, stats_: SpoolTimingStats, annotate_gaps: bool = True) -> None:
    """Plot a histogram of positive file gaps."""
    del annotate_gaps

    positive_gap = stats_.gap[stats_.gap > 0]
    minlim, maxlim = _positive_limits(positive_gap, lower_decades=1)
    bins = np.geomspace(minlim, maxlim, num=31)

    ax.hist(positive_gap, bins=bins, edgecolor="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _set_time_ticks(ax, minlim, maxlim)
    ax.set_xlim((minlim, maxlim))
    ax.set_xlabel("Gap between files")
    ax.set_ylabel("Number of files")
    ax.set_title("Gap distribution")


def _plot_duration_hist(
    ax, stats_: SpoolTimingStats, annotate_gaps: bool = True
) -> None:
    """Plot a histogram of file durations."""
    del annotate_gaps

    minlim, maxlim = _positive_limits(stats_.duration)
    bins = np.geomspace(minlim, maxlim, num=31)

    ax.hist(stats_.duration, bins=bins, edgecolor="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _set_time_ticks(ax, minlim, maxlim)
    ax.set_xlim((minlim, maxlim))
    ax.set_xlabel("File duration")
    ax.set_ylabel("Number of files")
    ax.set_title("Duration distribution")


_PLOTTERS = {
    "gap": _plot_gap,
    "duration": _plot_duration,
    "gap_hist": _plot_gap_hist,
    "duration_hist": _plot_duration_hist,
}


def _make_axes(plot_names: tuple[str, ...], figsize: tuple[float, float] | None = None):
    """Create one axis per requested plot and return ``(fig, axs)``."""
    n_plots = len(plot_names)
    if figsize is None:
        figsize = (10, max(3.5 * n_plots, 4))

    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        layout="constrained",
        figsize=figsize,
        squeeze=False,
    )
    axs = {name: ax for name, ax in zip(plot_names, axes.ravel())}
    return fig, axs


def viz_spool(
    spool,
    *,
    plots: str | Iterable[PlotName] | None = None,
    layout: LayoutName | None = None,
    method: OutlierMethod = "median",
    tolerance_percent: float = 20,
    annotate_gaps: bool = True,
    figsize: tuple[float, float] | None = None,
):
    """Visualize spool timing statistics.

    Parameters
    ----------
    spool
        A DASCore spool or pandas DataFrame containing spool contents. The
        contents must include ``time_min`` and ``time_max`` columns.
    plots
        One plot name or an iterable of plot names. Valid values are
        ``"gap"``, ``"duration"``, ``"gap_hist"``, and ``"duration_hist"``.
        If omitted, ``layout`` decides which plots are shown.
    layout
        Named plot selection used when ``plots`` is omitted. Valid values are
        ``"all"``, ``"timeseries"``, ``"histograms"``, and ``"summary"``.
        ``"summary"`` is currently equivalent to ``"all"``.
    method
        Outlier detection method: ``"mode"``, ``"mean"``, or ``"median"``.
    tolerance_percent
        Percentage deviation from the reference duration used to classify large
        gaps as outliers.
    annotate_gaps
        If true, annotate outlier gaps when fewer than 50 outliers are found.
    figsize
        Optional matplotlib figure size. If omitted, the size is based on the
        number of requested plots.

    Returns
    -------
    axs : dict[str, matplotlib.axes.Axes]
        Mapping from plot name to matplotlib axis.
    duration : numpy.ndarray
        File durations in seconds.
    gap : numpy.ndarray
        Gaps between consecutive files in seconds.
    outlier_index : numpy.ndarray
        Indices of detected gap outliers.
    """
    method = method.lower()
    df = _as_dataframe(spool)
    stats_ = _compute_timing_stats(df, method, tolerance_percent)
    plot_names = _normalize_plots(plots, layout)

    _, axs = _make_axes(plot_names, figsize=figsize)
    for name in plot_names:
        _PLOTTERS[name](axs[name], stats_, annotate_gaps=annotate_gaps)
        if name == "gap":
            rich_print(
                "Data gaps found at:\n   index\tfirst data after gap at\t\tgap length"
            )
            for i in stats_.outlier_index:
                thisgap = datetime.timedelta(seconds=stats_.gap[i])
                rich_print(f"{i:8d}\t{stats_.filestart[i]}\t{thisgap}")

    return axs, stats_.duration, stats_.gap, stats_.outlier_index


if __name__ == "__main__":
    # Example usage:

    from dascore.utils.hdf5 import HDFPatchIndexManager

    tmpfile = Path(
        r"C:\\Users\\andreasw\\OneDrive - NORSAR\\Fiber_Group"
        + r"\\FYBR_PROJECTS\\FibreEyes_NFR\\spools\\_dascore_index_Aurland.hdf5"
    )
    tmpfile = Path(
        r"C:\\Users\\andreasw\\OneDrive - NORSAR\\Fiber_Group"
        + r"\\FYBR_PROJECTS\\FibreEyes_NFR\\spools\\_dascore_index_Hoyanger.hdf5"
    )
    df = HDFPatchIndexManager(tmpfile).get_index()
    viz_spool(df, plots=["gap", "duration"])
    pass
