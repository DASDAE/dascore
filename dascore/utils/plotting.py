"""
Utilities for plotting with matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import ONE_BILLION
from dascore.utils.time import to_number


def _get_dim_label(patch, dim):
    """Create a label for the given dimension, including units if defined."""
    attrs = patch.attrs
    maybe_units = attrs.get(f"{dim}_units")
    unit_str = f"({maybe_units})" if maybe_units else ""
    return str(dim).capitalize() + unit_str


def _get_cmap(cmap):
    """Return a color map from a colormap or string."""
    if isinstance(cmap, str):  # get color map if a string was passed
        cmap = plt.get_cmap(cmap)
    return cmap


def _get_ax(ax):
    """Get an axis if ax is None"""
    if ax is None:
        _, ax = plt.subplots(1)
    return ax


def _get_format_string(time_array):
    """Determine the best format string."""
    default_formatter = ("%Y-", "%m-", "%dT", "%H:", "%M:", "%S.%f")
    td = time_array.max() - time_array.min()
    dt = dc.to_datetime64(td / dc.to_timedelta64(1))
    dt_str = str(dt).replace(":", "-").replace("T", "-").split("-")
    base = dc.to_datetime64(0)
    base_str = str(base).replace(":", "-").replace("T", "-").split("-")
    diffs = np.array([x != y for x, y in zip(dt_str, base_str)])
    ind_min = np.argmax(diffs)
    fmter = "".join(default_formatter[ind_min:])
    return fmter


def _get_extents(dims_r, coords):
    """Get the extents used for each dimension."""
    # need to reverse dims since extent is [left, right, bottom, top]
    # and we want first dim to go from top to bottom
    lims = {x: [] for x in dims_r}
    for dim in dims_r:
        array = coords[dim]
        lims[dim] += [array.min(), array.max()]
    # special handling for time
    if "time" in coords:  # convert dates to float in seconds
        denom = ONE_BILLION
        time_min = to_number(lims["time"][0]) / denom
        time_max = to_number(lims["time"][1]) / denom
        lims["time"] = [time_min, time_max]
    out = [x for dim in dims_r for x in lims[dim]]
    return out


def _strip_labels(labels, redundants=("0", ":", "T", ".")):
    """Strip all the trailing zeros from labels."""
    ar = np.array([list(x) for x in labels])
    redundant = (np.isin(ar, redundants)).all(axis=0)
    ind2keep = np.argmax(np.cumsum((~redundant).astype(int))) + 1
    return labels.str[:ind2keep].str.rstrip(".").str.rstrip(":").str.rstrip("T")


def _get_labels(ticks, time_fmt, dt_ns):
    """Get sensible labels for plots."""
    ticks_ns = dc.to_datetime64(ticks).astype(np.int64)
    out = np.round(ticks_ns / dt_ns).astype(int) * dt_ns
    labels = pd.to_datetime(out.astype("datetime64[ns]")).strftime(time_fmt)
    return labels


def _format_time_axis(ax, patch, dims_r, time_fmt, extents=None):
    """
    Function to handle formatting time axis for image-type plots.

    Tries to snape all axis labels to "nice" values and adds reference
    start time.
    """
    time = patch.coords["time"]
    if not time_fmt:
        time_fmt = _get_format_string(time)
    approx_dt = ((time[1:] - time[:-1]) / dc.to_timedelta64(1)).mean()
    dt_ns = np.round(approx_dt * ONE_BILLION, 6).astype(int)
    # determine which axis is x, tell mpl it's a date, get date axis
    axis_name = "x" if dims_r[0] == "time" else "y"
    # getattr(ax, f"{axis_name}axis_date")()
    ticks = getattr(ax, f"get_{axis_name}ticks")()
    labels = _get_labels(ticks, time_fmt, dt_ns)
    trimmed_labels = _strip_labels(labels)
    getattr(ax, f"set_{axis_name}ticks")(ticks, trimmed_labels, rotation=-20)
    # next add time label
    _add_time_axis_label(ax, patch, dims_r, dt_ns)
    # reset limits as these can get messed up when using custom labels.
    if extents is not None:
        ax.set_xlim(extents[0], extents[1])
        ax.set_ylim(extents[2], extents[3])


def _add_time_axis_label(ax, patch, dims_r, dt_ns):
    """Add a time axis label to plot."""
    # set time label with start/end time of trace
    start, end = patch.attrs["time_min"], patch.attrs["time_max"]
    if pd.isnull(start) or pd.isnull(end):
        return  # nothing to do if no start/end times in attrs
    # round start label to within 1 dt.
    start_ns = start.astype(int)
    new_start_ns = np.round(start_ns / dt_ns).astype(int) * dt_ns
    ser = pd.Series(str(new_start_ns.astype("datetime64[ns]")))
    new_start = _strip_labels(ser).iloc[0]
    x_or_y = ["x", "y"][dims_r.index("time")]
    time_label = f"Time\n (start: {new_start})"
    getattr(ax, f"set_{x_or_y}label")(time_label)
