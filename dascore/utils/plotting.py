"""
Utilities for plotting with matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates

import dascore as dc
from dascore.utils.time import to_number


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
        denom = 1_000_000_000
        time_min = to_number(lims["time"][0]) / denom
        time_max = to_number(lims["time"][1]) / denom
        lims["time"] = [time_min, time_max]
    out = [x for dim in dims_r for x in lims[dim]]
    return out


def _format_function(x, pos=None):
    x = mdates.num2date(x)
    if pos == 0:
        fmt = "%D %H:%M:%S.%f"
    else:
        fmt = "%H:%M:%S.%f"
    label = x.strftime(fmt)
    label = label.rstrip("0")
    label = label.rstrip(".")
    return label


def _strip_labels(labels):
    """Strip all the trailing zeros from labels."""
    ar = np.array([list(x) for x in labels])
    redundant = ar == "0"
    order = np.arange(ar.shape[1])
    order[~redundant.all(axis=0)] = order.max()
    ind2keep = order.min()
    return labels.str[:ind2keep].str.rstrip(".").str.rstrip(":").str.rstrip("T")


def _get_labels(ticks, time_fmt, dt):
    """Get sensible labels for plots."""
    dt_ns = np.round(dt * 1_000_000_000).astype(int)
    ticks_ns = dc.to_datetime64(ticks).astype(np.int64)
    out = np.round(ticks_ns / dt_ns).astype(int) * dt_ns
    labels = pd.to_datetime(out.astype("datetime64[ns]")).strftime(time_fmt)
    return labels


def _format_time_axis(ax, dims, time_fmt, time):
    """Format the time axis."""
    if not time_fmt:
        time_fmt = _get_format_string(time)

    approx_dt = ((time[1:] - time[:-1]) / dc.to_timedelta64(1)).mean()
    dt = np.round(approx_dt, 6)
    # determine which axis is x, tell mpl it's a date, get date axis
    axis_name = "x" if dims[0] == "time" else "y"
    # getattr(ax, f"{axis_name}axis_date")()
    ticks = getattr(ax, f"get_{axis_name}ticks")()
    labels = _get_labels(ticks, time_fmt, dt)
    trimmed_labels = _strip_labels(labels)
    getattr(ax, f"set_{axis_name}ticks")(ticks, trimmed_labels)


def _add_time_axis_label(ax, patch, dims_r):
    """Add a time axis label to plot."""
    # set time label with start/end time of trace
    start, end = patch.attrs["time_min"], patch.attrs["time_max"]
    if pd.isnull(start) or pd.isnull(end):
        return  # nothing to do if no start/end times in attrs
    x_or_y = ["x", "y"][dims_r.index("time")]
    time_label = f"Time (start: {start})"
    getattr(ax, f"set_{x_or_y}label")(time_label)
