"""
Utilities for plotting with matplotlib.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import dates as mdates


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


def _get_extents(dims_r, coords):
    """Get the extents used for each dimension."""
    # need to reverse dims since extent is [left, right, bottom, top]
    # and we want first dim to go from top to bottom
    lims = {x: [] for x in dims_r}
    for dim in dims_r:
        array = coords[dim]
        lims[dim] += [array.min(), array.max()]
    # special handling for time
    if "time" in coords:  # convert dates to mdates
        time_min = pd.Timestamp(lims["time"][0]).to_pydatetime()
        time_max = pd.Timestamp(lims["time"][1]).to_pydatetime()
        x_lims = [mdates.date2num(x) for x in [time_min, time_max]]
        lims["time"] = x_lims
    out = [x for dim in dims_r for x in lims[dim]]
    return out


def _format_time_axis(ax, dims, time_fmt):
    """Format the time axis."""
    # determine which axis is x, tell mpl it's a date, get date axis
    axis_name = "x" if dims[0] == "time" else "y"
    getattr(ax, f"{axis_name}axis_date")()
    date_ax = getattr(ax, f"{axis_name}axis")
    # apply specified format
    date_format = mdates.DateFormatter(time_fmt)
    date_ax.set_major_formatter(date_format)
    # Have MPL try to make the date formats fit better.
    fig = ax.get_figure()
    getattr(fig, f"autofmt_{axis_name}date", lambda: None)()
    getattr(ax, f"set_{axis_name}label")("time")


def _add_time_axis_label(ax, patch, dims_r):
    """Add a time axis label to plot."""
    # set time label with start/end time of trace
    start, end = patch.attrs["time_min"], patch.attrs["time_max"]
    if pd.isnull(start) or pd.isnull(end):
        return  # nothing to do if no start/end times in attrs
    x_or_y = ["x", "y"][dims_r.index("time")]
    start_str = str(start).split("T")[0]
    time_label = f"Time (start: {start_str})"
    getattr(ax, f"set_{x_or_y}label")(time_label)
