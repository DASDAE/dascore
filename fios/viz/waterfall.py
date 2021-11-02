"""
Module for waterfall plotting.
"""
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import dates as mdates

import fios


def _get_extents(dims, coords):
    """Get the extents used for each dimension."""
    # special handling for time
    lims = {x: [] for x in dims}
    for dim in dims:
        array = coords[dim]
        lims[dim] += [array.min(), array.max()]
    if "time" in coords:  # convert dates to mdates
        time_min = pd.Timestamp(lims["time"][0]).to_pydatetime()
        time_max = pd.Timestamp(lims["time"][1]).to_pydatetime()
        x_lims = [mdates.date2num(x) for x in [time_min, time_max]]
        lims["time"] = x_lims
    out = [x for dim in dims for x in lims[dim]]
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


def _add_time_axis_label(ax, patch):
    """Add a time axis label to plot."""
    # set time label with start/end time of trace
    start, end = patch.attrs["time_min"], patch.attrs["time_max"]
    if pd.isnull(start) or pd.isnull(end):
        return  # nothing to do if no start/end times in attrs
    x_or_y = ["x", "y"][patch.dims.index("time")]
    start_str = str(start).split("T")[0]
    end_str = str(end).split("T")[0]
    time_label = f"Time ({start_str} to {end_str})"
    getattr(ax, f"set_{x_or_y}label")(time_label)


def waterfall(
    patch: "fios.Patch",
    ax: Optional[plt.Axes] = None,
    cmap="bwr",
    timefmt="%H:%M:%S",
    show=False,
) -> plt.Figure:
    """
    Parameters
    ----------
    patch
        The Patch object.
    ax
        A matplotlib object, if None create one.
    cmap
        A matplotlib colormap string or instance.
    timefmt
        The format for the time axis.
    show
        If True, show the plot, else just return axis.
    """
    if ax is None:
        _, ax = plt.subplots(1)
    if isinstance(cmap, str):  # get color map if a string was passed
        cmap = plt.get_cmap(cmap)
    data = patch.data
    dims = patch.dims
    coords = {dim: patch.coords[dim] for dim in dims}
    extents = _get_extents(dims, coords)
    im = ax.imshow(data, extent=extents, aspect="auto", cmap=cmap, origin="lower")
    for dim, x in zip(dims, ["x", "y"]):
        getattr(ax, f"set_{x}label")(dim)
    if "time" in dims:
        _format_time_axis(ax, dims, timefmt)
        _add_time_axis_label(ax, patch)
    # add color bar
    ax.get_figure().colorbar(im)
    ax.invert_yaxis()  # invert y axis so origin is at top
    if show:
        plt.show()
    return ax
