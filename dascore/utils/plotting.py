"""
Utilities for plotting with matplotlib.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from dascore.utils.misc import suppress_warnings


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


def _get_extents(dims_r, coords):
    """Get the extents used for each dimension."""
    # need to reverse dims since extent is [left, right, bottom, top]
    # and we want first dim to go from top to bottom
    lims = {x: [] for x in dims_r}
    for dim in dims_r:
        array = coords[dim]
        lims[dim] += [array.min(), array.max()]
    # special handling for time
    if "time" in coords:
        # We can get a warning about loss of precision in ns, doesn't matter.
        with suppress_warnings(UserWarning):
            time_min = pd.to_datetime(lims["time"][0]).to_pydatetime()
            time_max = pd.to_datetime(lims["time"][1]).to_pydatetime()
        # convert to julian date to appease matplotlib
        lims["time"] = [mdates.date2num(time_min), mdates.date2num(time_max)]
    out = [x for dim in dims_r for x in lims[dim]]
    return out


def _format_time_axis(ax, dims_r):
    """
    Function to handle formatting time axis for image-type plots.

    Tries to snape all axis labels to "nice" values and adds reference
    start time.
    """
    axis_name = "x" if dims_r[0] == "time" else "y"
    # set date time formatting so MPL knows this axis is a date
    getattr(ax, f"{axis_name}axis_date")()
    # Set intelligent, zoom-in-able date formatter
    locator = getattr(ax, f"{axis_name}axis").get_major_locator()
    off_formats = ["", "%Y", "%Y-%m", "%Y-%m-%d", "%Y-%m-%d", "%Y-%m-%dT%H:%M"]
    date_format = mdates.ConciseDateFormatter(locator, offset_formats=off_formats)
    getattr(ax, f"{axis_name}axis").set_major_formatter(date_format)
    # Set a custom function for when mouse hovers to display full precision
    # see https://stackoverflow.com/a/32824933/3645626
    format_name = f"format_{axis_name}data"
    setattr(ax, format_name, lambda d: str(mdates.num2date(d)).split("+")[0][:-3])
