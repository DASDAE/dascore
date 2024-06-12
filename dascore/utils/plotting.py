"""Utilities for plotting with matplotlib."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dascore.units import get_quantity_str
from dascore.utils.misc import suppress_warnings


def _get_dim_label(patch, dim):
    """Create a label for the given dimension, including units if defined."""
    attrs = patch.attrs
    maybe_units = attrs.get(f"{dim}_units")
    unit_str = f"({get_quantity_str(maybe_units)})" if maybe_units else ""
    return str(dim) + unit_str


def _get_cmap(cmap):
    """Return a color map from a colormap or string."""
    if isinstance(cmap, str):  # get color map if a string was passed
        cmap = plt.get_cmap(cmap)
    return cmap


def _get_ax(ax):
    """Get an axis if ax is None."""
    if ax is None:
        _, ax = plt.subplots(1)
    return ax


def _get_extents(dims_r, coords):
    """Get the extents used for each dimension."""

    def _convert_datetimes(coords, lims):
        """Convert numpy datetimes to matplotlib style datettimes."""
        time_dims = [
            i for i, v in coords.items() if np.issubdtype(v.dtype, np.datetime64)
        ]
        for name in time_dims:
            # We can get a warning about loss of precision in ns, doesn't matter.
            with suppress_warnings(UserWarning):
                time_min = pd.to_datetime(lims[name][0]).to_pydatetime()
                time_max = pd.to_datetime(lims[name][1]).to_pydatetime()
            # convert to julian date to appease matplotlib
            lims[name] = [mdates.date2num(time_min), mdates.date2num(time_max)]

    def _convert_timedeltas(coords, lims):
        timedelta_dims = [
            i for i, v in coords.items() if np.issubdtype(v.dtype, np.timedelta64)
        ]
        for name in timedelta_dims:
            # We can get a warning about loss of precision in ns, doesn't matter.\
            low, high = lims[name]
            onesec = np.timedelta64(1, "s")
            # convert to julian date to appease matplotlib
            lims[name] = [low / onesec, high / onesec]

    # need to reverse dims since extent is [left, right, bottom, top]
    # and we want first dim to go from top to bottom
    lims = {x: [] for x in dims_r}
    for dim in dims_r:
        array = coords[dim]
        lims[dim] += [array.min(), array.max()]
    # find datetime coords and convert to numpy mtimes
    _convert_datetimes(coords, lims)
    _convert_timedeltas(coords, lims)
    out = [x for dim in dims_r for x in lims[dim]]
    return out


def _format_time_axis(ax, dim, axis_name):
    """
    Function to handle formatting time axis for image-type plots.

    Tries to snape all axis labels to "nice" values and adds reference
    start time.
    """
    # Set label to not include units
    getattr(ax, f"set_{axis_name}label")(dim)
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
