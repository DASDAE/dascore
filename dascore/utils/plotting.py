"""Utilities for plotting with matplotlib."""

from __future__ import annotations

import string

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dascore.exceptions import ParameterError
from dascore.units import Hz, get_quantity_str, maybe_convert_percent_to_fraction
from dascore.units import s as seconds
from dascore.utils.misc import suppress_warnings, tukey_fence
from dascore.utils.time import dtype_time_like


def _get_dim_label(patch, dim):
    """Create a label for the given dimension, including units if defined."""
    maybe_units = patch.get_coord(dim).units if dim in patch.coords else None
    if maybe_units == 1 / seconds:
        maybe_units = Hz
    return _get_label(string.capwords(str(dim)), maybe_units)


def _get_label(name, units, default=""):
    """
    Label a quantity as "name [units]", leaving out whichever part is unset.

    Returns default if neither is set.
    """
    name = str(name) if name else ""
    units = get_quantity_str(units) or ""
    if name and units:
        return f"{name} [{units}]"
    return name or units or default


def _get_data_label(patch, default=""):
    """
    Create a label for the patch data (its type and units).

    Returns default if the patch has neither a data_type nor data_units.
    """
    data_type = patch.attrs.get("data_type", "")
    return _get_label(data_type, patch.attrs.data_units, default)


def _validate_scale_type(scale_type):
    """Validate that scale_type is either 'relative' or 'absolute'."""
    valid_types = {"relative", "absolute"}
    if scale_type not in valid_types:
        msg = f"scale_type must be one of {valid_types}, but got '{scale_type}'"
        raise ParameterError(msg)


def _get_scale(scale, scale_type, data):
    """
    Calculate the colorbar limits based on scale and scale_type.
    """
    _validate_scale_type(scale_type)
    # Whatever form scale was given in, this makes it a list.
    scale = maybe_convert_percent_to_fraction(scale)
    match (scale, scale_type):
        # Case 1: Single value with relative scaling
        # Scale is symmetric around the mean, using fraction of dynamic range
        case (scale, "relative") if len(scale) == 1:
            scale = scale[0]
            # Zero gives one limit twice, and less than zero puts them in the
            # wrong order.
            if scale <= 0:
                msg = f"Relative scale value of {scale} must be greater than 0"
                raise ParameterError(msg)
            mod = 0.5 * (np.nanmax(data) - np.nanmin(data))
            if mod == 0:
                # Constant data, use small epsilon to avoid degenerate limits
                mod = 1e-10
            mean = np.nanmean(data)
            scale = np.asarray([mean - scale * mod, mean + scale * mod])
        # Case 2: No scale specified with relative scaling
        # Use Tukey's fence (C*IQR, C is normally 1.5) to exclude outliers.
        # This prevents a few extreme values from obscuring the majority of the
        # data at the cost of a slight performance penalty.
        case ([], "relative"):
            return tukey_fence(data)
        # Case 3: Sequence with relative scaling
        # Scale values represent fractions of the data range [0, 1]
        # and are mapped to [data_min, data_max]
        case (scale, "relative"):
            scale = np.array(scale)
            # Validate scale parameters
            if len(scale) != 2:
                msg = (
                    "Relative scale must be a number or a length-2 sequence, "
                    f"got {scale}"
                )
                raise ParameterError(msg)
            if np.any(scale < 0) or scale[0] > scale[1]:
                msg = (
                    "Relative scale values cannot be negative and the first "
                    f"value must be less than the second. You passed {scale}"
                )
                raise ParameterError(msg)
            dmin, dmax = np.nanmin(data), np.nanmax(data)
            data_range = dmax - dmin
            # Map [0, 1] to [data_min, data_max]
            scale = dmin + scale * data_range
        # Case 4: Absolute scaling
        case (scale, "absolute") if len(scale) == 1:
            scale = np.array([-abs(scale[0]), abs(scale[0])])
        # Case 5: Absolute scaling with a pair is used as the limits as is;
        # anything longer cannot be.
        case (scale, "absolute") if len(scale) > 2:
            msg = f"scale must be a number or a length-2 sequence, got {scale}"
            raise ParameterError(msg)

    # Scale values are used directly as colorbar limits
    return scale


def _add_colorbar(ax, im, data, label, scale=None):
    """
    Add a colorbar with the given label to the plot.

    When the limits leave data above or below them, extend triangles say so.
    Only a finite pair of limits is ever applied, so only that can clip.
    """
    above, below = False, False
    if scale is not None and len(scale) == 2 and np.all(np.isfinite(scale)):
        mi, mx = np.nanmin(data), np.nanmax(data)
        above = (mx > scale[1]) and not np.isclose(scale[1], mx)
        below = (mi < scale[0]) and not np.isclose(scale[0], mi)
    extend_map = {
        (True, True): "both",
        (True, False): "max",
        (False, True): "min",
    }
    extend = extend_map.get((above, below), "neither")
    cb = ax.get_figure().colorbar(
        im, ax=ax, fraction=0.05, pad=0.025, extend=extend, extendfrac=0.025
    )
    cb.set_label(label)


def _get_cmap(cmap):
    """Return a color map from a colormap or string."""
    if isinstance(cmap, str):  # get color map if a string was passed
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_over(cmap(1.0))
        cmap.set_under(cmap(0.0))
    return cmap


def _get_ax(ax):
    """Get an axis if ax is None."""
    if ax is None:
        _, ax = plt.subplots(1)
    return ax


def _maybe_invert_yaxis(ax, patch, dim, ascending=True):
    """
    Orient the y axis so the dimension it displays runs the standard way.

    Plots which put a patch dimension on the y axis share this rule, so a
    dimension is oriented the same way whichever plot draws it. Seismic
    displays put time on a downward axis (shot gathers, record sections),
    so a time-like dimension runs downward and any other runs upward.
    Distance in particular must not run downward: a wiggle plot draws its
    traces as offsets along the y axis, so flipping it would point positive
    amplitudes down as well, and distance carries no convention which asks
    for that.

    Parameters
    ----------
    ax
        The axis whose y axis may be inverted.
    patch
        The patch being plotted.
    dim
        The name of the dimension displayed on the y axis.
    ascending
        Whether the axis already places the dimension's values in ascending
        order. False flips the decision, for a plot whose y positions follow
        the array rather than the coordinate (wiggle offsets) and whose
        coordinate is reverse sorted.
    """
    invert = dtype_time_like(patch.get_coord(dim).dtype) != (not ascending)
    # invert_yaxis toggles, so check first; a caller can pass an axis which
    # is already time-down, and drawing on it must not flip it back.
    if invert and not ax.yaxis_inverted():
        ax.invert_yaxis()


def _cell_edge_limits(low, high, size):
    """
    Widen sample-centre limits to the outer edges of the cells.

    An image extent gives the outer edges of the image, so handing it the
    first and last coordinate values would draw every sample half a cell
    from where it belongs and squeeze the image into one cell less than it
    has. The mesh path already draws cells around their centres, and this
    is what keeps the two renderers agreeing.
    """
    if size < 2:
        return [low, high]
    half_cell = (high - low) / (2 * (size - 1))
    return [low - half_cell, high + half_cell]


def _get_extents(dims_r, coords):
    """Get the extents used for each dimension."""

    def _convert_datetimes(coords, lims):
        """Convert numpy datetimes to matplotlib style datetimes."""
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
        array = coords.get_array(dim) if hasattr(coords, "get_array") else coords[dim]
        # Use nanmin/nanmax to handle NaN/NaT values in coordinates
        with suppress_warnings(RuntimeWarning):
            array_min = np.nanmin(array)
            array_max = np.nanmax(array)
        # If all values are NaN/NaT, fall back to index-based extents
        if np.isnan(array_min) or np.isnan(array_max):
            array_min = 0
            array_max = len(array) - 1
        lims[dim] += _cell_edge_limits(array_min, array_max, len(array))
    # find datetime coords and convert to numpy mtimes
    _convert_datetimes(coords, lims)
    _convert_timedeltas(coords, lims)
    out = [x for dim in dims_r for x in lims[dim]]
    return out


def _format_time_axis(ax, dim, axis_name):
    """
    Function to handle formatting time axis for image-type plots.

    Tries to snap all axis labels to "nice" values and adds reference
    start time.
    """
    # Set label to not include units
    dim_str = string.capwords(str(dim))
    getattr(ax, f"set_{axis_name}label")(dim_str)
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
