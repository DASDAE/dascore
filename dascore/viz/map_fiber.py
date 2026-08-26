"""Module for waterfall plotting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import patch_function
from dascore.utils.plotting import (
    _get_ax,
    _get_cmap,
    _get_dim_label,
)


def _set_scale(im, scale, scale_type, color_coords):
    """Set the scale of the color bar based on scale and scale_type."""
    # check scale parameters
    if scale_type not in {"absolute", "relative"}:
        msg = f"scale_type must be 'absolute' or 'relative', got {scale_type!r}"
        raise ParameterError(msg)
    if not (isinstance(scale, float | int) or len(scale) == 2):
        msg = "scale must be a number or a length-2 sequence"
        raise ParameterError(msg)
    # make sure we have a len two array
    modifier = 1
    if scale_type == "relative":
        modifier = 0.5 * (np.nanmax(color_coords) - np.nanmin(color_coords))
        # only one scale parameter provided, center around mean
    if isinstance(scale, float):
        mean = np.nanmean(color_coords)
        scale = np.array([mean - scale * modifier, mean + scale * modifier])
    im.set_clim(scale)


def _get_colorbar_label(data_type, data_units):
    """Label the colorbar, leaving out whichever part is unset."""
    name = str(data_type) if data_type else ""
    units = str(data_units) if data_units else ""
    if name and units:
        return f"{name} ({units})"
    return name or units


def _get_data_to_color(patch, x):
    """The patch's own data, checked against the points being drawn."""
    points = np.shape(x)
    data = patch.data
    # An aggregated dimension is left as length one rather than squeezed
    # out, and such a patch does hold one value per channel, so measure the
    # dimensions which actually spread rather than the raw shape.
    spread = [size for size in data.shape if size != 1]
    if len(spread) > 1 or data.size != np.prod(points, dtype=int):
        msg = (
            "map_fiber draws one point per plotted coordinate, so coloring "
            "by data needs one value for each. The patch data has shape "
            f"{data.shape} and the plotted coordinates have shape {points}; "
            "reduce the patch to one value per channel first, for example "
            "with patch.std('time')."
        )
        raise ParameterError(msg)
    return data.reshape(points)


@patch_function()
def map_fiber(
    patch: PatchType,
    x: np.ndarray | str = "distance",
    y: np.ndarray | str = "distance",
    color: np.ndarray | str = "distance",
    ax: plt.Axes | None = None,
    cmap="cividis_r",
    scale: float | Sequence[float] | None = None,
    scale_type: Literal["relative", "absolute"] = "relative",
    show=False,
) -> plt.Axes:
    """
    Create a plot of the outline of the cable colorized by a given parameter.

    Parameters
    ----------
    patch
        The Patch object.
    x
        x coordinate: can be an array or a str representing a patch coordinate.
    y
        y coordinate: can be an array or a str representing a patch coordinate.
    color
        The color parameter to plot: can be an array, the name of a patch
        coordinate, or "data" to color by the patch's own data, which needs
        one value for each point drawn. A coordinate of that name wins.
    ax
        A matplotlib object, if None create one.
    cmap
        A matplotlib colormap string or instance. Set to None to not plot the
        colorbar.
    scale
        If not None, controls the saturation level of the colorbar.
        Values can either be a float, to set upper and lower limit to the same
        value centered around the mean of the data, or a length 2 tuple
        specifying upper and lower limits. See `scale_type` for controlling how
        values are scaled.
    scale_type
        Controls the type of scaling specified by `scale` parameter. Options
        are:
            relative - scale based on half the dynamic range in patch
            absolute - scale based on absolute values provided to `scale`
    show
        If True, show the plot, else just return axis.

    Examples
    --------
    >>> # Plot patch
    >>> import dascore as dc
    >>> patch = dc.get_example_patch("random_patch_with_lat_lon")
    >>> patch = patch.set_units(latitude="m", longitude="m")
    >>> _ = patch.viz.map_fiber("latitude", "longitude", "distance")
    >>>
    >>> # Color by the data itself, reduced to one value per channel.
    >>> reduced = patch.std("time").squeeze()
    >>> _ = reduced.viz.map_fiber("latitude", "longitude", "data")
    """
    dims = []
    if isinstance(x, str):
        if x not in patch.coords:
            msg = f"{x} not found in patch coordinates"
            raise ParameterError(msg)
        dims.append(x)
        x = patch.coords.get_array(x)
    if isinstance(y, str):
        if y not in patch.coords:
            msg = f"{y} not found in patch coordinates"
            raise ParameterError(msg)
        dims.append(y)
        y = patch.coords.get_array(y)
    if isinstance(color, str):
        if color in patch.coords:
            data_type = color
            data_units = patch.coords.coord_map[color].units
            color = patch.coords.get_array(color)
        elif color == "data":
            data_type = patch.attrs.data_type
            data_units = patch.attrs.data_units
            color = _get_data_to_color(patch, x)
        else:
            msg = (
                f"{color} not found in patch coordinates. Use 'data' to "
                "color by the patch's own data."
            )
            raise ParameterError(msg)
    else:
        data_type = ""
        data_units = ""

    ax = _get_ax(ax)
    cmap = _get_cmap(cmap)

    im = ax.scatter(x, y, c=color, cmap=cmap)

    # scale colorbar
    if scale is not None:
        _set_scale(im, scale, scale_type, color)

    # set axis labels
    for dim, x in zip(dims, ["x", "y"]):
        getattr(ax, f"set_{x}label")(_get_dim_label(patch, dim))

    # add color bar with title
    if cmap is not None:
        cb = ax.get_figure().colorbar(im, ax=ax, fraction=0.05, pad=0.025)
        cb.set_label(_get_colorbar_label(data_type, data_units))

    if show:
        plt.show()
    return ax
