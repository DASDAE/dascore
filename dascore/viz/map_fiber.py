"""Module for drawing a fiber where it lies, colored by a per-channel value."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import patch_function
from dascore.utils.plotting import (
    _add_colorbar,
    _get_ax,
    _get_cmap,
    _get_data_label,
    _get_dim_label,
    _get_label,
    _get_scale,
)


def _get_position(patch, position):
    """
    The values drawn along one axis, and the coordinate they came from.

    An array is drawn as given, masked points and all, and belongs to no
    coordinate, so its axis gets no label.
    """
    if not isinstance(position, str):
        return np.asanyarray(position), None
    if position not in patch.coords:
        msg = f"{position} not found in patch coordinates"
        raise ParameterError(msg)
    return patch.coords.get_array(position), position


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


def _get_color(patch, color, x):
    """The values to color by, and what the colorbar calls them."""
    if not isinstance(color, str):
        return np.asanyarray(color), ""
    if color in patch.coords:
        units = patch.coords.coord_map[color].units
        return patch.coords.get_array(color), _get_label(color, units)
    if color == "data":
        return _get_data_to_color(patch, x), _get_data_label(patch)
    msg = (
        f"{color} not found in patch coordinates. Use 'data' to "
        "color by the patch's own data."
    )
    raise ParameterError(msg)


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
        If not None, controls the saturation level of the colorbar. A single
        number is symmetric: with `scale_type="relative"` the limits sit that
        fraction of half the data range either side of the mean, and with
        `scale_type="absolute"` they are -abs(scale) and abs(scale). A pair
        of numbers gives the lower and upper limits: fractions of the data
        range, from 0 to 1, when relative, or the values themselves when
        absolute. Percent quantities, such as `10 * dc.units.percent`, are
        converted to fractions.
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
    x, x_dim = _get_position(patch, x)
    y, y_dim = _get_position(patch, y)
    color, label = _get_color(patch, color, x)

    ax = _get_ax(ax)
    im = ax.scatter(x, y, c=color, cmap=_get_cmap(cmap))

    if scale is not None:
        scale = _get_scale(scale, scale_type, color)
    # As in waterfall: limits are applied only where they are a finite pair.
    if scale is not None and len(scale) == 2 and np.all(np.isfinite(scale)):
        im.set_clim(np.asarray(scale))

    if x_dim is not None:
        ax.set_xlabel(_get_dim_label(patch, x_dim))
    if y_dim is not None:
        ax.set_ylabel(_get_dim_label(patch, y_dim))

    if cmap is not None:
        _add_colorbar(ax, im, color, label, scale)

    if show:
        plt.show()
    return ax
