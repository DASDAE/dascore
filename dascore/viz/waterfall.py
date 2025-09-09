"""Module for waterfall plotting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from dascore.constants import PatchType
from dascore.units import get_quantity_str
from dascore.utils.patch import patch_function
from dascore.utils.plotting import (
    _format_time_axis,
    _get_ax,
    _get_cmap,
    _get_dim_label,
    _get_extents,
)


def _set_scale(im, scale, scale_type, patch):
    """Set the scale of the color bar based on scale and scale_type."""
    # check scale parameters
    assert scale_type in {"absolute", "relative"}
    assert isinstance(scale, float | int) or len(scale) == 2
    # make sure we have a len two array
    data = patch.data
    modifier = 1
    if scale_type == "relative":
        modifier = 0.5 * (np.nanmax(data) - np.nanmin(data))
        # only one scale parameter provided, center around mean
    if isinstance(scale, float | int):
        mean = np.nanmean(patch.data)
        scale = np.asarray([mean - scale * modifier, mean + scale * modifier])
    im.set_clim(scale)


@patch_function()
def waterfall(
    patch: PatchType,
    ax: plt.Axes | None = None,
    cmap="bwr",
    scale: float | Sequence[float] | None = None,
    scale_type: Literal["relative", "absolute"] = "relative",
    log=False,
    show=False,
) -> plt.Axes:
    """
    Create a waterfall plot of the Patch data.

    Parameters
    ----------
    patch
        The Patch object.
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
    log
        If True, visualize the common logarithm of the absolute values of patch data.
    show
        If True, show the plot, else just return axis.

    Examples
    --------
    >>> # Plot the default patch
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> _ = patch.viz.waterfall(scale=0.1)
    """
    ax = _get_ax(ax)
    cmap = _get_cmap(cmap)
    data = np.log10(np.absolute(patch.data)) if log else patch.data
    dims = patch.dims
    assert len(dims) == 2, "Can only make waterfall plot of 2D Patch"
    dims_r = tuple(reversed(dims))
    coords = {dim: patch.coords.get_array(dim) for dim in dims}
    extents = _get_extents(dims_r, coords)
    im = ax.imshow(data, extent=extents, aspect="auto", cmap=cmap, origin="lower")
    # scale colorbar
    if scale is not None:
        _set_scale(im, scale, scale_type, patch)
    for dim, x in zip(dims_r, ["x", "y"]):
        getattr(ax, f"set_{x}label")(_get_dim_label(patch, dim))
        # format all dims which have time types.
        coord = patch.get_coord(dim)
        if np.issubdtype(coord.dtype, np.datetime64):
            _format_time_axis(ax, dim, x)
    # add color bar with title
    if cmap is not None:
        cb = ax.get_figure().colorbar(im, ax=ax, fraction=0.05, pad=0.025)
        data_type = str(patch.attrs["data_type"])
        data_units = get_quantity_str(patch.attrs.data_units) or ""
        dunits = f" ({data_units})" if (data_type and data_units) else f"{data_units}"
        if log:
            dunits = f"{dunits} - log_10"
        label = f"{data_type}{dunits}"
        cb.set_label(label)
    ax.invert_yaxis()  # invert y axis so origin is at top
    if show:
        plt.show()
    return ax
