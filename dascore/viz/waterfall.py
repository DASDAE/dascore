"""
Module for waterfall plotting.
"""
from typing import Literal, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from dascore.constants import PatchType
from dascore.utils.patch import patch_function
from dascore.utils.plotting import (
    _add_time_axis_label,
    _format_time_axis,
    _get_ax,
    _get_cmap,
    _get_extents,
)


def _set_scale(im, scale, scale_type, patch):
    """Set the scale of the color bar based on scale and scale_type."""
    # check scale paramters
    assert scale_type in {"absolute", "relative"}
    assert isinstance(scale, (float, int)) or len(scale) == 2
    # make sure we have a len two array
    data = patch.data
    modifier = 1
    if scale_type == "relative":
        modifier = 0.5 * (data.max() - data.min())
        # only one scale parameter provided, center around mean
    if isinstance(scale, float):
        mean = patch.data.mean()
        scale = np.array([mean - scale * modifier, mean + scale * modifier])
    im.set_clim(scale)


@patch_function()
def waterfall(
    patch: PatchType,
    ax: Optional[plt.Axes] = None,
    cmap="bwr",
    timefmt="%H:%M:%S",
    scale: Optional[Union[float, Sequence[float]]] = None,
    scale_type: Literal["relative", "absolute"] = "relative",
    colorbar=True,
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
    colorbar
        If True, show the color bar.
    show
        If True, show the plot, else just return axis.
    """
    ax = _get_ax(ax)
    cmap = _get_cmap(cmap)
    data = patch.data
    dims = patch.dims
    assert len(dims) == 2, "Can only make waterfall plot of 2D Patch"
    dims_r = tuple(reversed(dims))
    coords = {dim: patch.coords[dim] for dim in dims}
    extents = _get_extents(dims_r, coords)
    im = ax.imshow(data, extent=extents, aspect="auto", cmap=cmap, origin="lower")
    # scale colorbar
    if scale is not None:
        _set_scale(im, scale, scale_type, patch)
    for dim, x in zip(dims_r, ["x", "y"]):
        getattr(ax, f"set_{x}label")(str(dim).capitalize())
    if "time" in dims_r:
        _format_time_axis(ax, dims_r, timefmt)
        _add_time_axis_label(ax, patch, dims_r)
    # add color bar
    if colorbar:
        ax.get_figure().colorbar(im)
    ax.invert_yaxis()  # invert y axis so origin is at top
    if show:
        plt.show()
    return ax
