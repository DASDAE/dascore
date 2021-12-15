"""
Module for waterfall plotting.
"""
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

import dascore
from dascore.utils.patch import patch_function
from dascore.utils.plotting import (
    _add_time_axis_label,
    _format_time_axis,
    _get_ax,
    _get_cmap,
    _get_extents,
)


@patch_function()
def waterfall(
    patch: "dascore.Patch",
    ax: Optional[plt.Axes] = None,
    cmap="bwr",
    timefmt="%H:%M:%S",
    scale: Optional[Union[float, Sequence[float]]] = None,
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
        If not None, a tuple of fractions of min/max to scale colorbar.
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
        scale = np.array(scale) * np.ones(2)
        scale_val = np.max([np.abs(data.min()), np.abs(data.max())])
        im.set_clim(np.array([-scale_val * scale[0], scale_val * scale[1]]))
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
