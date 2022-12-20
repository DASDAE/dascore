"""
Module for wiggle plotting.
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from dascore.constants import PatchType
from dascore.utils.patch import patch_function
from dascore.utils.plotting import _add_time_axis_label, _format_time_axis, _get_ax


@patch_function()
def wiggle(
    patch: PatchType,
    dim="time",
    color="black",
    ax: Optional[plt.Axes] = None,
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
    color
        Color of wiggles
    timefmt
        The format for the time axis.
    show
        If True, show the plot, else just return axis.
    """
    ax = _get_ax(ax)
    data = patch.data
    dims = patch.dims
    assert len(dims) == 2, "Can only make wiggle plot of 2D Patch"
    patch = patch.transpose(..., "time")
    dims = patch.dims
    dims_r = tuple(reversed(dims))
    coords = {dim: patch.coords[dim] for dim in dims}

    if dim == "time":
        maxOfTraces = abs(data).max(axis=1)
        data = data / maxOfTraces[:, np.newaxis]
    else:
        maxOfTraces = abs(data).max(axis=0)
        data = data / maxOfTraces[np.newaxis, :]

    time = coords["time"]
    for a in range(len(data)):
        ax.plot(time, a + data[a], color, alpha=1, linewidth=1)
        wher = data[a] > 0
        ax.fill_between(
            time,
            a + 0 * data[a],
            a + data[a],
            color=color,
            where=wher,
            edgecolor=None,
            interpolate=True,
        )

    for dim, x in zip(dims_r, ["x", "y"]):
        getattr(ax, f"set_{x}label")(str(dim).capitalize())
    if "time" in dims_r:
        _format_time_axis(ax, dims_r, timefmt)
        _add_time_axis_label(ax, patch, dims_r)
    ax.invert_yaxis()  # invert y axis so origin is at top
    if show:
        plt.show()
    return ax
