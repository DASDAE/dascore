"""
Module for wiggle plotting.
"""
from datetime import timedelta
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from dascore.constants import ONE_BILLION, PatchType
from dascore.utils.patch import patch_function
from dascore.utils.plotting import _format_time_axis, _get_ax
from dascore.utils.time import to_number


@patch_function(required_dims=("time", "distance"))
def wiggle(
    patch: PatchType,
    dim="time",
    color="black",
    ax: Optional[plt.Axes] = None,
    timefmt=None,
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

    Examples
    --------
    >>> # Plot the default patch
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> _ = patch.viz.wiggle()
    """
    ax = _get_ax(ax)
    dims = patch.dims
    assert len(dims) == 2, "Can only make wiggle plot of 2D Patch"
    patch = patch.transpose(..., dim)
    data = patch.data
    dims = patch.dims
    dims_r = tuple(reversed(dims))
    coords = {dimen: patch.coords[dimen] for dimen in dims}

    if dim == "time":
        max_of_traces = abs(data).max(axis=1)
        data = data / max_of_traces[:, np.newaxis]
        axis_across_wiggles = coords["distance"]
        axis_along_wiggles = to_number(coords["time"]) / ONE_BILLION
        intervals = coords["distance"][1:] - coords["distance"][:-1]
    else:
        max_of_traces = abs(data).max(axis=0)
        data = data / max_of_traces[np.newaxis, :]
        axis_across_wiggles = to_number(coords["time"]) / ONE_BILLION
        axis_along_wiggles = coords["distance"]
        intervals = (coords["time"][1:] - coords["time"][:-1]) / np.timedelta64(1, "s")

    total_wiggles = len(data)
    data = -1 * data
    for a in range(total_wiggles):
        if dim == "distance":
            _intervals = intervals[min(a, total_wiggles - 2)]
            array_inputs = [timedelta(seconds=(_intervals * b)) for b in data[a]]
            array = np.array(array_inputs, dtype="timedelta64[ns]")
            wiggle = array + axis_across_wiggles[a]
        else:
            wiggle = (
                intervals[min(a, total_wiggles - 2)] * data[a] + axis_across_wiggles[a]
            )
        ax.plot(axis_along_wiggles, wiggle, color, alpha=1, linewidth=1)
        where = data[a] < 0
        ax.fill_between(
            axis_along_wiggles,
            np.array([axis_across_wiggles[a]] * len(data[a])),
            wiggle,
            color=color,
            where=where,
            edgecolor=None,
            interpolate=True,
        )

    for dim, x in zip(dims_r, ["x", "y"]):
        getattr(ax, f"set_{x}label")(str(dim).capitalize())
    if "time" in dims_r:
        _format_time_axis(ax, patch, dims_r, timefmt)
    ax.invert_yaxis()  # invert y axis so origin is at top
    if show:
        plt.show()
    return ax
