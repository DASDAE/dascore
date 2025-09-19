"""Module for wiggle plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dascore.constants import PatchType
from dascore.utils.patch import patch_function
from dascore.utils.plotting import (
    _format_time_axis,
    _get_ax,
    _get_dim_label,
)
from dascore.utils.time import dtype_time_like


def _get_offsets_factor(patch, dim, scale, other_labels):
    """Get the offsets and scale the data."""
    dim_axis = patch.get_axis(dim)
    # get and apply scale_factor. This controls how far apart the wiggles are.
    diffs = np.max(patch.data, axis=dim_axis) - np.min(patch.data, axis=dim_axis)
    offsets = (np.median(diffs) * scale) * np.arange(len(other_labels))
    # add scale factor to data
    data_scaled = offsets[None, :] + patch.data
    return offsets, data_scaled


def _shade(offsets, ax, data_scaled, color, wiggle_labels):
    """Shades the part of the waveforms under the offset line."""
    for i in range(len(offsets)):
        ax.fill_between(
            wiggle_labels,
            offsets[i],
            data_scaled[:, i],
            where=(data_scaled[:, i] > offsets[i]),
            color=color,
            alpha=0.6,
        )


def _format_y_axis_ticks(ax, offsets, other_axis_ticks, max_ticks=10):
    """Format the Y axis tick labels."""
    # make sure not printing too many digits on the figure
    if not dtype_time_like(other_axis_ticks):
        other_axis_ticks = np.around(other_axis_ticks, decimals=2)
    # set the offset
    ax.set_yticks(offsets, other_axis_ticks)
    min_bins = min(len(other_axis_ticks), max_ticks)
    plt.locator_params(axis="y", nbins=min_bins)


@patch_function()
def wiggle(
    patch: PatchType,
    dim="time",
    scale=1,
    alpha=0.2,
    color="black",
    shade=False,
    ax: plt.Axes | None = None,
    show=False,
) -> plt.Figure:
    """
    Create a wiggle plot of patch data.

    Parameters
    ----------
    patch
        The Patch object.
    dim
        The dimension along which samples are connected.
    scale
        The scale (or gain) of the waveforms. A value of 1 indicates waveform
        centroids are separated by the average total waveform excursion.
    alpha
        Opacity of the wiggle lines.
    color
        Color of wiggles
    shade
        If True, shade all values of each trace which are less than the mean
        trace value.
    ax
        A matplotlib object, if None ne will be created.
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
    assert len(patch.dims) == 2, "Can only make wiggle plot of 2D Patch"
    # After transpose selected dim must be axis 0 and other axis 1
    patch = patch.transpose(dim, ...)
    other_dim = next(iter(set(patch.dims) - {dim}))
    # values for axis which is connected
    connect_axis_ticks = patch.coords.get_array(dim)
    # values for y axis (not connected)
    other_axis_ticks = patch.coords.get_array(other_dim)
    offsets, data_scaled = _get_offsets_factor(patch, dim, scale, other_axis_ticks)
    # now plot, add labels, etc.
    ax.plot(connect_axis_ticks, data_scaled, color=color, alpha=alpha)
    # shade negative part of waveforms if desired
    if shade:
        _shade(offsets, ax, data_scaled, color, connect_axis_ticks)
    _format_y_axis_ticks(ax, offsets, other_axis_ticks)
    for dim, x in zip(patch.dims, ["x", "y"]):
        getattr(ax, f"set_{x}label")(_get_dim_label(patch, dim))
        # format all dims which have time types.
        if np.issubdtype(patch.get_coord(dim).dtype, np.datetime64):
            _format_time_axis(ax, dim, x)
    if show:
        plt.show()
    ax.invert_yaxis()  # invert y so its consistent with waterfall
    return ax
