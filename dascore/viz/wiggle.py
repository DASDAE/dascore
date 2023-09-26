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


def _get_offsets_factor(patch, dim, scale, other_labels):
    """Get the offsets and scale the data."""
    dim_axis = patch.dims.index(dim)
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


@patch_function()
def wiggle(
    patch: PatchType,
    dim="time",
    scale=1,
    color="black",
    shade=False,
    ax: plt.Axes | None = None,
    show=False,
) -> plt.Figure:
    """
    Parameters
    ----------
    patch
        The Patch object.
    dim
        The dimension along which samples are connected.
    scale
        The scale (or gain) of the waveforms. A value of 1 indicates waveform
        centroids are separated by the average total waveform excursion.
    color
        Color of wiggles
    shade
        If True, shade all values of each trace which are less than the mean
        trace value.
    ax
        A matplotlib object, if None create one.
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
    wiggle_labels = patch.coords.get_array(dim)
    other_labels = patch.coords.get_array(other_dim)
    offsets, data_scaled = _get_offsets_factor(patch, dim, scale, other_labels)
    # now plot, add labels, etc.
    plt.plot(wiggle_labels, data_scaled, color=color, alpha=0.2)
    # shade negative part of waveforms if desired
    if shade:
        _shade(offsets, ax, data_scaled, color, wiggle_labels)
    # we need to tweak the other labels
    ax.set_yticks(offsets, other_labels)
    # reduce density of labels to 30 for better clarity
    plt.locator_params(axis="y", nbins=min(len(other_labels), 30))
    for dim, x in zip(patch.dims, ["x", "y"]):
        getattr(ax, f"set_{x}label")(_get_dim_label(patch, dim))
        # format all dims which have time types.
        if np.issubdtype(patch.get_coord(dim).dtype, np.datetime64):
            _format_time_axis(ax, dim, x)
    if show:
        plt.show()
    return ax
