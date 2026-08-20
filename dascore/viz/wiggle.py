"""Module for wiggle plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import patch_function
from dascore.utils.plotting import (
    _format_time_axis,
    _get_ax,
    _get_data_label,
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
    """Shades the part of each waveform above its offset line."""
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


def _wiggle_1d(patch, ax, alpha, color, shade):
    """Plot a 1D patch as a single trace against its only coordinate."""
    dim = patch.dims[0]
    x_values = patch.coords.get_array(dim)
    ax.plot(x_values, patch.data, color=color, alpha=alpha)
    if shade:
        _shade(np.array([0]), ax, patch.data[:, None], color, x_values)
    ax.set_xlabel(_get_dim_label(patch, dim))
    ax.set_ylabel(_get_data_label(patch, default="amplitude"))
    if np.issubdtype(patch.get_coord(dim).dtype, np.datetime64):
        _format_time_axis(ax, dim, "x")
    return ax


def _wiggle_2d(patch, ax, dim, scale, alpha, color, shade):
    """Plot each trace of a 2D patch offset from the others."""
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
    # shade the part of each wiggle above its offset if desired
    if shade:
        _shade(offsets, ax, data_scaled, color, connect_axis_ticks)
    _format_y_axis_ticks(ax, offsets, other_axis_ticks)
    for dim, x in zip(patch.dims, ["x", "y"]):
        getattr(ax, f"set_{x}label")(_get_dim_label(patch, dim))
        # format all dims which have time types.
        if np.issubdtype(patch.get_coord(dim).dtype, np.datetime64):
            _format_time_axis(ax, dim, x)
    ax.invert_yaxis()  # invert y so it's consistent with waterfall
    return ax


@patch_function()
def wiggle(
    patch: PatchType,
    dim: str = "time",
    scale: float = 1,
    alpha: float | None = None,
    color: str = "black",
    shade: bool = False,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create a wiggle plot of patch data.

    Length one dimensions are squeezed out first. A patch left with a single
    dimension (e.g., an OTDR trace stored as ``(time: 1, distance: N)``) is
    drawn as one line against that dimension, with the data type and units
    (or "amplitude" if the patch has neither) on the y axis. A patch left with
    two dimensions is drawn as one wiggle per trace.

    Parameters
    ----------
    patch
        The Patch object.
    dim
        The dimension along which samples are connected. Ignored if only
        one dimension remains after squeezing.
    scale
        The scale (or gain) of the waveforms. A value of 1 indicates waveform
        centroids are separated by the average total waveform excursion.
    alpha
        Opacity of the wiggle lines. Defaults to 0.2 for 2D patches, where
        neighboring wiggles overlap, and 1.0 for a single trace.
    color
        Color of wiggles
    shade
        If True, shade all values of each trace which are greater than the
        trace offset (zero for a single trace).
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
    >>>
    >>> # A single trace plots as one line
    >>> trace = patch.select(distance=0, samples=True)
    >>> _ = trace.viz.wiggle()
    """
    # A length one dimension has nothing to connect, so drop it rather than
    # drawing a separate (one-sample) wiggle for every sample along the other.
    # Only exactly length one dims are dropped; Patch.squeeze can't remove
    # empty dims and those just plot nothing.
    squeezable = [x for x in patch.dims if len(patch.get_coord(x)) == 1]
    if len(squeezable) == patch.ndim:
        msg = "Cannot make wiggle plot of a Patch with a single sample."
        raise ParameterError(msg)
    if squeezable:
        patch = patch.squeeze(squeezable)
    if patch.ndim == 2 and dim not in patch.dims:
        msg = (
            f"dim {dim!r} is not a dimension of the patch after squeezing "
            f"length one dimensions; it must be one of {patch.dims}."
        )
        raise ParameterError(msg)
    if patch.ndim > 2:
        msg = (
            "Can only make wiggle plot of a 1D or 2D Patch, but after "
            f"squeezing length one dimensions patch has dims {patch.dims}."
        )
        raise ParameterError(msg)
    # Create the axis only once the patch is known to be plottable so a
    # rejected call doesn't leak an empty figure.
    ax = _get_ax(ax)
    if patch.ndim == 1:
        alpha = 1.0 if alpha is None else alpha
        _wiggle_1d(patch, ax, alpha, color, shade)
    else:
        alpha = 0.2 if alpha is None else alpha
        _wiggle_2d(patch, ax, dim, scale, alpha, color, shade)
    if show:
        plt.show()
    return ax
