"""Module for wiggle plotting."""

from __future__ import annotations

import string

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.misc import suppress_warnings
from dascore.utils.patch import patch_function
from dascore.utils.plotting import (
    _format_time_axis,
    _get_ax,
    _get_data_label,
    _get_dim_label,
    _maybe_invert_yaxis,
)
from dascore.utils.time import dtype_time_like


def _get_offsets_factor(patch, dim, scale, other_labels):
    """Get the offsets and scale the data."""
    dim_axis = patch.get_axis(dim)
    # get and apply scale_factor. This controls how far apart the wiggles are.
    # NaN samples are skipped so one gap doesn't blank the whole plot.
    with suppress_warnings(RuntimeWarning):  # all-NaN traces
        mx = np.nanmax(patch.data, axis=dim_axis)
        mn = np.nanmin(patch.data, axis=dim_axis)
        separation = np.nanmedian(mx - mn) * scale
    if not np.isfinite(separation):  # no trace has a finite range
        separation = 0.0
    offsets = separation * np.arange(len(other_labels))
    # add scale factor to data
    data_scaled = offsets[None, :] + patch.data
    return offsets, data_scaled


def _get_plot_values(ax, patch, dim):
    """
    Get the values of a coordinate as floats for the x axis.

    The conversion is matplotlib's own, the one ax.plot would apply, so
    datetimes become date numbers (which xaxis_date, set up later by
    _format_time_axis, expects) and anything else with a registered
    converter is handled the same way. Floats can then share arrays with
    the trace data.
    """
    values = patch.coords.get_array(dim)
    ax.xaxis.update_units(values)
    return np.asarray(ax.xaxis.convert_units(values), dtype=float)


def _plot_traces(ax, x, data_scaled, color, alpha):
    """
    Draw each column of data_scaled as a line against x.

    A single LineCollection is much faster to build than one Line2D per
    trace. Each trace is still rendered as its own path, so overlapping
    traces darken with alpha like separate lines do. (One NaN separated
    Line2D would draw faster for very long traces, but Agg composites a
    path once, so overlaps wouldn't darken.)
    """
    n_samples, n_traces = data_scaled.shape
    segments = np.empty((n_traces, n_samples, 2))
    segments[:, :, 0] = x[None, :]
    segments[:, :, 1] = data_scaled.T
    # A (n_traces, n_samples, 2) array is a valid segment sequence.
    collection = LineCollection(segments, colors=color, alpha=alpha)  # ty: ignore[invalid-argument-type]
    ax.add_collection(collection)
    # Collections don't update the view limits on their own.
    ax.autoscale_view()


def _shade(ax, x, offsets, data_scaled, color):
    """
    Shade the part of each trace above its offset line.

    Builds one polygon per trace (rather than calling fill_between for each)
    with a vertex wherever the trace crosses its offset, like
    fill_between(interpolate=True), so the shading stops at the crossing
    rather than at the neighboring sample.
    """
    # Height of each sample above its offset; non-finite samples sit on it.
    d = data_scaled - offsets[None, :]
    d = np.where(np.isfinite(d), d, np.nan)
    above = np.fmax(d, 0)
    d0, d1 = d[:-1], d[1:]
    x0, x1 = x[:-1, None], x[1:, None]
    # Between samples add a vertex on the offset at the crossing. Next to a
    # non-finite sample drop to the offset at the finite one, so gaps aren't
    # shaded. Elsewhere repeat the next sample (a zero length edge).
    finite = np.isfinite(d0) & np.isfinite(d1)
    cross = finite & ((d0 > 0) != (d1 > 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        x_mid = np.where(cross, x0 + (x1 - x0) * d0 / (d0 - d1), x1)
    x_mid = np.where(np.isnan(d1) & np.isfinite(d0), x0, x_mid)
    y_mid = np.where(cross | ~finite, 0.0, above[1:])
    n_samples, n_traces = d.shape
    xs = np.empty((2 * n_samples - 1, n_traces))
    ys = np.empty_like(xs)
    xs[0::2], ys[0::2] = x[:, None], above
    xs[1::2], ys[1::2] = x_mid, y_mid
    # Assemble the polygons, closing each along its offset line.
    verts = np.empty((n_traces, 2 * n_samples + 1, 2))
    verts[:, 1:-1, 0] = xs.T
    verts[:, 1:-1, 1] = (ys + offsets[None, :]).T
    verts[:, [0, -1], 0] = x[[0, -1]]
    verts[:, [0, -1], 1] = offsets[:, None]
    # A (n_traces, n_verts, 2) array is a valid vertex sequence.
    poly = PolyCollection(verts, facecolors=color, edgecolors="none", alpha=0.6)  # ty: ignore[invalid-argument-type]
    ax.add_collection(poly)
    # The fill reaches the offset line, which may be outside the data range
    # the lines were scaled to.
    ax.autoscale_view()


def _format_y_axis_ticks(ax, offsets, other_axis_ticks, max_ticks=10):
    """Put at most max_ticks trace labels on the Y axis."""
    # make sure not printing too many digits on the figure
    if not dtype_time_like(other_axis_ticks):
        other_axis_ticks = np.around(other_axis_ticks, decimals=2)
    # Only create the ticks which will be shown; matplotlib builds a Tick
    # for every location passed to set_yticks, which dominates the run time
    # with thousands of traces. The step keeps the positions the previous
    # set-all-then-locator_params(nbins) approach produced.
    step = max(int(0.99 + len(offsets) / max_ticks), 1)
    ax.set_yticks(offsets[::step], other_axis_ticks[::step])


def _wiggle_1d(patch, ax, alpha, color, shade):
    """Plot a 1D patch as a single trace against its only coordinate."""
    dim = patch.dims[0]
    x_values = _get_plot_values(ax, patch, dim)
    ax.plot(x_values, patch.data, color=color, alpha=alpha)
    if shade:
        _shade(ax, x_values, np.array([0.0]), patch.data[:, None], color)
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
    connect_axis_values = _get_plot_values(ax, patch, dim)
    # values for y axis (not connected)
    other_axis_ticks = patch.coords.get_array(other_dim)
    offsets, data_scaled = _get_offsets_factor(patch, dim, scale, other_axis_ticks)
    # now plot, add labels, etc.
    _plot_traces(ax, connect_axis_values, data_scaled, color, alpha)
    # shade the part of each wiggle above its offset if desired
    if shade:
        _shade(ax, connect_axis_values, offsets, data_scaled, color)
    _format_y_axis_ticks(ax, offsets, other_axis_ticks)
    for name, x in zip(patch.dims, ["x", "y"], strict=True):
        is_time = np.issubdtype(patch.get_coord(name).dtype, np.datetime64)
        label = string.capwords(name) if is_time else _get_dim_label(patch, name)
        getattr(ax, f"set_{x}label")(label)
    # Only the connected (x) axis is a continuous time axis; the y axis
    # holds trace offsets whose tick labels were set above, and a date
    # formatter there would overwrite them.
    if np.issubdtype(patch.get_coord(dim).dtype, np.datetime64):
        _format_time_axis(ax, dim, "x")
    # The y axis holds the trace offsets, so it is other_dim which decides.
    _maybe_invert_yaxis(ax, patch, other_dim)
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

    Notes
    -----
    - Traces are offset along the y axis and positive amplitudes point up,
      as they point right in a conventional (time down) wiggle display.

    - As in [waterfall](`dascore.viz.waterfall`), the y axis is
      inverted only when it is "time-like", meaning when the traces are
      stacked along time rather than along distance. If you don't want that,
      invert the y axis of the returned axis object.
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
    # An empty dimension has nothing to draw; the offsets can't be computed
    # from zero samples so just hand back the empty axis.
    if 0 in patch.shape:
        return ax
    if patch.ndim == 1:
        alpha = 1.0 if alpha is None else alpha
        _wiggle_1d(patch, ax, alpha, color, shade)
    else:
        alpha = 0.2 if alpha is None else alpha
        _wiggle_2d(patch, ax, dim, scale, alpha, color, shade)
    if show:
        plt.show()
    return ax
