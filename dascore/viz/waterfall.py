"""Module for waterfall plotting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dascore.constants import DEFAULT_COLORMAPS, PatchType
from dascore.exceptions import ParameterError
from dascore.utils.gaps import get_gap_edges, is_monotonic_and_finite
from dascore.utils.patch import patch_function
from dascore.utils.plotting import (
    _add_colorbar,
    _format_time_axis,
    _get_ax,
    _get_cmap,
    _get_data_label,
    _get_dim_label,
    _get_extents,
    _get_scale,
    _maybe_invert_yaxis,
)
from dascore.utils.time import is_datetime64
from dascore.viz._labels import (
    draw_labels,
    image_cell_edges,
    label_plan,
    label_runs,
    mesh_cell_edges,
)


def _validate_gap_factor(gap_factor):
    """Validate the factor used to identify coordinate gaps."""
    if not np.isfinite(gap_factor) or gap_factor <= 1:
        msg = "gap_factor must be a finite number greater than 1"
        raise ParameterError(msg)


def _validate_patch_dims(patch):
    """Validate that patch is 2D for waterfall plotting."""
    if patch.ndim != 2:
        # Try squeezing out degenerate dims to visualize.
        patch = patch.squeeze()
        if patch.ndim != 2:
            dims = patch.dims
            msg = (
                f"Can only make waterfall plot of 2D Patch, "
                f"but got {patch.ndim}D patch with dims {dims}"
            )
            raise ParameterError(msg)
    return patch


def _format_axis_labels(ax, patch, dims_r):
    """
    Format axis labels and handle time-like axes.
    """
    for dim, x in zip(dims_r, ["x", "y"], strict=True):
        getattr(ax, f"set_{x}label")(_get_dim_label(patch, dim))
        # Check if special formatting is needed to make date times label correctly.
        dtype = patch.get_coord(dim).dtype
        if is_datetime64(dtype):
            _format_time_axis(ax, dim, x)
        if x == "y":
            _maybe_invert_yaxis(ax, patch, dim)


def _get_waterfall_colormap(patch, cmap=None):
    """
    Select a default colormap based on datatype
    """
    if cmap is None:
        this_type = str(patch.attrs.get("data_type", "")).lower()
        cmap = DEFAULT_COLORMAPS.get(this_type, "bwr")  # defaults to "bwr"
    return _get_cmap(cmap)


def _insert_gap_bands(data, gap_mask, axis):
    """Insert masked bands into an array at each coordinate gap."""
    if not np.any(gap_mask):
        return data
    old_size = data.shape[axis]
    new_size = old_size + np.count_nonzero(gap_mask)
    new_shape = list(data.shape)
    new_shape[axis] = new_size
    out = np.ma.masked_all(new_shape, dtype=data.dtype)
    new_indices = np.arange(old_size) + np.cumsum(
        np.concatenate(([0], gap_mask.astype(int)))
    )
    indexer = [slice(None)] * data.ndim
    indexer[axis] = new_indices
    out[tuple(indexer)] = data
    return out


def _plot_with_mesh(ax, data, dims, coords, cmap, gap_color, gap_factor):
    """Plot irregularly sampled data using a quadrilateral mesh.

    Returns the mesh and, per dimension, the cell edges it was drawn from
    with the gaps they opened, so a caller marking the same cells reads
    them off the mesh rather than working them out again.
    """
    mesh_data = np.ma.asarray(data)
    edges = {}
    cells = {}
    mesh_gap_factor = gap_factor if gap_color is not None else None
    for axis, dim in enumerate(dims):
        dim_edges, gap_mask = get_gap_edges(coords[dim], mesh_gap_factor)
        if gap_color is not None:
            mesh_data = _insert_gap_bands(mesh_data, gap_mask, axis)
        edges[dim] = dim_edges
        cells[dim] = (dim_edges, gap_mask)

    if gap_color is not None:
        cmap = cmap.with_extremes(bad=gap_color)
    mesh = ax.pcolormesh(
        edges[dims[1]],
        edges[dims[0]],
        mesh_data,
        cmap=cmap,
        shading="flat",
        edgecolors="none",
        linewidth=0,
        antialiased=False,
    )
    return mesh, cells


@patch_function()
def waterfall(
    patch: PatchType,
    ax: plt.Axes | None = None,
    cmap: str | None = None,
    scale: float | Sequence[float] | None = None,
    scale_type: Literal["relative", "absolute"] = "relative",
    interpolation: str | None = "antialiased",
    interpolation_stage: str = "auto",
    gap_color: str | Sequence[float] | None = None,
    gap_factor: float = 1.5,
    log: bool = False,
    cbar: bool = True,
    show: bool = False,
    label_coord: str | None = None,
) -> plt.Axes:
    """
    Create a waterfall plot of the Patch data.

    Evenly sampled coordinates use ``imshow``. Finite, monotonic irregular
    coordinates use ``pcolormesh`` so cells follow their coordinate values;
    incomplete or nonmonotonic coordinates fall back to ``imshow``.

    Parameters
    ----------
    patch
        The Patch object.
    ax
        A matplotlib object, if None create one.
    cmap
        Matplotlib colormap. None selects one from the patch ``data_type``.
    scale
        Color limits. A scalar produces symmetric limits: a fraction of half
        the data range around its mean when relative, or ``±abs(scale)`` when
        absolute. A relative pair maps fractions from 0 to 1 onto the data
        minimum and maximum; an absolute pair gives the limits directly.
        Percent quantities are converted to fractions.
    scale_type
        Interpret ``scale`` as ``"relative"`` fractions or ``"absolute"``
        values.
    interpolation
        Passed to matplotlib ``imshow``. ``"antialiased"`` handles large
        arrays well; None can help if they look smeared. Ignored by
        ``pcolormesh``.
    interpolation_stage
        ``imshow`` interpolation stage: ``"data"``, ``"rgba"``, or
        ``"auto"``. Ignored by ``pcolormesh``.
    gap_color
        Color for gaps in irregular coordinates. A color inserts masked cells
        at detected gaps; None bridges them with adjacent cells. Existing NaN
        or masked data use the same color. Applies only to ``pcolormesh``.
    gap_factor
        Intervals larger than this multiple of the median are gaps. Must exceed 1,
        even when ``gap_color`` is None and it has no visual effect. Mixed
        sampling rates may classify coarser regions as gaps; increase this value
        or plot or resample those regions separately.
    log
        If True, visualize the common logarithm of the absolute values of patch data.
        To avoid log(0), the abs(array) is cast to float64 and a small value
        added.
    cbar
        Whether to draw a colorbar.
    show
        Whether to show the plot.
    label_coord
        Coordinate whose values label stretches along one plotted dimension,
        drawn on its spines without tinting the data. Its legend is outside a
        figure created here, or in the upper-right of a supplied ``ax``. String
        and numeric values are categories; a boolean coordinate marks only True
        stretches. Empty strings, NaN, and False are omitted. A coordinate that
        is a dimension, spans both dimensions, has no labels, exceeds 20 labels,
        or changes more than 200 times raises `ParameterError` before drawing.

    Examples
    --------
    >>> # Plot with default scaling (uses 1.5*IQR fence to exclude outliers)
    >>> import dascore as dc
    >>> from dascore.units import percent
    >>> patch = dc.get_example_patch("example_event_1").normalize("time")
    >>> _ = patch.viz.waterfall()
    >>>
    >>> _ = patch.viz.waterfall(scale=(0.1, 0.9), scale_type="relative")
    >>> _ = patch.viz.waterfall(scale=10 * percent)
    >>> _ = patch.viz.waterfall(scale=(-0.5, 0.5), scale_type="absolute")
    >>> _ = patch.viz.waterfall(log=True)
    >>>
    >>> from dascore.examples import inventory_patch_pair
    >>> zoned, inventory = inventory_patch_pair()
    >>> _ = zoned.enrich(inventory).viz.waterfall(label_coord="zone")
    >>>
    >>> ax = patch.viz.waterfall()
    >>> ax.invert_yaxis()

    Notes
    -----
    Empty dimensions raise `ParameterError`. Time-like Y axes are inverted by
    seismic convention; call ``ax.invert_yaxis()`` to undo this. Since version
    0.1.13, ``scale=None`` uses a statistical fence to limit outliers; use
    ``scale=1.0`` for the full data range.
    """
    if 0 in patch.shape:
        msg = "Cannot plot a Patch with an empty dimension."
        raise ParameterError(msg)
    # Validate inputs
    patch = _validate_patch_dims(patch)
    _validate_gap_factor(gap_factor)
    dims = patch.dims
    dims_r = tuple(reversed(dims))
    # Before an axes exists, so a refused label_coord leaves no figure
    # behind: both what the coordinate is and what it states are settled
    # here, since either can be grounds for refusing it.
    plan, runs = None, None
    if label_coord is not None:
        plan = label_plan(patch, label_coord, dims_r)
        runs = label_runs(patch.coords.get_array(plan.name), plan.name)
    # Setup axes and data. A figure this call built is one whose room a
    # legend may take; any other belongs to the caller.
    owned = ax is None
    ax = _get_ax(ax)
    if log:
        data = np.log10(np.abs(patch.data) + np.finfo(np.float64).eps)
    else:
        data = patch.data
    dim_coords = {dim: patch.get_coord(dim) for dim in dims}
    coords = {dim: np.asarray(coord) for dim, coord in dim_coords.items()}
    cmap = _get_waterfall_colormap(patch, cmap)
    scale = _get_scale(scale, scale_type, data)
    label_edges = None
    use_image = all(coord.evenly_sampled for coord in dim_coords.values())
    if use_image or not all(is_monotonic_and_finite(x) for x in coords.values()):
        extents = _get_extents(dims_r, coords)
        with mpl.rc_context({"image.resample": True}):
            im = ax.imshow(
                data,
                extent=extents,
                aspect="auto",
                cmap=cmap,
                origin="lower",
                interpolation=interpolation,
                interpolation_stage=interpolation_stage,
            )
        if plan is not None:
            label_edges = image_cell_edges(
                extents, dims_r, plan.dim, len(coords[plan.dim])
            )
    else:
        im, cells = _plot_with_mesh(
            ax,
            data,
            dims,
            coords,
            cmap,
            gap_color=gap_color,
            gap_factor=gap_factor,
        )
        if plan is not None:
            label_edges = mesh_cell_edges(*cells[plan.dim])
    if scale is not None and len(scale) == 2 and np.all(np.isfinite(scale)):
        im.set_clim(np.asarray(scale))
    # Format axis labels and handle time-like dimensions
    _format_axis_labels(ax, patch, dims_r)
    # Add colorbar if requested
    if cbar:
        label = _get_data_label(patch)
        if log:
            label = f"{label} - log_10"
        _add_colorbar(ax, im, data, label, scale)
    # Label lines come last so the legend is placed beyond a colorbar which
    # has already taken its room.
    if plan is not None and runs is not None:
        assert label_edges is not None, "a plan is only made where edges are"
        draw_labels(ax, plan, runs, label_edges, owned=owned)
    if show:
        plt.show()
    return ax
