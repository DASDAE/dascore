"""Module for waterfall plotting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.units import get_quantity_str, maybe_convert_percent_to_fraction
from dascore.utils.misc import tukey_fence
from dascore.utils.patch import patch_function
from dascore.utils.plotting import (
    _format_time_axis,
    _get_ax,
    _get_cmap,
    _get_dim_label,
    _get_extents,
)
from dascore.utils.time import dtype_time_like, is_datetime64


def _validate_scale_type(scale_type):
    """Validate that scale_type is either 'relative' or 'absolute'."""
    valid_types = {"absolute", "relative"}
    if scale_type not in valid_types:
        msg = f"scale_type must be one of {valid_types}, " f"but got '{scale_type}'"
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


def _get_scale(scale, scale_type, data):
    """
    Calculate the color bar scale limits based on scale and scale_type.
    """
    _validate_scale_type(scale_type)
    # This ensures we have a list of the previous scale parameters.
    scale = maybe_convert_percent_to_fraction(scale)
    match (scale, scale_type):
        # Case 1: Single value with relative scaling
        # Scale is symmetric around the mean, using fraction of dynamic range
        case (scale, "relative") if len(scale) == 1:
            scale = scale[0]
            if scale == 0:
                msg = (
                    "Relative scale value of 0 would produce degenerate colorbar limits"
                )
                raise ParameterError(msg)
            mod = 0.5 * (np.nanmax(data) - np.nanmin(data))
            if mod == 0:
                # Constant data, use small epsilon to avoid degenerate limits
                mod = 1e-10
            mean = np.nanmean(data)
            scale = np.asarray([mean - scale * mod, mean + scale * mod])
        # Case 2: No scale specified with relative scaling
        # Use Tukey's fence (C*IQR, C is normally 1.5) to exclude outliers.
        # This prevents a few extreme values from obscuring the majority of the
        # data at the cost of a slight performance penalty.
        case ([], "relative"):
            return tukey_fence(data)
        # Case 3: Sequence with relative scaling
        # Scale values represent fractions of the data range [0, 1]
        # and are mapped to [data_min, data_max]
        case (scale, "relative"):
            scale = np.array(scale)
            # Validate scale parameters
            if len(scale) != 2 or np.any(scale < 0) or scale[0] > scale[1]:
                msg = (
                    "Relative scale values cannot be negative and the first "
                    f"value must be less than the second. You passed {scale}"
                )
                raise ParameterError(msg)
            dmin, dmax = np.nanmin(data), np.nanmax(data)
            data_range = dmax - dmin
            # Map [0, 1] to [data_min, data_max]
            scale = dmin + scale * data_range
        # Case 4: Absolute scaling
        case (scale, "absolute") if len(scale) == 1:
            scale = np.array([-abs(scale[0]), abs(scale[0])])
        # Case 5: Absolute scaling with sequence: no match needed.

    # Scale values are used directly as colorbar limits
    return scale


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
        # Invert the y axis so origin is at the top. This follows the
        # convention for seismic shot gathers where time increases downward.
        if x == "y" and dtype_time_like(dtype):
            ax.invert_yaxis()


def _add_colorbar(ax, im, patch, log):
    """
    Add a colorbar with appropriate labels to the plot.
    """
    cb = ax.get_figure().colorbar(im, ax=ax, fraction=0.05, pad=0.025)
    data_type = str(patch.attrs.get("data_type", ""))
    data_units = get_quantity_str(patch.attrs.data_units) or ""
    dunits = f" ({data_units})" if (data_type and data_units) else f"{data_units}"
    if log:
        dunits = f"{dunits} - log_10"
    label = f"{data_type}{dunits}"
    cb.set_label(label)


@patch_function()
def waterfall(
    patch: PatchType,
    ax: plt.Axes | None = None,
    cmap: str = "bwr",
    scale: float | Sequence[float] | None = None,
    scale_type: Literal["relative", "absolute"] = "relative",
    interpolation: str | None = "antialiased",
    log: bool = False,
    show: bool = False,
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
        value centered around the mean of the data, a length 2 tuple
        specifying upper and lower limits, or None, which will automatically
        determine limits based on a quartile fence. (uses q1 - 1.5 * (q3 - q1)
        and q3 + 1.5 * (q3 - q1)).
    scale_type
        Controls the type of scaling specified by `scale` parameter. Options
        are:
            relative - scale based on half the dynamic range in patch
            absolute - scale based on absolute values provided to `scale`
    interpolation
        A value fed to matplotlib's imshow to handle downsampling large arrays,
        which is relevant for DAS. Usually, "antialiased" works well, but if the
        data look smeared disabling interpolation with None might help. Other
        options are available, see matplotlib's documentation for more details.
    log
        If True, visualize the common logarithm of the absolute values of patch data.
    show
        If True, show the plot, else just return axis.

    Examples
    --------
    >>> # Plot with default scaling (uses 1.5*IQR fence to exclude outliers)
    >>> import dascore as dc
    >>> from dascore.units import percent
    >>> patch = dc.get_example_patch("example_event_1").normalize("time")
    >>> _ = patch.viz.waterfall()
    >>>
    >>> # Use relative scaling with a tuple to show a specific fraction
    >>> # of data range. Scale values of (0.1, 0.9) map to 10% and 90%
    >>> # of the data's [min, max] range
    >>> _ = patch.viz.waterfall(scale=0.1, scale_type="relative")
    >>> # Likewise, percent units can be used for additional clarity
    >>> _ = patch.viz.waterfall(scale=10*percent, scale_type="absolute")
    >>>
    >>> # Use relative scaling with a tuple to show the middle 80% of data range
    >>> # Scale values of (0.1, 0.9) map to 10% and 90% of [data_min, data_max]
    >>> _ = patch.viz.waterfall(scale=(0.1, 0.9), scale_type="relative")
    >>>
    >>> # Use absolute scaling to set specific colorbar limits
    >>> # This directly sets the colorbar limits to [-0.5, 0.5]
    >>> _ = patch.viz.waterfall(scale=(-0.5, 0.5), scale_type="absolute")
    >>>
    >>> # Visualize data on a logarithmic scale
    >>> # Useful for data spanning multiple orders of magnitude
    >>> _ = patch.viz.waterfall(log=True)
    >>>
    >>> # Compare scale types: relative vs absolute
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    >>> # Relative: 0.5 means ±50% of dynamic range around mean
    >>> _ = patch.viz.waterfall(scale=0.5, scale_type="relative", ax=ax1)
    >>> _ = ax1.set_title("Relative scaling (scale=0.5)")
    >>> # Absolute: 0.5 means colorbar limits are [-0.5, 0.5]
    >>> _ = patch.viz.waterfall(scale=0.5, scale_type="absolute", ax=ax2)
    >>> _ = ax2.set_title("Absolute scaling (scale=0.5)")
    >>>
    >>> # Undo Y axis inversion which occurs when time is on the Y
    >>> ax = patch.viz.waterfall()
    >>> ax.invert_yaxis()

    Notes
    -----
    - The Y axis is automatically inverted if it is "time-like". This is to
      be consistent with standard seismic plotting convention. If you don't
      want this, simply invert the y axis of the returned axis object as
      shown in the example section.

    - Changes to default scale behavior: Until DASCore version 0.1.13, the
      default behavior when scale=None was to scale along the entire range of
      the data. However, very often in real data a few anomalously large or
      small values would obscure most of the patch details. In version 0.1.13
      the default behavior is to now use a statistical fence to avoid the
      problem. To get the old behavior, simply set scale=1.0.
    """
    # Validate inputs
    patch = _validate_patch_dims(patch)
    # Setup axes and data
    ax = _get_ax(ax)
    cmap = _get_cmap(cmap)
    data = np.log10(np.absolute(patch.data)) if log else patch.data
    dims = patch.dims
    dims_r = tuple(reversed(dims))
    coords = {dim: patch.coords.get_array(dim) for dim in dims}
    # Plot using imshow and set colorbar limits
    extents = _get_extents(dims_r, coords)
    scale = _get_scale(scale, scale_type, data)
    with mpl.rc_context({"image.resample": True}):
        im = ax.imshow(
            data,
            extent=extents,
            aspect="auto",
            cmap=cmap,
            origin="lower",
            # Note: these parameters are so that matplotlib versions > 3.10 behave
            # like matplotlib < 3.9, which tends to work better for visualizing big
            # DAS data. See #512. We might consider making these parameters of the
            # waterfall function in the future.
            interpolation=interpolation,
            interpolation_stage="data",
        )
    if scale is not None and len(scale) == 2 and np.all(np.isfinite(scale)):
        im.set_clim(np.asarray(scale))
    # Format axis labels and handle time-like dimensions
    _format_axis_labels(ax, patch, dims_r)
    # Add colorbar if requested
    if cmap is not None:
        _add_colorbar(ax, im, patch, log)
    if show:
        plt.show()
    return ax
