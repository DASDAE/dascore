"""Module for waterfall plotting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.units import get_quantity_str
from dascore.utils.patch import patch_function
from dascore.utils.plotting import (
    _format_time_axis,
    _get_ax,
    _get_cmap,
    _get_dim_label,
    _get_extents,
)


def _get_scale(scale, scale_type, patch):
    """Set the scale of the color bar based on scale and scale_type."""
    # check scale parameters
    assert scale_type in {"absolute", "relative"}
    data = patch.data

    match (scale, scale_type):
        # Floating point value, relative type and zero mean
        case (scale, "relative") if isinstance(scale, float | int):
            mod = 0.5 * (np.nanmax(data) - np.nanmin(data))
            mean = np.nanmean(patch.data)
            scale = np.asarray([mean - scale * mod, mean + scale * mod])
        # No scale specified and relative, use fence.
        case (None, "relative"):
            q2, q3 = np.nanpercentile(data, [25, 75])
            diff = q3 - q2
            scale = np.asarray([q2 - diff * 1.5, q3 + diff * 1.5])
        # Here scale must be a list, tuple, or array.
        case (scale, "relative"):
            # Raise if invalid scale params.
            scale = np.array(scale)
            if np.any(scale < 0) or scale[0] > scale[1] or len(scale) != 2:
                msg = (
                    "Relative scale values cannot be negative and the first "
                    f"value must be less than the second. You passed {scale}"
                )
                raise ParameterError(msg)
            dmin, dmax = np.nanmin(data), np.nanmax(data)
            data_range = dmax - dmin
            scale = dmin + scale * data_range
    return scale


@patch_function()
def waterfall(
    patch: PatchType,
    ax: plt.Axes | None = None,
    cmap="bwr",
    scale: float | Sequence[float] | None = None,
    scale_type: Literal["relative", "absolute"] = "relative",
    log=False,
    show=False,
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
        Controls the saturation level of the colorbar. Values can be:
        - A float to set upper and lower limit to the same value.
          This will be cenetered around 0 by default.
        - A length 2 tuple to independantly set the lower and uper limit
        - None means use the 1.5*IQR fence to set the upper and lower limits.
    scale_type
        Controls the type of scaling specified by `scale` parameter. Options
        are:
        - relative - scale based on half the dynamic range in patch
        - absolute - scale based on absolute values provided to `scale`
    log
        If True, visualize the common logarithm of the absolute values of patch data.
    show
        If True, show the plot, else just return axis.

    Examples
    --------
    >>> # Plot the default patch
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> _ = patch.viz.waterfall()

    Notes
    -----
    Until DASCore version 0.1.13, the default behavior when scale=None was to
    scale along the entire range of the data. However, very often in real data
    a few anonomously large or small would obscur most of the patch details.
    In version 0.1.13 the default behavior is to now use a stiatical fence to
    avoid the problem. To get the old behavior, simply set scale=1.0.
    """
    ax = _get_ax(ax)
    cmap = _get_cmap(cmap)
    data = np.log10(np.absolute(patch.data)) if log else patch.data
    dims = patch.dims
    assert len(dims) == 2, "Can only make waterfall plot of 2D Patch"
    dims_r = tuple(reversed(dims))
    coords = {dim: patch.coords.get_array(dim) for dim in dims}
    # Plot using imshow, set colorbar limits.
    extents = _get_extents(dims_r, coords)
    scale = _get_scale(scale, scale_type, patch)
    im = ax.imshow(data, extent=extents, aspect="auto", cmap=cmap, origin="lower")
    im.set_clim(scale)

    # format all dims which have time types.
    for dim, x in zip(dims_r, ["x", "y"]):
        getattr(ax, f"set_{x}label")(_get_dim_label(patch, dim))
        coord = patch.get_coord(dim)
        if np.issubdtype(coord.dtype, np.datetime64):
            _format_time_axis(ax, dim, x)
            # Invert the y axis so origin is at the top. This is convention
            # for seismic shot gathers, but if axis is not time-like is just
            # annoying.
            if x == "y":
                ax.invert_yaxis()
    # add color bar with title
    if cmap is not None:
        cb = ax.get_figure().colorbar(im, ax=ax, fraction=0.05, pad=0.025)
        data_type = str(patch.attrs["data_type"])
        data_units = get_quantity_str(patch.attrs.data_units) or ""
        dunits = f" ({data_units})" if (data_type and data_units) else f"{data_units}"
        if log:
            dunits = f"{dunits} - log_10"
        label = f"{data_type}{dunits}"
        cb.set_label(label)
    if show:
        plt.show()
    return ax
