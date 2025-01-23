"""A spectrogram visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt

from dascore.constants import PatchType
from dascore.utils.patch import patch_function


def _get_other_dim(dim, dims):
    if not isinstance(dim, str):
        raise TypeError(f"Expected 'dim' to be a string, got {type(dim).__name__}.")
    if dim not in dims:
        raise ValueError(f"The dimension '{dim}' is not in patch's dimensions {dims}.")
    if len(dims) == 1:
        return None
    else:
        return dims[0] if dims[1] == dim else dims[1]


@patch_function()
def spectrogram(
    patch: PatchType,
    ax: plt.Axes | None = None,
    dim="time",
    aggr_domain="frequency",
    cmap="bwr",
    scale: float | Sequence[float] | None = None,
    scale_type: Literal["relative", "absolute"] = "relative",
    log=False,
    show=False,
) -> plt.Axes:
    """
    Plot a spectrogram of a patch.

    Parameters
    ----------
    patch
        The Patch object.
    ax
        A matplotlib object, if None create one.
    dim
        Dimension along which spectogram is being plotted.
        Default is "time"
    aggr_domain
        "time" or "frequency" in which the mean value of the other
        dimension is caluclated. No need to specify if other dimension's
        coord size is 1.
        Default is "frequency"
    cmap
        A matplotlib colormap string or instance. Set to None to not plot the
        colorbar.
    scale
        If not None, controls the saturation level of the colorbar.
        Values can either be a float, to set upper and lower limit to the same
        value centered around the mean of the data, or a length 2 tuple
        specifying upper and lower limits. See `scale_type` for controlling how
        values are scaled.
    scale_type
        Controls the type of scaling specified by `scale` parameter. Options
        are:
            relative - scale based on half the dynamic range in patch
            absolute - scale based on absolute values provided to `scale`
    log
        If True, visualize the common logarithm of the absolute values of patch data.
    show
        If True, show the plot, else just return axis.
    """
    dims = patch.dims
    if len(dims) > 2 or len(dims) < 1:
        raise ValueError("Can only make spectogram of 1D or 2D patches.")

    other_dim = _get_other_dim(dim, dims)
    if other_dim is not None:
        if aggr_domain == "time":
            patch_aggr = patch.aggregate(other_dim, method="mean", dim_reduce="squeeze")
            spec = patch_aggr.spectrogram(dim)
        elif aggr_domain == "frequency":
            _spec = patch.spectrogram(dim).squeeze()
            spec = _spec.aggregate(other_dim, method="mean").squeeze()
        else:
            raise ValueError(
                f"The aggr_domain '{aggr_domain}' should be 'time' or 'frequency.'"
            )
    else:
        spec = patch.spectrogram(dim)
    return spec.viz.waterfall(
        ax=ax, cmap=cmap, scale=scale, scale_type=scale_type, log=log, show=show
    )
