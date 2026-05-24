"""A visualization of spectra."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib import ticker

from dascore.constants import PatchType
from dascore.utils.patch import patch_function
from dascore.utils.plotting import _get_dim_label


@patch_function()
def spectraplot(
    patch: PatchType,
    ax: plt.Axes | None = None,
    cmap="turbo",
    scale: float | Sequence[float] | None = (0.5, 1),
    scale_type: Literal["relative", "absolute"] = "relative",
    interpolation: str | None = "bilinear",
    log=False,
    show=False,
    **kwargs,
) -> plt.Axes:
    """
    Plot spectra of a patch
    """
    ax = patch.viz.waterfall(
        cmap=cmap,
        ax=ax,
        show=show,
        scale=scale,
        scale_type=scale_type,
        interpolation=interpolation,
        log=False,
    )

    """
    Customisations
    """
    dims = patch.coords.dims
    is_fft_dim = [d.startswith("ft_") for d in dims]
    if all(x is False for x in is_fft_dim):
        raise Exception("Patch does not contain a Fourier-transformed coordinate")

    fft_axis = is_fft_dim.index(True)
    fft_dim = dims[fft_axis]

    """
    Format axis labels.
    """
    for dim, x in zip(dims, ["y", "x"], strict=True):
        if dim != fft_dim:
            continue

        # replace labels with publication-ready descriptors
        label = getattr(ax, f"get_{x}label")()
        label = _get_dim_label(patch, dim)
        label = label.replace("Ft_time", "Frequency").replace(
            "Ft_distance", "Wavennumber"
        )
        getattr(ax, f"set_{x}label")(label)

        if log:
            lim = list(getattr(ax, f"get_{x}lim")())
            lim[0] = patch.get_coord(dim).step
            getattr(ax, f"set_{x}scale")("log")
            getattr(ax, f"set_{x}lim")(lim)  # set axis-limits tight

            # Make ticks human-readable
            formatter = ticker.FuncFormatter(lambda tick, _: f"{tick:g}")
            # formatter = ticker.ScalarFormatter()
            getattr(ax, f"{x}axis").set_major_formatter(formatter)
            # ax.autoscale(enable=True, axis=x, tight=True)

    return ax
