"""A visualization of spectra."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib import ticker

from dascore.constants import PatchType
from dascore.exceptions import CoordError
from dascore.utils.patch import patch_function
from dascore.utils.plotting import _get_dim_label

label_replacements = {"Ft_time": "Frequency", "Ft_distance": "Wavenumber"}


@patch_function()
def specplot(
    patch: PatchType,
    ax: plt.Axes | None = None,
    cmap=None,
    scale: float | Sequence[float] | None = (0, 1),
    scale_type: Literal["relative", "absolute"] = "relative",
    interpolation: str | None = "bilinear",
    log: bool = False,
    show: bool = False,
    **kwargs,
) -> plt.Axes:
    """
    Plot the spectrum contained in a Fourier-transformed patch.

    This function wraps :meth:`Patch.viz.waterfall` and automatically
    identifies the Fourier-transformed coordinate. The corresponding axis label
    is replaced with a publication-friendly descriptor (e.g. ``Frequency`` or
    ``Wavenumber``). Optionally, the Fourier axis can be displayed
    on a logarithmic scale.

    Parameters
    ----------
    patch
        The patch containing spectral data. At least one coordinate must
        represent a Fourier-transformed dimension (``ft_*``).
    ax
        Existing matplotlib axes to draw on. If omitted, a new axes is
        created.
    cmap
        Colormap passed to [waterfall](`dascore.viz.waterfall`).
    scale
        Scaling limits passed to [waterfall](`dascore.viz.waterfall`).
        Default is [0, 1], showing the full data ran
    scale_type
        Scaling mode passed to [waterfall](`dascore.viz.waterfall`).
    interpolation
        Interpolation method used for image rendering. The default here is ``bilinear``
        for a smoother look than waterfall's default ``antialiased``
    log
        If True, display the Fourier-transformed axis on a logarithmic
        scale.
        For a distance coordinate, positive and negative wavenumbers are shown, while
        for a time coordinate only positive frequencies are shown.
    show
        If True, show the plot, else just return axis.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the spectrum plot.




    Examples
    --------
    Plot a frequency spectrum:

    >>> import dascore as dc
    >>> patch = dc.examples.example_event_2().decimate(time=10).T
    >>> spec = patch.dft("time").abs()
    >>> ax = spec.viz.specplot(cmap='turbo')

    Compute a frequency-wavenumber spectrum and display results on a logarithmic scale:

    >>> fk_patch = patch.dft(("time", "distance")).abs()
    >>> ax = fk_patch.viz.specplot(log=True, cmap='inferno')
    """
    ax = patch.viz.waterfall(
        cmap=cmap,
        ax=ax,
        scale=scale,
        scale_type=scale_type,
        interpolation=interpolation,
        log=False,
        show=False,
    )

    # check if patch actually has a fourier-transformed dimension
    dims = patch.coords.dims
    is_fft_dim = [d.startswith("ft_") for d in dims]
    if not any(is_fft_dim):
        raise CoordError("Patch does not contain a Fourier-transformed coordinate")

    # Format axis labels.
    fft_dims = {d for d in dims if d.startswith("ft_")}
    for dim, axis_name in zip(dims, ["y", "x"], strict=True):
        if dim not in fft_dims:
            continue

        # replace labels with publication-ready descriptors
        label = _get_dim_label(patch, dim)
        for key, value in label_replacements.items():
            label = label.replace(key, value)

        getattr(ax, f"set_{axis_name}label")(label)

        if log:
            # prevent changes in axis limits
            step = patch.get_coord(dim).step
            if step <= 0:
                raise ValueError(
                    f"Cannot use log scale for coordinate '{dim}' with step={step}"
                )
            lim = list(getattr(ax, f"get_{axis_name}lim")())

            if dim == "ft_distance":
                getattr(ax, f"set_{axis_name}scale")(
                    "symlog", linscale=0.5, linthresh=step * 20
                )
            else:
                getattr(ax, f"set_{axis_name}scale")("log")
                lim[0] = step
                getattr(ax, f"set_{axis_name}lim")(lim)  # set axis-limits tight

            # Make ticks human-readable
            formatter = ticker.FuncFormatter(lambda tick, _: f"{tick:g}")
            getattr(ax, f"{axis_name}axis").set_major_formatter(formatter)

    if show:
        plt.show()
    return ax
