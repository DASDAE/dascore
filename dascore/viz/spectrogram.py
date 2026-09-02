"""A spectrogram visualization: a short-time Fourier transform, drawn."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.units import Quantity, percent
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


def _spectrogram_patch(
    patch: PatchType, dim: str, other_dim: str | None, aggr_domain: str, **stft_kwargs
):
    """
    Return the power the spectrogram draws: |STFT|², one other dimension averaged.

    The other dimension is averaged before the transform (`aggr_domain="time"`)
    or after it (`"frequency"`); a one-sample dimension is squeezed either way.
    """
    if other_dim is None:
        return patch.stft(**stft_kwargs).abs() ** 2
    if aggr_domain == "time":
        averaged = patch.aggregate(other_dim, method="mean", dim_reduce="squeeze")
        return averaged.stft(**stft_kwargs).abs() ** 2
    if aggr_domain == "frequency":
        power = (patch.stft(**stft_kwargs).abs() ** 2).squeeze()
        # A length one other_dim is squeezed out above, and a dimension
        # of one sample has nothing to average over anyway.
        if other_dim in power.dims:
            power = power.aggregate(other_dim, method="mean").squeeze()
        return power
    msg = f"The aggr_domain '{aggr_domain}' should be 'time' or 'frequency'."
    raise ValueError(msg)


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
    *,
    taper_window: Any = "hann",
    overlap: Quantity | int | None = 50 * percent,
    nfft: int | Quantity | None = None,
    samples: bool = False,
    detrend: bool = False,
    **kwargs,
) -> plt.Axes:
    """
    Plot a spectrogram of a patch.

    Parameters
    ----------
    patch : PatchType
        The Patch object.
    ax : matplotlib.axes.Axes or None, optional
        A matplotlib axis object. If None, creates a new axis.
    dim : str, optional
        Dimension along which the spectrogram is being plotted.
        Default is "time".
    aggr_domain : str, optional
        "time" or "frequency" in which the mean value of the other
        dimension is calculated. No need to specify if the other
        dimension's coordinate size is 1. Default is "frequency".
    cmap : str or matplotlib.colors.Colormap, optional
        A matplotlib colormap string or instance. Set to None to not plot the
        colorbar. Default is "bwr".
    scale : float, tuple of floats, or None, optional
        If not None, controls the saturation level of the colorbar.
        Values can be a single float or a length-2 tuple specifying upper
        and lower limits. See `scale_type` for more details.
    scale_type : {"relative", "absolute"}, optional
        Specifies the type of scaling:
            - "relative": Scale based on half the dynamic range in the patch.
            - "absolute": Scale based on absolute values provided to `scale`.
        Default is "relative".
    log : bool, optional
        If True, visualize the common logarithm of the absolute values of patch data.
    show : bool, optional
        If True, show the plot. Otherwise, just return the axis.
    taper_window, overlap, nfft, samples, detrend
        Passed to [Patch.stft](`dascore.Patch.stft`), and read as it reads them.
    **kwargs
        The window, as [Patch.stft](`dascore.Patch.stft`) takes it: the
        dimension and its length, such as ``time=0.5`` (seconds) or
        ``time=256, samples=True``. With none given, a 256 sample window along
        `dim`.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Create a spectrogram plot
    >>> ax = patch.viz.spectrogram(show=False)
    >>>
    >>> # Half second windows, zero padded to 512 point FFTs, log colours.
    >>> ax = patch.viz.spectrogram(time=0.5, nfft=512, log=True)

    Notes
    -----
    This is [Patch.stft](`dascore.Patch.stft`) followed by
    [Patch.viz.waterfall](`dascore.viz.waterfall`), with one other
    dimension averaged away, and the values drawn are |STFT|² in the scaling
    `stft` uses. Before DASCore 0.1.22 it called `scipy.signal.spectrogram`
    directly, which differed in more than scaling: it removed the mean of
    each window (`detrend="constant"`), tapered with a ``("tukey", 0.25)``
    window, overlapped by an eighth of the window, took no windows past the
    ends of the data, and spelled its arguments as scipy does (`nperseg`,
    `noverlap`). Now the taper is hann, the overlap half, windows reach the
    ends as `stft`'s do, nothing is detrended unless `detrend=True` is
    passed through, and the window, overlap, taper and FFT length are given
    as `stft` takes them.
    """
    dims = patch.dims
    if len(dims) > 2 or len(dims) < 1:
        raise ValueError("Can only make spectrogram of 1D or 2D patches.")
    other_dim = _get_other_dim(dim, dims)
    if not kwargs:
        # scipy's old default, or the whole of a shorter patch.
        kwargs, samples = {dim: min(256, len(patch.get_coord(dim)))}, True
    elif dim not in kwargs:
        msg = (
            f"The window is given along {sorted(kwargs)} but the spectrogram is "
            f"of {dim!r}; give the window as {dim}=..."
        )
        raise ParameterError(msg)
    spec = _spectrogram_patch(
        patch,
        dim,
        other_dim,
        aggr_domain,
        taper_window=taper_window,
        overlap=overlap,
        nfft=nfft,
        samples=samples,
        detrend=detrend,
        **kwargs,
    )
    return spec.viz.waterfall(
        ax=ax, cmap=cmap, scale=scale, scale_type=scale_type, log=log, show=show
    )
