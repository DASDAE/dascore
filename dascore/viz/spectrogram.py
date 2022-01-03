"""
A spectrogram visualization
"""
from typing import Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram

from dascore.constants import PatchType
from dascore.utils.patch import patch_function
from dascore.utils.plotting import _get_ax, _get_cmap


def _get_1d_array_dim_name(patch):
    """
    Return the flattened (1D) array and dim name.
    Raises a Value Error array is higher than 1D.
    """
    data = patch.data
    if len(data.shape) == 1:  # if array only has one dimension.
        return data, patch.dims[0]
    shape = np.array(data.shape)
    not_one_len = shape > 1
    if np.sum(not_one_len) != 1:
        msg = f"Spectrogram requires 1D patch, not {data.shape}"
        raise ValueError(msg)
    name = patch.dims[np.argmax(not_one_len)]
    out = data.flatten()
    return out, name


@patch_function()
def spectrogram(
    patch: PatchType,
    ax: Optional[plt.Axes] = None,
    cmap="bwr",
    log=False,
    show=False,
) -> plt.Axes:
    """
    Plot a spectrogram of a patch along a collapsed dimension.

    Parameters
    ----------
    ax
        If not None, an axis on which to plot.
    cmap
        A matplotlob color map or code.
    show
        If True call plt.show() else just return axis.
    """
    data, name = _get_1d_array_dim_name(patch)
    d_dim = patch.attrs[f"d_{name}"]
    if name == "time":
        d_dim = d_dim / np.timedelta64(1, "s")
    freqs, dim_x, spec = scipy_spectrogram(data, 1 / d_dim)
    ax = _get_ax(ax)
    norm_class = colors.LogNorm if log else colors.Normalize
    norm = norm_class(vmin=np.min(spec), vmax=np.max(spec))
    cmap = _get_cmap(cmap)
    ax.pcolormesh(dim_x, freqs, spec, shading="gouraud", cmap=cmap, norm=norm)
    ax.set_ylabel(f"Frequency ({name}) [Hz]")
    ax.set_xlabel(f"{name.capitalize()}")
    if name == "time":
        time_str = str(patch.attrs["time_min"]).split(".")[0]
        ax.set_xlabel(f"{name.capitalize()} from {time_str}")
    if show:
        plt.show()
    return ax
