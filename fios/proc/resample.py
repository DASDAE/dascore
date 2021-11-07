"""
Methods for resampling.
"""
import warnings

import numpy as np
import scipy.signal
from scipy.fftpack import hilbert
from scipy.signal import (
    cheb2ord,
    cheby2,
    convolve,
    get_window,
    iirfilter,
    remez,
    sosfilt,
    zpk2sos,
)

import fios


def decimate(
    patch: "fios.Patch", factor: int, dim: str = "time", lowpass: bool = True
) -> "fios.Patch":
    """
    Decimate a patch along a dimension.

    Parameters
    ----------
    factor
        The decimation factor (e.g., 10)
    dim
        dimension along which to decimate.
    lowpass
        If True, first apply a low-pass (anti-alis) filter.
    """
    if lowpass:
        raise NotImplementedError("working on it")
    kwargs = {dim: slice(None, None, factor)}
    out = patch._data_array.sel(**kwargs)
    return fios.Patch(out)


def detrend(patch: "fios.Patch", dim="time", type="linear") -> "fios.Patch":
    """Perform detrending along a given dimension."""
    assert dim in patch.dims
    axis = patch.dims.index(dim)
    out = scipy.signal.detrend(patch.data, axis=axis, type=type)
    return patch.new(data=out)
