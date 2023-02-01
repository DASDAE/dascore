"""
dascore Filtering module.

Much of this code was inspired by ObsPy's filtering module created by:
Tobias Megies, Moritz Beyreuther, Yannik Behr
"""

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import iirfilter, medfilt2d, sosfilt, sosfiltfilt, zpk2sos

import dascore
from dascore.constants import PatchType
from dascore.exceptions import FilterValueError
from dascore.utils.patch import patch_function


def _check_filter_kwargs(kwargs):
    """Check filter kwargs and return dim name and filter range."""
    if len(kwargs) != 1:
        msg = "pass filter requires you specify one dimension and filter range."
        raise FilterValueError(msg)
    dim = list(kwargs.keys())[0]
    filt_range = kwargs[dim]
    if not isinstance(filt_range, Sequence) or len(filt_range) != 2:
        msg = f"filter range must be a length two sequence not {filt_range}"
        raise FilterValueError(msg)
    filt1, filt2 = filt_range
    if all([pd.isnull(x) for x in filt_range]):
        msg = (
            f"pass filter requires at least one filter limit, "
            f"you passed {filt_range}"
        )
        raise FilterValueError(msg)

    return dim, filt1, filt2


def _check_sobel_args(dim, mode, cval):
    """Check Sobel filter kwargs and return"""
    mode_options = {
        "reflect",
        "constant",
        "nearest",
        "mirror",
        "wrap",
        "grid-constant",
        "grid-mirror",
        "grid-wrap",
    }
    if not isinstance(dim, str):
        msg = "dim parameter should be a string."
        raise FilterValueError(msg)
    if not isinstance(mode, str):
        msg = "mode parameter should be a string."
        raise FilterValueError(msg)
    if not isinstance(cval, (float, int)):
        msg = "cval parameter should be a float or an int."
        raise FilterValueError(msg)
    if mode not in mode_options:
        msg = f"The valid values for modes are {mode_options}."
        raise FilterValueError(msg)

    return dim, mode, cval


def _get_sampling_rate(patch, dim):
    """Get sampling rate, as a float from sampling period along a dimension."""
    d_dim = patch.attrs[f"d_{dim}"]
    if dim == "time":  # get time in to seconds
        d_dim = d_dim / np.timedelta64(1, "s")
    return 1.0 / d_dim


def _check_filter_range(niquest, low, high, filt_min, filt_max):
    """Simple check on filter parameters."""
    # ensure filter bounds are within niquest
    if low is not None and ((0 > low) or (low > 1)):
        msg = f"possible filter bounds are [0, {niquest}] you passed {filt_min}"
        raise FilterValueError(msg)
    if high is not None and ((0 > high) or (high > 1)):
        msg = f"possible filter bounds are [0, {niquest}] you passed {filt_max}"
        raise FilterValueError(msg)
    if high is not None and low is not None and high <= low:
        msg = (
            "Low filter param must be less than high filter param, you passed:"
            f"filt_min = {filt_min}, filt_max = {filt_max}"
        )
        raise FilterValueError(msg)


def _get_sos(sr, filt_min, filt_max, corners):
    """Get second order sections from sampling rate and filter bounds."""
    niquest = 0.5 * sr
    low = None if pd.isnull(filt_min) else filt_min / niquest
    high = None if pd.isnull(filt_max) else filt_max / niquest
    _check_filter_range(niquest, low, high, filt_min, filt_min)

    if (low is not None) and (high is not None):  # apply bandpass
        z, p, k = iirfilter(
            corners, [low, high], btype="band", ftype="butter", output="zpk"
        )
    elif low is not None:
        z, p, k = iirfilter(
            corners, low, btype="highpass", ftype="butter", output="zpk"
        )
    else:
        assert high is not None
        z, p, k = iirfilter(
            corners, high, btype="lowpass", ftype="butter", output="zpk"
        )
    return zpk2sos(z, p, k)


@patch_function()
def pass_filter(patch: PatchType, corners=4, zerophase=True, **kwargs) -> PatchType:
    """
    Apply a Butterworth pass filter (bandpass, highpass, or lowpass).

    Parameters
    ----------
    corners
        The number of corners for the filter.
    zerophase
        If True, apply the filter twice.
    **kwargs
        Used to specify the dimension and frequency, wavelength, or equivilent
        limits.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()

    >>>  # 1. Apply bandpass filter along time axis from 1 to 100 Hz
    >>> bandpassed = pa.pass_filter(time=(1, 100))

    >>>  # 2. Apply lowpass filter along distance axis for wavelengths less than 100m
    >>> lowpassed = pa.pass_filter(distance=(None, 1/100))
    """
    dim, filt_min, filt_max = _check_filter_kwargs(kwargs)
    axis = patch.dims.index(dim)
    sr = _get_sampling_rate(patch, dim)
    # get niquest and low/high in terms of niquest
    sos = _get_sos(sr, filt_min, filt_max, corners)
    if zerophase:
        out = sosfiltfilt(sos, patch.data, axis=axis)
    else:
        out = sosfilt(sos, patch.data, axis=axis)
    return dascore.Patch(
        data=out, coords=patch.coords, attrs=patch.attrs, dims=patch.dims
    )


@patch_function()
def sobel_filter(patch: PatchType, dim: str, mode="reflect", cval=0.0) -> PatchType:
    """
    Apply a Sobel filter.

    Parameters
    ----------
    **kwargs
        Used to specify the dimension, mode, and for the 'constant' mode, cval.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()

    >>>  # 1. Apply Sobel filter using the default parameter values.
    >>> sobel_default = pa.sobel_filter(dim='time', mode='reflect', cval=0.0)

    >>>  # 2. Apply Sobel filter with arbitrary parameter values.
    >>> sobel_arbitrary = pa.sobel_filter(dim='time', mode='constant', cval=1)

    >>> # 3. Apply Sobel filter along both axes
    >>> sobel_time_space = pa.sobel_filter('time',).sobel_filter('distance')
    """
    dim, mode, cval = _check_sobel_args(dim, mode, cval)
    axis = patch.dims.index(dim)
    out = ndimage.sobel(patch.data, axis=axis, mode=mode, cval=cval)
    return dascore.Patch(
        data=out, coords=patch.coords, attrs=patch.attrs, dims=patch.dims
    )


#
# @patch_function()
# def stop_filter(patch: PatchType, corners=4, zerophase=True, **kwargs) -> PatchType:
#     """
#     Apply a Butterworth band stop filter or (highpass, or lowpass).
#
#     Parameters
#     ----------
#     corners
#         The number of corners for the filter.
#     zerophase
#         If True, apply the filter twice.
#     **kwargs
#         Used to specify the dimension and frequency, wavenumber, or equivalent
#         limits.
#
#
#     """
#     dim, filt_min, filt_max = _check_filter_kwargs(kwargs)
#     axis = patch.dims.index(dim)
#     sr = _get_sampling_rate(patch, dim)
#     # get niquest and low/high in terms of niquest
#     sos = _get_sos(sr, filt_min, filt_max, corners)
#     out = sosfilt(sos, patch.data, axis=axis)
#     if zerophase:
#         out = sosfilt(sos, out[::-1])[::-1]
#     return dascore.Patch(data=out, coords=patch.coords, attrs=patch.attrs)


@patch_function()
def median_filter(patch: PatchType, kernel_size=3) -> PatchType:
    """
    Apply 2-D median filter

    Parameters
    ----------
    kernel_size: array_like, optional
        A scalar or a list of length 2, giving the size of the median filter window
        in each dimension. Elements of kernel_size should be odd. If kernel_size is
        a scalar, then this scalar is used as the size in each dimension. Default is
        a kernel of size (3, 3).

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()

    >>>  # 1. Apply median filter with 9 time intervals and 5 channels
    >>> filtered_pa = pa.median_filter((9,5))

    Written by Ge Jin (gjin@mines.edu)

    """
    # get niquest and low/high in terms of niquest
    out = medfilt2d(patch.data, kernel_size=kernel_size)
    return dascore.Patch(
        data=out, coords=patch.coords, attrs=patch.attrs, dims=patch.dims
    )
