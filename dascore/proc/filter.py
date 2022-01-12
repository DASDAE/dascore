"""
dascore Filtering module.

Much of this code was inspired by ObsPy's filtering module created by:
Tobias Megies, Moritz Beyreuther, Yannik Behr
"""

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import cheb2ord, cheby2, iirfilter, sosfilt, sosfiltfilt, zpk2sos

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
    """
    Get second order sections from sampling rate and filter bounds.
    """
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

    >>>  # 2. Apply lowpass filter along distance axis for wavelengths less than 1000m
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
    return dascore.Patch(data=out, coords=patch.coords, attrs=patch.attrs)


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


def _lowpass_cheby_2(data, freq, df, maxorder=12, axis=0):
    """
    Cheby2-Lowpass Filter used for pre-conditioning decimation.

    Based on Obspy's implementation found here:
    https://docs.obspy.org/master/_modules/obspy/signal/filter.html#lowpass_cheby_2
    """
    nyquist = df * 0.5
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 96, 1e99
    ws = freq / nyquist  # stop band frequency
    wp = ws  # pass band frequency
    # raise for some bad scenarios
    while True:
        if order <= maxorder:
            break
        wp = wp * 0.99
        order, wn = cheb2ord(wp, ws, rp, rs, analog=0)
    z, p, k = cheby2(order, rs, wn, btype="low", analog=0, output="zpk")
    sos = zpk2sos(z, p, k)
    return sosfilt(sos, data, axis=axis)
