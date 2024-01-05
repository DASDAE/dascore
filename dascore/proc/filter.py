"""
dascore Filtering module.

Much of this code was inspired by ObsPy's filtering module created by:
Tobias Megies, Moritz Beyreuther, Yannik Behr
"""
from __future__ import annotations

import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter as np_gauss
from scipy.ndimage import median_filter as nd_median_filter
from scipy.signal import iirfilter, sosfilt, sosfiltfilt, zpk2sos
from scipy.signal import savgol_filter as np_savgol_filter

import dascore
from dascore.constants import PatchType, samples_arg_description
from dascore.exceptions import FilterValueError
from dascore.units import get_filter_units
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import check_filter_kwargs, check_filter_range
from dascore.utils.patch import (
    get_dim_sampling_rate,
    get_multiple_dim_value_from_kwargs,
    patch_function,
)


def _check_sobel_args(dim, mode, cval):
    """Check Sobel filter kwargs and return."""
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
    if not isinstance(cval, float | int):
        msg = "cval parameter should be a float or an int."
        raise FilterValueError(msg)
    if mode not in mode_options:
        msg = f"The valid values for modes are {mode_options}."
        raise FilterValueError(msg)

    return dim, mode, cval


def _get_sos(sr, filt_min, filt_max, corners):
    """Get second order sections from sampling rate and filter bounds."""
    nyquist = 0.5 * sr
    low = None if pd.isnull(filt_min) else filt_min / nyquist
    high = None if pd.isnull(filt_max) else filt_max / nyquist
    check_filter_range(nyquist, low, high, filt_min, filt_max)

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
        Used to specify the dimension and frequency, wavelength, or equivalent
        limits.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()

    >>>  # Apply bandpass filter along time axis from 1 to 100 Hz
    >>> bandpassed = pa.pass_filter(time=(1, 100))

    >>>  # Apply lowpass filter along distance axis for wavelengths less than 100m
    >>> lowpassed = pa.pass_filter(distance=(None, 1/100))
    >>> # Note that None and ... both indicate open intervals
    >>> assert pa.pass_filter(time=(None, 90)) == pa.pass_filter(time=(..., 90))

    >>> # Optionally, units can be specified for a more expressive API.
    >>> from dascore.units import m, ft, s, Hz
    >>> # Filter from 1 Hz to 10 Hz in time dimension
    >>> lp_units = pa.pass_filter(time=(1 * Hz, 10 * Hz))
    >>> # Filter wavelengths 50m to 100m
    >>> bp_m = pa.pass_filter(distance=(50 * m, 100 * m))
    >>> # filter wavelengths less than 200 ft
    >>> lp_ft = pa.pass_filter(distance=(200 * ft, ...))
    """
    dim, (arg1, arg2) = check_filter_kwargs(kwargs)
    axis = patch.dims.index(dim)
    coord_units = patch.coords.coord_map[dim].units
    filt_min, filt_max = get_filter_units(arg1, arg2, to_unit=coord_units, dim=dim)
    sr = get_dim_sampling_rate(patch, dim)
    # get nyquist and low/high in terms of nyquist
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
    dim
        The dimension along which to apply
    mode
        Determines how the input array is extended when the filter
        overlaps with a border.
    cval
        Fill value when mode="constant".

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()

    >>>  # 1. Apply Sobel filter using the default parameter values.
    >>> sobel_default = pa.sobel_filter(dim='time', mode='reflect', cval=0.0)

    >>>  # 2. Apply Sobel filter with arbitrary parameter values.
    >>> sobel_arbitrary = pa.sobel_filter(dim='time', mode='constant', cval=1)

    >>> # 3. Apply Sobel filter along both axes
    >>> sobel_time_space = pa.sobel_filter('time').sobel_filter('distance')
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
#     # get nyquist and low/high in terms of nyquist
#     if zerophase:


def _create_size_and_axes(patch, kwargs, samples):
    """
    Return a tuple of (size) and (axes).

    Note: size will always have the same size as the patch, but
    1s will be used if axis is not used.
    """
    dimfo = get_multiple_dim_value_from_kwargs(patch, kwargs)
    axes = [x["axis"] for x in dimfo.values()]
    size = [1] * len(patch.dims)
    for dim, info in dimfo.items():
        axis = info["axis"]
        coord = patch.get_coord(dim)
        window = coord.get_sample_count(info["value"], samples=samples)
        size[axis] = window
    return tuple(size), tuple(axes)


@patch_function()
@compose_docstring(sample_explination=samples_arg_description)
def median_filter(
    patch: PatchType, samples=False, mode="reflect", cval=0.0, **kwargs
) -> PatchType:
    """
    Apply 2-D median filter.

    Parameters
    ----------
    patch
        The patch to filter
    samples
        {sample_explination}
    mode
        The mode for handling edges.
    cval
        The constant value for when mode == constant.
    **kwargs
        Used to specify the shape of the median filter in each dimension.
        See examples for more info.

    Examples
    --------
    >>> import dascore
    >>> from dascore.units import m, s
    >>> pa = dascore.get_example_patch()

    >>>  # 1. Apply median filter only over time dimension with 0.10 sec window
    >>> filtered_pa_1 = pa.median_filter(time=0.1)

    >>>  # 2. Apply median filter over both time and distance
    >>>  # using a 0.1 second time window and 2 m distance window
    >>> filtered_pa_2 = pa.median_filter(time=0.1 * s, distance=2 * m)

    >>>  # 3. Apply median filter with 3 time samples and 4 distance samples
    >>> filtered_pa = pa.median_filter(
    ...     time=3, distance=4, samples=True,
    ... )

    Notes
    -----
    See scipy.ndimage.median_filter for more info on implementation
    and arguments.

    Values specified with kwargs should be small, for example < 10 samples
    otherwise this can take a long time and use lots of memory.
    """
    size, _ = _create_size_and_axes(patch, kwargs, samples)
    new_data = nd_median_filter(patch.data, size=size, mode=mode, cval=cval)
    return patch.update(data=new_data)


@patch_function()
@compose_docstring(sample_explination=samples_arg_description)
def savgol_filter(
    patch: PatchType, polyorder, samples=False, mode="interp", cval=0.0, **kwargs
) -> PatchType:
    """
    Applies Savgol filter along spenfied dimensions.

    The filter will be applied over each selected dimension sequentially.

    Parameters
    ----------
    patch
        The patch to filter
    polyorder
        Order of polynomial
    samples
        If True samples are specified
        If False coordinate of dimension
    mode
        The mode for handling edges.
    cval
        The constant value for when mode == constant.
    **kwargs
        Used to specify the shape of the savgol filter in each dimension.

    Notes
    -----
    See scipy.signal.savgol_filter for more info on implementation
    and arguments.

    Examples
    --------
    >>> import dascore
    >>> from dascore.units import m, s
    >>> pa = dascore.get_example_patch()
    >>>
    >>> # Apply second order polynomial Savgol filter
    >>> # over time dimension with 0.10 sec window.
    >>> filtered_pa_1 = pa.savgol_filter(polyorder=2, time=0.1)
    >>>
    >>> # Apply Savgol filter over distance dimension using a 5 sample
    >>> # distance window.
    >>> filtered_pa_2 = pa.median_filter(distance=5, samples=True, polyorder=2)
    >>>
    >>> # Combine distance and time filter
    >>> filtered_pa_3 = pa.savgol_filter(distance=10, time=0.1, polyorder=4)
    """
    data = patch.data
    size, axes = _create_size_and_axes(patch, kwargs, samples)
    for ax in axes:
        data = np_savgol_filter(
            x=patch.data,
            window_length=size[ax],
            polyorder=polyorder,
            mode=mode,
            cval=cval,
            axis=ax,
        )
    return patch.update(data=data)


@patch_function()
@compose_docstring(sample_explination=samples_arg_description)
def gaussian_filter(
    patch: PatchType, samples=False, mode="reflect", cval=0.0, truncate=4.0, **kwargs
) -> PatchType:
    """
    Applies a Gaussian filter along specified dimensions.

    Parameters
    ----------
    patch
        The patch to filter
    samples
        If True samples are specified
        If False coordinate of dimension
    mode
        The mode for handling edges.
    cval
        The constant value for when mode == constant.
    truncate
        Truncate the filter kernel length to this many standard deviations.
    **kwargs
        Used to specify the sigma value (standard deviation) for desired
        dimensions.

    Examples
    --------
    >>> import dascore
    >>> from dascore.units import m, s
    >>> pa = dascore.get_example_patch()
    >>>
    >>> # Apply Gaussian smoothing along time axis.
    >>> pa_1 = pa.gaussian_filter(time=0.1)
    >>>
    >>> # Apply Gaussian filter over distance dimension
    >>> # using a 3 sample standard deviation.
    >>> pa_2 = pa.gaussian_filter(samples=True, distance=3)
    >>>
    >>> # Apply filter to time and distance axis.
    >>> pa_3 = pa.gaussian_filter(time=0.1, distance=3)

    Notes
    -----
    See scipy.ndimage.gaussian_filter for more info on implementation
    and arguments.
    """
    size, axes = _create_size_and_axes(patch, kwargs, samples)
    used_size = tuple(size[x] for x in axes)
    data = np_gauss(
        input=patch.data,
        sigma=used_size,
        axes=axes,
        mode=mode,
        cval=cval,
        truncate=truncate,
    )
    return patch.update(data=data)
