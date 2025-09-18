"""
dascore Filtering module.

Much of this code was inspired by ObsPy's filtering module created by:
Tobias Megies, Moritz Beyreuther, Yannik Behr
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter as np_gauss
from scipy.ndimage import median_filter as nd_median_filter
from scipy.signal import filtfilt, iirfilter, iirnotch, sosfilt, sosfiltfilt, zpk2sos
from scipy.signal import savgol_filter as np_savgol_filter

import dascore as dc
from dascore.constants import PatchType, samples_arg_description
from dascore.exceptions import FilterValueError, ParameterError, UnitError
from dascore.units import (
    convert_units,
    get_filter_units,
    get_inverted_quant,
    invert_quantity,
    quant_sequence_to_quant_array,
)
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import (
    broadcast_for_index,
    check_filter_kwargs,
    check_filter_range,
)
from dascore.utils.patch import (
    get_dim_axis_value,
    get_dim_sampling_rate,
    patch_function,
)
from dascore.utils.time import to_float


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
        The number of corners for the filter. Default is 4.
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
    axis = patch.get_axis(dim)
    coord_units = patch.coords.coord_map[dim].units
    filt_min, filt_max = get_filter_units(arg1, arg2, to_unit=coord_units, dim=dim)
    sr = get_dim_sampling_rate(patch, dim)
    # get nyquist and low/high in terms of nyquist
    sos = _get_sos(sr, filt_min, filt_max, corners)
    if zerophase:
        out = sosfiltfilt(sos, patch.data, axis=axis)
    else:
        out = sosfilt(sos, patch.data, axis=axis)
    return dc.Patch(data=out, coords=patch.coords, attrs=patch.attrs, dims=patch.dims)


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
    axis = patch.get_axis(dim)
    out = ndimage.sobel(patch.data, axis=axis, mode=mode, cval=cval)
    return dc.Patch(data=out, coords=patch.coords, attrs=patch.attrs, dims=patch.dims)


def _create_size_and_axes(patch, kwargs, samples):
    """
    Return a tuple of (size) and (axes).

    Note: size will always have the same size as the patch, but
    1s will be used if axis is not used.
    """
    dimfo = get_dim_axis_value(patch, kwargs=kwargs, allow_multiple=True)
    axes = [x.axis for x in dimfo]
    size = [1] * len(patch.dims)
    for dim, axis, value in dimfo:
        coord = patch.get_coord(dim)
        window = coord.get_sample_count(value, samples=samples)
        size[axis] = window
    return tuple(size), tuple(axes)


@patch_function()
@compose_docstring(sample_explanation=samples_arg_description)
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
def notch_filter(patch: PatchType, q, **kwargs) -> PatchType:
    """
    Apply a second-order IIR notch digital filter on patch's data.

    A notch filter is a band-stop filter with a narrow bandwidth (high quality factor).
    It rejects a narrow frequency band and leaves the rest of the spectrum
    little changed.

    Parameters
    ----------
    patch
        The patch to filter
    q
        Quality factor (float). A higher Q value means a narrower notch,
        which targets the specific frequency more precisely.
        A lower Q results in a wider notch, meaning a broader range of
        frequencies around the specific frequency will be attenuated.
    **kwargs
        Used to specify the dimension(s) and associated frequency and/or wavelength
        (or equivalent values) for the filter.

    See Also
    --------
    [scipy.signal.iirnotch](https://docs.scipy.org/doc/scipy/reference
    /generated/scipy.signal.iirnotch.html).

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()

    >>>  # Apply a notch filter along time axis to remove 60 Hz
    >>> filtered = pa.notch_filter(time=60, q=30)

    >>>  # Apply a notch filter along distance axis to remove 5 m wavelength
    >>> filtered = pa.notch_filter(distance=0.2, q=30)

    >>>  # Apply a notch filter along both time and distance axes
    >>> filtered = pa.notch_filter(time=60, distance=0.2, q=30)

    >>> # Optionally, units can be specified for a more expressive API.
    >>> from dascore.units import m, ft, s, Hz
    >>> # Apply a notch filter along time axis to remove 60 Hz
    >>> filtered = pa.notch_filter(time=60 * Hz, q=30)
    >>> # Apply a notch filter along distance axis to remove 5 m wavelength
    >>> filtered = pa.notch_filter(distance=5 * m, q=30)
    """
    data = patch.data
    dinfo = get_dim_axis_value(patch, kwargs=kwargs, allow_multiple=True)
    for dim, axis, value in dinfo:
        coord = patch.get_coord(dim)
        # Invert units if needed
        if isinstance(value, dc.units.Quantity) and coord.units is not None:
            value, _ = get_inverted_quant(value, coord.units)
        # Check valid parameters
        w0 = to_float(value)
        sr = get_dim_sampling_rate(patch, dim)
        nyquist = 0.5 * sr
        if w0 > nyquist:
            msg = f"possible filter values are in [0, {nyquist}] you passed {w0}"
            raise FilterValueError(msg)
        b, a = iirnotch(w0, Q=q, fs=sr)
        data = filtfilt(b, a, data, axis=axis)
    return dc.Patch(data=data, coords=patch.coords, attrs=patch.attrs, dims=patch.dims)


@patch_function()
@compose_docstring(sample_explanation=samples_arg_description)
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
    >>> filtered_pa_2 = pa.savgol_filter(distance=5, samples=True, polyorder=2)
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
@compose_docstring(sample_explanation=samples_arg_description)
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


@patch_function()
@compose_docstring(sample_explanation=samples_arg_description)
def slope_filter(
    patch: PatchType,
    filt: Sequence[float],
    dims: tuple[str, str] = ("distance", "time"),
    directional: bool = False,
    notch: bool = False,
) -> PatchType:
    """
    Filter the patch over certain slopes in the 2D Fourier domain.

    Most commonly this is used as an F-K (frequency wavenumber)
    filter to attenuate energy with specified apparent velocities.

    Parameters
    ----------
    patch
        The patch to filter.
    filt
        A length 4 array of the form [va, vb, vc, vd]. If notch is False,
        the filter selects the apparent velocities between 'vb' and 'vc'
        with tapering boundaries from 'va' to 'vb' and from 'vc' to 'vd'.
    dims
        The dimensions used to determine slope. The first dim is in the
        numerator and the second in the denominator. (eg distance, time)
        represents a velocity since distance/time has units of |L|/|T|
        (commonly m/s).
    directional
        If True, the filter should be considered direction. That is to say,
        the sign of the values in `filt` indicate the direction (towards or
        away) with increasing coordinate values.
        This can be used for up/down or left/right separation, assuming a
        near-linear fiber layout.
    notch
        If True, the filter represents a notch, meaning the slopes
        specified by the inner `filt` parameters are attenuated rather
        than those outside of them.

    Examples
    --------
    >>> # Example 1: Compare slope filtered patch to Non-filtered.
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>>
    >>> import dascore as dc
    >>> from dascore.units import Hz
    >>>
    >>>
    >>> # Apply taper function and bandpass filter along time axis from 1 to 500 Hz
    >>> patch = (
    ...     dc.get_example_patch('example_event_1')
    ...     .set_units(distance='m', time='s')
    ... 	.taper(time=0.05)
    ... 	.pass_filter(time=(1*Hz, 500*Hz))
    ... )
    >>> filt = np.array([2e3,2.2e3,8e3,2e4])
    >>> # Apply fk filter
    >>> patch_filtered = patch.slope_filter(
    ...     filt=filt,
    ... 	directional=False,
    ...     notch=False
    ... )
    >>> # Plot results
    >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    >>> ax1 = patch.viz.waterfall(ax=ax1, scale=0.5)
    >>> _ = ax1.set_title('Raw')
    >>> ax2 = patch_filtered.viz.waterfall(ax=ax2, scale=0.5)
    >>> _ = ax2.set_title('Filtered')
    >>>
    >>> # Example 2: Notch filter
    >>> patch_filtered = patch.slope_filter(filt=filt, notch=True)
    >>>
    >>> # Example 3: specify units
    >>> filt = np.array([2e3,2.2e3,8e3,2e4]) * dc.get_unit("m/s")
    >>> patch_filtered = patch.slope_filter(filt=filt)

    The [FK recipe](`docs/recipes/fk.qmd`) provides addtional examples.
    """

    def _check_inputs(patch, filt, dims):
        """Ensure inputs are valid."""
        sorted_filt = np.all(filt[:-1] <= filt[1:])
        if not (sorted_filt and len(filt) == 4):
            msg = f"filt must be a sorted length 4 sequence. You passed {filt}"
            raise ParameterError(msg)
        if missing := set(dims) - set(patch.coords.coord_map):
            msg = f"Cant apply slope filter. {missing} are missing from patch."
            raise ParameterError(msg)

    def _get_taper_mask(filt, slope, notch):
        """Get a mask for applying taper and attenuation."""
        fac = np.where(
            (slope >= filt[0]) & (slope <= filt[1]),
            1.0 - np.sin(0.5 * np.pi * (slope - filt[0]) / (filt[1] - filt[0])),
            1.0,
        )
        fac = np.where((slope >= filt[1]) & (slope <= filt[2]), 0.0, fac)
        fac = np.where(
            (slope >= filt[2]) & (slope <= filt[3]),
            np.sin(0.5 * np.pi * (slope - filt[2]) / (filt[3] - filt[2])),
            fac,
        )
        fac = fac if notch else 1.0 - fac
        return fac

    def _get_slope_array(dft_patch, directional, freq_dims):
        """Get an array which specifies slope."""
        dim1, dim2 = freq_dims[-1], freq_dims[-2]
        dims = dft_patch.dims
        ndims = dft_patch.ndim
        coord1 = dft_patch.get_array(dim1)
        coord2 = dft_patch.get_array(dim2) + sys.float_info.epsilon
        # Need to add appropriate blank dims to keep overall shape of patch.
        ax1, ax2 = dims.index(dim1), dims.index(dim2)
        shape_1 = broadcast_for_index(ndims, ax1, value=slice(None), fill=None)
        shape_2 = broadcast_for_index(ndims, ax2, value=slice(None), fill=None)
        # Then just allow broadcasting to do its magic
        slope = coord1[shape_1] / coord2[shape_2]
        if not directional:
            slope = np.abs(slope)
        return slope

    def _maybe_transform_units(filt, dft_patch, freq_dims):
        """Handle units on filter."""
        # Hand the units/partial units in sequence.
        units = getattr(filt, "units", None)
        try:
            filt = np.array(filt)
        except ValueError:
            filt = quant_sequence_to_quant_array(filt)
        if units:
            filt = filt * dc.get_quantity(units)
        if not isinstance(filt, dc.units.Quantity):
            return filt
        array, units = filt.magnitude, filt.units
        coord_unit_1 = dft_patch.get_coord(freq_dims[-1]).units
        coord_unit_2 = dft_patch.get_coord(freq_dims[-2]).units
        if not (coord_unit_1 and coord_unit_2):
            msg = (
                f"Units of {units} specified in Patch.slope_filter, but units "
                f"are not defined for both specified dimensions: {dims}."
            )
            raise UnitError(msg)
        new_units = coord_unit_1 / coord_unit_2
        # Determine if we need to flip units.
        if new_units.dimensionality == (1 / units).dimensionality:
            array, units = np.sort(1 / array), invert_quantity(units)
        out = convert_units(array, new_units, units)
        return out

    _check_inputs(patch, filt, dims)
    freq_dims = tuple(f"ft_{x}" for x in dims)
    dft_patch = patch.dft.func(patch, dims)
    transformed = patch is not dft_patch

    slope = _get_slope_array(dft_patch, directional, freq_dims)
    filt = _maybe_transform_units(filt, dft_patch, freq_dims)

    mask = _get_taper_mask(filt, slope, notch)
    new_data = dft_patch.data * mask
    out = dft_patch.update(data=new_data)
    if transformed:
        out = out.idft().real()
    return out
