"""Module for re-sampling patches."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import decimate as scipy_decimate

import dascore as dc
import dascore.compat as compat
from dascore.constants import PatchType
from dascore.exceptions import FilterValueError
from dascore.units import get_filter_units
from dascore.utils.patch import (
    get_dim_axis_value,
    get_start_stop_step,
    patch_function,
)
from dascore.utils.time import to_int, to_timedelta64


def _apply_scipy_decimation(patch, factor, ftype, axis):
    """
    Apply decimation along axis.
    """
    try:
        data = scipy_decimate(patch.data, factor, ftype=ftype, axis=axis)
    except ValueError as e:
        msg = (
            "Scipy decimation failed. This can happen for dimensions with"
            "few elements. Consider setting filter_type to False. The raised "
            f"exception was {e}"
        )
        raise FilterValueError(msg)
    return data


@patch_function()
def decimate(
    patch: PatchType,
    filter_type: Literal["iir", "fir", None] = "iir",
    copy=True,
    **kwargs,
) -> PatchType:
    """
    Decimate a patch along a dimension.

    Parameters
    ----------
    filter_type
        filter type to use to avoid aliasing. Options are:
            iir - infinite impulse response
            fir - finite impulse response
            None - No pre-filtering, not recommended, may cause aliasing
    copy
        If True, copy the decimated data array. This is needed if you want
        the old array to get gc'ed to free memory otherwise a view is returned.
        Only applies when filter_type == None.
    **kwargs
        Used to pass dimension and factor. For example `time=10` is 10x
        decimation along the time axis.

    Notes
    -----
    - Simply uses scipy.signal.decimate if filter_type is specified.
      Otherwise,just slice data long specified dimension only including
      every n samples.

    - If the decimation dimension is small, this can fail due to lack of
      padding values.

    Examples
    --------
    # Simple example using iir
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> decimated_irr = patch.decimate(time=10, filter_type='iir')
    >>> # Example using fir along distance dimension
    >>> decimated_fir = patch.decimate(distance=10, filter_type='fir')
    """
    dim, axis, factor = get_dim_axis_value(patch, kwargs=kwargs)[0]
    coords, slices = patch.coords.decimate(**{dim: int(factor)})
    # Apply scipy.signal.decimate and get new coords
    if filter_type:
        data = _apply_scipy_decimation(patch.data, factor, ftype=filter_type, axis=axis)
    else:  # No filter, simply slice along specified dimension.
        data = patch.data[slices]
        # Need to copy so array isn't a slice and holds onto reference of parent
        data = np.array(data) if copy else data
    # Update delta_dim since spacing along dimension has changed.
    return patch.new(data=data, coords=coords)


@patch_function()
def interpolate(patch: PatchType, kind: str | int = "linear", **kwargs) -> PatchType:
    """
    Set coordinates of patch along a dimension using interpolation.

    Parameters
    ----------
    patch
        The patch object to which interpolation is applied.
    kind
        The type of interpolation. See Notes for more details.
        If a string, the following are supported:
            linear - linear interpolation between a pair of points.
            nearest - use the nearest sample for interpolation.
        If an int, it specifies the order of spline to use. EG 1 is a linear
            spline, 2 is quadratic, 3 is cubic, etc.

    **kwargs
        Used to specify dimension and interpolation values. Use a value of
        None to "snap" coordinate to evenly sampled points along coordinate.

    Notes
    -----
    This function just uses scipy's interp1d function under the hood.
    See scipy.interpolate.interp1d for information.

    See also [snap](`dascore.core.Patch.snap_coords`).

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # up-sample time coordinate
    >>> time = patch.coords.get_array('time')
    >>> new_time = np.arange(time.min(), time.max(), 0.5*patch.attrs.time_step)
    >>> patch_uptime = patch.interpolate(time=new_time)
    >>> # interpolate unevenly sampled dim to evenly sampled
    >>> patch = dc.get_example_patch("wacky_dim_coords_patch")
    >>> patch_time_even = patch.interpolate(time=None)
    """
    dim, axis, samples = get_dim_axis_value(patch, kwargs=kwargs)[0]
    # if samples is None, get evenly sampled coords along dimension.
    if samples is None:
        coord = patch.coords.coord_map[dim]
        samples = coord.snap().values
    # we need to make sure only real numbers are used, interp1d doesn't support
    # datetime64 yet.
    coord_num = to_int(patch.coords.get_array(dim))
    samples_num = to_int(samples)
    func = compat.interp1d(
        coord_num, patch.data, axis=axis, kind=kind, fill_value="extrapolate"
    )
    out = func(samples_num)
    cm = patch.coords
    associated_dims = cm.dim_map[dim]
    coord_new = dc.core.get_coord(data=samples)
    cm_new = cm.update(**{dim: (associated_dims, coord_new)})
    return patch.new(data=out, coords=cm_new)


@patch_function()
def resample(
    patch: PatchType, window=None, interp_kind="linear", samples=False, **kwargs
) -> PatchType:
    """
    Resample along a single dimension using Fourier Method and interpolation.

    The dimension which should be resampled is passed as kwargs. The key
    is the dimension name and the value is the new sampling period.

    Since Fourier methods only support adding or removing an integer number
    of frequency bins, the exact desired sampling rate is often not achievable
    with resampling alone. If the fourier resampling doesn't produce the exact
    result, an interpolation (see [interpolate](`dascore.proc.interpolate`))
    is used to achieve the desired sampling rate.

    Parameters
    ----------
    patch
        The patch to resample.
    window
        The Fourier-domain window that tapers the Fourier spectrum. See
        scipy.signal.resample for details. Only used if method == 'fft'.
    interp_kind
        The interpolation type if output of fourier resampling doesn't produce
        exactly the right sampling rate.
    samples
        If true, the values in kwargs represent the number of samples along
        The specified dimension.
    **kwargs
        keyword arguments to specify dimension and new sampling value. Units
        can also be used to specify sampling_period or frequency.

    Notes
    -----
    - Unless `samples` is `True`, this function requires a sampling_period.
    - The resulting Patch can be slightly shorter than the input Patch.

    Examples
    --------
    >>> # resample a patch along time dimension to 10 ms
    >>> import numpy as np
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> new = patch.resample(time=np.timedelta64(10, 'ms'))
    >>> # Resample time dimension to 50 Hz
    >>> from dascore.units import Hz
    >>> new = patch.resample(time=(50 * Hz))
    >>> # Resample distance dimension to a sampling period of 15m
    >>> from dascore.units import m
    >>> new = patch.resample(distance=15 * m)
    >>> # Resample time axis such that there are 50 samples total
    >>> new = patch.resample(time=50, samples=True)

    See Also
    --------
    [decimate](`dascore.proc.resample.decimate`)
    [interpolate](`dascore.proc.resample.interpolate`)
    """
    dim, axis, value = get_dim_axis_value(patch, kwargs=kwargs)[0]
    coord = patch.get_coord(dim, require_sorted=True, require_evenly_sampled=True)
    new_step = None
    if not samples:
        step = coord.step
        coord_units = dc.get_quantity(coord.units)
        # inverse coord unit to trick filter units into giving correct units.
        if coord_units is not None:
            coord_units = 1 / coord_units
        new_step, _ = get_filter_units(value, value, to_unit=coord_units)
        # nasty hack so that ints/floats get converted to seconds.
        if isinstance(step, np.timedelta64):
            new_step = to_timedelta64(new_step)
        current_sig_len = patch.data.shape[axis]
        new_len = current_sig_len * (step / new_step)
    else:
        new_len = value
    # do the resampling
    data, new_coord = compat.resample(
        patch.data, int(np.round(new_len)), t=coord, axis=axis, window=window
    )
    cm = patch.coords.update(**{dim: new_coord})
    out = patch.new(data=data, coords=cm)
    # Interpolate if new sampling rate is not very close to desired sampling rate.
    if not samples and not np.isclose(new_len, np.round(new_len)):
        start, stop, step = get_start_stop_step(out, dim)
        new_coord = np.arange(start, stop, new_step)
        out = interpolate(out, kind=interp_kind, **{dim: new_coord})
    return out
