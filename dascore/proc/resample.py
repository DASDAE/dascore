"""Module for re-sampling patches."""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import decimate as scipy_decimate

import dascore as dc
import dascore.compat as compat
from dascore.constants import PatchType
from dascore.units import get_filter_units
from dascore.utils.patch import (
    get_dim_value_from_kwargs,
    get_start_stop_step,
    patch_function,
)
from dascore.utils.time import to_int, to_timedelta64


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
    Simply uses scipy.signal.decimate if filter_type is specified. Otherwise,
    just slice data long specified dimension only including every n samples.

    Examples
    --------
    # Simple example using iir
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> decimated_irr = patch.decimate(time=10, filter_type='iir')
    >>> # Example using fir along distance dimension
    >>> decimated_fir = patch.decimate(distance=10, filter_type='fir')
    """
    dim, axis, factor = get_dim_value_from_kwargs(patch, kwargs)
    coords, slices = patch.coords.decimate(**{dim: int(factor)})
    # Apply scipy.signal.decimate and get new coords
    if filter_type:
        data = scipy_decimate(patch.data, factor, ftype=filter_type, axis=axis)
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
    >>> time = patch.coords['time']
    >>> new_time = np.arange(time.min(), time.max(), 0.5*patch.attrs.time_step)
    >>> patch_uptime = patch.interpolate(time=new_time)
    >>> # interpolate unevenly sampled dim to evenly sampled
    >>> patch = dc.get_example_patch("wacky_dim_coords_patch")
    >>> patch_time_even = patch.interpolate(time=None)
    """
    dim, axis, samples = get_dim_value_from_kwargs(patch, kwargs)
    # if samples is None, get evenly sampled coords along dimension.
    if samples is None:
        coord = patch.coords.coord_map[dim]
        samples = coord.snap().values
    # we need to make sure only real numbers are used, interp1d doesn't support
    # datetime64 yet.
    coord_num = to_int(patch.coords[dim])
    samples_num = to_int(samples)
    func = compat.interp1d(
        coord_num, patch.data, axis=axis, kind=kind, fill_value="extrapolate"
    )
    out = func(samples_num)
    new_coords = dict(patch.coords.coord_map)
    new_coords[dim] = samples
    return patch.new(data=out, coords=new_coords)


@patch_function()
def resample(
    patch: PatchType, window=None, interp_kind="linear", **kwargs
) -> PatchType:
    """
    Resample along a single dimension using Fourier Method and interpolation.

    The dimension which should be resampled is passed as kwargs. The key
    is the dimension name and the value is the new sampling period.

    Since Fourier methods only support adding or removing an integer number
    of frequency bins, the exact desired sampling rate is often not achievable
    with resampling alone. If the fourier resampling doesn't produce the exact
    an interpolation (see [interpolate](`dascore.proc.interpolate`)) is used to
    achieve the desired sampling rate.

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
    **kwargs
        keyword arguments to specify dimension and new sampling value. Units
        can also be used to specify sampling_period or frequency.

    Notes
    -----
    Unlike [iresample](`dascore.proc.iresample`) this function requires a
    sampling_period.

    Often the resulting Patch will be slightly shorter than the input Patch.

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

    See Also
    --------
    [decimate](`dascore.proc.resample.decimate`)
    [interpolate](`dascore.proc.resample.interpolate`)
    [iresample](`dascore.proc.resample.iresample`)
    """
    dim, axis, new_d_dim = get_dim_value_from_kwargs(patch, kwargs)
    d_dim = patch.attrs[f"{dim}_step"]  # current sampling rate
    coord_units = dc.get_quantity(patch.coords.coord_map[dim].units)
    # inverse coord unit to trick filter units into giving correct units.
    if coord_units is not None:
        coord_units = 1 / coord_units
    new_d_dim, _ = get_filter_units(new_d_dim, new_d_dim, to_unit=coord_units)
    # nasty hack so that ints/floats get converted to seconds.
    if isinstance(d_dim, np.timedelta64):
        new_d_dim = to_timedelta64(new_d_dim)
    current_sig_len = patch.data.shape[axis]
    new_len = current_sig_len * (d_dim / new_d_dim)
    out = iresample.func(patch, window=window, **{dim: int(np.round(new_len))})
    # Interpolate if new sampling rate is not very close to desired sampling rate.
    if not np.isclose(new_len, np.round(new_len)):
        start, stop, step = get_start_stop_step(out, dim)
        new_coord = np.arange(start, stop, new_d_dim)
        out = interpolate(out, kind=interp_kind, **{dim: new_coord})
    return out


@patch_function()
def iresample(patch: PatchType, window=None, **kwargs) -> PatchType:
    """
    Resample a patch along a single dimension using Fourier Method.

    Unlike [resample](`dascore.proc.resample`) this function requires the
    number of samples for the selected dimension.

    Parameters
    ----------
    patch
        The patch to resample.
    window
        The Fourier-domain window that tapers the Fourier spectrum. See
        scipy.signal.resample for details.
    **kwargs
         keyword arguments to specify dimension.

    Notes
    -----
    Simply uses scipy.signal.resample.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # Resample time axis such that there are 50 samples total
    >>> new = patch.iresample(time=50)

    See Also
    --------
    [resample](`dascore.proc.resample.resample`)
    [decimate](`dascore.proc.resample.decimate`)
    """
    dim, axis, new_length = get_dim_value_from_kwargs(patch, kwargs)
    coord = patch.coords[dim]
    data, new_coord = compat.resample(
        patch.data, new_length, t=coord, axis=axis, window=window
    )
    # update coordinates
    new_coords = patch.coords.update_coords(**{dim: new_coord})
    # update attributes
    new_attrs = dict(patch.attrs)
    new_attrs[f"{dim}_step"] = new_coord[1] - new_coord[0]
    new_attrs[f"{dim}_max"] = np.max(new_coord)
    return patch.new(data=data, coords=new_coords)
