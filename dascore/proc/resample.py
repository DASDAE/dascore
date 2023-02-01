"""
Module for re-sampling patches.
"""
from typing import Literal, Union

import numpy as np
from scipy.signal import decimate as scipy_decimate

import dascore as dc
import dascore.compat as compat
from dascore.constants import PatchType
from dascore.exceptions import FilterValueError
from dascore.utils.misc import check_evenly_sampled
from dascore.utils.patch import (
    get_dim_value_from_kwargs,
    get_start_stop_step,
    patch_function,
)
from dascore.utils.time import to_number, to_timedelta64


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
            None - No pre-filtering
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
    # Apply scipy.signal.decimate and geet new coords
    if filter_type:
        if filter_type == "IRR" and factor > 13:
            msg = (
                "IRR filter is unstable for decimation factors above"
                " 13. Call decimate multiple times."
            )
            raise FilterValueError(msg)
        data = scipy_decimate(patch.data, factor, ftype=filter_type, axis=axis)
        coords = {x: patch.coords[x] for x in patch.dims}
        coords[dim] = coords[dim][::factor]
    else:  # No filter, simply slice along specified dimension.
        dar = patch._data_array.sel(**{dim: slice(None, None, factor)})
        # Need to copy so array isn't a slice and holds onto reference of parent
        data = dar.data if not copy else dar.data.copy()
        coords = dar.coords
    # Update delta_dim since spacing along dimension has changed.
    attrs = dict(patch.attrs)
    attrs[f"d_{dim}"] = patch.attrs[f"d_{dim}"] * factor
    out = dc.Patch(data=data, coords=coords, attrs=attrs, dims=patch.dims)
    return out


@patch_function()
def interpolate(
    patch: PatchType, kind: Union[str, int] = "linear", **kwargs
) -> PatchType:
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
        Used to specify dimension and interpolation values.

    Notes
    -----
    This function just uses scipy's interp1d function under the hood.
    See scipy.interpolate.interp1d for information.

    Values for interpolation must be evenly-spaced.
    """
    dim, axis, samples = get_dim_value_from_kwargs(patch, kwargs)
    # ensure samples are evenly sampled
    check_evenly_sampled(samples)
    # we need to make sure only real numbers are used, interp1d doesn't support
    # datetime64 yet.
    coord_num = to_number(patch.coords[dim])
    samples_num = to_number(samples)
    func = compat.interp1d(coord_num, patch.data, axis=axis, kind=kind)
    out = func(samples_num)
    # update attributes
    new_attrs = dict(patch.attrs)
    new_attrs[f"d_{dim}"] = np.median(np.diff(samples))
    new_attrs[f"min_{dim}"] = np.min(samples)
    new_attrs[f"max_{dim}"] = np.max(samples)
    # update coordinates
    new_coords = {x: patch.coords[x] for x in patch.dims}
    new_coords[dim] = samples
    kwargs = dict(data=out, attrs=new_attrs, dims=patch.dims, coords=new_coords)
    return patch.__class__(**kwargs)


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
        keyword arguments to specify

    Notes
    -----
    Unlike [iresample](`dascore.proc.iresample`) this function requires a
    sampling_period.

    Often the resulting Patch will be slightly shorter than the input Patch.

    Examples
    --------
    # resample a patch along time dimension to 10 ms
    import dascore as dc
    patch = dc.get_example_patch()
    new = patch.resample(time=np.timedelta64(10, 'ms'))

    See Also
    --------
    [decimate](`dascore.proc.resample.decimate`)
    [interpolate](`dascore.proc.resample.interpolate`)
    [iresample](`dascore.proc.resmaple.irsample`)
    """
    dim, axis, new_d_dim = get_dim_value_from_kwargs(patch, kwargs)
    d_dim = patch.attrs[f"d_{dim}"]  # current sampling rate
    # nasty hack so that ints/floats get converted to seconds.
    if isinstance(d_dim, np.timedelta64):
        new_d_dim = to_timedelta64(new_d_dim)
    # dim_range = int(compat.floor((dim_stop - dim_start) / val))
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
    >>> new = patch.iresample(time=50)

    See Also
    --------
    [iresample](`dascore.proc.resmaple.irsample`)
    [decimate](`dascore.proc.resample.decimate`)
    """
    dim, axis, new_length = get_dim_value_from_kwargs(patch, kwargs)
    coord = patch.coords[dim]
    out, new_coord = compat.resample(
        patch.data, new_length, t=coord, axis=axis, window=window
    )
    # update coordinates
    new_coords = {x: patch.coords[x] for x in patch.dims}
    new_coords[dim] = new_coord
    # update attributes
    new_attrs = dict(patch.attrs)
    new_attrs[f"d_{dim}"] = new_coord[1] - new_coord[0]
    new_attrs[f"{dim}_max"] = np.max(new_coord)
    patch_inputs = dict(data=out, attrs=new_attrs, dims=patch.dims, coords=new_coords)
    return patch.__class__(**patch_inputs)
