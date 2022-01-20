"""
Module for applying decimation to Patches.
"""
import numpy as np

import dascore
import dascore.compat as compat
from dascore.constants import PatchType
from dascore.exceptions import FilterValueError
from dascore.proc.filter import _get_sampling_rate, _lowpass_cheby_2
from dascore.utils.misc import all_close, get_dim_value_from_kwargs
from dascore.utils.patch import patch_function
from dascore.utils.time import is_datetime64, to_number


@patch_function()
def decimate(
    patch: PatchType,
    factor: int,
    dim: str = "time",
    lowpass: bool = True,
    copy=True,
) -> PatchType:
    """
    Decimate a patch along a dimension.

    Parameters
    ----------
    factor
        The decimation factor (e.g., 10)
    dim
        dimension along which to decimate.
    lowpass
        If True, first apply a low-pass (anti-alis) filter. Uses
        :func:`dascore.proc.filter._lowpass_cheby_2`
    copy
        If True, copy the decimated data array. This is needed if you want
        the old array to get gc'ed to free memory otherwise a view is returned.
    """
    # Note: We can't simply use scipy.signal.decimate due to this issue:
    # https://github.com/scipy/scipy/issues/15072
    if lowpass:
        # get new niquest
        if factor > 16:
            msg = (
                "Automatic filter design is unstable for decimation "
                + "factors above 16. Manual decimation is necessary."
            )
            raise FilterValueError(msg)
        sr = _get_sampling_rate(patch, dim)
        freq = sr * 0.5 / float(factor)
        fdata = _lowpass_cheby_2(patch.data, freq, sr, axis=patch.dims.index(dim))
        patch = dascore.Patch(fdata, coords=patch.coords, attrs=patch.attrs)

    kwargs = {dim: slice(None, None, factor)}
    dar = patch._data_array.sel(**kwargs)
    # need to create a new xarray so the old, probably large, numpy array
    # gets gc'ed, otherwise it stays in memory (if lowpass isn't called)
    data = dar.data if not copy else dar.data.copy()
    attrs = dar.attrs
    # update delta_dim since spacing along dimension has changed
    d_attr = f"d_{dim}"
    attrs[d_attr] = patch.attrs[d_attr] * factor

    return dascore.Patch(data=data, coords=dar.coords, attrs=dar.attrs)


@patch_function()
def interpolate(patch: PatchType, kind: str = "linear", **kwargs) -> PatchType:
    """
    Set coordinates of patch along a dimension using interpolation.

    Parameters
    ----------
    patch
        The patch object to which interpolation is applied.
    kind
        The type of interpolation. See Notes for more details.
    **kwargs
        Used to specify dimension and interpolation values.

    Notes
    -----
    This function just uses scipy's interp1d function under the hood.
    See :func:`scipy.inpterp.interp1d for information.

    Values for interpolation must be evenly-spaced.
    """
    dim, axis, samples = get_dim_value_from_kwargs(patch, kwargs)
    is_time = is_datetime64()

    # ensure samples are evenly sampled
    diff = np.diff(samples)
    if not all_close(diff, np.mean(diff)):
        unique_diffs = np.unique(diff)
        msg = (
            "Interpolate requires evenly sampled data. The values you passed "
            f"have the following unique differences: {unique_diffs}"
        )
        raise FilterValueError(msg)

    coord = patch.coords[dim]
    func = compat.interp1d(coord, patch.data, axis=axis, kind=kind)
    out = func(samples)
    # update attributes
    new_attrs = dict(patch.attrs)
    new_attrs[f"d_{dim}"] = np.median(diff)
    new_attrs[f"min_{dim}"] = np.min(samples)
    new_attrs[f"max_{dim}"] = np.max(samples)
    # update coordinates
    new_coords = {x: patch.coords[x] for x in patch.dims}
    new_coords[dim] = samples
    kwargs = dict(data=out, attrs=new_attrs, dims=patch.dims, coords=new_coords)
    return patch.__class__(**kwargs)


@patch_function()
def resample(patch: PatchType, method, window=None, **kwargs) -> PatchType:
    """
    Resample a patch along a single dimension using Fourier Method.

    The dimension which should be resampled is passed as kwargs. The key
    is the

    Unlike :func: `dascore.proc.iresample` this function requires a
    sampling_period.

    Parameters
    ---------
    patch
        The patch to resample.
    method
        Indicates the resampling method. Must be one of the following:
            'interpolation' - `scipy.ndimage.zoom()`
            'poly' - `scipy.signal.resample_poly()`
            'fft' - `scipy.signal.resample()`
    window
        The Fourier-domain window that tapers the Fourier spectrum. See
        :func:`scipy.signal.resample` for details. Only used if method == 'fft'.
    **kwargs
        keyword arguments to specify

    Notes
    -----
    Normally uses scipy.signal.resample

    Examples
    --------
    import dascore as dc
    patch = dc.get_example_patch()
    new = patch.resample(time=np.timedelta64(10, 'ms'))

    See Also
    --------
    iresample: :func: `~dascore.proc.resmaple.isample`
    decimate: :func: `~dascore.proc.resample.decimate`
    """
    dim, axis, val = get_dim_value_from_kwargs(patch, kwargs)
    d_dim = patch.attrs[f"d_{dim}"]
    # dim_range = int(compat.floor((dim_stop - dim_start) / val))
    sig_len = patch.data.shape[axis]
    new_len = int(np.round(sig_len * (val / d_dim)))
    # since new_len is rounded find the actual sampling rate
    # actual_ds =
    # Resample
    out = _resample_data(patch.data, new_len, axis, method)
    new_attrs = dict(patch.attrs)
    new_attrs[f"d_{dim}"] = val
    new_coords = {x: patch.coords[x] for x in patch.dims}
    new_coords[dim] = np.arange(out.shape[axis]) * val
    kwargs = dict(data=out, attrs=new_attrs, dims=patch.dims, coords=new_coords)
    return patch.__class__(**kwargs)


@patch_function()
def iresample(patch: PatchType, window=None, **kwargs) -> PatchType:
    """
    Resample a patch along a single dimension using Fourier Method.

    Unlike :func: `dascore.proc.resample` this function requires the
    number of samples for the selected dimension.

    Parameters
    ---------
    patch
        The patch to resample.
    window
        The Fourier-domain window that tapers the Fourier spectrum. See
        :func:`scipy.signal.resample` for details.
    **kwargs
         keyword arguments to specify

    Notes
    -----
    Normally uses scipy.signal.resample

    Examples
    --------
    import dascore as dc
    patch = dc.get_example_patch()
    new = patch.iresample(time=50)

    See Also
    --------
    iresample: :func: `~dascore.proc.resmaple.isample`
    decimate: :func: `~dascore.proc.resample.decimate`
    """
    dim, axis, val = get_dim_value_from_kwargs(patch, kwargs)
    out = compat.resample(patch.data, val, axis=axis, window=window)
    new_d_dim = patch.attrs[f"d_{dim}"] * (patch.data.shape[axis] / val)
    new_attrs = dict(patch.attrs)
    new_attrs[f"d_{dim}"] = new_d_dim
    new_coords = {x: patch.coords[x] for x in patch.dims}
    new_coords[dim] = np.arange(out.shape[axis]) * new_d_dim
    patch_inputs = dict(data=out, attrs=new_attrs, dims=patch.dims, coords=new_coords)
    return patch.__class__(**patch_inputs)


def _resample_data(data, desired_length, axis, method):
    """Resample data to desired length"""
    if method.lower() == "fft":
        out = compat.resample(data, desired_length, axis=axis)
    elif method.lower() == "poly":
        out = compat.resample_poly(data, desired_length, data.size[axis])
    elif method.lower() == "interpolation":
        zoom_factors = [
            1 if i != axis else desired_length / data.shape[axis]
            for i in range(len(data.shape))
        ]
        out = compat.zoom(data, zoom_factors)
    else:
        msg = f"Method {method} is not a supported resampling method."
        raise NotImplementedError(msg)

    return out
