"""Module for calculating cross-correlation over time or distance."""
from __future__ import annotations

import numpy as np
from scipy.fftpack import next_fast_len

import dascore as dc
from dascore.constants import PatchType
from dascore.units import Quantity
from dascore.utils.misc import broadcast_for_index
from dascore.utils.patch import (
    _get_dx_or_spacing_and_axes,
    get_dim_value_from_kwargs,
    patch_function,
)


def _get_correlated_coord(old_coord, data_shape, real=True):
    """Get the new coordinate which corresponds to correlated values."""
    step = old_coord.step
    one_sided_len = data_shape // 2
    new = dc.core.get_coord(
        start=-one_sided_len * step,
        stop=(one_sided_len + 1) * step,
        step=step,
    )
    assert len(new) == data_shape, "failed to create correlated coord"
    return new


def _shift(data, cx_len, axis):
    """
    Re-assemble fft data so zero lag is in center.

    Also accounts for padding for fast fft.
    """
    ndims = len(data.shape)
    start_slice = slice(-np.floor(cx_len / 2).astype(int), None)
    start_ind = broadcast_for_index(ndims, axis, start_slice)
    stop_slice = slice(None, np.ceil(cx_len / 2).astype(int))
    stop_ind = broadcast_for_index(ndims, axis, stop_slice)
    data1 = data[start_ind]
    data2 = data[stop_ind]
    data = np.concatenate([data1, data2], axis=axis)
    return data


@patch_function()
def correlate(
    patch: PatchType, lag: int | float | Quantity | None = None, samples=False, **kwargs
) -> PatchType:
    """
    Correlate a single row/column in a 2D patch with every other row/column.

    Parameters
    ----------
    patch : PatchType
        The input data patch to be cross-correlated. Must be 2-dimensional.
    lag :
        An optional argument to save only certain lag times instead of full
        output.
    samples : bool, optional (default = False)
        If True, the argument specified in kwargs refers to the *sample* not
        value along that axis. See examples for details.
    **kwargs
        Additional arguments to specify cross correlation dimension and the
        master source, to which
        we cross-correlate all other channels/time samples.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.units import m, s

    >>> patch = dc.get_example_patch()

    >>> # Example 1
    >>> # Calculate cc for all channels as receivers and
    >>> # the 10 m channel as the master channel. The new patch has dimensions
    >>> # (lag_time, distance)
    >>> cc_patch = patch.correlate(distance = 10 * m)

    >>> # Example 2
    >>> # Calculate cc within (-2,2) sec of lag for all channels as receivers and
    >>> # the 10 m channel as the master channel.
    >>> cc_patch = patch.correlate(distance = 10 * m, lag = 2 * s)

    >>> # Example 3
    >>> # Use 2nd channel (python is 0 indexed) along distance as master channel
    >>> cc_patch = patch.correlate(distance=1, samples=True)

    >>> # Example 4
    >>> # Correlate along time dimension
    >>> cc_patch = patch.correlate(time=100, samples=True)

    Notes
    -----
    The cross-correlation is performed in the frequency domain for efficiency
    reasons.

    The output dimension is opposite of the one specified in kwargs, has
    the units of float, and the string "lag_" prepended. For example, "lag_time".
    """
    assert len(patch.dims) == 2, "must be a 2D patch"
    dim, source_axis, source = get_dim_value_from_kwargs(patch, kwargs)
    # get the axis and coord over which fft should be calculated.
    fft_axis = next(iter(set(range(len(patch.dims))) - {source_axis}))
    fft_dim = patch.dims[fft_axis]
    # ensure coordinate is evenly spaced
    _get_dx_or_spacing_and_axes(patch, fft_dim, require_evenly_spaced=True)
    fft_coord = patch.get_coord(fft_dim)
    # get the coordinate which contains the source
    coord_source = patch.get_coord(dim)
    index_source = coord_source.get_next_index(source, samples=samples)
    # get the closest fast length. Some padding is applied in the fft to avoid
    # inefficient lengths. Note: This is not always a power of 2.
    cx_len = patch.shape[fft_axis] * 2 - 1
    fast_len = next_fast_len(cx_len)
    # determine proper fft, ifft functions based on data being real or complex
    is_real = not np.issubdtype(patch.data.dtype, np.complexfloating)
    fft_func = np.fft.rfft if is_real else np.fft.fft
    ifft_func = np.fft.irfft if is_real else np.fft.ifft
    # perform ffts and get source array (a sub-slice of larger fft)
    fft = fft_func(patch.data, axis=fft_axis, n=fast_len)
    ndims = len(patch.shape)
    slicer = slice(index_source, index_source + 1)
    inds = broadcast_for_index(ndims, axis=source_axis, value=slicer)
    source_fft = fft[inds]
    # perform correlation in freq domain and transform back to time domain
    fft_prod = fft * np.conj(source_fft)
    # the n parameter needs to be odd so we have a 0 lag time. This only
    # applies to real fft
    n_out = fast_len if (not is_real or fast_len % 2 != 0) else fast_len - 1
    corr_array = ifft_func(fft_prod, axis=fft_axis, n=n_out)
    corr_data = _shift(corr_array, cx_len, axis=fft_axis)
    # get new coordinate along correlation dimension
    new_coord = _get_correlated_coord(fft_coord, corr_data.shape[fft_axis])
    coords = patch.coords.update(**{fft_dim: new_coord}).rename_coord(
        **{fft_dim: f"lag_{fft_dim}"}
    )
    out = dc.Patch(coords=coords, data=corr_data, attrs=patch.attrs)
    if lag is not None:
        out = out.select(**{f"lag_{fft_dim}": (-lag, +lag)})
    return out
