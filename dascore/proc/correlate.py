"""Module for calculating cross-correlation over time or distance."""
from __future__ import annotations

import numpy as np
from scipy.fftpack import next_fast_len

import dascore as dc
from dascore.constants import PatchType
from dascore.utils.misc import broadcast_for_index
from dascore.utils.patch import (
    get_dim_value_from_kwargs,
    patch_function,
)


def _get_correlated_coord(old_coord, data_shape):
    """Get the new coordinate which corresponds to correlated values."""
    step = old_coord.step
    one_sided_len = data_shape // 2
    new = dc.core.get_coord(
        start=-one_sided_len * step,
        stop=(one_sided_len + 1) * step,
        step=step,
    )
    return new.change_length(data_shape)


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


def _correlate_fft(patch, fft_axis, fft_dim):
    """Perform a padded FFT for correlations."""
    is_real = not np.issubdtype(patch.data.dtype, np.complexfloating)
    # perform FFT (since the input is not already in the frequency domain)
    fft_func = np.fft.rfft if is_real else np.fft.fft
    # get the closest fast length. Some padding is applied in the fft to avoid
    # inefficient lengths. Note: This is not always a power of 2.
    cx_len = patch.shape[fft_axis] * 2 - 1
    fast_len = next_fast_len(cx_len)
    # perform ffts and get source array (a sub-slice of larger fft)
    fft = fft_func(patch.data, axis=fft_axis, n=fast_len)
    # ensure coordinate is evenly spaced
    fft_coord = patch.get_coord(fft_dim)
    return fft, fft_coord, cx_len, fast_len, is_real


def _get_fft_array(patch, fft_axis, fft_dim):
    """Get the array info from pre-transformed patch."""
    fft = patch.data
    fft_coord = patch.get_coord(fft_dim)
    cx_len = patch.shape[fft_axis]
    fast_len = cx_len
    # If fft axis is strictly positive a real dft was used.
    is_real = fft_coord.min() >= 0
    return fft, fft_coord, cx_len, fast_len, is_real


def _get_source_fft(patch, fft, dim, source, source_axis, fft_axis, samples):
    """Get an array of coordinate sources.
    This function will place the new sources in a third dimension so
    they broadcast with the original fft matrix.
    """
    # get the coordinate which contains the source
    ndim = patch.ndim
    coord_source = patch.get_coord(dim)
    index_source = coord_source.get_next_index(source, samples=samples)
    slicer = slice(index_source, index_source + 1)
    inds = broadcast_for_index(ndim, axis=source_axis, value=slicer)
    flat_fft = fft[inds]
    # The new dimension should be the old source dimension.
    source_fft = np.expand_dims(flat_fft, axis=source_axis)
    # print(source_fft)
    return source_fft


@patch_function()
def correlate(
    patch: PatchType,
    samples=False,
    idft=True,
    **kwargs,
) -> PatchType:
    """
    Correlate row/column (virtual sources) in a 2D patch with other
    rows/columns (virtual receivers).

    Parameters
    ----------
    patch : PatchType
        The input data patch to be cross-correlated. Must be 2-dimensional.
        The patch can be in time or frequency domains.
    samples : bool, optional (default = False)
        If True, the argument specified in kwargs refers to the *sample* not
        value along that axis. See examples for details.
    idft : bool, optional (default = Ture)
        If True, it applies idft and return results in time domain.
    **kwargs
        Additional arguments to specify cc dimension and the
        master source(s), to which we want to cross-correlate all other
        channels/time samples (with or without a step_size).
        If the master source is an array, the function will compute cc for all
        the half of all possible pairs (i.e., only a-b and not b-a).
        This will result in a 3D patch.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.units import m, s

    >>> patch = dc.get_example_patch()

    >>> # Example 1
    >>> # Calculate cc for all channels as receivers and
    >>> # the 10 m channel as the master channel.
    >>> cc_patch = patch.correlate(distance = 10 * m)

    >>> # Example 2
    >>> # Calculate cc within (-2,2) sec of lag for all channels as receivers and
    >>> # the 10 m channel as the master channel. The new patch has dimensions
    >>> # (lag_time, distance)
    >>> cc_patch = patch.correlate(distance = 10 * m, lag = 2 * s)

    >>> # Example 3
    >>> # Use 2nd channel (python is 0 indexed) along distance as master channel
    >>> cc_patch = patch.correlate(distance=1, samples=True)

    >>> # Example 4
    >>> # Correlate along time dimension
    >>> cc_patch = patch.correlate(time=100, samples=True)

    >>> # Example 5
    >>> # Calculate cc of a patch for all channels as receivers and
    >>> # the 10 m channel as the master channel and result data in frequency domain.
    >>> # The new patch has dimensions (ft_time, distance).
    >>> cc_patch = patch.correlate(distance = 10 * m, idft = False)

    # Example 6
    # Calculate cc of channel numbers [1,3,7,8,20] as master channels
    # and every fourth channel as receivers.
    # The new patch has dimensions (lag_time, distance, source_distance).
    cc_patch = patch.correlate(distance = [1,3,7,8,20], step_size = 4)

    Notes
    -----
    1- The cross-correlation is performed in the frequency domain.

    2- The output dimension is opposite of the one specified in kwargs, has
    the units of float, and the string "lag_" prepended. For example, "lag_time".

    3- If the patch is in the frequency domain, it needs to be zero-padded in
    the time domain first. If not, first apply the
    [idft](`dascore.transform.fourier.idft`) function to the patch, and then
    use the correlate function, which automatically handles the zero padding.
    """
    assert len(patch.dims) == 2, "must be a 2D patch."
    dim, source_axis, source = get_dim_value_from_kwargs(patch, kwargs)
    # Get the axis and coord over which fft should be calculated.
    fft_axis = next(iter(set(range(len(patch.dims))) - {source_axis}))
    fft_dim = patch.dims[fft_axis]
    # Determine if the DFT needs to be performed or just extract dft array.
    dft_func = _get_fft_array if fft_dim.startswith("ft_") else _correlate_fft
    fft, fft_coord, cx_len, fast_len, is_real = dft_func(patch, fft_axis, fft_dim)
    # Get the sources.
    source_fft = _get_source_fft(
        patch, fft, dim, source, source_axis, fft_axis, samples
    )
    # Perform correlation in freq domain. The last dim corresponds to sources.
    fft_prod = fft[..., None] * np.conj(source_fft)
    # the n parameter needs to be odd so we have a 0 lag time. This only
    # applies to real fft
    n_out = fast_len if (not is_real or fast_len % 2 != 0) else fast_len - 1

    assert fft_prod * n_out
    #
    # if idft:
    #     ifft_func = np.fft.irfft if is_real else np.fft.ifft
    #     corr_array = ifft_func(fft_prod, axis=fft_axis, n=n_out)
    #     corr_data = _shift(corr_array, cx_len, axis=fft_axis)
    #     # get new coordinate along correlation dimension
    #     new_coord = _get_correlated_coord(fft_coord, corr_data.shape[fft_axis])
    #     coords = patch.coords.update(**{fft_dim: new_coord}).rename_coord(
    #         **{fft_dim: f"lag_{fft_dim}"}
    #     )
    #     out = dc.Patch(coords=coords, data=corr_data, attrs=patch.attrs)
    #     else:
    #         corr_array = np.real(ifft_func(fft_prod, axis=fft_axis, n=n_out))
    #         corr_data = _shift(corr_array, cx_len, axis=fft_axis)
    #         # get new coordinate along correlation dimension
    #         new_coord = _get_correlated_coord(fft_coord, corr_data.shape[fft_axis])
    #         coords = patch.coords.update(**{fft_dim: new_coord}).rename_coord(
    #             **{fft_dim: f"{fft_dim}"}
    #         )
    #         out = dc.Patch(coords=coords, data=corr_data, attrs=patch.attrs)
    #         return out
    # else:
    #     dims = fft_dim
    #     _, axes = _get_dx_or_spacing_and_axes(patch, dims, require_evenly_spaced=True)
    #     # get new coordinates
    #     old_cm = patch.coords.disassociate_coord(dims)
    #     new_cm = old_cm.get_coord_tuple_map()
    #     ft = FourierTransformatter()
    #     name = ft.rename_dims(fft_dim)[0]
    #     coord = _get_correlated_coord(fft_coord, fft_prod.shape[fft_axis])
    #     new_cm[name] = (name, coord)
    #     new_dims = ft.rename_dims(patch.dims, index=axes)
    #     new_coords = get_coord_manager(new_cm, dims=new_dims)
    #     # get attributes
    #     attrs = _get_dft_attrs(patch, dims, new_coords)
    #     # return the frequency domain data directly without taking the inverse FFT
    #     return patch.new(fft_prod, coords=new_coords, attrs=attrs)
