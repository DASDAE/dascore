"""Module for calculating cross-correlation over time or distance."""
from __future__ import annotations

import numpy as np
from scipy.fftpack import next_fast_len

import dascore as dc
from dascore.constants import PatchType
from dascore.core.coordmanager import get_coord_manager
from dascore.transform.fourier import _get_dft_attrs
from dascore.units import Quantity
from dascore.utils.misc import broadcast_for_index
from dascore.utils.patch import (
    _get_dx_or_spacing_and_axes,
    get_dim_value_from_kwargs,
    patch_function,
)
from dascore.utils.transformatter import FourierTransformatter


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


@patch_function()
def correlate(
    patch: PatchType,
    lag: int | float | Quantity | None = None,
    step_size: int = 1,
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
    lag :
        An optional argument to save only certain lags instead of full
        output. It can't be used if idft = False.
    step_size : int, optional (default = None)
        An optional argument to skip rows/columns for the virtual receiver.
        A step_size = 2 means performing cross correlation (cc)
        with every other channels instead of the full array.
        This will reduce computation and storage cost if needed.
    samples : bool, optional (default = False)
        If True, the argument specified in kwargs refers to the *sample* not
        value along that axis. See examples for details.
    idft : bool, optional (default = Ture)
        If True, it applies idft and return results in time domain.
    **kwargs
        Additional arguments to specify cc dimension and the
        master source(s), to which we want to cross-correlate all other
        channels/time samples (with or without a step_size).
        If the master source is an array, the function will compute cc for all the
        half of all possible pairs (i.e., only a-b and not b-a).
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
    1- The cross-correlation is performed in the frequency domain for efficiency
    reasons.

    2- The output dimension is opposite of the one specified in kwargs, has
    the units of float, and the string "lag_" prepended. For example, "lag_time".

    3- If the patch is in the frequency domain, it needs to be zero-padded in
    the time domain first. If not, first apply the
    [idft](`dascore.transform.fourier.idft`) function to the patch, and then
    use the correlate function, which automatically handles the zero padding.
    """
    assert len(patch.dims) == 2, "must be a 2D patch."
    if not idft and lag is not None:
        msg = "lag argument can't be used when idft is set to False."
        raise ValueError(msg)
    dim, source_axis, source = get_dim_value_from_kwargs(patch, kwargs)
    # get the axis and coord over which fft should be calculated.
    fft_axis = next(iter(set(range(len(patch.dims))) - {source_axis}))
    fft_dim = patch.dims[fft_axis]
    # get the coordinate which contains the source
    coord_source = patch.get_coord(dim)
    index_source = coord_source.get_next_index(source, samples=samples)

    if step_size is not None and step_size > 1:
        step = patch.get_coord(dim).step * step_size
        new_coord_step = dc.core.get_coord(
            start=patch.get_coord(dim)[0],
            stop=patch.get_coord(dim)[-1],
            step=step,
        )
        coords = patch.coords.update(**{dim: new_coord_step})
        attrs = patch.attrs
        patch_new_data = patch.data.take(
            range(0, patch.data.shape[source_axis], step_size), axis=source_axis
        )
        patch = patch.update(data=patch_new_data, coords=coords, attrs=attrs)

    # determine whether fft is needed
    # determine proper fft, idft functions based on data being real or complex
    if (dim == "distance" and fft_dim != "ft_time") or (
        dim == "time" and fft_dim != "ft_distance"
    ):
        needed_fft = True
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

    else:
        needed_fft = False
        is_real = not np.min(patch.coords.get_array(fft_dim)) < 0
        # assume input is already in the frequency domain, skip FFT
        fft = patch.data
        fft_coord = patch.get_coord(fft_dim)  # edit: needs adjustment?

        cx_len = patch.shape[fft_axis]
        fast_len = cx_len

    ndims = len(patch.shape)
    slicer = slice(index_source, index_source + 1)
    inds = broadcast_for_index(ndims, axis=source_axis, value=slicer)
    source_fft = fft[inds]
    # perform correlation in freq domain and transform back to time domain
    fft_prod = fft * np.conj(source_fft)
    # the n parameter needs to be odd so we have a 0 lag time. This only
    # applies to real fft
    n_out = fast_len if (not is_real or fast_len % 2 != 0) else fast_len - 1

    if idft:
        ifft_func = np.fft.irfft if is_real else np.fft.ifft
        if needed_fft:
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
        else:
            corr_array = np.real(ifft_func(fft_prod, axis=fft_axis, n=n_out))
            corr_data = _shift(corr_array, cx_len, axis=fft_axis)
            # get new coordinate along correlation dimension
            new_coord = _get_correlated_coord(fft_coord, corr_data.shape[fft_axis])
            coords = patch.coords.update(**{fft_dim: new_coord}).rename_coord(
                **{fft_dim: f"{fft_dim}"}
            )
            out = dc.Patch(coords=coords, data=corr_data, attrs=patch.attrs)
            return out
    else:
        dims = fft_dim
        _, axes = _get_dx_or_spacing_and_axes(patch, dims, require_evenly_spaced=True)
        # get new coordinates
        old_cm = patch.coords.disassociate_coord(dims)
        new_cm = old_cm.get_coord_tuple_map()
        ft = FourierTransformatter()
        name = ft.rename_dims(fft_dim)[0]
        coord = _get_correlated_coord(fft_coord, fft_prod.shape[fft_axis])
        new_cm[name] = (name, coord)
        new_dims = ft.rename_dims(patch.dims, index=axes)
        new_coords = get_coord_manager(new_cm, dims=new_dims)
        # get attributes
        attrs = _get_dft_attrs(patch, dims, new_coords)
        # return the frequency domain data directly without taking the inverse FFT
        return patch.new(fft_prod, coords=new_coords, attrs=attrs)
