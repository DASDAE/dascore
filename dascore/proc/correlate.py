"""Module for calculating cross-correlation over time or distance."""
from __future__ import annotations

import numpy as np
import scipy
from scipy.fftpack import next_fast_len

from dascore.constants import PatchType
from dascore.units import Quantity
from dascore.utils.misc import broadcast_for_index
from dascore.utils.patch import get_dim_value_from_kwargs, patch_function


@patch_function()
def correlate(
    patch: PatchType, lag: int | float | Quantity | None = None, samples=False, **kwargs
) -> PatchType:
    """
    A function which takes the DAS data in time domain, calculates cross-correlation
    (cc) in frequency domain, and returns the results back in time domain.

    Parameters
    ----------
    patch : PatchType
        The input data patch to be cross-correlated.
    lag :
        An optional argument to save trimmed cc results instead of full output.
    samples : bool, optional (default = False)
        {sample_explination}
    **kwargs
        Additional arguments to specify cc dimension and the master source, to which
        we cross-correlate all other channels/time samples.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()

    >>> # Example 1
    >>> # Calculate cc for all channels as receivers and
    >>> # the 10 m channel as the master channel.
    >>> cc_patch = patch.correlate(distance = 10 * m)

    >>> # Example 2
    >>> # Calculate cc within (-2,2) sec of lag for all channels as receivers and
    >>> # the 10 m channel as the master channel.
    >>> cc_patch = patch.correlate(distance = 10 * m, lag = 2 * s)

    Notes
    -----
    The cross-correlation is performed in the frequency domain for efficiency
    reasons.
    """
    assert len(patch.dims) == 2, "must be a 2D patch"
    dim, source_axis, source = get_dim_value_from_kwargs(patch, kwargs)
    fft_axis = next(iter(set(range(len(patch.dims))) - {source_axis}))
    # get the coordinate which contains the source
    coord_source = patch.get_coord(dim)
    index_source = coord_source.get_next_index(source, samples=samples)
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
    cor_array = ifft_func(fft_prod, axis=fft_axis, n=cx_len)
    out = np.fft.fftshift(cor_array, axes=fft_axis)
    # get new coordinate along correlation dimension
    # new = patch.new(data=out)

    sampling_interval = patch.attrs["time_step"] / np.timedelta64(1, "s")

    num_ch = len(patch.coords.get_array("distance"))
    num_samples = len(patch.coords.get_array("time"))

    # Make Patch to be in ("distance", "time")
    patch_dist_time = patch.convert_units(distance="m").transpose("distance", "time")

    # Do fft transfrom
    # NEED TO MODIFY THE CODE TO WORK IN DISTANCE AXIS AS WELL
    # MAYBE RFFT INSTEAD?
    num_fft = int(scipy.fftpack.next_fast_len(int(num_samples)))
    num_fft_half = num_fft // 2
    fft_rec = scipy.fftpack.fft(patch_dist_time.data, num_fft, axis=source_axis)[
        :, :num_fft_half
    ]
    fft_src = fft_rec[index_source]

    # Convert all 2D arrays into 1D to speed up
    corr = np.zeros(num_ch * num_fft, dtype=np.complex64)

    # Reshape fft_src to be a 2D array with shape (1, -1)
    fft_src_2d = fft_src[np.newaxis, :]

    # Use broadcasting to multiply fft_src_2d with fft_rec
    corr = fft_src_2d * fft_rec

    # Remove the mean in freq domain (spike at t=0)
    corr[:, :num_fft_half] -= np.mean(
        corr[:, :num_fft_half], axis=source_axis, keepdims=True
    )

    # Process the negative frequencies
    corr[:, -(num_fft_half) + 1 :] = np.flip(
        np.conj(corr[:, :num_fft_half]), axis=source_axis
    )

    # Set the zero-frequency component to zero
    corr[:, 0] = complex(0, 0)

    # Take the inverse FFT
    inverse_fft_result = scipy.fftpack.ifft(corr, n=num_fft, axis=source_axis)

    # Shift the zero-frequency component to the center
    shifted_result = np.fft.ifftshift(inverse_fft_result)

    # Extract the real part
    corr_time_real = np.real(shifted_result)

    # Pick data in the defined lag range
    t = np.arange(-num_fft, num_fft) * sampling_interval
    ind = np.where(np.abs(t) <= lag)[0]
    corr_time_real = corr_time_real[:, ind]

    # NEED TO DOUBLE-CHECK NEW_COORDS
    if patch.dims == ("distance", "time"):
        new_coords = patch_dist_time.coords.correlate(**{dim: int(lag)})
        out = patch_dist_time.new(data=corr_time_real, coords=new_coords)
    else:
        new_coords = patch.coords.correlate(**{dim: int(lag)})
        out = patch.new(data=corr_time_real.T, coords=new_coords)

    return out
