"""Module for performing cross-correlation on a Patch."""
from __future__ import annotations

import numpy as np
import scipy

import dascore as dc
from dascore.constants import PatchType
from dascore.utils.patch import get_dim_value_from_kwargs, patch_function

@patch_function()
def correlate(patch: PatchType, source=int, samples=False, **kwargs
) -> PatchType:
    """
    This function does the cross-correlation in freq. domain. it takes advantage of the linear relationship of ifft, so that
    stacking is performed in spectrum domain first to reduce the total number of ifft.

    Parameters
    ----------


    Return
    ------
    patch_corr: 2D matrix of the averaged of cross-correlation functions in time domain
    
    Examples
    --------
    # Simple example for cross-correlation
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    """

    dim, axis, lag = get_dim_value_from_kwargs(patch, kwargs) # Note: Need to use axis instead of 0 or 1

    sampling_interval = patch.attrs['time_step'] / np.timedelta64(1, 's')

    num_ch = len(patch.coords.get_array("distance"))
    num_samples = len(patch.coords.get_array("time"))
    
    # Do fft transfrom
    # fft_rec = patch.dft(dim="time")
    # num_fft = fft_rec.shape[1]
    # num_fft_half = num_fft//2
    # fft_rec_data = fft_rec.data[:,:num_fft_half]
    # fft_src = fft_rec.select(distance=(source,source), samples=True)
    # fft_src_data = fft_src.data[:,:num_fft_half]

    # Note: Assuming data s in ("distance", "time") - need to generalize

    num_fft = int(scipy.fftpack.next_fast_len(int(num_samples)))
    num_fft_half = num_fft//2
    fft_rec = scipy.fftpack.fft(patch.data, num_fft, axis=1)[:,:num_fft_half]
    fft_src = fft_rec[source] 

    # Convert all 2D arrays into 1D to speed up
    corr = np.zeros(num_ch * num_fft, dtype=np.complex64)
    # Duplicate fft_src for num_ch rows
    # fft_src_data_1 = np.ones(shape=(num_ch,1))*fft_src.reshape(1,fft_src.size)  
    # corr = fft_src_data_1.reshape(fft_src_data_1.size,)*fft_rec.reshape(fft_rec.size,)
    # corr  = corr.reshape(num_ch, num_fft)

    # Reshape fft_src to be a 2D array with shape (1, -1)
    fft_src_2d = fft_src[np.newaxis, :]

    # Use broadcasting to multiply fft_src_2d with fft_rec
    corr = fft_src_2d * fft_rec

    # Loop through each cross correlation
    # corr_patch = np.zeros(shape=(num_ch,num_fft),dtype=np.float64)  
    # pre_corr = np.zeros(num_fft,dtype=np.complex64)
    # for i in range(num_ch):
    #     pre_corr[:num_fft_half] = corr[i,:]
    #     pre_corr[:num_fft_half] = pre_corr[:num_fft_half]-np.mean(pre_corr[:num_fft_half])   # remove the mean in freq domain (spike at t=0)
    #     pre_corr[-(num_fft_half)+1:] = np.flip(np.conj(pre_corr[-(num_fft_half)+1:]),axis=0)
    #     pre_corr[0]=complex(0,0)
    #     corr_patch[i,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(pre_corr, num_fft, axis=0)))

    # Remove the mean in freq domain (spike at t=0)
    corr[:, :num_fft_half] -= np.mean(corr[:, :num_fft_half], axis=1, keepdims=True)

    # Process the negative frequencies
    corr[:, -(num_fft_half)+1:] = np.flip(np.conj(corr[:, :num_fft_half]), axis=1)

    # Set the zero-frequency component to zero
    corr[:, 0]=complex(0,0)

    # Take the inverse FFT
    inverse_fft_result = scipy.fftpack.ifft(corr, n=num_fft, axis=1)

    # Shift the zero-frequency component to the center
    shifted_result = np.fft.ifftshift(inverse_fft_result)

    # Extract the real part
    corr_time_real = np.real(shifted_result)

    # Pick data in the defined lag range
    t = np.arange(-num_fft,num_fft)*sampling_interval
    ind = np.where(np.abs(t) <= lag)[0]
    corr_time_real = corr_time_real[:,ind]

    new_coords = patch.coords.correlate(**{dim: int(lag)})

    return patch.new(data=corr_time_real, coords=new_coords)
