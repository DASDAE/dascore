"""Module for calculating cross-correlation over all Patch's channels."""
from __future__ import annotations

import numpy as np
import scipy

import dascore as dc
from dascore.constants import PatchType
from dascore.units import Quantity
from dascore.utils.patch import get_dim_value_from_kwargs, patch_function

@patch_function()
def correlate(patch: PatchType, source : int | float | Quantity, samples=False, **kwargs
) -> PatchType:
    """
    This function takes the DAS data in time domain, calculates cross-correlation (cc) 
    in freq. domain, and returns the results back in time domain. 

    Parameters
    ----------
    patch : PatchType
        The input data patch to be cross-correlated.
    source :
        Virtual source, to which we cross-correlate all other channels/time samples.
    samples : bool, optional (default = False)
        {sample_explination}
    **kwargs
        Additional arguments to specify both dimension and lag for the cross-correlation.
        Refer to the examples section for detailed usage.
    
    Examples
    --------
    # Simple example for cross-correlation
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # Calculate cc with (-2,2) sec of lag for all channels as receivers and 
    >>> # the at 10 m as the master channel. 
    >>> cc_patch = patch.correlate(source = 10 * m, time = 2 * s)

    Notes
    -----
    The cross-correlation is performed in the frequency domain for efficiency reasons.
    """
    assert len(patch.dims) == 2, "must be 2D patch"
    dim, _, lag = get_dim_value_from_kwargs(patch, kwargs) 
    other_dim = list(set(patch.dims) - {dim})[0]
    other_axis = patch.dims.index(other_dim)

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
    fft_rec = scipy.fftpack.fft(patch.data, num_fft, axis=other_axis)[:,:num_fft_half]
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
    corr[:, :num_fft_half] -= np.mean(corr[:, :num_fft_half], axis=other_axis, keepdims=True)

    # Process the negative frequencies
    corr[:, -(num_fft_half)+1:] = np.flip(np.conj(corr[:, :num_fft_half]), axis=other_axis)

    # Set the zero-frequency component to zero
    corr[:, 0]=complex(0,0)

    # Take the inverse FFT
    inverse_fft_result = scipy.fftpack.ifft(corr, n=num_fft, axis=other_axis)

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
