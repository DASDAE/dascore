"""
Modules for Fourier transforms.
"""
import numpy as np

from dascore.constants import PatchType
from dascore.utils.misc import _get_sampling_rate
from dascore.utils.patch import patch_function


@patch_function()
def rfft(patch: PatchType, dim="time") -> PatchType:
    """
    Perform a real fourier transform along the specified dimension.

    Parameters
    ----------
    patch
        The Patch object to transform.
    dim
        The dimension along which the transform is applied.

    Notes
    -----
    The real Fourier Transform is only appropriate for real-valued data arrays.
    """
    data = patch.data
    axis = patch.dims.index(dim)
    new_dim_name = f"frequency_{dim}"
    sr = 1 / _get_sampling_rate(patch.attrs[f"d_{dim}"])
    freqs = np.fft.rfftfreq(data.shape[axis], sr)
    new_data = np.fft.rfft(data, axis=axis)
    new_dims = [(x if x != dim else new_dim_name) for x in patch.dims]
    new_coords = {new_dim_name: freqs}
    new_coords.update({x: patch.coords[x] for x in patch.dims if x != dim})
    return patch.__class__(data=new_data, coords=new_coords, dims=new_dims)
