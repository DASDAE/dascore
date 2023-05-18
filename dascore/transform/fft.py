"""
Modules for Fourier transforms.
"""
import numpy as np

from dascore.constants import PatchType
from dascore.utils.misc import _get_sampling_rate
from dascore.utils.patch import patch_function
from dascore.utils.transformatter import FourierTransformatter


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
    ft = FourierTransformatter()
    data = patch.data
    axis = patch.dims.index(dim)
    sr = 1 / _get_sampling_rate(patch.attrs[f"d_{dim}"])
    freqs = np.fft.rfftfreq(data.shape[axis], sr)
    new_data = np.fft.rfft(data, axis=axis)
    new_dim_name = ft._forward_rename(patch.dims[axis])
    dims, attrs = ft.transform_dims_and_attrs(patch.dims, patch.attrs, index=axis)
    new_coords = {new_dim_name: freqs}
    new_coords.update({x: patch.coords[x] for x in patch.dims if x != dim})
    return patch.__class__(data=new_data, coords=new_coords, dims=dims, attrs=attrs)
