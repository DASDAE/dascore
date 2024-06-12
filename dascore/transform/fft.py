"""
Deprecated module for Fourier transforms. Use
[fourier](`dascore.transform.fourier`) instead.
"""

from __future__ import annotations

import warnings
from operator import mul

import numpy as np

from dascore.constants import PatchType
from dascore.core.coords import get_coord
from dascore.units import get_quantity
from dascore.utils.patch import _get_data_units_from_dims, patch_function
from dascore.utils.time import to_float
from dascore.utils.transformatter import FourierTransformatter


@patch_function()
def rfft(patch: PatchType, dim="time") -> PatchType:
    """
    Perform a real fourier transform along the specified dimension.

    DEPRECATED FUNCTION: Use [dft](`dascore.transform.fourier.dft`) instead.
    This function is not scaled as detailed in the dascore documentation.
    """
    msg = "The Patch transform rfft is deprecated. Use dft instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    assert dim in patch.dims
    axis = patch.dims.index(dim)

    ft = FourierTransformatter()
    data = patch.data
    sr = 1 / to_float(patch.attrs[f"{dim}_step"])
    freqs = np.fft.rfftfreq(data.shape[axis], sr)
    new_data = np.fft.rfft(data, axis=axis)
    # get new dims and data units
    new_dims = ft.rename_dims(dim)
    new_data_units = _get_data_units_from_dims(patch, dim, mul)
    attrs = patch.attrs.update(data_units=new_data_units)
    dims = [x if i != axis else new_dims[0] for i, x in enumerate(patch.dims)]
    # get new coord
    units = get_quantity(patch.coords.coord_map[dim].units)
    coord = get_coord(data=freqs, units=None if units is None else 1 / units)
    new_coords = {new_dims[0]: coord}
    new_coords.update({x: patch.coords.get_array(x) for x in patch.dims if x != dim})
    return patch.__class__(data=new_data, coords=new_coords, dims=dims, attrs=attrs)
