"""
Modules for Fourier transforms.



"""
from __future__ import annotations

from operator import mul

import numpy as np

from dascore.constants import PatchType
from dascore.core.coords import get_coord
from dascore.units import get_quantity
from dascore.utils.patch import (
    _get_data_units_from_dims,
    _get_dx_or_spacing_and_axes,
    patch_function,
)
from dascore.utils.time import to_float
from dascore.utils.transformatter import FourierTransformatter


@patch_function()
def rfft(patch: PatchType, dim: str | None) -> PatchType:
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

    This implementation uses a unitary transform (norm='ortho') such that
    Parseval's theorem holds.
    """

    axis = patch.dims.index(dim)
    _get_dx_or_spacing_and_axes(patch, dim)
    data = patch.data
    sr = 1 / to_float(patch.attrs[f"d_{dim}"])
    freqs = np.fft.rfftfreq(data.shape[axis], sr)
    new_data = np.fft.rfft(data, axis=axis, norm="ortho")
    # get new dims/attrs
    ft = FourierTransformatter()
    new_dim_name = ft._forward_rename(patch.dims[axis])
    dims, attrs = ft.transform_dims_and_attrs(patch.dims, patch.attrs, index=axis)
    # get new coord
    new_units = _get_data_units_from_dims(patch, dims, mul)
    attrs = patch.attrs.update(data_units=new_units)
    units = get_quantity(patch.coords.coord_map[dim].units)
    coord = get_coord(values=freqs, units=None if units is None else 1 / units)
    new_coords = {new_dim_name: coord}
    new_coords.update({x: patch.coords[x] for x in patch.dims if x != dim})
    return patch.__class__(data=new_data, coords=new_coords, dims=dims, attrs=attrs)


# @patch_function()
# def irfft(patch: PatchType, dim="time") -> PatchType:
#     """
#     Perform an inverse real fourier transform along the specified dimension.
#
#     Parameters
#     ----------
#     patch
#         The Patch object to transform.
#     dim
#         The dimension along which the transform is applied.
#
#     Notes
#     -----
#     The inverse real Fourier Transform is only appropriate for real-valued
#     data arrays.
#
#     This implementation uses a unitary transform (norm='ortho') such that
#     Parseval's theorem holds.
#     """
#     ft = FourierTransformatter()
#     data = patch.data
#     axis = patch.dims.index(dim)
#     sr = 1 / _get_sampling_rate(patch.attrs[f"d_{dim}"])
#     freqs = np.fft.rfftfreq(data.shape[axis], sr)
#     new_data = np.fft.rfft(data, axis=axis, norm="ortho")
#     # get new dims/attrs
#     new_dim_name = ft._forward_rename(patch.dims[axis])
#     dims, attrs = ft.transform_dims_and_attrs(patch.dims, patch.attrs, index=axis)
#     # get new coord
#     units = get_quantity(patch.coords.coord_map[dim].units)
#     coord = get_coord(values=freqs, units=None if units is None else 1 / units)
#     new_coords = {new_dim_name: coord}
#     new_coords.update({x: patch.coords[x] for x in patch.dims if x != dim})
#     return patch.__class__(data=new_data, coords=new_coords, dims=dims, attrs=attrs)
