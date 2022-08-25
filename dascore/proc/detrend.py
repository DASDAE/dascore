"""
Module for detrending.
"""
from scipy.signal import detrend as scipy_detrend

from dascore.constants import PatchType
from dascore.utils.patch import patch_function


@patch_function()
def detrend(patch: PatchType, dim, type="linear") -> PatchType:
    """Perform detrending along a given dimension."""
    assert dim in patch.dims
    axis = patch.dims.index(dim)
    out = scipy_detrend(patch.data, axis=axis, type=type)
    return patch.new(data=out)
