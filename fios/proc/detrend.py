"""
Module for detrending.
"""
import scipy

from fios.constants import PatchType
from fios.utils.patch import patch_function


@patch_function()
def detrend(patch: PatchType, dim="time", type="linear") -> PatchType:
    """Perform detrending along a given dimension."""
    assert dim in patch.dims
    axis = patch.dims.index(dim)
    out = scipy.signal.detrend(patch.data, axis=axis, type=type)
    return patch.new(data=out)
