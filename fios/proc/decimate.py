"""
Module for applying decimation to Patches.
"""
import fios
from fios.constants import PatchType
from fios.utils.patch import patch_function


@patch_function()
def decimate(
    patch: PatchType, factor: int, dim: str = "time", lowpass: bool = True
) -> PatchType:
    """
    Decimate a patch along a dimension.

    Parameters
    ----------
    factor
        The decimation factor (e.g., 10)
    dim
        dimension along which to decimate.
    lowpass
        If True, first apply a low-pass (anti-alis) filter.
    """
    if lowpass:
        raise NotImplementedError("working on it")
    kwargs = {dim: slice(None, None, factor)}
    out = patch._data_array.sel(**kwargs)
    return fios.Patch(out)
