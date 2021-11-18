"""
Module for applying decimation to Patches.
"""
import fios
from fios.constants import PatchType
from fios.utils.patch import patch_function


@patch_function()
def decimate(
    patch: PatchType,
    factor: int,
    dim: str = "time",
    lowpass: bool = True,
    copy=True,
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
    copy
        If True, copy the decimated data array. This is needed if you want
        the old array to get gc'ed to free memory otherwise a view is returned.
    """
    if lowpass:
        raise NotImplementedError("working on it")
    kwargs = {dim: slice(None, None, factor)}
    out = patch._data_array.sel(**kwargs)
    # need to create a new xarray so the old, probably large, numpy array
    # gets gc'ed, otherwise it stays in memory.
    data = out.data if copy is False else out.data.copy()
    attrs = out.attrs
    # update delta_dim since spacing along dimension has changed
    d_attr = f"d_{dim}"
    attrs[d_attr] = patch.attrs[d_attr] * factor
    return fios.Patch(data=data, coords=out.coords, attrs=out.attrs)
