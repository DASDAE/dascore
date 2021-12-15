"""
Module for applying decimation to Patches.
"""

import dascore
from dascore.constants import PatchType
from dascore.exceptions import FilterValueError
from dascore.proc.filter import _get_sampling_rate, _lowpass_cheby_2
from dascore.utils.patch import patch_function


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
        If True, first apply a low-pass (anti-alis) filter. Uses
        :func:`dascore.proc.filter._lowpass_cheby_2`
    copy
        If True, copy the decimated data array. This is needed if you want
        the old array to get gc'ed to free memory otherwise a view is returned.
    """
    # Note: We can't simply use scipy.signal.decimate due to this issue:
    # https://github.com/scipy/scipy/issues/15072
    if lowpass:
        # get new niquest
        if factor > 16:
            msg = (
                "Automatic filter design is unstable for decimation "
                + "factors above 16. Manual decimation is necessary."
            )
            raise FilterValueError(msg)
        sr = _get_sampling_rate(patch, dim)
        freq = sr * 0.5 / float(factor)
        fdata = _lowpass_cheby_2(patch.data, freq, sr, axis=patch.dims.index(dim))
        patch = dascore.Patch(fdata, coords=patch.coords, attrs=patch.attrs)

    kwargs = {dim: slice(None, None, factor)}
    dar = patch._data_array.sel(**kwargs)
    # need to create a new xarray so the old, probably large, numpy array
    # gets gc'ed, otherwise it stays in memory (if lowpass isn't called)
    data = dar.data if not copy else dar.data.copy()
    attrs = dar.attrs
    # update delta_dim since spacing along dimension has changed
    d_attr = f"d_{dim}"
    attrs[d_attr] = patch.attrs[d_attr] * factor

    return dascore.Patch(data=data, coords=dar.coords, attrs=dar.attrs)


#
#
# @patch_function()
# def decimate(
#     patch: PatchType,
#     factor: int,
#     dim: str = "time",
#     low_pass=True,
# ) -> PatchType:
#     """
#     Decimate a patch along a dimension by first lowpassing the data.
#
#     Parameters
#     ----------
#     factor
#         The decimation factor (e.g., 10)
#     dim
#         dimension along which to decimate.
#
#     Note
#     ----
#     Simply calls scipy.signal.decimate on the data.
#     """
#     breakpoint()
#     axis = patch.dims.index(dim)
#     new_data = scipy_decimate(patch.data, factor, axis=axis)
#     # update coordinates
#     new_coord = patch.coords[dim][::factor]
#     assert len(new_coord) == new_data.shape[axis]
#     new_coords = update_coords(patch.coords, patch.dims, **{dim: new_coord})
#     # update delta_dim since spacing along dimension has changed
#     attrs = dict(patch.attrs)
#     d_attr = f"d_{dim}"
#     attrs[d_attr] = patch.attrs[d_attr] * factor
#     return dascore.Patch(data=new_data, coords=new_coords, attrs=attrs)
