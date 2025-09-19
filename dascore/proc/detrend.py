"""Module for detrending."""

from __future__ import annotations

from scipy.signal import detrend as scipy_detrend

from dascore.constants import PatchType
from dascore.utils.patch import patch_function


@patch_function()
def detrend(patch: PatchType, dim, type="linear") -> PatchType:
    """
    Perform detrending along a given dimension (distance or time) of a patch.

    Parameters
    ----------
    dim
        The dimension ("distance" or "time") along where detrending is applied.
    type
        Specifies least-squares fit type for detrend,
        with "linear" (default) or "constant" as options.

    Returns
    -------
    The Patch instance after applying the detrend function.

    Examples
    --------
    >>> import dascore # import dascore library
    >>> pa = dascore.get_example_patch() # generate example patch
    >>> out = pa.detrend("time") # detrend along the time dimension
    """
    assert dim in patch.dims
    axis = patch.get_axis(dim)
    out = scipy_detrend(patch.data, axis=axis, type=type)
    return patch.new(data=out)
