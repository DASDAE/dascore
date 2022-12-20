"""
Module for detrending.
"""
from scipy.signal import detrend as scipy_detrend

from dascore.constants import PatchType
from dascore.utils.patch import patch_function


@patch_function()
def detrend(patch: PatchType, dim, type="linear") -> PatchType:
    """
    Perform detrending along a given dimension (distance or time) of a patch.


    Parameters
    ----------
    PatchType
        The patch instance you're applying the detrend function.
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
    >>> out = dascore.proc.detrend(pa, "distance") # detrend in the distance dimension
    """
    assert dim in patch.dims
    axis = patch.dims.index(dim)
    out = scipy_detrend(patch.data, axis=axis, type=type)
    return patch.new(data=out)
