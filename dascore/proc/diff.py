"""
Module for performing differentiation on patches.
"""
from dascore import patch_function
from dascore.constants import PatchType


@patch_function()
def diff(patch: PatchType, dim: str) -> PatchType:
    """
    Differentiate along specified dimension.

    Parameters
    ----------
    patch
        The patch to differentiate
    dim
        The dimension along which to differentiate.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> integrated = patch.diff(dim='time')
    """
