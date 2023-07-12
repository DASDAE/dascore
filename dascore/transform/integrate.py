"""
Module for performing integration on patches.
"""
from dascore.constants import PatchType
from dascore.utils.patch import patch_function


@patch_function()
def integrate(patch: PatchType, dim: str, keep_dims=False) -> PatchType:
    """
    Integrate along a specified dimension using simpson's composite rule.

    Parameters
    ----------
    patch
        Patch object for integration.
    dim
        The dimension along which to integrate.
    keep_dims
        If True, collapse (remove) the integration dimension.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> integrated = patch.integrate(dim='time')
    """
