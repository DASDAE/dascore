"""
Module for performing differentiation on patches.
"""
from __future__ import annotations

import findiff

from dascore.constants import PatchType
from dascore.utils.patch import patch_function


@patch_function()
def differentiate(patch: PatchType, dim: str | None, order=2) -> PatchType:
    """
    Differentiate along specified dimension using centeral diferences.

    Parameters
    ----------
    patch
        The patch to differentiate.
    dim
        The dimension along which to differentiate. If None,

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> integrated = patch.diff(dim='time')
    """
    differ = findiff.FinDiff(1, 2, order)
    print(differ)
    # new = differ(patch.data)
    # attrs = dict(patch.attrs)
