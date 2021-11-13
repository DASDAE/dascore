"""
Implements time and space filters.
"""
from typing import Literal, Optional

from fios.constants import PatchType
from fios.utils import register_method


@register_method()
def bandpass(
    data_array: PatchType,
    dim: Literal["time", "distance"] = "time",
    start: Optional[float] = None,
    stop: Optional[float] = None,
) -> PatchType:
    """
    Apply a filter along the time dimension.

    Returns
    -------

    """


@register_method()
def bandstop(
    data_array: PatchType,
    dim: Literal["time", "distance"] = "time",
    start: Optional[float] = None,
    stop: Optional[float] = None,
) -> PatchType:
    """
    Apply a filter along the time dimension.

    Returns
    -------

    """
