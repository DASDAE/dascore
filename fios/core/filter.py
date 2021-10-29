"""
Implements time and space filters.
"""
from typing import Optional, Literal

from fios.core import DataArray
from fios.utils import register_method


@register_method()
def bandpass(
    data_array: DataArray,
    dim: Literal["time", "distance"] = "time",
    start: Optional[float] = None,
    stop: Optional[float] = None,
) -> DataArray:
    """
    Apply a filter along the time dimension.

    Returns
    -------

    """


@register_method()
def bandstop(
    data_array: DataArray,
    dim: Literal["time", "distance"] = "time",
    start: Optional[float] = None,
    stop: Optional[float] = None,
) -> DataArray:
    """
    Apply a filter along the time dimension.

    Returns
    -------

    """
