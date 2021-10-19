"""
Functionality for changing the data shape.
"""
from typing import Optional

import numpy as np

from dfs.core import DataArray
from dfs.utils import register_method


@register_method()
def trim_by_time(
    data_array: DataArray,
    start_time: Optional[np.datetime64] = None,
    end_time: Optional[np.datetime64] = None,
) -> DataArray:
    """
    Trim a data array by its time variable.

    Parameters
    ----------
    data_array
        The data array containing distributed fiber data.
    start_time
        The time the trim should start.
    end_time
        The time the trim should end.

    Returns
    -------
    Trimmed data array.

    Notes
    -----
    Following xarray indexing, times are inclusive.
    """
    return _generic_trim(data_array, "time", start_time, end_time)


def trim_by_distance(
    data_array: DataArray,
    start_distance: Optional[np.datetime64] = None,
    end_distance: Optional[np.datetime64] = None,
) -> DataArray:
    """
    Trim the Data Array by distance along the fiber.

    Parameters
    ----------
    data_array
        The data array to trim.
    start_distance
        The start of the trimming measured in m from interrogator.
    end_distance
        The end of the trimming measured in m from interrogator.

    Returns
    -------
    Trimmed Data Array.

    Notes
    -----
    Following xarray indexing, times are inclusive.

    """
    return _generic_trim(data_array, "distance", start_distance, end_distance)


def _generic_trim(data_array, dim_name, start, stop):
    """Private Generic trim function."""
    array = data_array[dim_name]
    if start is None:
        start = array.min().values
    if stop is None:
        stop = array.max().values
    kwarg = {dim_name: slice(start, stop)}
    return data_array.sel(**kwarg)
