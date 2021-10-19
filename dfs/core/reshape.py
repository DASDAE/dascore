"""
Functionality for changing the data shape.
"""
from typing import Optional

import numpy as np
import pandas as pd

from dfs.core import DataArray
from dfs.utils import register_method


@register_method()
def trim_by_time(
    data_array: DataArray,
    start_time: Optional[np.datetime64] = None,
    end_time: Optional[np.datetime64] = None,
):
    """
    Trim by time.
    """
    # ensure both start and end times are defined.
    time = data_array['time']
    if start_time is None:
        start_time = time.min().values
    if end_time is None:
        end_time = time.max().values
    out = data_array.sel(time=(slice(start_time, end_time)))
    return out


# @register_method()
# def trim(
#     data_array:DataArray,
#     start_time: Optional[np.datetime64]=None,
#     end_time: Optional[np.datetime64]=None,
#     start_length: Optional[float] = None,
#     end_length: Optional[float] = None,
#     start_channel: Optional[int] = None,
#     end_channel: Optional[int] = None,
# ) -> DataArray:
#     """
#     Trim the DataArray based on space, time, or channel number.
#
#     Parameters
#     ----------
#     data_array
#         A data array of distributed fiber data.
#     starttime
#         The starttime to which the data should be trimed.
#
#     Returns
#     -------
#
#     """
#     breakpoint()
