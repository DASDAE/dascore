"""
Functionality for changing the data shape.
"""
from typing import Optional

import numpy as np

from fios.core import DataArray
from fios.utils import register_method


def _generic_trim(data_array, dim_name, start, stop):
    """Private Generic trim function."""
    array = data_array[dim_name]
    if start is None:
        start = array.min().values
    if stop is None:
        stop = array.max().values
    kwarg = {dim_name: slice(start, stop)}
    return data_array.sel(**kwarg)
