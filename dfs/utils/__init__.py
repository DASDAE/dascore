"""
Utilities for dfs
"""
from collections import defaultdict
from typing import Optional, Union

from .time import to_datetime64

DFS_METHODS = defaultdict(dict)  # a dict for storing dtx attributes


def register_method(
        attr_name:Optional[Union[str, callable]]=None,
        namespace="dfs",
):
    """
    Register a function as a specific type of accessor.

    Parameters
    ----------
    attr_name
        The to which the method is registered. If none, just use the function
        name.
    namespace
        The type of data expected. Options are:
            dfs - generic distributed fiber sensing data
            dts - distributed temperature sensing data
            das - distributed acoustic data
            dss - distributed strain data
    """
    def decorator(func):
        DFS_METHODS[namespace][attr_name] = func
        return func

    return decorator


import dfs.utils.accessor  # NOQA this needs to be at the end of the file.
