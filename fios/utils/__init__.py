"""
Utilities for fios
"""
from collections import defaultdict
from typing import Optional, Union

from .time import to_datetime64

DFS_METHODS = defaultdict(dict)  # a dict for storing dtx attributes


def register_method(
    attr_name: Optional[Union[str, callable]] = None,
    namespace="fios",
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
            fios - generic distributed fiber sensing data
            dts - distributed temperature sensing data
            das - distributed acoustic data
            dss - distributed strain data
    """

    def decorator(func):
        if attr_name is None:
            name = func.__name__
        else:
            name = attr_name
        DFS_METHODS[namespace][name] = func
        return func

    return decorator
