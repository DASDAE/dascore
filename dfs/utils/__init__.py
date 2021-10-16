"""
Utilities for dfs
"""
from .time import to_datetime64


DTX_METHODS = {}  # a dict for storing dtx attributes


def dtx_method(attr_name):
    """decorator for adding a function to data array accessor. The
    first argument must always be the data array"""

    def decorator(func):
        DTX_METHODS[attr_name] = func
        return func

    return decorator
