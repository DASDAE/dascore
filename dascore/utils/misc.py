"""
Misc Utilities.
"""
import functools
from typing import Optional, Union

import numpy as np
import pandas as pd


def register_func(list_or_dict: Union[list, dict], key: Optional[str] = None):
    """
    Decorator for registering a function name in a list or dict.

    If list_or_dict is a list only append the name of the function. If it is
    as dict append name (as key) and function as the value.

    Parameters
    ----------
    list_or_dict
        A list or dict to which the wrapped function will be added.
    key
        The name to use, if different than the name of the function.
    """

    def wrapper(func):
        name = key or func.__name__
        if hasattr(list_or_dict, "append"):
            list_or_dict.append(name)
        else:
            list_or_dict[name] = func
        return func

    return wrapper


def append_func(some_list):
    """Decorator to append a function to a list."""

    def _func(func):
        some_list.append(func)
        return func

    return _func


class _NameSpaceMeta(type):
    """Metaclass for namespace class"""

    def __setattr__(self, key, value):
        if callable(value):
            value = _pass_through_method(value)
        super(_NameSpaceMeta, self).__setattr__(key, value)


class MethodNameSpace(metaclass=_NameSpaceMeta):
    """A namespace for class methods."""

    def __init__(self, obj):
        self._obj = obj

    def __init_subclass__(cls, **kwargs):
        """wrap all public methods."""
        for key, val in vars(cls).items():
            if callable(val):  # passes to _NameSpaceMeta settattr
                setattr(cls, key, val)


def _pass_through_method(func):
    """Decorator for marking functions as methods on namedspace parent class."""

    @functools.wraps(func)
    def _func(self, *args, **kwargs):
        obj = getattr(self, "_obj")
        return func(obj, *args, **kwargs)

    return _func


def get_slice(array, cond=Optional[tuple]) -> slice:
    """
    Return a slice object which meets conditions in cond on array.

    This is useful for determining how a particular dimension should be
    trimmed based on a coordinate (array) and an interval (cond).

    Parameters
    ----------
    array
        Any array with sorted values, but the array can have zeros
        at the end up to the sorted segment. Eg [1,2,3, 0, 0] works.
    cond
        An interval for which the array is to be indexed. End points are
        inclusive.

    Examples
    --------
    >>> import numpy as np
    >>> ar = np.arange(100)
    >>> array_slice = get_slice(ar, cond=(1, 10))
    """
    if cond is None:
        return slice(None, None)
    assert len(cond) == 2, "you must pass a length 2 tuple to get_slice."
    start, stop = None, None
    if not pd.isnull(cond[0]):
        start = np.searchsorted(array, cond[0], side="left")
        start = start if start != 0 else None
    if not pd.isnull(cond[1]):
        stop = np.searchsorted(array, cond[1], side="right")
        stop = stop if stop != len(array) else None
    # check for and handle zeroed end values
    if array[-1] <= array[0]:
        increasing_segment_end = np.argmin(np.diff(array))
        out = get_slice(array[:increasing_segment_end], cond)
        stop = np.min([out.stop or len(array), increasing_segment_end])
    return slice(start, stop)


def _get_sampling_rate(sampling_period):
    """
    Get a sampling rate as a float.
    """
    if np.issubdtype(sampling_period.dtype, np.timedelta64):
        num = np.timedelta64(1, "s")
    else:
        num = 1
    return num / sampling_period
