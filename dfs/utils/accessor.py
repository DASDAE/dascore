import functools

import xarray as xr

from dfs.utils import DTX_METHODS


def _wrap_accessor_func(func):
    """
    Decorator to bind a function to the data_array contained by DetexAccessor.

    wrap a function that takes a data array as the first argument,
    bind the data array to first argument, update docs string and return
    """
    functools.wraps(func)

    def new_func(self, *args, **kwargs):
        out = func(self._obj, *args, **kwargs)
        return out

    new_func.__doc__ = func.__doc__  # set doc string
    return new_func


def _add_wrapped_methods(cls):
    """add the methods marked with the dtx decorator to class def"""
    for name, func in DTX_METHODS.items():
        setattr(cls, name, _wrap_accessor_func(func))
    return cls


@xr.register_dataarray_accessor("dtx")
@_add_wrapped_methods
class DetexAccessor:
    """
    Class for registering some detex specific functionality on the data array
    """

    def __init__(self, xarray_object):
        self._obj = xarray_object
