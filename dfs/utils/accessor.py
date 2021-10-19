"""
Module to enable adding methods into the xarray namesapces.
"""
import functools
from typing import Literal

import xarray as xr

from dfs.utils import DFS_METHODS


def register_func(
    func_type: Literal["dfs", "dts", "das", "dss"],
):
    """
    Register a function as a specific type of accessor.

    Parameters
    ----------
    func_type
        The type of data expected. Options are:
        dfs - generic distributed fiber sensing data
        dts - distributed temperature sensing data
        das - distributed acoustic data
        dss - distributed strain data
    """


def _wrap_accessor_func(func):
    """
    Decorator to bind a function to the data_array contained by DetexAccessor.

    wrap a function that takes a data array as the first argument,
    bind the data array to first argument, update docs string and return
    """

    @functools.wraps(func)
    def new_func(self, *args, **kwargs):
        out = func(self._obj, *args, **kwargs)
        return out

    new_func.__doc__ = func.__doc__  # set doc string
    return new_func


def _add_wrapped_methods(cls):
    """add the methods marked with the dtx decorator to class def"""
    for name, func in DFS_METHODS.items():
        setattr(cls, name, _wrap_accessor_func(func))
    return cls


@xr.register_dataarray_accessor("dfs")
@_add_wrapped_methods
class DistributedFiberAccessor:
    """
    Class for registering distributed fiber methods.
    """

    def __init__(self, xarray_object):
        self._obj = xarray_object


@xr.register_dataarray_accessor("das")
@_add_wrapped_methods
class AcousticAccessor:
    """
    Class for registering attributes specific to distributed acoustic sensing.
    """

    def __init__(self, xarray_object):
        self._obj = xarray_object


@xr.register_dataarray_accessor("dts")
@_add_wrapped_methods
class TemperatureAccessor:
    """
    Class for registering attributes specific to distributed temperature sensing.
    """

    def __init__(self, xarray_object):
        self._obj = xarray_object


@xr.register_dataarray_accessor("dss")
@_add_wrapped_methods
class StrainAccessor:
    """
    Class for registering attributes specific to distributed strain sensing.
    """

    def __init__(self, xarray_object):
        self._obj = xarray_object
