"""
Module to enable adding methods into the xarray namesapces.
"""
import functools
from typing import Literal

import xarray as xr

from fios.utils import DFS_METHODS


def _wrap_accessor_func(func):
    """
    Decorator to bind a function to the data_array contained by Accessor.

    Wrap a function that takes a data array as the first argument,
    bind the data array to first argument, update docs string and return
    """

    @functools.wraps(func)
    def new_func(self, *args, **kwargs):
        out = func(self._obj, *args, **kwargs)
        return out

    return new_func


def _add_wrapped_methods(cls):
    """add the methods marked with the dtx decorator to class def"""
    namesapce = getattr(cls, "_namespace", None)
    sub_dict = DFS_METHODS.get(namesapce, {})
    for name, func in sub_dict.items():
        setattr(cls, name, _wrap_accessor_func(func))
    return cls


class _BaseAccessor:
    """
    Accessor class for fios.
    """

    _namespace = ""
    _accessors = {}

    def __init_subclass__(cls, **kwargs):
        cls._accessors[cls.__name__] = cls

    def __init__(self, xarray_object):
        self._obj = xarray_object

    def __getattr__(self, item):
        sub_dict = DFS_METHODS.get(self._namespace, {})
        if item not in sub_dict:
            msg = f"{self} has no attribute {item}"
            raise AttributeError(msg)
        # if the item is found, bind it as an instance and return.
        if item in sub_dict:
            new_func = _wrap_accessor_func(sub_dict[item])
            setattr(self.__class__, item, new_func)
        return getattr(self, item)


@xr.register_dataarray_accessor("fios")
@_add_wrapped_methods
class DistributedFiberAccessor(_BaseAccessor):
    """
    Class for registering distributed fiber methods.
    """

    _namespace = "fios"


@xr.register_dataarray_accessor("das")
@_add_wrapped_methods
class AcousticAccessor(_BaseAccessor):
    """
    Class for registering attributes specific to distributed acoustic sensing.
    """

    _namespace = "das"


@xr.register_dataarray_accessor("dts")
@_add_wrapped_methods
class TemperatureAccessor(_BaseAccessor):
    """
    Class for registering attributes specific to distributed temperature sensing.
    """

    _namespace = "dts"


@xr.register_dataarray_accessor("dss")
@_add_wrapped_methods
class StrainAccessor(_BaseAccessor):
    """
    Class for registering attributes specific to distributed strain sensing.
    """

    _namespace = "das"
