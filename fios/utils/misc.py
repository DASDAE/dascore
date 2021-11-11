"""
Misc Utilities.
"""
import functools
from typing import Optional, Union


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
    def __init__(self, obj):
        self._obj = obj

    def __init_subclass__(cls, **kwargs):
        """wrap all public methods."""
        for key, val in vars(cls).items():
            if callable(val):  # passes to _NameSpaceMeta settattr
                setattr(cls, key, val)

    #


def _pass_through_method(func):
    """Decorator for marking functions as methods on namedspace parent class."""

    @functools.wraps(func)
    def _func(self, *args, **kwargs):
        obj = getattr(self, "_obj")
        return func(obj, *args, **kwargs)

    return _func


def pass_through_method(attr_name: str):
    """Decorator for binding method to an attribute rather than self."""

    def _wrap(func):
        @functools.wraps(func)
        def _func(self, *args, **kwargs):
            obj = getattr(self, attr_name)
            return func(obj, *args, **kwargs)

        return _func

    return _wrap
