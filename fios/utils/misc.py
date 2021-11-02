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


def pass_through_method(attr_name: str):
    """Decorator for binding method to an attribute rather than self."""

    def _wrap(func):
        @functools.wraps(func)
        def _func(self, *args, **kwargs):
            obj = getattr(self, attr_name)
            return func(obj, *args, **kwargs)

        return _func

    return _wrap
