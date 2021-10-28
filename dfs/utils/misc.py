"""
Misc Utilities.
"""

from typing import Union, Optional


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
