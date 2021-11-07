"""
Utilities for working with the Patch class.
"""
from typing import Any, Dict, Tuple, Union, Sequence

from fios.constants import PatchType
from fios.exceptions import PatchDimError, PatchAttributeError

attr_type = Union[Dict[str, Any], str, Sequence[str], None]


import functools
import inspect
from typing import Union, Mapping, Callable, Optional

from typing_extensions import Literal


def _func_and_kwargs_str(func: Callable, patch, *args, **kwargs) -> str:
    """
    Get a str rep of the function and input args.
    """
    callargs = inspect.getcallargs(func, patch, *args, **kwargs)
    callargs.pop("patch")
    kwargs_ = callargs.pop("kwargs", {})
    arguments = []
    arguments += [f"{k}={repr(v)}" for k, v in callargs.items() if v is not None]
    arguments += [f"{k}={repr(v)}" for k, v in kwargs_.items() if v is not None]
    arguments.sort()
    out = f"{func.__name__}::"
    if arguments:
        out += f"{':'.join(arguments)}"
    return out


def _get_history_str(patch: PatchType, func, *args, _history="full", **kwargs) -> str:
    """
    Log history of a function being called on a Patch object.

    Parameters
    ----------
    patch
        The patch which will track history.
    func
        A callable which takes the patch as the first argument.
    *args
        The arguments passed to the function.
    _history
        String specifying how the history is to be recorded.
    **kwargs
        kwargs for func.
    """
    if _history is None:
        return None
    if _history == "full":
        history_str = _func_and_kwargs_str(func, patch, *args, **kwargs)
    else:
        history_str = str(func.__name__)
    return history_str


def check_patch_dims(
    patch: PatchType, required_dims: Optional[Tuple[str, ...]]
) -> PatchType:
    """
    Check if a patch object has required dimensions, else raise PatchDimError.

    Parameters
    ----------
    patch
        The input patch
    required_dims
        A tuple of required dimensions.
    """
    if required_dims is None:
        return patch
    dim_set = set(patch.dims)
    missing_dims = set(required_dims) - set(patch.dims)
    if not set(required_dims).issubset(dim_set):
        msg = f"patch is missing required dims: {tuple(missing_dims)}"
        raise PatchDimError(msg)
    return patch


def check_patch_attrs(patch: PatchType, required_attrs: attr_type) -> PatchType:
    """
    Check for expected attributes.

    Parameters
    ----------
    patch
        The patch to validate
    required_attrs
        The expected attrs. Can be a sequence or mapping. If sequence, only
        check that attrs exist. If mapping also check that values are equal.
    """
    if required_attrs is None:
        return patch
    # test that patch attr mapping is equal
    patch_attrs = patch.attrs
    if isinstance(required_attrs, Mapping):
        sub = {i: patch_attrs[i] for i in required_attrs}
        if sub != dict(required_attrs):
            msg = f"Patch's attrs {sub} are not required attrs: {required_attrs}"
            raise PatchAttributeError(msg)
    else:
        missing = set(required_attrs) - set(patch.attrs)
        if missing:
            msg = f"Patch is missing the following attributes: {missing}"
            raise PatchAttributeError(msg)
    return patch


def patch_function(
    required_dims: Optional[Union[Tuple[str, ...], Callable]] = None,
    required_attrs: attr_type = None,
    history: Literal["full", "method_name", None] = "full",
):
    """
    Decorator to mark a function as a patch method.

    Parameters
    ----------
    required_dims
        A tuple of dimensions which must be found in the Patch.
    required_attrs
        A dict of attributes which must be found in the Patch and whose
        values must be equal to those provided.
    history
        Specifies how to track history on Patch.
            Full - Records function name and str version of input arguments.
            method_name - Only records method name. Useful if args are long.
            None - Function call is not recorded in history attribute.

    Examples
    -------
    1. A patch method which requires dimensions (time, distance)
    >>> @fios.patch_function(dims=('time', 'distance'))
    >>> def do_something(patch):
    >>>     ...   # raises a PatchCoordsError if patch doesn't have time,
    >>>     #  distance

    2. A patch method which requires an attribute 'data_type' == 'DAS'
    >>> @fios.patch_function(required_attrs={'data_type': 'DAS'})
    >>> def do_another_thing(patch):
    >>>     ...  # raise PatchAttributeError if patch doesn't have attribute
    >>>     # called "data_type" or its values is not equal to "DAS".
    """

    def _wrapper(func):
        @functools.wraps(func)
        def _func(patch, *args, **kwargs):
            check_patch_dims(patch, required_dims)
            check_patch_attrs(patch, required_attrs)
            hist_str = _get_history_str(patch, func, *args, _history=history, **kwargs)
            out: PatchType = func(patch, *args, **kwargs)
            # attach history string. Consider something a bit less hacky.
            if hist_str:
                out._data_array.attrs["history"].append(hist_str)
            return out

        return _func

    if callable(required_dims):  # the decorator is used without parens
        return patch_function()(required_dims)

    return _wrapper
