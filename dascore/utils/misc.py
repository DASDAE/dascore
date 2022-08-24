"""
Misc Utilities.
"""
import contextlib
import functools
import os
import warnings
from typing import Iterable, Optional, Union

import numpy as np
import numpy.typing as nt
import pandas as pd

from dascore.exceptions import ParameterError, PatchDimError


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


@contextlib.contextmanager
def suppress_warnings(category=Warning):
    """
    Context manager for suppressing warnings.

    Parameters
    ----------
    category
        The types of warnings to suppress. Must be a subclass of Warning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=category)
        yield
    return None


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


def get_dim_value_from_kwargs(patch, kwargs):
    """
    Assert that kwargs contain one value and it is a dimension of patch.

    Several patch functions allow passing values via kwargs which are dimension
    specific. This function allows for some sane validation of such functions.

    Return the name of the dimension, its axis position, and its value.
    """
    dims = patch.dims
    overlap = set(dims) & set(kwargs)
    if len(kwargs) != 1 or not overlap:
        msg = (
            "You must use exactly one dimension name in kwargs. "
            f"You passed the following kwargs: {kwargs} to a patch with "
            f"dimensions {patch.dims}"
        )
        raise PatchDimError(msg)
    dim = list(overlap)[0]
    axis = dims.index(dim)
    return dim, axis, kwargs[dim]


def all_close(ar1, ar2):
    """
    Return True if ar1 is allcose to ar2.

    Just uses numpy.allclose unless ar1 is a datetime, in which case
    strict equality is used.
    """
    is_date = np.issubdtype(ar1.dtype, np.datetime64)
    is_timedelta = np.issubdtype(ar1.dtype, np.timedelta64)
    if is_date or is_timedelta:
        return np.all(ar1 == ar2)
    else:
        return np.allclose(ar1, ar2)


def check_evenly_sampled(array: nt.ArrayLike):
    """
    Check if an array is evenly sampled.

    If not raise a ParameterError.
    """
    diff = np.diff(array)
    if not all_close(diff, np.mean(diff)):
        unique_diffs = np.unique(diff)
        msg = (
            "The passed array is not evenly sampled. The values you passed"
            f"have the following unique differences: {unique_diffs}"
        )
        raise ParameterError(msg)


def iter_files(
    paths: Union[str, Iterable[str]],
    ext: Optional[str] = None,
    mtime: Optional[float] = None,
    skip_hidden: bool = True,
) -> Iterable[str]:
    """
    Use os.scan dir to iter files, optionally only for those with given
    extension (ext) or modified times after mtime

    Parameters
    ----------
    paths
        The path to the base directory to traverse. Can also use a collection
        of paths.
    ext : str or None
        The extensions to map.
    mtime : int or float
        Time stamp indicating the minimum mtime.
    skip_hidden : bool
        If True skip files or folders (they begin with a '.')

    Yields
    ------
    Paths, as strings, meeting requirements.
    """
    try:  # a single path was passed
        for entry in os.scandir(paths):
            if entry.is_file() and (ext is None or entry.name.endswith(ext)):
                if mtime is None or entry.stat().st_mtime >= mtime:
                    if entry.name[0] != "." or not skip_hidden:
                        yield entry.path
            elif entry.is_dir() and not (skip_hidden and entry.name[0] == "."):
                yield from iter_files(
                    entry.path, ext=ext, mtime=mtime, skip_hidden=skip_hidden
                )
    except TypeError:  # multiple paths were passed
        for path in paths:
            yield from iter_files(path, ext, mtime, skip_hidden)
    except NotADirectoryError:  # a file path was passed, just return it
        yield paths


def iterate(obj):
    """
    Return an iterable from any object.

    If a string, do not iterate characters, return str in tuple.

    *This is how iteration *should* work in python.
    """
    if obj is None:
        return ()
    if isinstance(obj, str):
        return (obj,)
    return obj if isinstance(obj, Iterable) else (obj,)


class CacheDescriptor:
    """
    A descriptor for storing information in an instance-level cache (mapping).
    """

    def __init__(self, cache_name, func_name, args=None, kwargs=None):
        self._cache_name = cache_name
        self._func_name = func_name
        self._args = () if args is None else args
        self._kwargs = {} if kwargs is None else kwargs

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        cache = getattr(instance, self._cache_name)
        if self._name not in cache:
            func = getattr(instance, self._func_name)
            out = func(*self._args, **self._kwargs)
            cache[self._name] = out
        return cache[self._name]

    def __set__(self, instance, value):
        cache = getattr(instance, self._cache_name)
        cache[self._name] = value
