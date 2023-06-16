"""
Misc Utilities.
"""
import contextlib
import functools
import importlib
import os
import warnings
from types import ModuleType
from typing import Iterable, Optional, Union

import numpy as np
import numpy.typing as nt
import pandas as pd

from dascore.exceptions import MissingOptionalDependency, ParameterError


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


def get_slice_from_monotonic(array, cond=Optional[tuple]) -> slice:
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
    >>> from dascore.utils.misc import get_slice_from_monotonic
    >>> ar = np.arange(100)
    >>> array_slice = get_slice_from_monotonic(ar, cond=(1, 10))
    """
    # TODO do we still need this or can we just use coordinates?
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
        out = get_slice_from_monotonic(array[:increasing_segment_end], cond)
        stop = np.min([out.stop or len(array), increasing_segment_end])
    return slice(start, stop)


def broadcast_for_index(
    n_dims: int, axis: int, value: Union[slice, int], fill_none=False
):
    """
    For a given shape of array, return empty slices except for slice axis.

    Parameters
    ----------
    n_dims
        The number of dimensions in the array that will be indexed.
    axis
        The axis number.
    value
        A slice object.
    fill_none
        If True, fill non axis dims with None, else slice(None)
    """
    fill = None if fill_none else slice(None)
    return tuple(fill if x != axis else value for x in range(n_dims))


def _get_sampling_rate(sampling_period):
    """
    Get a sampling rate as a float.
    """
    if np.issubdtype(sampling_period.dtype, np.timedelta64):
        num = np.timedelta64(1, "s")
    else:
        num = 1
    return num / sampling_period


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
    """A descriptor for storing infor in an instance cache (mapping)."""

    def __init__(self, cache_name, func_name, args=None, kwargs=None):
        self._cache_name = cache_name
        self._func_name = func_name
        self._args = () if args is None else args
        self._kwargs = {} if kwargs is None else kwargs

    def __set_name__(self, owner, name):
        """Method to set the name of the description on the instance."""
        self._name = name

    def __get__(self, instance, owner):
        """Get contents of the cache."""
        cache = getattr(instance, self._cache_name)
        if self._name not in cache:
            func = getattr(instance, self._func_name)
            out = func(*self._args, **self._kwargs)
            cache[self._name] = out
        return cache[self._name]

    def __set__(self, instance, value):
        """Set the cache contents."""
        cache = getattr(instance, self._cache_name)
        cache[self._name] = value


def optional_import(package_name: str) -> ModuleType:
    """
    Import a module and return the module object if installed, else raise error.

    Parameters
    ----------
    package_name
        The name of the package which may or may not be installed. Can
        also be sub-packages/modules (eg dascore.core).

    Raises
    ------
    MissingOptionalDependency if the package is not installed.

    Examples
    --------
    >>> from dascore.utils.misc import optional_import
    >>> from dascore.exceptions import MissingOptionalDependency
    >>> # import a module (this is the same as import dascore as dc)
    >>> dc = optional_import('dascore')
    >>> try:
    ...     optional_import('boblib5')  # doesn't exist so this raises
    ... except MissingOptionalDependency:
    ...     pass
    """
    try:
        mod = importlib.import_module(package_name)
    except ImportError:
        msg = (
            f"{package_name} is not installed but is required for the "
            f"requested functionality"
        )
        raise MissingOptionalDependency(msg)
    return mod


def _query_trims_range(query_tuple, lim1, lim2, spacing):
    """
    Return True if a query tuple should trim the range of limits.

    Parameters
    ----------
    query_tuple
        A tuple of (min, max) where both min and max are defined or
        one is None.
    lim1
        The lower limit.
    lim2
        The upper limit.
    spacing
        The spacing along dimension.
    """
    assert len(query_tuple) == 2, "only length two sequence allowed to specify range"
    q1, q2 = query_tuple
    if q1 is not None and q1 >= (lim1 + spacing):
        return True
    if q2 is not None and q2 <= (lim2 - spacing):
        return True
    return False


def trim_attrs_get_inds(attrs, dim_length, **kwargs):
    """
    Trim a dimension in attrs and get a slice for trimming data.

    Parameters
    ----------
    attrs
        A dict-able object which contains {dim}_min, {dim}_max, d_{dim}.
    dim_length
        The length of the data along the specified dimension.
    **kwargs
        Used to specify which dimension to trim (dim=(start, stop)).
    """
    # TODO Do we still need this or can we just use Coords?
    if not kwargs:
        return attrs, dim_length
    assert len(kwargs) == 1, "exactly one dimension allowed."
    dim = list(kwargs)[0]
    value = kwargs[dim]
    old_start, old_stop = attrs[f"{dim}_min"], attrs[f"{dim}_max"]
    spacing = attrs[f"d_{dim}"]
    if not _query_trims_range(value, old_start, old_stop, spacing):
        return slice(None), attrs
    out = dict(attrs)
    # get new start/stop values
    start_ind, stop_ind = 0, dim_length
    if value[0] is not None and value[0] > old_start:
        diff = value[0] - old_start
        start_ind = np.ceil(diff / spacing)
        out[f"{dim}_min"] = start_ind * spacing + old_start
    if value[1] is not None and value[1] < old_stop:
        diff = old_stop - value[1]
        diff_samples = -np.floor(diff / spacing)
        stop_ind = dim_length + diff_samples
        out[f"{dim}_max"] = old_stop + diff_samples * spacing
    return slice(int(start_ind), int(stop_ind)), attrs.__class__(**out)
