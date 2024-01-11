"""Misc Utilities."""
from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.special import factorial

import dascore as dc
from dascore.constants import WARN_LEVELS
from dascore.exceptions import (
    FilterValueError,
    MissingOptionalDependencyError,
    ParameterError,
)
from dascore.utils.progress import track


class _Sentinel:
    """Sentinel for key checks."""


def register_func(list_or_dict: list | dict, key=None):
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


def _pass_through_method(func):
    """Decorator for marking functions as methods on namedspace parent class."""

    @functools.wraps(func)
    def _func(self, *args, **kwargs):
        obj = self._obj
        return func(obj, *args, **kwargs)

    return _func


class _NameSpaceMeta(type):
    """Metaclass for namespace class."""

    def __setattr__(cls, key, value):
        if callable(value):
            value = _pass_through_method(value)
        super().__setattr__(key, value)


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


def warn_or_raise(
    msg: str,
    exception: type[Exception] = Exception,
    warning: type[Warning] = UserWarning,
    behavior: WARN_LEVELS = "warn",
):
    """
    A helper function to issues a warning, raise an exception or do nothing.

    Parameters
    ----------
    msg
        The message to attach to warning or exception.
    exception
        The exception class to raise.
    warning
        The type of warning to use. Must be a subclass of Warning.
    behavior
        If None, do nothing. If
    """
    if not behavior:
        return
    if behavior == "raise":
        raise exception(msg)
    warnings.warn(msg, warning)


class MethodNameSpace(metaclass=_NameSpaceMeta):
    """A namespace for class methods."""

    def __init__(self, obj):
        self._obj = obj

    def __init_subclass__(cls, **kwargs):
        """Wrap all public methods."""
        for key, val in vars(cls).items():
            if callable(val):  # passes to _NameSpaceMeta settattr
                setattr(cls, key, val)


def broadcast_for_index(
    n_dims: int,
    axis: int | Sequence[int],
    value: slice | int | None,
    fill=slice(None),
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
    fill
        The default values for non-axis entries.
    """
    axes = set(iterate(axis))
    return tuple(fill if x not in axes else value for x in range(n_dims))


def all_close(ar1, ar2):
    """
    Return True if ar1 is allcose to ar2.

    Just uses numpy.allclose unless ar1 is a datetime, in which case
    strict equality is used.
    """
    ar1, ar2 = np.array(ar1), np.array(ar2)
    try:
        return np.allclose(ar1, ar2)
    except TypeError:
        return np.all(ar1 == ar2)


def iter_files(
    paths: str | Iterable[str],
    ext: str | None = None,
    mtime: float | None = None,
    skip_hidden: bool = True,
) -> Iterable[str]:
    """
    Use os.scan dir to iter files, optionally only for those with given
    extension (ext) or modified times after mtime.

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
    >>> from dascore.exceptions import MissingOptionalDependencyError
    >>> # import a module (this is the same as import dascore as dc)
    >>> dc = optional_import('dascore')
    >>> try:
    ...     optional_import('boblib5')  # doesn't exist so this raises
    ... except MissingOptionalDependencyError:
    ...     pass
    """
    try:
        mod = importlib.import_module(package_name)
    except ImportError:
        msg = (
            f"{package_name} is not installed but is required for the "
            f"requested functionality"
        )
        raise MissingOptionalDependencyError(msg)
    return mod


def get_middle_value(array):
    """Get the middle value in the differences array without changing dtype."""
    array = np.sort(np.array(array))
    last_ind = len(array) - 1
    ind = int(np.floor(last_ind / 2))
    return np.sort(array)[ind]


def all_diffs_close_enough(diffs):
    """Check if all the diffs are 'close' handling timedeltas."""
    if not len(diffs):
        return False
    diffs = np.array(diffs)
    is_dt = np.issubdtype(diffs.dtype, np.timedelta64)
    is_td = np.issubdtype(diffs.dtype, np.datetime64)
    if is_td or is_dt:
        diffs = diffs.astype(np.int64).astype(np.float64)
    med = np.median(diffs)
    # Note: The rtol parameter here is a bit arbitrary; it was set
    # based on experience but there is probably a better way to do this.
    return np.allclose(diffs, med, rtol=0.001)


def unbyte(byte_or_str: bytes | str) -> str:
    """Ensure a string is given by str or possibly bytes."""
    if isinstance(byte_or_str, bytes | np.bytes_):
        byte_or_str = byte_or_str.decode("utf8")
    return byte_or_str


def _get_stencil_weights(array, ref_point, order):
    """
    Computes the derivative stencil weights.

    Parameters
    ----------
        array
            An array representing the stencil domain.
        ref_point
            The point in the domain to base the stencil weights on.
        order
            The order of the derivative.

    Returns
    -------
        The vector of stencil weights.
    """
    ell = np.arange(len(array))
    assert order in ell, "Order must be in domain"
    mat = (((array - ref_point)[:, np.newaxis] ** ell) / factorial(ell)).T
    weights = solve(mat, ell == order)
    return weights.flatten()


def _maybe_make_parent_directory(path):
    """Maybe make parent directories."""
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    return


def get_stencil_coefs(order, derivative=2):
    """Get centered coefficients for a derivative of specified order and derivative."""
    dx = np.arange(-order, order + 1)
    return _get_stencil_weights(dx, 0, derivative)


def get_parent_code_name(levels: int = 2) -> str:
    """Get the name of the calling function/class levels up in stack."""
    stack = inspect.currentframe()
    for _ in range(levels):
        stack = stack.f_back
    return stack.f_code.co_name


def to_str(val):
    """Convert value to string."""
    # This is primarily used to avoid lambdas which can cause issues
    # in pickling.
    return str(val)


def maybe_get_attrs(obj, attr_map: Mapping):
    """Maybe get attributes from object (if they exist)."""
    out = {}
    for old_name, new_name in attr_map.items():
        if hasattr(obj, old_name):
            value = getattr(obj, old_name)
            out[new_name] = unbyte(value)
    return out


def maybe_get_items(obj, attr_map: Mapping):
    """Maybe get items from a mapping (if they exist)."""
    out = {}
    for old_name, new_name in attr_map.items():
        if not (value := obj.get(old_name, None)):
            continue
        out[new_name] = unbyte(value)
    return out


@cache
def _get_compiled_suffix_prefix_regex(
    suffixes: str | tuple[str],
    prefixes: str | tuple[str] | None,
):
    """Get a compiled regex which matches the form prefixes_suffixes."""
    suffixes = iterate(suffixes)
    pattern = rf".*_({'|'.join(iterate(suffixes))})"
    if prefixes is not None:
        pattern = rf"({'|'.join(iterate(prefixes))})" + pattern
    return re.compile(pattern)


def _matches_prefix_suffix(input_str, suffixes, prefixes=None):
    """Determine if a string matches given prefixes_suffixes."""
    regex = _get_compiled_suffix_prefix_regex(suffixes, prefixes)
    return bool(re.match(regex, input_str))


def is_valid_coord_str(input_str, prefixes=None):
    """Return True if an input string is valid for representing coord info."""
    _valid_keys = tuple(dc.core.CoordSummary.model_fields)
    return _matches_prefix_suffix(input_str, _valid_keys, prefixes)


def separate_coord_info(
    obj,
    dims: tuple[str] | None = None,
    required: Sequence[str] | None = None,
    cant_be_alone: tuple[str] = ("units", "dtype"),
) -> tuple[dict, dict]:
    """
    Separate coordinate information from attr dict.

    These can be in the flat-form (ie {time_min, time_max, time_step, ...})
    or a nested coord: {coords: {time: {min, max, step}}

    Parameters
    ----------
    obj
        The object or model to
    dims
        The dimension to look for.
    required
        If provided, the required attributes (e.g., min, max, step).
    cant_be_alone
        names which cannot be on their own.

    Returns
    -------
    coord_dict and attrs_dict.
    """

    def _meets_required(coord_dict):
        """Return True coord dict meets the minimum required keys."""
        if not coord_dict:
            return False
        if not required and (set(coord_dict) - cant_be_alone):
            return True
        return set(coord_dict).issuperset(required)

    def _get_dims(obj):
        """Try to ascertain dims from keys in obj."""
        potential_keys = defaultdict(set)
        for key in obj:
            if not is_valid_coord_str(key):
                continue
            potential_keys[key.split("_")[0]].add(key.split("_")[1])
        return tuple(i for i, v in potential_keys.items() if _meets_required(v))

    def _get_coords_from_top_level(obj, out, dims):
        """First get coord info from top level."""
        for dim in iterate(dims):
            potential_coord = {
                i.split("_")[1]: v for i, v in obj.items() if is_valid_coord_str(i, dim)
            }
            # nasty hack for handling d_{dim} for backward compatibility.
            if (bad_name := f"d_{dim}") in obj:
                msg = f"d_{dim} is deprecated, use {dim}_step"
                warnings.warn(msg, DeprecationWarning, stacklevel=3)
                potential_coord["step"] = obj[bad_name]

            if _meets_required(potential_coord):
                out[dim] = potential_coord

    def _get_coords_from_coord_level(obj, out):
        """Get coords from coordinate level."""
        coords = obj.get("coords", {})
        if hasattr(coords, "to_summary_dict"):
            coords = coords.to_summary_dict()
        for key, value in coords.items():
            if hasattr(value, "to_summary"):
                value = value.to_summary()
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            if _meets_required(value):
                out[key] = value

    def _pop_keys(obj, out):
        """Pop out old keys for attrs, and unused keys from out."""
        # first coord subdict
        obj.pop("coords", None)
        # then top-level
        for coord_name, sub_dict in out.items():
            for thing_name in sub_dict:
                obj.pop(f"{coord_name}_{thing_name}", None)
            if "step" in sub_dict:
                obj.pop(f"d_{coord_name}", None)

    # sequence of short-circuit checks
    out = {}
    required = set(required) if required is not None else set()
    cant_be_alone = set(cant_be_alone)
    if obj is None:
        return out, {}
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    if dims is None:
        dims = _get_dims(obj)
    # this is already a dict of coord info.
    if set(dims) == set(obj):
        return obj, {}
    obj = dict(obj)
    _get_coords_from_coord_level(obj, out)
    _get_coords_from_top_level(obj, out, dims)
    _pop_keys(obj, out)
    return out, obj


def cached_method(func):
    """
    Cache decorated method.

    Simply uses the id of self for the key rather than hashing it.
    We can't use functools.cache due to pydantic #6787.
    """
    sentinel = object()  # unique object for cache misses.

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_cache"):
            self._cache = {}
        cache = self._cache
        if not (args or kwargs):
            key = id(func)
        else:
            key = (id(func), *args)
            if kwargs:
                for item in kwargs.items():
                    key += item
        out = cache.get(key, sentinel)
        if out is not sentinel:
            return out
        out = func(self, *args, **kwargs)
        cache[key] = out
        return out

    return wrapper


class _MapFuncWrapper:
    """A class for unwrapping spools to base applies."""

    def __init__(self, func, kwargs, progress=True):
        self._func = func
        self._kwargs = kwargs
        self._progress = progress

    def __call__(self, spool):
        iterable = spool
        # in order to handle multiprocessing, we apply a secret tag of "_progress"
        # to the first spool. This way only the first spool displays the
        # the progress bar. A huge hack, maybe there is a better way? See #265.
        if not getattr(spool, "_no_progress", False):
            desc = f"Applying {self._func.__name__} to spool"
            iterable = track(spool, desc) if self._progress else spool
        return [self._func(x, **self._kwargs) for x in iterable]


def _spool_map(spool, func, size=None, client=None, progress=True, **kwargs):
    """
    Map a func over a spool.

    Parameters
    ----------
    spool
        The spool object ot apply func to
    size
        The number of patches for each spool (ie chunksize)
    client
        An object with a map method for applying concurrency.
    progress
        If True, display a progress bar.
    **kwargs
        Keywords passed to func.
    """
    # no client; simple for loop.
    desc = f"Applying {func.__name__} to spool"
    if client is None:
        iterable = track(spool, desc) if progress else spool
        return [func(patch, **kwargs) for patch in iterable]
    # Now things get interesting. We need to split the spool here
    # so that patches don't get serialized.
    if size is None:
        size = len(spool) / os.cpu_count()
    spools = list(spool.split(size=size))
    # this is a hack to get the progress bar to work. Essentially, we just
    # add a secret flag to all but one spool so that progress bar is only
    # displayed in one thread/process.
    for sub_spool in spools[1:]:
        sub_spool._no_progress = True
    new_func = _MapFuncWrapper(func, kwargs, progress=progress)
    return [x for y in client.map(new_func, spools) for x in y]


def _dict_list_diffs(dict_list):
    """Return the keys which are not equal dicts in a list."""
    out = set()
    first = dict_list[0]
    first_keys = set(first)
    for other in dict_list[1:]:
        if other == first:
            continue
        other_keys = set(other)
        out |= (other_keys - first_keys) | (first_keys - other_keys)
        common_keys = other_keys & first_keys
        for key in common_keys:
            if first[key] != other[key]:
                out.add(key)
    return sorted(out)


def sanitize_range_param(select) -> tuple:
    """Given a slice or tuple, check and return slice or tuple."""
    # handle slices, need to convert to tuple
    if isinstance(select, slice):
        if select.step is not None:
            msg = (
                "Step not supported in select/filtering. Use decimate for "
                "proper down-sampling."
            )
            raise ParameterError(msg)
        select = (select.start, select.stop)
    # convert ellipses or ellipses values
    if select is None or select is Ellipsis:
        select = (None, None)
    # validate length (only length 2 allowed)
    if len(select) != 2:
        msg = "Range indices must be a length 2 sequence."
        raise ParameterError(msg)
    # swap out ellipses for None so downstream funcs dont have to
    select = tuple(None if x is ... else x for x in select)
    return select


def check_filter_kwargs(kwargs):
    """Check filter kwargs and return dim name and filter range."""
    if len(kwargs) != 1:
        msg = "pass filter requires you specify one dimension and filter range."
        raise FilterValueError(msg)
    dim = next(iter(kwargs.keys()))
    filt_range = kwargs[dim]
    # strip out units if used.
    mags = tuple([getattr(x, "magnitude", x) for x in filt_range])
    if not isinstance(filt_range, Sequence) or len(filt_range) != 2:
        msg = f"filter range must be a length two sequence not {filt_range}"
        raise FilterValueError(msg)
    if all([pd.isnull(x) for x in mags]):
        msg = (
            f"pass filter requires at least one filter limit, "
            f"you passed {filt_range}"
        )
        raise FilterValueError(msg)

    return dim, filt_range


def check_filter_range(nyquist, low, high, filt_min, filt_max):
    """Simple check on filter parameters."""
    # ensure filter bounds are within nyquist
    if low is not None and ((0 > low) or (low > 1)):
        msg = f"possible filter bounds are [0, {nyquist}] you passed {filt_min}"
        raise FilterValueError(msg)
    if high is not None and ((0 > high) or (high > 1)):
        msg = f"possible filter bounds are [0, {nyquist}] you passed {filt_max}"
        raise FilterValueError(msg)
    if high is not None and low is not None and high <= low:
        msg = (
            "Low filter param must be less than high filter param, you passed:"
            f"filt_min = {filt_min}, filt_max = {filt_max}"
        )
        raise FilterValueError(msg)
