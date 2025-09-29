"""Misc Utilities."""

from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import os
import re
import warnings
from collections.abc import Generator, Iterable, Mapping, Sequence, Sized
from functools import cache
from io import IOBase
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.special import factorial

import dascore as dc
from dascore.compat import is_array
from dascore.constants import WARN_LEVELS
from dascore.exceptions import (
    FilterValueError,
    MissingOptionalDependencyError,
    ParameterError,
)
from dascore.utils.progress import track


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
    if not behavior or behavior == "ignore":
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
    ar1, ar2 = np.asarray(ar1), np.asarray(ar2)
    if not ar1.shape == ar2.shape:
        return False
    ar1_null = pd.isnull(ar1)
    ar2_null = pd.isnull(ar2)
    try:
        close = np.isclose(ar1, ar2)
        bools = close | ar1_null | ar2_null
        return np.all(bools)
    except TypeError:
        return np.all(ar1 == ar2)


def _all_null(maybe_ar):
    """Return True if values is nullish, or all sub-values nullish if sequence."""
    out = pd.isnull(maybe_ar)
    out = out.all() if hasattr(out, "all") else out
    return out


def _get_nullish(dtype=np.floating):
    """Get nullish values for a given dtype."""
    if np.issubdtype(dtype, np.datetime64):
        return np.datetime64("NaT")
    elif np.issubdtype(dtype, np.timedelta64):
        return np.timedelta64("NaT")
    return np.nan


def _iter_filesystem(
    paths: str | Path | Iterable[str | Path],
    ext: str | None = None,
    timestamp: float | None = None,
    skip_hidden: bool = True,
    include_directories: bool = False,
) -> Generator[str, str, None]:
    """
    Iterate contents of a filesystem like thing.

    Options allow for filtering and terminating early.

    Parameters
    ----------
    paths
        The path to the base directory to traverse. Can also use a collection
        of paths.
    ext : str or None
        The extensions of files to return.
    timestamp : int or float
        Time stamp indicating the minimum mtime to scan.
    skip_hidden : bool
        If True skip files or folders (they begin with a '.')
    include_directories
        If True, also yield directories. In this case, a "skip" can be
        passed back to the generator to indicate the rest of the directory
        contents should be skipped.

    Yields
    ------
    Paths, as strings, meeting requirements.
    """
    # handle returning directories if requested.
    if include_directories and os.path.isdir(paths):
        if not (skip_hidden and str(paths).startswith(".")):
            signal = yield paths
            if signal is not None and signal == "skip":
                yield None
                return
    try:  # a single path was passed
        for entry in os.scandir(paths):
            if entry.is_file() and (ext is None or entry.name.endswith(ext)):
                if timestamp is None or entry.stat().st_mtime >= timestamp:
                    if entry.name[0] != "." or not skip_hidden:
                        yield entry.path
            elif entry.is_dir() and not (skip_hidden and entry.name[0] == "."):
                yield from _iter_filesystem(
                    entry.path,
                    ext=ext,
                    timestamp=timestamp,
                    skip_hidden=skip_hidden,
                    include_directories=include_directories,
                )
    except (TypeError, AttributeError):  # multiple paths were passed
        for path in paths:
            yield from _iter_filesystem(path, ext, timestamp, skip_hidden)
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


def optional_import(
    package_name: str, on_missing: Literal["raise", "warn", "ignore"] = "raise"
) -> ModuleType | None:
    """
    Import a module and return the module object if installed.

    If not installed, raise an Error or return None.

    Parameters
    ----------
    package_name
        The name of the package which may or may not be installed. Can
        also be sub-packages/modules (eg dascore.core).
    on_missing
        If "raise" raise an Error if missing, if "warn" or "ignore",
        return None.

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
    >>>
    >>> bob = optional_import('boblib5', on_missing="ignore")
    >>> assert bob is None
    """
    try:
        mod = importlib.import_module(package_name)
    except ImportError:
        msg = (
            f"{package_name} is not installed but is required for the "
            f"requested functionality."
        )
        warn_or_raise(msg, MissingOptionalDependencyError, behavior=on_missing)
        mod = None
    return mod


def get_middle_value(array):
    """Get the middle value in the differences array without changing dtype."""
    array = np.sort(np.asarray(array))
    last_ind = len(array) - 1
    ind = int(np.floor(last_ind / 2))
    return np.sort(array)[ind]


def all_diffs_close_enough(diffs):
    """Check if all the diffs are 'close' handling timedeltas."""
    if not len(diffs):
        return False
    diffs = np.asarray(diffs)
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


def yield_sub_sequences(sequence, length=None):
    """Yield subsequences of a sequence for specified length."""
    length = length if length is not None else len(sequence)
    for i in range(0, len(sequence), length):
        yield sequence[i : i + length]


def maybe_get_items(
    obj, attr_map: Mapping[str, str], unpack_names: None | set[str] = None
):
    """
    Maybe get items from a mapping (if they exist).

    Parameters
    ----------
    obj
        Any map like object.
    attr_map
        A mapping of {current_name: output_name}
    unpack_names
        A set of names which should be unpacked (ie collapse 0d arrays).
    """
    unpack_names = set() if unpack_names is None else unpack_names
    out = {}
    for old_name, new_name in attr_map.items():
        if not (value := obj.get(old_name, None)):
            continue
        val = unbyte(value)
        out[new_name] = _maybe_unpack(val) if old_name in unpack_names else val
    return out


def _maybe_unpack(maybe_array):
    """Unpack an array like object if it is size one, else return input."""
    size = getattr(maybe_array, "size", 0)
    if size == 1:
        maybe_array = maybe_array[0]
    return maybe_array


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
    # convert ellipses or ellipses values
    if select is None or select is Ellipsis:
        select = (None, None)
    # we allow a len(2) list here to not break old codes, but encourage a tuple.
    if not isinstance(select, (tuple | slice | list)) and select is not ...:
        msg = "Range values must be a tuple or slice."
        raise ParameterError(msg)
    # handle slices, need to convert to tuple
    if isinstance(select, slice):
        if select.step is not None:
            msg = (
                "Step not supported in select/filtering. Use decimate for "
                "proper down-sampling."
            )
            raise ParameterError(msg)
        select = (select.start, select.stop)

    # validate length (only length 2 allowed)
    if len(select) != 2:
        msg = "Range indices must be a length 2 sequence."
        raise ParameterError(msg)
    # swap out ellipses for None so downstream funcs dont have to
    select = tuple(None if x is ... else x for x in select)
    return select


def check_filter_sequence(filt_range):
    """Ensure the filter sequence is the right shape."""
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
    return filt_range


def check_filter_kwargs(kwargs):
    """Check filter kwargs and return dim name and filter range."""
    if len(kwargs) != 1:
        msg = "pass filter requires you specify one dimension and filter range."
        raise FilterValueError(msg)
    dim = next(iter(kwargs.keys()))
    filt_range = check_filter_sequence(kwargs[dim])
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


def _merge_tuples(dims1, dims2):
    """Merge tuples together, preserving order where possible."""
    dims = dict.fromkeys(dims1)
    dims.update(dict.fromkeys(dims2))
    out = tuple(dims.keys())
    return out


def _validate_sample_values(value):
    """
    Validate values, or ranges, which represent samples.
    """
    slice_ = _to_slice(value)
    start, stop = slice_.start, slice_.stop
    if not all(
        isinstance(v, (int | np.integer | type(None) | type(Ellipsis)))
        for v in (start, stop)
    ):
        msg = "When samples=True, values must be integers."
        raise ParameterError(msg)


def _to_slice(limits):
    """Convert slice or two len tuple to slice."""
    if isinstance(limits, slice):
        return limits
    # ints should be interpreted as Slice(int, int+1) to not collapse dim.
    if isinstance(limits, int):
        if limits == -1:  # -1 case needs open interval to work
            return slice(-1, None)
        return slice(limits, limits + 1)
    if limits is ... or limits is None:
        return slice(None, None)
    assert isinstance(limits, Sized) and len(limits) == 2
    val1, val2 = limits
    start = None if val1 is ... or val1 is None else val1
    stop = None if val2 is ... or val2 is None else val2
    return slice(start, stop)


def _apply_union_indexers(indexer, array):
    """
    Apply indexers to array getting the union of indices.

    For the case of multiple int arrays we actually don't want numpy's
    advanced indexing feature here but rather the union of the array.
    For example ar = [[1,2,3], [4,5,6], [7,8,9]]; ar[[0,1], [0,2]] returns
    [1, 6] but we want [[1, 3], [4,6]], so we have to break the index apart.
    We also want row/column independent boolean indexing, so whenever there
    is more than one array in the indexer we need to apply each independently.
    """
    if array is None:  # no array passed, just return.
        return array
    array_count = sum(is_array(x) for x in indexer)
    if array_count > 1:
        out = array
        ndim = len(out.shape)
        for axis, ind in enumerate(indexer):
            out = out[broadcast_for_index(ndim, axis, ind)]
    else:
        out = array[indexer]
    return out


def _maybe_array_to_slice(int_array, data_len):
    """
    Maybe convert an array of ints (indices) to a slice if it is sorted.
    """
    if len(int_array) < 2:
        return int_array
    diff = int_array[-1] - int_array[0]
    int_array_len = len(int_array)
    if diff == len(int_array) - 1:
        if np.all(int_array[1:] > int_array[:-1]):
            # this spans the whole array, use empty slice.
            if int_array_len == data_len:
                return slice(None)
            # otherwise return sub-slice.
            return slice(int_array[0], int_array[-1] + 1)
    return int_array


def to_object_array(object_sequence):
    """
    Convert a sequence of objects to a numpy array of objects.

    This is useful, eg, for storing an object array in a dataframe.
    """
    out = np.empty(len(object_sequence), dtype=object)
    out[:] = object_sequence
    return out


def get_buffer_size(fid: IOBase):
    """
    Get the size of a buffer in bytes.

    Parameters
    ----------
    fid
        A buffered reader, e.g. from open(file) as fid.
    """
    path = getattr(fid, "name", None)
    if path is None:
        cur = fid.tell()
        fid.seek(0, 2)  # end
        file_size = fid.tell()
        fid.seek(cur, 0)
    else:
        file_size = Path(path).stat().st_size
    return file_size


def maybe_mem_map(fid: IOBase, dtype="<u1") -> np.ndarray | np.memmap:
    """
    Try to get a memory map array from fid, otherwise just return array.

    Parameters
    ----------
    fid
        A buffered reader, e.g. from open(file) as fid.
    """
    try:
        raw = np.memmap(fid.name, dtype=dtype, mode="r")
    except (AttributeError, TypeError, ValueError):
        # Fallback: read into memory
        fid.seek(0)
        raw = np.frombuffer(fid.read(), dtype=dtype)
    return raw


def deep_equality_check(obj1, obj2, visited=None):
    """
    Deep equality comparison for dictionaries and nested objects.

    Handles circular references, numpy arrays, pandas DataFrames,
    and objects with __dict__ attributes. This function provides
    comprehensive equality checking that goes beyond Python's
    default equality operators.

    Parameters
    ----------
    obj1, obj2
        The objects to compare. Can be dictionaries, objects with __dict__,
        numpy arrays, pandas DataFrames, or any other objects.
    visited
        Set to track visited object pairs for circular reference detection.
        Internal parameter used during recursion.

    Returns
    -------
    bool
        True if the objects are deeply equal, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.misc import deep_equality_check
    >>>
    >>> # Basic usage
    >>> assert deep_equality_check({"a": 1}, {"a": 1})
    >>>
    >>> # With numpy arrays
    >>> dict1 = {"arr": np.array([1, 2, 3])}
    >>> dict2 = {"arr": np.array([1, 2, 3])}
    >>> assert deep_equality_check(dict1, dict2)
    """

    def _robust_equality_check(obj1, obj2):
        """Robust equality check to also handle arrays."""
        try:
            equal = obj1 == obj2
            # Handle numpy arrays and other array-like objects
            if hasattr(equal, "all"):
                result = equal.all()
                # Handle case where .all() returns a Series (e.g., pandas DataFrame)
                # In such cases, call .all() again to get a boolean
                if hasattr(result, "all"):
                    result = result.all()
                # Ensure we return a Python bool, not numpy.bool_
                return bool(result)
            return bool(equal)
        except (ValueError, TypeError):
            # For objects that can't be compared, fall back to False
            return False

    if visited is None:
        visited = set()
    # Create unique identifiers for the objects to detect cycles
    # Use the objects themselves for more accurate cycle detection
    obj1_id = id(obj1)
    obj2_id = id(obj2)
    pair_id = (obj1_id, obj2_id)
    # If we've already started comparing these exact objects,
    # avoid infinite recursion
    if pair_id in visited or (obj2_id, obj1_id) in visited:
        return True  # Equal for circular refs to avoid infinite recursion
    visited.add(pair_id)
    try:
        if not isinstance(obj1, Mapping) or not isinstance(obj2, Mapping):
            # Non-dict comparison, handle arrays and other types
            return _robust_equality_check(obj1, obj2)

        if (set1 := set(obj1)) != set(obj2):
            return False
        for key in set1:
            val1, val2 = obj1[key], obj2[key]
            # Check for object identity first to handle self-references
            if val1 is val2:
                continue
            elif isinstance(val1, Mapping) and isinstance(val2, Mapping):
                if not deep_equality_check(val1, val2, visited):
                    return False
            # this is primarily for dataframes which have equals method.
            elif hasattr(val1, "equals") and hasattr(val2, "equals"):
                if not val1.equals(val2):
                    return False
            # Handle object comparison carefully to avoid infinite recursion
            elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
                # For objects with __dict__, use recursive comparison
                if not deep_equality_check(val1.__dict__, val2.__dict__, visited):
                    return False
            else:
                if not _robust_equality_check(val1, val2):
                    return False
        return True
    finally:
        visited.remove(pair_id)
