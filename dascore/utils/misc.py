"""Misc Utilities."""

from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import itertools
import math
import os
import re
import warnings
from collections.abc import Generator, Iterable, Mapping, Sequence, Sized
from functools import cache
from io import IOBase
from pathlib import Path
from types import ModuleType
from typing import Literal, TypeVar, get_args, overload

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.special import factorial

from dascore.compat import UPath, is_array
from dascore.config import config_context, get_config
from dascore.constants import (
    PROGRESS_LEVELS,
    WARN_LEVELS,
    WARNING_ACTIONS,
    max_lens,
)
from dascore.exceptions import (
    FilterValueError,
    InvalidInventoryError,
    MissingOptionalDependencyError,
    ParameterError,
)
from dascore.utils.paths import (
    coerce_to_local_path,
    coerce_to_upath,
    is_local_path,
    is_pathlike,
)
from dascore.utils.progress import track, validate_progress_level

_T = TypeVar("_T")


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
        if isinstance(list_or_dict, list):
            list_or_dict.append(name)
        else:
            list_or_dict[name] = func
        return func

    return wrapper


@contextlib.contextmanager
def suppress_warnings(
    category=Warning,
    message: str | None = None,
    action: WARNING_ACTIONS = "ignore",
    record: bool = False,
):
    """
    Context manager for configuring warnings inside a local scope.

    Parameters
    ----------
    category
        The types of warnings to suppress. Must be a subclass of Warning.
    message
        Optional regex used to match warning messages.
    action
        The warning action to apply, such as ``"ignore"``, ``"error"``,
        or ``"always"``.
    record
        If True, yield the recorded warning list from ``catch_warnings``.
    """
    with warnings.catch_warnings(record=record) as caught:
        if message is None:
            warnings.simplefilter(action, category=category)
        else:
            warnings.filterwarnings(action, message=message, category=category)
        yield caught if record else None


def validate_warn_level(behavior, name: str = "behavior") -> WARN_LEVELS:
    """
    Ensure a warn-level argument is one of the supported values.

    Called where the argument arrives rather than where it is acted on:
    only an incompatible patch reaches `warn_or_raise`, so validating
    there would accept a retired spelling until the data made it matter.
    `name` is the parameter the caller spells it as, so the refusal names
    the argument the user actually passed.
    """
    if behavior not in get_args(WARN_LEVELS):
        msg = f"{name} must be one of {get_args(WARN_LEVELS)}, got {behavior!r}."
        raise ParameterError(msg)
    return behavior


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
        "warn" to issue the warning, "raise" to raise the exception, and
        "ignore" to do nothing. Anything else raises a ParameterError,
        so a retired spelling cannot quietly pick a behavior.
    """
    validate_warn_level(behavior)
    if behavior == "ignore":
        return
    if behavior == "raise":
        raise exception(msg)
    warnings.warn(msg, warning)


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
    Return True if ar1 is all close to ar2.

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
    """
    Return the value which stands for a missing entry of this dtype.

    Time-like dtypes get NaT at their own resolution rather than a fixed
    one, so a datetime64[us] array is filled with microsecond NaT and
    never has a nanosecond value cast into it. Everything else gets NaN,
    which means an integer array widens to float to hold the result.
    """
    if np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64):
        return np.array("NaT").astype(dtype)[()]
    return np.nan


def _iter_filesystem(
    paths: str | Path | UPath | Iterable[str | Path | UPath],
    ext: str | None = None,
    timestamp: float | None = None,
    skip_hidden: bool = True,
    include_directories: bool = False,
    _warned_timestamp_paths: set[str] | None = None,
) -> Generator[Path | UPath | None, str | None, None]:
    """
    Iterate contents of a filesystem like thing.

    Options allow for filtering and terminating early.

    Parameters
    ----------
    paths
        The path to the base directory to traverse. Can also use a collection
        of paths when ``include_directories`` is False.
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
    _warned_timestamp_paths
        Internal accumulator used to avoid repeating timestamp-filter warnings
        while recursing through one remote traversal.

    Yields
    ------
    Local paths as ``Path`` objects and remote paths as ``UPath`` objects.
    """
    warned_timestamp_paths = (
        set() if _warned_timestamp_paths is None else _warned_timestamp_paths
    )

    if is_pathlike(paths):
        if is_local_path(paths):
            yield from _iter_local_filesystem(
                coerce_to_local_path(paths),
                ext=ext,
                timestamp=timestamp,
                skip_hidden=skip_hidden,
                include_directories=include_directories,
                warned_timestamp_paths=warned_timestamp_paths,
            )
            return

        yield from _iter_remote_filesystem(
            coerce_to_upath(paths),
            ext=ext,
            timestamp=timestamp,
            skip_hidden=skip_hidden,
            include_directories=include_directories,
            warned_timestamp_paths=warned_timestamp_paths,
        )
        return

    for path in paths:
        yield from _iter_filesystem(
            path,
            ext=ext,
            timestamp=timestamp,
            skip_hidden=skip_hidden,
            include_directories=include_directories,
            _warned_timestamp_paths=warned_timestamp_paths,
        )


def _name_is_hidden(path_like) -> bool:
    """Return True if a path-like object's display name starts with a dot."""
    name = getattr(path_like, "name", "") or Path(str(path_like)).name
    return name.startswith(".")


def _passes_iter_filesystem_filters(
    path_like,
    *,
    ext: str | None,
    timestamp: float | None,
    skip_hidden: bool,
    warned_timestamp_paths: set[str],
    on_timestamp_failure: str,
) -> bool:
    """Apply extension, hidden-file, and timestamp filters for traversal."""
    if ext is not None and not str(path_like).endswith(ext):
        return False
    if skip_hidden and _name_is_hidden(path_like):
        return False
    if timestamp is None:
        return True
    try:
        return path_like.stat().st_mtime >= timestamp
    except (AttributeError, NotImplementedError, OSError):
        if on_timestamp_failure not in warned_timestamp_paths:
            msg = (
                f"Remote filesystem path {on_timestamp_failure} does not "
                "expose reliable mtime; ignoring timestamp filter."
            )
            warnings.warn(msg, UserWarning)
            warned_timestamp_paths.add(on_timestamp_failure)
        return True


def _iter_local_filesystem(
    path,
    *,
    ext: str | None,
    timestamp: float | None,
    skip_hidden: bool,
    include_directories: bool,
    warned_timestamp_paths: set[str],
) -> Generator[Path | None, str | None, None]:
    """Traverse one local path or directory tree."""
    # Local paths use scandir directly so we can recurse cheaply and keep the
    # hidden/timestamp checks close to the yielded entries.
    path = Path(path)
    if include_directories and os.path.isdir(path):
        if not (skip_hidden and path.name.startswith(".")):
            signal = yield path
            if signal is not None and signal == "skip":
                yield None
                return
    try:
        for entry in os.scandir(path):
            if entry.is_file() and (ext is None or entry.name.endswith(ext)):
                if timestamp is None or entry.stat().st_mtime >= timestamp:
                    if entry.name[0] != "." or not skip_hidden:
                        yield Path(entry.path)
            elif entry.is_dir() and not (skip_hidden and entry.name[0] == "."):
                yield from _iter_local_filesystem(
                    Path(entry.path),
                    ext=ext,
                    timestamp=timestamp,
                    skip_hidden=skip_hidden,
                    include_directories=include_directories,
                    warned_timestamp_paths=warned_timestamp_paths,
                )
    except NotADirectoryError:
        if _passes_iter_filesystem_filters(
            path,
            ext=ext,
            timestamp=timestamp,
            skip_hidden=skip_hidden,
            warned_timestamp_paths=warned_timestamp_paths,
            on_timestamp_failure=str(path),
        ):
            yield path


def _iter_remote_filesystem(
    path: UPath,
    *,
    ext: str | None,
    timestamp: float | None,
    skip_hidden: bool,
    include_directories: bool,
    warned_timestamp_paths: set[str],
) -> Generator[UPath | None, str | None, None]:
    """Traverse one remote path defensively across backend quirks."""
    # Remote traversal has to be more defensive because some backends have
    # partial directory support and may not expose reliable stat metadata.
    remote_is_dir = path.is_dir()
    if include_directories and remote_is_dir:
        if not (skip_hidden and _name_is_hidden(path)):
            signal = yield path
            if signal is not None and signal == "skip":
                yield None
                return
    # Some remote backends can report directory URLs as both files and
    # directories. Prefer directory traversal so recursion still works.
    elif path.is_file():
        if _passes_iter_filesystem_filters(
            path,
            ext=ext,
            timestamp=timestamp,
            skip_hidden=skip_hidden,
            warned_timestamp_paths=warned_timestamp_paths,
            on_timestamp_failure=str(path),
        ):
            yield path
        return
    # Only probe directory contents after ruling out the simple file case; some
    # remote backends raise here instead of answering is_dir/is_file.
    try:
        entries = iter(path.iterdir())
        first_entry = next(entries)
    except StopIteration:
        return
    except (
        AttributeError,
        FileNotFoundError,
        NotADirectoryError,
        NotImplementedError,
        OSError,
    ):
        if not path.exists():
            return
        raise
    # Once we have one entry, recurse through remote directories and apply the
    # same file filters to leaf files.
    for entry in itertools.chain((first_entry,), entries):
        if skip_hidden and _name_is_hidden(entry):
            continue
        entry_is_file = entry.is_file()
        entry_is_dir = entry.is_dir()
        if not entry_is_file and not entry_is_dir:
            try:
                next(entry.iterdir())
            except (AttributeError, NotImplementedError, OSError, StopIteration):
                entry_is_dir = False
            else:
                entry_is_dir = True
        if entry_is_dir:
            yield from _iter_remote_filesystem(
                entry,
                ext=ext,
                timestamp=timestamp,
                skip_hidden=skip_hidden,
                include_directories=include_directories,
                warned_timestamp_paths=warned_timestamp_paths,
            )
            continue
        if not entry_is_file:
            continue
        if _passes_iter_filesystem_filters(
            entry,
            ext=ext,
            timestamp=timestamp,
            skip_hidden=skip_hidden,
            warned_timestamp_paths=warned_timestamp_paths,
            on_timestamp_failure=str(path),
        ):
            yield entry


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


# Import names whose installable package name differs from the import name.
_INSTALL_NAMES = {
    "google.protobuf": "protobuf",
    "yaml": "pyyaml",
}


def _get_install_name(import_name: str) -> str:
    """Get the package to install which provides the module import_name."""
    parts = import_name.split(".")
    # Search most to least specific so sub-modules resolve to their parent.
    for stop in range(len(parts), 0, -1):
        if (name := _INSTALL_NAMES.get(".".join(parts[:stop]))) is not None:
            return name
    return parts[0]


def _get_install_message(packages: str | Iterable[str]) -> str:
    """Get a message telling the user how to install one or more packages."""
    names = " ".join(sorted(iterate(packages)))
    return f"Install with `pip install {names}` or `uv pip install {names}`."


def _is_missing_module(import_name: str, error: ImportError) -> bool:
    """Determine if an ImportError means import_name itself is not installed."""
    # Only the import machinery proves a module is absent, and it names the
    # module it failed to find. Any other error (eg an installed package
    # raising from its __init__) means the failure happened inside code which
    # is installed, as does one naming a different module.
    if not isinstance(error, ModuleNotFoundError) or not (failed := error.name or ""):
        return False
    return import_name == failed or import_name.startswith(f"{failed}.")


@overload
def optional_import(
    package_name: str,
    on_missing: Literal["raise"] = "raise",
    required_for: str = "the requested functionality",
) -> ModuleType: ...


@overload
def optional_import(
    package_name: str,
    on_missing: Literal["warn", "ignore"],
    required_for: str = "the requested functionality",
) -> ModuleType | None: ...


def optional_import(
    package_name: str,
    on_missing: WARN_LEVELS = "raise",
    required_for: str = "the requested functionality",
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
    required_for
        A string indicating what this import is required for.

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
    except ImportError as ex:
        install_name = _get_install_name(package_name)
        if _is_missing_module(package_name, ex):
            msg = (
                f"{package_name} is not installed but is required for "
                f"{required_for}. {_get_install_message(install_name)}"
            )
        else:
            # The package is installed; something it imports is not, so
            # installing it again wouldn't help.
            install_name = None
            msg = (
                f"{package_name} could not be imported ({ex}) but is "
                f"required for {required_for}."
            )
        # Raise here (rather than with warn_or_raise) so the install name is
        # attached to the exception for callers which aggregate them.
        if on_missing == "raise":
            raise MissingOptionalDependencyError(msg, install_name=install_name) from ex
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
    is_td = np.issubdtype(diffs.dtype, np.timedelta64)
    is_dt = np.issubdtype(diffs.dtype, np.datetime64)
    if is_dt or is_td:
        null_mask = np.isnat(diffs)
        diffs = diffs[~null_mask]
        diffs = diffs.astype(np.int64).astype(np.float64)
    else:
        diffs = diffs[~np.isnan(diffs)]
    if not len(diffs):
        return False
    med = np.median(diffs)
    # Note: The rtol parameter here is a bit arbitrary; it was set
    # based on experience but there is probably a better way to do this.
    return np.allclose(diffs, med, rtol=0.001)


@overload
def unbyte(byte_or_str: bytes) -> str: ...


@overload
def unbyte(byte_or_str: _T) -> _T: ...


def unbyte(byte_or_str) -> str | _T:
    """
    Decode a bytes value, passing anything else through unchanged.

    Callers use this to normalize values which may or may not have come
    from a binary file, so the non-bytes case is the common one.
    """
    if isinstance(byte_or_str, bytes | np.bytes_):
        return byte_or_str.decode("utf8")
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
    frame = inspect.currentframe()
    for _ in range(levels):
        frame = frame.f_back if frame is not None else None
    if frame is None:  # asked for a frame above the top of the stack
        return "<unknown>"
    return frame.f_code.co_name


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
    obj, attr_map: Mapping[str, str], unpack_names: set[str] | None = None
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
        if (value := obj.get(old_name, None)) is None:
            continue
        val = unbyte(value)
        out[new_name] = _maybe_unpack(val) if old_name in unpack_names else val
    return out


def _maybe_unpack(maybe_array):
    """Unpack a single-element array-like object, else return input unchanged."""
    size = getattr(maybe_array, "size", 0)
    shape = getattr(maybe_array, "shape", ())
    if size == 1 and shape:
        maybe_array = maybe_array[0]
    return maybe_array


@cache
def _get_compiled_suffix_prefix_regex(
    suffixes: str | tuple[str, ...],
    prefixes: str | tuple[str, ...] | None,
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


def _locked(lock_name: str):
    """
    Run the decorated method while holding one of its owner's locks.

    Parameters
    ----------
    lock_name
        Name of the attribute (on self or cls) holding the lock, eg "_lock".
        It is looked up on each call so the owner can swap in a fresh lock
        (eg after a fork).

    Notes
    -----
    Not for generator functions; the lock would be held across the
    consumer's iteration rather than the function body.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(self, *args, **kwargs):
            with getattr(self, lock_name):
                return func(self, *args, **kwargs)

        return _wrapper

    return _decorator


def _reinit_after_fork(func):
    """
    Register a callable to run in the child process after a fork.

    A fork can copy a lock while another thread holds it. That thread does
    not exist in the child, so the inherited copy would never be released.
    Hooks registered here install fresh locks in the child.
    """
    if hasattr(os, "register_at_fork"):  # not available on windows
        os.register_at_fork(after_in_child=func)
    return func


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


def _callable_name(func) -> str:
    """
    Return a human-readable name for a callable's progress label.

    A `functools.partial` is named for the function it wraps. A callable
    object and a `Task` have no `__name__`; a `Task` standing for more than
    one operation names itself with `node_name`, and anything else falls
    back to its class name.
    """
    while isinstance(func, functools.partial):
        func = func.func
    name = getattr(func, "__name__", None) or getattr(func, "node_name", None)
    return name or type(func).__name__


class _MapFuncWrapper:
    """A class for unwrapping spools to base applies."""

    def __init__(self, func, kwargs, config, progress: PROGRESS_LEVELS = "standard"):
        self._func = func
        self._kwargs = kwargs
        self._progress = progress
        # Bind the config active at map() call time so workers (threads or
        # pickled into processes) apply the same config the caller had, rather
        # than a fresh default or a scoped override that would not propagate.
        self._config = config

    def __call__(self, spool):
        with config_context(self._config):
            iterable = spool
            # in order to handle multiprocessing, we apply a secret tag of
            # "_progress" to the first spool. This way only the first spool
            # displays the progress bar. A huge hack, maybe there is a better
            # way? See #265.
            if not getattr(spool, "_no_progress", False):
                desc = f"Applying {_callable_name(self._func)} to spool"
                iterable = track(spool, desc, self._progress)
            return [self._func(x, **self._kwargs) for x in iterable]


def _spool_map(
    spool,
    func,
    size=None,
    client=None,
    progress: PROGRESS_LEVELS = "standard",
    **kwargs,
):
    """
    Map a func over a spool.

    Parameters
    ----------
    spool
        The spool object to apply func to
    size
        The number of patches for each spool (ie chunksize)
    client
        An object with a map method for applying concurrency.
    progress
        Controls the progress bar; see `PROGRESS_LEVELS`.
    **kwargs
        Keywords passed to func.
    """
    # Normalized before branching: with a client and an empty spool no
    # patch is ever tracked, so an unsupported level would be accepted or
    # refused depending on how much data there was.
    progress = validate_progress_level(progress)
    # no client; simple for loop.
    if client is None:
        desc = f"Applying {_callable_name(func)} to spool"
        return [func(patch, **kwargs) for patch in track(spool, desc, progress)]
    # Now things get interesting. We need to split the spool here
    # so that patches don't get serialized.
    if size is None:
        # split takes a positive patch count, so round up rather than hand
        # it a float, and keep an empty spool from asking for zero.
        size = max(1, math.ceil(len(spool) / (os.cpu_count() or 1)))
    spools = list(spool.split(size=size))
    # this is a hack to get the progress bar to work. Essentially, we just
    # add a secret flag to all but one spool so that progress bar is only
    # displayed in one thread/process.
    for sub_spool in spools[1:]:
        sub_spool._no_progress = True
    new_func = _MapFuncWrapper(func, kwargs, get_config(), progress=progress)
    return [x for y in client.map(new_func, spools) for x in y]


def is_range(value) -> bool:
    """True for a 2-tuple range (a ``(start, stop)`` selector)."""
    return isinstance(value, tuple) and len(value) == 2


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
    # swap out ellipses for None so downstream funcs don't have to
    select = tuple(None if x is ... else x for x in select)
    return select


class _CanonicalRange:
    """
    A numeric coordinate range with unit-bearing bounds in canonical SI.

    ``magnitudes`` and ``units`` align per bound: a bound that carried
    units holds its SI magnitude beside its base unit string, a bare
    bound holds its raw value beside None, and an open bound is None in
    both. The exact per-patch re-select defers its representation until
    the target patch is known: on a unit-bearing coordinate, canonical
    bounds become quantities (`Patch.select` converts them to native
    units) while bare bounds stay raw and mean the coordinate's own
    units, exactly as `Patch.select` reads them. Unitless coordinates
    get bare magnitudes for every bound (documented policy: unitless
    values cannot be proven incompatible and read SI magnitudes
    directly). Units travel per bound so the residual preserves each
    bound's own meaning — a metre query must never trim a seconds
    coordinate as 1-2 s, and a bare bound beside a metre bound must not
    turn into metres.
    """

    __slots__ = ("magnitudes", "units")

    def __init__(self, magnitudes: tuple, units: tuple):
        assert len(magnitudes) == len(units)
        self.magnitudes = magnitudes
        self.units = units

    def __eq__(self, other) -> bool:
        """Value equality so equal selections compare equal (spool __eq__)."""
        if not isinstance(other, _CanonicalRange):
            return NotImplemented
        return (self.magnitudes, self.units) == (other.magnitudes, other.units)

    def __hash__(self) -> int:
        return hash((self.magnitudes, self.units))

    def for_patch_coord(self, coord) -> tuple:
        """Return the range in the representation this coord needs."""
        # deferred: dascore.units imports this module at import time
        from dascore.units import get_quantity  # noqa: PLC0415

        coord_units = getattr(coord, "units", None)
        if coord_units is None:
            # unitless coords: bare magnitudes for every bound
            return self.magnitudes
        return tuple(
            mag if mag is None or unit is None else mag * get_quantity(unit)
            for mag, unit in zip(self.magnitudes, self.units, strict=True)
        )

    def magnitudes_in(self, unit: str) -> tuple:
        """Return the bound magnitudes re-expressed in one target unit.

        Canonical bounds convert; bare bounds already mean the target's
        native units and stay raw. This is the one kernel every
        envelope-side consumer shares, so the bare-stays-native rule
        lives in exactly one place.
        """
        # deferred: dascore.units imports this module at import time
        from dascore.units import convert_units  # noqa: PLC0415

        return tuple(
            mag
            if mag is None or bound_units is None
            else convert_units(mag, to_units=unit, from_units=bound_units)
            for mag, bound_units in zip(self.magnitudes, self.units, strict=True)
        )


def _canonical_range(value) -> _CanonicalRange | None:
    """Return the canonical per-bound form of a numeric range, or None.

    None also covers the all-bare range: bare bounds mean native units
    and never canonicalize, so a range only becomes a _CanonicalRange
    when at least one bound carries units.
    """
    if not is_range(value):
        return None
    magnitudes = []
    units = []
    for bound in value:
        if bound is None or bound is Ellipsis:
            magnitudes.append(None)
            units.append(None)
        elif hasattr(bound, "units"):  # pint quantity -> SI magnitude
            base = bound.to_base_units()
            magnitudes.append(float(base.magnitude))
            units.append(str(base.units))
        elif isinstance(bound, bool | np.bool_):
            return None
        elif isinstance(bound, np.datetime64 | np.timedelta64):
            # Time bounds are never a canonical-SI numeric range. timedelta64
            # needs naming here because numpy makes it an np.integer subclass,
            # so it would otherwise reach float() below and raise.
            return None
        elif isinstance(bound, int | float | np.integer | np.floating):
            magnitudes.append(float(bound))
            units.append(None)
        else:  # datetimes, strings: not a numeric range
            return None
    if all(mag is None for mag in magnitudes):
        return None
    if all(unit is None for unit in units):
        return None
    return _CanonicalRange(tuple(magnitudes), tuple(units))


def express_range_for_coord(value, coord):
    """
    Re-express one index-side range in what a patch coordinate needs.

    A range with any unit-bearing bound carries a `_CanonicalRange`,
    whose representation can be chosen only here, where the coordinate
    is finally known (a bare bound riding inside one stays raw there
    too). Everything else — all-bare value ranges (native units),
    sample-index ranges, times, strings — passes through untouched.
    """
    if isinstance(value, _CanonicalRange):
        return value.for_patch_coord(coord)
    return value


def order_range_tuple(range_tuple):
    """Ensure finite range bounds are in increasing order."""
    val_min, val_max = range_tuple
    if val_min is not None and val_max is not None and val_max < val_min:
        return val_max, val_min
    return val_min, val_max


def check_filter_sequence(filt_range):
    """Ensure the filter sequence is the right shape."""
    if not isinstance(filt_range, Sequence) or len(filt_range) != 2:
        msg = (
            f"filter range must be a length two sequence of (low, high), "
            f"not {filt_range}. Use None or ... for an open end, "
            f"e.g. time=(None, 10) for a 10 Hz lowpass."
        )
        raise FilterValueError(msg)
    # strip out units if used.
    mags = tuple([getattr(x, "magnitude", x) for x in filt_range])
    if all([pd.isnull(x) for x in mags]):
        msg = f"pass filter requires at least one filter limit, you passed {filt_range}"
        raise FilterValueError(msg)
    return filt_range


def raise_on_extra_kwargs(kwargs, accepted: str) -> None:
    """
    Raise ParameterError when a call received keyword arguments it does not take.

    Parameters
    ----------
    kwargs
        The leftover keyword arguments.
    accepted
        What the call does accept, for the message.
    """
    if kwargs:
        msg = (
            f"Unexpected keyword argument(s) {sorted(kwargs)}; "
            f"only {accepted} are accepted."
        )
        raise ParameterError(msg)


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
    if isinstance(limits, int | np.integer):
        # Convert numpy scalars to Python ints to avoid dtype overflow on +1.
        limits = int(limits)
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
    for num, value in enumerate(object_sequence):
        out[num] = value
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
    if path is not None and is_local_path(path):
        try:
            return Path(path).stat().st_size
        except (OSError, TypeError, ValueError):
            pass
    cur = fid.tell()
    fid.seek(0, 2)  # end
    file_size = fid.tell()
    fid.seek(cur, 0)
    return file_size


def maybe_mem_map(fid: IOBase, dtype="<u1") -> np.ndarray | np.memmap:
    """
    Try to get a memory map array from fid, otherwise just return array.

    Parameters
    ----------
    fid
        A buffered reader, e.g. from open(file) as fid.
    """
    # File objects backed by memory (BytesIO and friends) have no usable
    # name, so there is nothing to map; they read into memory below.
    name = getattr(fid, "name", None)
    if name is not None:
        try:
            return np.memmap(name, dtype=dtype, mode="r")
        except (AttributeError, OSError, TypeError, ValueError):
            # A name which is not a mappable path -- an fd number, an empty
            # file, a path already unlinked, a filesystem which cannot map --
            # falls back rather than failing a read the handle can still do.
            pass
    fid.seek(0)
    return np.frombuffer(fid.read(), dtype=dtype)


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


def get_2d_line_intersection(p1, p2, p3, p4):
    """
    Return intersection point of two lines (p1,p2) and (p3,p4).

    Parameters
    ----------
    p1, p2, p3, p4
        Each a pair of (x, y) coordinates.

    Returns
    -------
    point
        (x, y) intersection point. x and y are nan if lines are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if np.isclose(denom, 0):
        return np.array([np.nan, np.nan])

    num_x = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    num_y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = num_x / denom
        py = num_y / denom
    return np.array([px, py])


def tukey_fence(data, fence_multiplier=1.5) -> np.ndarray:
    """
    Apply Tukey's fence to determine data range without outliers.
    """
    with suppress_warnings(
        category=RuntimeWarning, message="All-NaN (?:slice|axis) encountered"
    ):
        q1, q3 = np.nanpercentile(data, [25, 75])
        dmin, dmax = np.nanmin(data), np.nanmax(data)
        diff = q3 - q1  # Interquartile range (IQR)
        q_lower = np.nanmax([q1 - diff * fence_multiplier, dmin])
        q_upper = np.nanmin([q3 + diff * fence_multiplier, dmax])
    lower_and_top = np.asarray([q_lower, q_upper])
    return lower_and_top


def is_strictly_monotonic(values, increasing: bool | None = None) -> bool:
    """
    Return True if a 1D sequence is strictly monotonic.

    Parameters
    ----------
    values
        The sequence to test. Sequences shorter than two elements are
        trivially monotonic; multidimensional inputs never are.
    increasing
        If None either direction qualifies, else require strictly increasing
        (True) or strictly decreasing (False) values.
    """
    data = np.asarray(values)
    if data.ndim != 1:
        return False
    view1, view2 = data[:-1], data[1:]
    try:
        # Ascending is by far the common case; answer without a second pass.
        if increasing is not False and bool(np.all(view2 > view1)):
            return True
        if increasing is True:
            return False
        return bool(np.all(view1 > view2))
    except TypeError:  # values which do not support comparison
        return False


_CODE_RE = re.compile(r"[A-Za-z0-9-]+")
_LOCATION_RE = re.compile(r"[A-Za-z0-9-]*")
# The tokens of an acquisition_key, in order. Only location may be blank.
ACQUISITION_KEY_PARTS = ("network", "fiber_array", "location", "acquisition")


def check_code(value: str, allow_blank: bool = False) -> str:
    """
    Validate a single code token of an acquisition key.

    Parameters
    ----------
    value
        The token to validate.
    allow_blank
        If True an empty token is valid, as it is for location codes.
    """
    pattern = _LOCATION_RE if allow_blank else _CODE_RE
    if pattern.fullmatch(value) is None:
        blank = " (or blank)" if allow_blank else ""
        msg = f"Invalid code {value!r}; codes use letters, digits, and '-'{blank}."
        raise InvalidInventoryError(msg)
    return value


def validate_acquisition_key(value: str) -> str:
    """
    Validate a composite acquisition key.

    An acquisition key names an inventory acquisition as
    network.fiber_array.location.acquisition. An empty string means unset,
    which is how patches with no inventory identity are spelled.

    Both the inventory model and `PatchAttrs` use this function so a code
    legal in one is legal in the other.
    """
    if not value:
        return value
    parts = value.split(".")
    if len(parts) != len(ACQUISITION_KEY_PARTS):
        expected = ".".join(ACQUISITION_KEY_PARTS)
        msg = (
            f"Invalid acquisition_key {value!r}; got {len(parts)} dot separated "
            f"codes but expected {len(ACQUISITION_KEY_PARTS)} ({expected})."
        )
        raise InvalidInventoryError(msg)
    for part, name in zip(parts, ACQUISITION_KEY_PARTS, strict=True):
        check_code(part, allow_blank=name == "location")
    # PatchAttrs bounds the field, so a key too long to store must not read
    # as merely unknown to the inventory; the two have to agree on legality.
    if len(value) > max_lens["acquisition_key"]:
        msg = (
            f"Invalid acquisition_key {value!r}; it is {len(value)} characters "
            f"and the limit is {max_lens['acquisition_key']}."
        )
        raise InvalidInventoryError(msg)
    return value


@cache
def glob_to_regex(pattern: str) -> re.Pattern:
    """
    Translate a glob to the regex which matches what SQLite's GLOB does.

    Every glob DASCore reads -- a spool query, a dataframe filter, a
    string coordinate -- means what SQLite's GLOB means, since the index
    answers some of those queries in SQL and one pattern must not mean two
    things depending on which side answers it. SQLite is not `fnmatch`: a
    class is negated with `[^...]`, where fnmatch spells that `[!...]` and
    reads a leading `!` as a literal. An unterminated class matches
    nothing, as it does in SQLite; a class a regex cannot express at all
    (a reversed range, which SQLite reads leniently) matches nothing here
    rather than being guessed at.
    """
    out, index, size = [], 0, len(pattern)
    while index < size:
        char = pattern[index]
        if char in "*?":
            out.append(".*" if char == "*" else ".")
        elif char == "[":
            # A class may open with '^' to negate and may hold ']' as its
            # first member; the class ends at the next ']' after those.
            end = index + 1 + pattern[index + 1 : index + 2].count("^")
            end += pattern[end : end + 1].count("]")
            end = pattern.find("]", end)
            if end < 0:
                return _MATCHES_NOTHING
            body = pattern[index + 1 : end]
            negate, body = body.startswith("^"), body.removeprefix("^")
            # A ']' opening a class is one of its members, and SQLite takes
            # it as a plain one: it is not the low end of a range, so the
            # dash which may follow it is a member too.
            leading = ""
            if body.startswith("]"):
                leading, body = re.escape("]"), body[1:]
            out.append(f"[{'^' if negate else ''}{leading}{_class_body(body)}]")
            index = end
        else:
            out.append(re.escape(char))
        index += 1
    # SQLite's wildcards cross a newline like any other byte. The flag goes
    # inline, not into re.compile, so it survives a caller that hands the
    # pattern text on to something else.
    return re.compile(r"(?s)" + "".join(out) + r"\Z")


# A pattern which matches nothing, for a glob SQLite would not read.
_MATCHES_NOTHING = re.compile(r"(?!)")


def _class_body(body: str) -> str:
    """
    Escape the members of a glob character class for a regex one.

    Ranges stay ranges, since a glob class means them, with each endpoint
    escaped on its own — escaping the body wholesale would turn the dash
    of a range into a member. A range's low endpoint is also emitted as a
    member: SQLite tests it before testing the range, so `[z-a]` matches
    `z` where the reversed range alone matches nothing.
    """
    out, index, size = [], 0, len(body)
    while index < size:
        low = body[index]
        if index + 2 < size and body[index + 1] == "-":
            high = body[index + 2]
            out.append(re.escape(low))
            if low <= high:
                out.append(f"{re.escape(low)}-{re.escape(high)}")
            index += 3
            continue
        out.append(re.escape(low))
        index += 1
    return "".join(out)
