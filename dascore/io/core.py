"""
Base functionality for reading, writing, determining file formats, and scanning
Das Data.
"""

from __future__ import annotations

import inspect
import re
import warnings
from collections import defaultdict
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
)
from contextlib import suppress
from dataclasses import replace
from functools import cached_property, wraps
from pathlib import Path
from threading import RLock
from typing import (
    Literal,
    Protocol,
    TypeVar,
    cast,
    get_type_hints,
)

import numpy as np
import pandas as pd

import dascore as dc
from dascore.compat import Progress, UPath
from dascore.constants import (
    PROGRESS_LEVELS,
    float_select_type,
    path_types,
    time_select_type,
)
from dascore.core.coords import CoordSegmented
from dascore.core.source import PatchSource
from dascore.core.spool import Spool
from dascore.core.summary import PatchSummary, normalize_source_patch_key
from dascore.exceptions import (
    DependencyError,
    InvalidFiberFileError,
    InvalidFiberIOError,
    MissingOptionalDependencyError,
    MissingPatchError,
    ParameterError,
    PatchAttributeError,
    RemoteCacheError,
    UnknownFiberFormatError,
)
from dascore.io.utils import selection_windows
from dascore.utils.downloader import resolve_example_uri
from dascore.utils.identity import (
    ids_enabled,
    source_patch_id,
)
from dascore.utils.io import (
    IOResourceManager,
    _normalize_source_patch_keys,
    get_handle_from_resource,
    release_handle,
)
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    _apply_union_indexers,
    _get_install_message,
    _get_install_name,
    _iter_filesystem,
    _locked,
    _reinit_after_fork,
    cached_method,
    iterate,
    warn_or_raise,
)
from dascore.utils.paths import (
    coerce_to_local_path,
    coerce_to_upath,
    is_example_uri,
    is_local_path,
)
from dascore.utils.pd import filter_df
from dascore.utils.plugins import FIBER_IO_GROUP, get_entry_point_loaders
from dascore.utils.progress import track
from dascore.utils.remote_io import (
    get_remote_cache_scope,
    remote_cache_scope,
    suppress_gc_pause_warning,
)

# What the scan dispatchers accept: one resource or patch, or an
# iterable of them (`_iterate_scan_inputs` flattens its input with
# `iterate` before resolving each element). The dispatcher walks its
# input twice, once to size the progress bar, so one-shot iterators
# (e.g. generators) are materialized up front rather than silently
# scanning nothing (see #818).
ScanInput = (
    path_types
    | dc.Patch
    | dc.Spool
    | IOResourceManager
    | Iterable[path_types | dc.Patch | IOResourceManager]
)


def _validate_metadata(patch):
    """Require a data-less Patch at the reader's metadata boundary."""
    if not isinstance(patch, dc.Patch) or patch._data is not None:
        msg = "FiberIO.get_metadata() must return data-less Patch objects."
        raise TypeError(msg)
    return patch


def _resolve_read_spool(spool, source_patch_key: object = "") -> dc.Patch:
    """
    Resolve one patch from a read result by source identity.

    Readers that consume source_patch_key may return the single matching
    patch without preserving that reload metadata on it; only trust that
    when the patch doesn't claim a different identity.
    """
    source_patch_key = normalize_source_patch_key(source_patch_key)
    if source_patch_key and len(spool) == 1:
        found = normalize_source_patch_key((spool[0]._source or PatchSource()).key)
        if found == source_patch_key or (not found and not source_patch_key.isdigit()):
            return spool[0]
    return _select_patch_from_spool(spool, source_patch_key=source_patch_key)


def _select_patch_from_spool(spool, source_patch_key: object = "") -> dc.Patch:
    """Select one loaded patch from a spool using source identity."""
    if len(spool) == 0:
        # Iteration skips these with a warning, see #583.
        msg = (
            "No patch remained after applying load filters; the requested "
            "range may have trimmed it to nothing."
        )
        raise MissingPatchError(msg)
    if source_patch_key not in (None, ""):
        source_patch_key = str(source_patch_key)
        # Native source ids are preserved on patch attrs by their readers.
        matches = [
            patch
            for patch in spool
            if normalize_source_patch_key((patch._source or PatchSource()).key)
            == source_patch_key
        ]
        if len(matches) == 1:
            return matches[0]
        # Synthesized ids are positional within the full source read.
        try:
            index = int(source_patch_key)
        except (TypeError, ValueError):
            index = None
        if index is not None and 0 <= index < len(spool):
            return spool[index]
        if len(spool) == 1 and spool[0].get_patch_name() == source_patch_key:
            return spool[0]
        msg = "Patch could not be uniquely resolved after applying load filters."
        raise PatchAttributeError(msg)
    if len(spool) == 1:
        return spool[0]
    msg = "Patch could not be uniquely resolved after applying load filters."
    raise PatchAttributeError(msg)


def _get_reloadable_source_path(
    resource, fallback: str | Path | UPath | None = None
) -> UPath | str:
    """Return a normalized reloadable path for resources that expose one."""
    candidates = [fallback, resource]
    for name in ("source", "filename", "name", "path"):
        candidates.append(getattr(resource, name, None))
    for candidate in candidates:
        if candidate in {None, ""}:
            continue
        if isinstance(candidate, IOResourceManager):
            candidate = candidate.source
        # A reloadable path names the file, never the examples:// name,
        # which no filesystem knows how to reopen.
        candidate = resolve_example_uri(candidate)
        if isinstance(candidate, str | Path | UPath):
            return coerce_to_upath(candidate)
    return ""


class _FiberIOManager:
    """
    A structure for intelligently storing, loading, and return FiberIO objects.

    This should only be used in conjunction with `FiberIO`.
    """

    def __init__(self, entry_point: str):
        self._entry_point = entry_point
        # One lock guards all mutable state below; it is held for the whole
        # of load_plugins so no caller can observe a half-loaded format.
        self._lock = RLock()
        self._loaded_eps: set[str] = set()
        # Formats whose load attempt finished, successfully or not. The
        # outcome lives in _format_version/_failed_formats.
        self._loaded_formats: set[str] = set()
        # True once no format is left to load; keeps the (hot) repeat call
        # to load_plugins() off the lock entirely.
        self._all_loaded = False
        self._failed_formats: set[str] = set()
        # Plain dicts, not defaultdicts: these are shared state, and a
        # missing-key read must not register anything.
        self._format_version: dict[str, dict[str, FiberIO]] = {}
        self._extension_list: dict[str, list[FiberIO]] = {}
        # This is a dict of {input_type: {fiberio, ...}}
        self._fiber_io_by_input_type: dict[str, set[FiberIO]] = {}
        self._fiber_io_name_ver = set()
        # Snapshots derived from the registry; cleared when it changes.
        # Kept as two dicts rather than one keyed by a discriminating
        # prefix so each stays a single value type.
        self._input_type_cache: dict[str, frozenset[FiberIO]] = {}
        self._prioritized_cache: dict[str, tuple[FiberIO, ...]] = {}

    def __getstate__(self) -> dict:
        """Return copy/pickle state without the process-local lock."""
        with self._lock:
            state = dict(self.__dict__)
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state with a fresh process-local lock."""
        self.__dict__.update(state)
        self._lock = RLock()

    @cached_property
    def _eps(self):
        """
        Get the unloaded entry points registered to this domain into a dict of
        {name: ep}.
        """
        return pd.Series(get_entry_point_loaders(FIBER_IO_GROUP))

    @cached_property
    @_locked("_lock")
    def known_formats(self) -> frozenset[str]:
        """Return names of known formats."""
        formats = [name.split("__", maxsplit=1)[0] for name in self._eps.index]
        return frozenset(formats) | frozenset(self._format_version)

    @property
    @_locked("_lock")
    def unloaded_formats(self) -> list[str]:
        """Return names of known formats which have not been loaded."""
        loaded_or_failed = set(self._format_version) | self._failed_formats
        return sorted(self.known_formats - loaded_or_failed)

    @_locked("_lock")
    def _get_fiber_io_by_input_type(self, input_type) -> frozenset[FiberIO]:
        """Get a set of FiberIO instances that meet input type."""
        if (cached := self._input_type_cache.get(input_type)) is None:
            if (out := self._fiber_io_by_input_type.get(input_type)) is None:
                out = set()
                for input_set in self._fiber_io_by_input_type.values():
                    out |= input_set
            cached = self._input_type_cache[input_type] = frozenset(out)
        return cached

    @_locked("_lock")
    def _get_prioritized_list(self, input_type="file") -> tuple[FiberIO, ...]:
        """Yield a prioritized list of fiber_ios."""
        if (cached := self._prioritized_cache.get(input_type)) is not None:
            return cached
        # must load all plugins before getting list
        self.load_plugins()
        priority_fiber_ios = []
        second_class_fiber_ios = []
        for format_name in self.known_formats:
            if not (unsorted := self._format_version.get(format_name)):
                continue
            keys = sorted(unsorted, reverse=True)
            fiber_ios = [unsorted[key] for key in keys]
            priority_fiber_ios.append(fiber_ios[0])
            if len(fiber_ios) > 1:
                second_class_fiber_ios.extend(fiber_ios[1:])
        maybe_ios = priority_fiber_ios + second_class_fiber_ios
        # Now filter to input_type
        valid_fiberio_by_type = self._get_fiber_io_by_input_type(input_type)
        out = tuple(x for x in maybe_ios if x in valid_fiberio_by_type)
        # And return fiberIOs that much the input type.
        self._prioritized_cache[input_type] = out
        return out

    def load_plugins(self, format: str | None = None):
        """Load plugin for specific format or ensure all formats are loaded."""
        # A format only lands in _loaded_formats (or _all_loaded) once every
        # one of its entry points is registered, so these fast paths (and
        # the lock below) keep multi-version formats from being seen half
        # loaded.
        if self._all_loaded or (format is not None and format in self._loaded_formats):
            return
        with self._lock:
            # Anything already registered (directly, or by another thread
            # while this one waited) is not pending; it only needs stamping.
            pending = set(self.unloaded_formats)
            formats = {format} if format is not None else pending
            self._loaded_formats |= formats
            # known_formats is fixed once computed, so what stays pending
            # here stays pending until it is loaded.
            self._all_loaded = not (pending - formats)
            if not (todo := formats & pending):
                return  # nothing left to load; already registered or failed
            # Plugin imports deliberately run under the lock: tracking
            # in-flight formats instead would need a claim/wait graph for a
            # step which happens once per format. The cost is that a thread
            # importing a module which defines a FiberIO waits here until
            # any in-progress load finishes.
            for form in todo:
                entries = [name for name in self._eps.index if name.startswith(form)]
                for name, loader in self._eps.loc[entries].items():
                    fiberio = self._load_entry_point(name, loader)
                    if fiberio is not None:
                        self.register_fiberio(fiberio)
                if form not in self._format_version:
                    self._failed_formats.add(form)
            # The selected format(s) should now be loaded
            assert formats.isdisjoint(self.unloaded_formats)

    def _load_entry_point(self, name: str, loader) -> FiberIO | None:
        """Load one FiberIO entry point, skipping broken registrations."""
        try:
            return loader()()
        except Exception as exc:
            msg = (
                f"Failed to load FiberIO plugin {name!r} "
                f"({exc.__class__.__name__}: {exc}); skipping it. "
                "This can happen when an entry point from a previous install "
                "is stale; reinstalling dascore (or the package providing the "
                "plugin) may fix it."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            return None

    @_locked("_lock")
    def register_fiberio(self, fiberio: FiberIO):
        """Register a new fiber IO to manage."""
        format_name, ver = fiberio.name.upper(), fiberio.version
        id_tuple = (format_name, ver)
        if id_tuple in self._fiber_io_name_ver:
            return
        self._loaded_eps.add(fiberio.name)
        for ext in iter(fiberio.preferred_extensions):
            self._extension_list.setdefault(ext, []).append(fiberio)
        self._format_version.setdefault(format_name, {})[ver] = fiberio
        self._fiber_io_by_input_type.setdefault(fiberio.input_type, set()).add(fiberio)
        self._fiber_io_name_ver.add(id_tuple)
        # Snapshots derived from the registry are now stale.
        self._input_type_cache.clear()
        self._prioritized_cache.clear()

    @cached_method
    def get_fiberio(
        self,
        *,
        format: str | None = None,
        version: str | None = None,
        extension: str | None = None,
    ) -> FiberIO:
        """
        Return the most likely fiber_io for given inputs.

        If no such fiber_io exists, raise UnknownFiberFormat error.

        Parameters
        ----------
        format
            The format string indicating the format name
        version
            The version string of the format
        extension
            The extension of the file.
        """
        iterator = self.yield_fiberio(
            format=format,
            version=version,
            extension=extension,
        )
        fiber_io = next(iterator, None)
        # yield_fiberio raises rather than yield nothing for a format or
        # version it does not know, and with nothing named at all it yields
        # the whole registry, which is never empty.
        assert fiber_io is not None, "no fiber_io for the requested inputs"
        return fiber_io

    def yield_fiberio(
        self,
        format: str | None = None,
        version: str | None = None,
        extension: str | None = None,
        fiber_io_hint: dict[str, FiberIO] | None = None,
        input_type: str | None = None,
    ) -> Generator[FiberIO, None, None]:
        """
        Yields fiber IO object based on input priorities.

        The order is sorted in likelihood of the fiber_io being correct. For
        example, if file format is specified but file_version is not, all
        fiber_ios for the format will be yielded with the newest versions
        first in the list.

        If neither version nor format are specified but extension is all fiber_ios
        specifying the extension will be first in the list, sorted by format name
        and format version.

        If nothing is specified, all fiber_ios will be returned starting with
        the newest (the highest version) of each fiber_io, followed by older
        versions.

        Parameters
        ----------
        format
            The format string indicating the format name.
        version
            The version string of the format
        extension
            The extension of the file.
        fiber_io_hint
            If not None, a suspected fiber_io to use first. This is an
            optimization for file archives which tend to have many files of
            the same format.
        """
        fiber_io_hint = {} if fiber_io_hint is None else fiber_io_hint
        if version and not format:
            msg = "Providing only a version is not sufficient to determine format"
            raise UnknownFiberFormatError(msg)
        elif format is not None:
            self.load_plugins(format)
            yield from self._yield_format_version(format, version)
            return
        if input_type is not None and (out := fiber_io_hint.get(input_type)):
            yield out
        if extension is not None:
            yield from self._yield_extensions(extension, input_type)
        else:
            yield from self._get_prioritized_list(input_type)

    def _yield_format_version(self, format, version):
        """Yield file format/version prioritized fiber_ios."""
        assert isinstance(format, str), "Only works once format is known."
        format = format.upper()
        self.load_plugins(format)
        with self._lock:
            # Snapshot; the generator must not read shared state while paused.
            fiber_ios = dict(self._format_version.get(format, {}))
        # no format found
        if not fiber_ios:
            format_list = list(self.known_formats)
            msg = f"Unknown format {format}, known formats are {format_list}"
            raise UnknownFiberFormatError(msg)
        # a version is specified
        if version:
            fiber_io = fiber_ios.get(version, None)
            if fiber_io is None:
                msg = (
                    f"Format {format} has no version: [{version}] "
                    f"known versions of this format are: {list(fiber_ios)}"
                )
                raise UnknownFiberFormatError(msg)
            yield fiber_io
            return
        # reverse sort fiber_ios and yield latest version first.
        for fiber_io in dict(sorted(fiber_ios.items(), reverse=True)).values():
            yield fiber_io
        return

    def _yield_extensions(self, extension, input_type=None):
        """Generator to get fiber_io prioritized by preferred extensions."""
        has_yielded = set()
        self.load_plugins()
        potential_fiberios = self._get_fiber_io_by_input_type(input_type)
        with self._lock:
            extension_fiberios = tuple(self._extension_list.get(extension, ()))
        for fiber_io in extension_fiberios:
            if fiber_io in potential_fiberios:
                yield fiber_io
            has_yielded.add(fiber_io)
        for fiber_io in self._get_prioritized_list(input_type):
            if fiber_io not in has_yielded:
                yield fiber_io

    def _get_format(
        self,
        path: path_types | IOResourceManager,
        file_format: str | None = None,
        file_version: str | None = None,
        fiber_io_hint: dict[str, FiberIO] | None = None,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Return the file's format name and version.

        See [`dascore.io.core.get_format`](`dascore.io.core.get_format`)
        for docs.
        """
        # Probing must not announce a remote gc pause: the resource is not
        # known to be HDF5 yet, and under warnings-as-errors the warning
        # would be caught by the robustness handler below and read as
        # "wrong format", silently skipping the reader which does match.
        with IOResourceManager(path) as man, suppress_gc_pause_warning():
            # The source may still be an examples:// name if a manager was
            # handed in already wrapping one; the checks below need a path.
            path = resolve_example_uri(man.source)
            if isinstance(path, UPath):
                exists = path.exists()
                suffix = path.suffix
            else:
                local_path = (
                    coerce_to_local_path(path)
                    if is_local_path(path)
                    else coerce_to_upath(path)
                )
                exists = local_path.exists()
                suffix = local_path.suffix
            if not exists:
                raise FileNotFoundError(f"{path} does not exist.")
            # get extension (str minus .)
            ext = suffix[1:] if suffix else None
            input_type = self._get_input_type_name(path)
            iterator = self.yield_fiberio(
                file_format,
                file_version,
                extension=ext,
                fiber_io_hint=fiber_io_hint,
                input_type=input_type,
            )
            for fiber_io in iterator:
                # We need to wrap this in try except to make it robust to what
                # may happen in each fiber_ios get_format method, many of which
                # may be third party code.
                func = fiber_io.get_format
                required_type = _required_resource_type(fiber_io.get_version)
                func_input = None
                try:
                    # Get resource has to be in the try block because it can also
                    # raise, in which case the format doesn't belong.
                    func_input = man.get_resource(required_type)
                    format_version = func(func_input, _pre_cast=True)
                except RemoteCacheError:
                    # A remote fetch failure is a real error, not a "wrong
                    # format" signal, so it must propagate rather than be
                    # swallowed by the robustness handler below.
                    raise
                # For robustness, we need to catch everything else here.
                except Exception:
                    continue
                finally:
                    # If file handle-like seek back to 0 so it can be reused.
                    getattr(func_input, "seek", lambda x: None)(0)
                if format_version:
                    return format_version
            else:
                msg = f"Could not determine file format of {man.source}"
                raise UnknownFiberFormatError(msg)

    def _get_input_type_name(self, obj):
        """Get the name of the IO type."""
        # This effectively acts as a dispatch to determine which type of
        # FiberIO could possibly read the obj.
        out = "file"
        if isinstance(obj, str | Path | UPath):
            path = coerce_to_upath(obj)
            if path.exists():
                out = "directory" if path.is_dir() else "file"
        return out


# ------------- Protocol for File Format support


class _TypeCasterMethod(Protocol):
    """
    A FiberIO method wrapped by the type caster.

    The caster stamps these markers onto the wrapped method so the io
    machinery can find the original function and the resource type the
    method wants its input coerced to.
    """

    func: Callable
    _type_caster_wrapped: bool
    _required_type: type | None

    def __call__(self, *args, **kwargs): ...


def _required_resource_type(method) -> type | None:
    """
    Return the resource type a FiberIO method's caster coerces its input to.

    None when the method's resource parameter carries no type hint, or
    when the method was never wrapped at all (only the base FiberIO's
    own methods, which __init_subclass__ does not visit).
    """
    return getattr(method, "_required_type", None)


def _type_caster(func, sig, required_type, arg_name):
    """A decorator for casting types for arguments of cast ind."""
    fun_name = func.__name__

    # this is a subclass of a FiberIO subclass and its key methods
    # have already been wrapped. Just return.
    if getattr(func, "_type_caster_wrapped", False):
        return func

    @wraps(func)
    def _wrapper(*args, _pre_cast=False, **kwargs):
        """Wraps args but performs coercion to get proper stream."""
        # TODO look at replacing this with pydantic's type_guard thing.

        # this allows us to fast-track calls from generic functions
        if required_type is None or _pre_cast:
            return func(*args, **kwargs)
        bound = sig.bind(*args, **kwargs)
        new_kw = bound.arguments
        resource = new_kw.pop(arg_name)
        new_resource = None
        try:
            new_resource = get_handle_from_resource(resource, required_type)
            new_kw[arg_name] = new_resource
            # kwargs is included in bound arguments, need to re-attach
            new_kw.update(new_kw.pop("kwargs", {}))
            out = func(**new_kw)
        except BaseException as e:
            # A handle created here must be released even on failure,
            # including on KeyboardInterrupt: leaking a remote handle leaves
            # garbage collection paused for as long as the traceback is
            # retained. Abort rather than close, so a failed remote write
            # discards its temp file instead of uploading a partial one.
            if new_resource is not None and new_resource is not resource:
                with suppress(Exception):
                    release_handle(new_resource, abort=True)
            # get_format reports "not my format" by returning False rather
            # than raising, so an ordinary Exception becomes False here.
            # Everything else propagates, including a BaseException raised
            # inside get_format: the catch is only this wide so the cleanup
            # above runs on a KeyboardInterrupt, not to swallow one.
            if fun_name not in {"get_format", "get_version"} or not isinstance(
                e, Exception
            ):
                raise
            out = None if fun_name == "get_version" else False
        else:
            # if a new file handle was created we need to close it now. But it
            # shouldn't close any passed in, that should happen up the stack.
            if new_resource is not resource:
                release_handle(new_resource)
        return out

    # attach the function and required type for later use
    caster = cast(_TypeCasterMethod, _wrapper)
    caster.func = func
    # subclasses of FIBERIO subclasses can wrap this twice, so we mark
    # it to avoid that scenario.
    caster._type_caster_wrapped = True
    # also specify required type
    caster._required_type = required_type

    return caster


def _is_wrapped_func(func1, func2):
    """Small helper function to determine if func1 is func2, unwrapping decorators."""
    func = func1
    while hasattr(func, "func") or hasattr(func, "__func__"):
        func = getattr(func, "func", func)
        func = getattr(func, "__func__", func)
    return func is func2


class FiberIO:
    """
    Interface for a fiber data format; subclass it to add format support.
    """

    name: str = ""
    version: str = ""
    preferred_extensions: tuple[str, ...] = ()
    # Whether this format expects a directory or a single file.
    input_type: Literal["file", "directory"] = "file"
    # True when a single resource can hold more than one patch.
    multi_patch_write: bool = False
    # True when a written patch may keep gapped (segmented) dimensional
    # coordinates; otherwise write splits or refuses them.
    segmented_write: bool = False

    manager = _FiberIOManager(FIBER_IO_GROUP)

    # Methods using automatic type casting and the parameter index to cast.
    _automatic_type_casters = FrozenDict(
        {
            "read_array": 1,
            "get_metadata": 1,
            "write": 2,
            "get_version": 1,
        }
    )

    def get_version(self, resource) -> str | None:
        """Return this family's file version, or None for another family."""
        msg = f"FiberIO: {self.name} has no get_version method"
        raise NotImplementedError(msg)

    def get_metadata(self, resource, *, snap: bool = True) -> list[dc.Patch]:
        """
        Return one data-less patch per logical patch in a resource.

        The coordinates declare the array's dimensions and shape, and the
        patch declares its dtype. Multi-patch readers put their logical key
        in `PatchSource`. The framework supplies the source path, format,
        and version. `snap=False` preserves stored coordinate values when
        available; neither setting includes unwritten samples.
        """
        msg = f"FiberIO: {self.name} has no get_metadata method"
        raise NotImplementedError(msg)

    def read_array(
        self, resource, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Read one logical patch's array over half-open positional windows.

        `windows` maps dimension names to `(start, stop)` sample indices.
        Missing dimensions are returned whole. The array follows the
        dimensions, shape, and dtype declared by `get_metadata`, including
        format-specific layout and scaling. `key` identifies the logical
        patch in a multi-patch resource; single-patch formats ignore it.
        Readers which cannot slice storage decode and then slice here.
        """
        msg = f"FiberIO: {self.name} has no read_array method"
        raise NotImplementedError(msg)

    def read(
        self,
        resource,
        *,
        snap: bool | None = None,
        samples: bool = False,
        source_patch_key: str | Iterable[str] = "",
        **select,
    ) -> dc.Spool:
        """
        Read patches by selecting metadata and loading the resulting array windows.

        Coordinate queries are resolved on the original sample grid without
        recording a processing operation. `source_patch_key` selects logical
        patches, and matching attribute queries filter metadata before data
        are read. `snap` defaults to True; the older `snap_dims` spelling is
        also accepted, with explicit `snap` taking precedence. `samples=True`
        interprets coordinate selections as sample indices. `source_patch_key`
        accepts one logical key or an iterable of keys.
        """
        provenance_source = select.pop("_provenance_source", None)
        snap_dims = select.pop("snap_dims", True)
        snap = snap_dims if snap is None else snap
        wanted = _normalize_source_patch_keys(source_patch_key)
        relative = select.pop("relative", False)
        out = []
        with IOResourceManager(resource) as manager:
            metadata_resource = manager.get_resource(
                _required_resource_type(self.get_metadata)
            )
            metadata_func = cast(_TypeCasterMethod, self.get_metadata)
            patches = [
                _validate_metadata(patch)
                for patch in metadata_func(metadata_resource, snap=snap, _pre_cast=True)
            ]
            getattr(metadata_resource, "seek", lambda x: None)(0)
            origins = [patch._source or PatchSource() for patch in patches]
            if provenance_source is not None:
                patches = _stamp_source_ids(
                    patches, self.name, self.version, provenance_source
                )
            for index, (patch, origin) in enumerate(zip(patches, origins, strict=True)):
                source = patch._source or PatchSource()
                key = source.key or (str(index) if len(patches) > 1 else "")
                if wanted and (source.key or str(index)) not in wanted:
                    continue
                queries = {
                    name: value
                    for name, value in select.items()
                    if name in patch.coords.coord_map
                    and patch.coords.coord_map[name].ndim == 1
                    and value is not None
                }
                flat_attrs = patch.attrs.flat_dump()
                attr_queries = {
                    name: value
                    for name, value in select.items()
                    if name in flat_attrs
                    and not (pd.api.types.is_scalar(value) and pd.isna(value))
                }
                if (
                    attr_queries
                    and not filter_df(pd.DataFrame([flat_attrs]), **attr_queries)[0]
                ):
                    continue
                coords, indexers = patch.coords.select_indexers(
                    relative=relative, samples=samples, **queries
                )
                if not coords.size and queries:
                    continue
                windows, residual = selection_windows(patch.coords, indexers)
                array_resource = manager.get_resource(
                    _required_resource_type(self.read_array)
                )
                getattr(array_resource, "seek", lambda x: None)(0)
                array_func = cast(_TypeCasterMethod, self.read_array)
                # Directory member paths pin array loading to the metadata entry,
                # while public keys and source-derived IDs retain their ordinals.
                array_key = (
                    origin.path
                    if self.input_type == "directory" and origin.path
                    else key
                )
                data = array_func(
                    array_resource, windows, key=array_key, _pre_cast=True
                )
                expected = (
                    tuple(stop - start for start, stop in windows.values())
                    if patch.dims
                    else patch.shape
                )
                if data.shape != expected or np.dtype(data.dtype) != np.dtype(
                    patch.dtype
                ):
                    msg = (
                        f"{self.name}.read_array returned {data.shape}/{data.dtype}; "
                        f"metadata declared {expected}/{patch.dtype}."
                    )
                    raise InvalidFiberIOError(msg)
                data = _apply_union_indexers(residual, data)
                out.append(
                    patch.new(data=data, coords=coords, source=replace(source, key=key))
                )
        return dc.spool(out)

    def scan(
        self, resource, *, snap: bool = True, timestamp=None, **kwargs
    ) -> list[dc.Patch]:
        """Return data-less patches; the dispatcher attaches source provenance."""
        if self.input_type == "directory":
            resource = coerce_to_upath(resource)
            resource = resource if resource.is_dir() else resource.parent
        with IOResourceManager(resource) as manager:
            # Cast outside the best-effort scan handler: invalid input types
            # are caller errors, not an unsupported file variant.
            metadata_resource = (
                resource
                if kwargs.get("_pre_cast")
                else manager.get_resource(_required_resource_type(self.get_metadata))
            )
            metadata_kwargs = {"snap": snap}
            if getattr(self.get_metadata, "_type_caster_wrapped", False):
                metadata_kwargs["_pre_cast"] = True
            try:
                patches = self.get_metadata(metadata_resource, **metadata_kwargs)
            except NotImplementedError as exc:
                if _is_wrapped_func(self.get_metadata, FiberIO.get_metadata):
                    raise
                warnings.warn(str(exc), UserWarning, stacklevel=2)
                return []
        if timestamp is not None:
            patches = [
                patch
                for patch in patches
                if self._updated_after(
                    (patch._source or PatchSource()).path or resource, timestamp
                )
            ]
        return patches

    def write(self, spool: dc.Patch | dc.Spool, resource, **kwargs):
        """Write the spool to a resource (eg path, stream, etc.)."""
        msg = f"FiberIO: {self.name} has no write method"
        raise NotImplementedError(msg)

    def get_format(self, resource, **kwargs) -> tuple[str, str] | Literal[False]:
        """Derive the format name and version from the family detector."""
        detector = cast(_TypeCasterMethod, self.get_version)
        version = detector(resource, _pre_cast=kwargs.get("_pre_cast", False))
        return (self.name, version) if version is not None else False

    @property
    def implements_write(self) -> bool:
        """Return whether the reader supports writing."""
        return not _is_wrapped_func(self.write, FiberIO.write)

    @classmethod
    def get_supported_io_table(cls):
        """Return the supported formats, versions, and optional write capability."""
        cls.manager.load_plugins()
        return pd.DataFrame(
            [
                {"name": name, "version": version, "write": io.implements_write}
                for name, versions in cls.manager._format_version.items()
                for version, io in versions.items()
            ]
        )

    def _updated_after(self, resource, timestamp):
        """Determine if the resource was updated after specified mtime."""
        if not timestamp:
            return True
        is_remote = not is_local_path(resource)
        try:
            path = (
                coerce_to_upath(resource)
                if is_remote
                else coerce_to_local_path(resource)
            )
            return path.stat().st_mtime > timestamp
        except Exception:
            if not is_remote:
                return False
            warnings.warn(
                "Remote path backend does not expose reliable mtime; "
                "continuing scan without timestamp filtering.",
                UserWarning,
                stacklevel=2,
            )
            return True

    def __hash__(self):
        """FiberIO instances should be uniquely defined by (format, version)."""
        return hash((self.name, self.version))

    def __init_subclass__(cls, **kwargs):
        """Hook for registering subclasses."""
        # check that the subclass is valid
        if not cls.name:
            msg = "You must specify the file format with the name field."
            raise InvalidFiberIOError(msg)
        # register fiber_io
        parent = cls.__mro__[1]
        assert issubclass(parent, FiberIO)  # only FiberIO subclasses get here
        parent.manager.register_fiberio(cls())
        # decorate methods for type-casting
        for name, param_ind in cls._automatic_type_casters.items():
            method = getattr(cls, name)
            sig = inspect.signature(method)
            arg_name = list(sig.parameters)[param_ind]
            required_type = get_type_hints(method).get(arg_name)
            method_wrapped = _type_caster(method, sig, required_type, arg_name)
            setattr(cls, name, method_wrapped)


@_reinit_after_fork
def _reinit_manager_lock():
    """Install a fresh lock on the FiberIO manager; see _reinit_after_fork."""
    FiberIO.manager._lock = RLock()


# What a reader which keeps its own patch ids leaves behind for `read` to
# find. Only a format which stores an id sets it; see the DASDAE reader.
STORED_PATCH_ID = "_stored_patch_id"


def _source_stats(source) -> tuple[int | None, int | None]:
    """
    Return a source's size and modification time, or nothing for both.

    Nothing is not a failure: a stream and some remote backends have
    neither, and an id which says so is better than one which pretends
    the fields were equal.

    A remote source is stat-ed too. One metadata request is nothing
    beside reading the bytes, and an object rewritten under the same key
    would otherwise keep the id of what it replaced.

    A directory is covered by its members rather than by itself: a
    directory's own mtime moves when members come and go, but not when
    one of them is rewritten, which is exactly the case worth catching.
    """
    try:
        path = (
            coerce_to_local_path(source)
            if is_local_path(source)
            else coerce_to_upath(source)
        )
        if path.is_dir():
            return _directory_stats(path)
        return _size_and_mtime(path.stat())
    except Exception:
        # A source which will not answer is one with no size and no
        # mtime, which is what the id then says of it.
        return None, None


def _is_hidden(relative) -> bool:
    """Return True when a path, or any directory above it, is hidden."""
    return any(part.startswith(".") for part in relative.parts)


def _size_and_mtime(stat) -> tuple[int | None, int | None]:
    """Return one stat result's size and modification time in nanoseconds."""
    size = getattr(stat, "st_size", None)
    mtime = getattr(stat, "st_mtime_ns", None)
    if mtime is None and (seconds := getattr(stat, "st_mtime", None)) is not None:
        mtime = int(seconds * 1_000_000_000)
    return (None if size is None else int(size), None if mtime is None else int(mtime))


def _directory_stats(path) -> tuple[int, int]:
    """
    Return the total size and latest modification time of a directory.

    A directory-format source is one scan unit made of many files, and
    the two numbers stand for all of them: a member rewritten in place
    moves the latest mtime, and one which changes length moves the total
    even if a clock does not. A directory's own stat says neither, which
    is why it is not used.

    Hidden members are skipped, as they are in the index's own manifest
    over a directory-format unit -- and so is anything under a hidden
    directory, which is hidden for the same reason its parent is.
    """
    stats = [
        _size_and_mtime(x.stat())
        for x in path.rglob("*")
        if x.is_file() and not _is_hidden(x.relative_to(path))
    ]
    return (
        sum(size or 0 for size, _ in stats),
        max((mtime or 0 for _, mtime in stats), default=0),
    )


def _source_path_string(source) -> str:
    """
    Return how a source spells itself, or nothing if it does not.

    An open file names the path it was opened on, and a manager names
    what it was built around, so reading a file by handle is reading the
    same data as reading it by name.
    """
    if isinstance(source, IOResourceManager):
        source = source.source
    # An id names the file an examples:// name resolves to, not the name.
    source = resolve_example_uri(source)
    if isinstance(source, str | Path | UPath):
        return _canonical_path(source)
    for attribute in ("_dascore_source_path", "name", "filename"):
        if value := getattr(source, attribute, ""):
            # A file object opened on a descriptor names an int, which is
            # not a path and is not the same one twice.
            if isinstance(value, str | Path | UPath):
                return _canonical_path(value)
    return ""


def _canonical_path(path) -> str:
    """
    Return the one spelling of a path an id is derived from.

    A local path resolves, so a relative spelling, an absolute one and the
    one a spool absolutizes out of its index all name a single datum --
    which is what lets a patch scanned through a spool and the same patch
    read straight off disk agree about which data they are.

    A URI is left alone: it is already absolute, and resolving one would
    only mangle it.
    """
    text = str(path)
    if not is_local_path(text):
        return text
    try:
        # `coerce_to_local_path` rather than `Path`: a local file may be
        # spelled as a `file://` URI, which `Path` would read as a
        # relative directory called `file:` and resolve against the
        # working directory.
        return str(coerce_to_local_path(text).resolve())
    except Exception:
        # A path the filesystem will not answer for is still a path, and
        # a spelling nothing can canonicalize is better than none.
        return text


def source_identity(source) -> tuple[str, int | None, int | None]:
    """
    Return what a source is: its canonical path, its size and its mtime.

    The three fields of a derived id which come from the source rather
    than from the reader; see
    [`source_patch_id`](`dascore.utils.identity.source_patch_id`).
    """
    if not (path := _source_path_string(source)):
        return "", None, None
    return path, *_source_stats(path)


def _stamp_source_ids(
    patches: list[dc.Patch], file_format: str, file_version: str, source
) -> list[dc.Patch]:
    """Attach source metadata and preserve stored or source-derived patch IDs."""
    path, size_bytes, mtime_ns = source_identity(source)
    reload_path = str(_get_reloadable_source_path(source) or "")
    out = []
    for index, patch in enumerate(patches):
        origin = replace(
            patch._source or PatchSource(),
            path=reload_path,
            format=file_format,
            version=file_version,
        )
        attrs = patch.attrs
        if path and ids_enabled():
            stored = attrs.get(STORED_PATCH_ID, "")
            patch_id = stored or source_patch_id(
                replace(origin, path=path),
                size_bytes,
                mtime_ns,
                ordinal=index,
            )
            # Validate IDs supplied by a reader; derived IDs are trusted strings
            # and need no second validation of every scientific attribute.
            attrs = (
                attrs.update(patch_id=patch_id)
                if stored
                else attrs.model_copy(update={"patch_id": patch_id})
            )
        if hasattr(attrs, STORED_PATCH_ID):
            attrs = attrs.drop(STORED_PATCH_ID)
        out.append(patch.new(attrs=attrs, source=origin))
    return out


def read(
    path: path_types | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    time: time_select_type | None = None,
    distance: float_select_type | None = None,
    **kwargs,
) -> dc.Spool:
    """
    Read a fiber file.

    For most cases, [`dascore.spool`](`dascore.spool`) is preferable to
    this function.

    Parameters
    ----------
    path
        A path to the file to read.
    file_format
        A string indicating the file format. If not provided dascore will
        try to estimate the format.
    file_version
        An optional string indicating the format version.
    time
        An optional tuple of time ranges.
    distance
        An optional tuple of distances.
    *kwargs
        All kwargs are passed to the format-specific read functions.

    Notes
    -----
    This function loads the requested samples immediately. Use
    [`spool`](`dascore.spool`) to defer loading until patches are accessed.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>>
    >>> patch = dc.read("examples://terra15_das_1_trimmed.hdf5")
    """
    # Held because `path` is reassigned to whatever the reader wanted; the
    # id names the source the caller asked for, not the handle it became.
    # An examples:// name resolves first so the id names the file, not the URI.
    source = path = resolve_example_uri(path)
    with remote_cache_scope("read"):
        with IOResourceManager(path) as man:
            inferred_format = not file_format or not file_version
            if not file_format or not file_version:
                file_format, file_version = get_format(
                    man,
                    file_format=file_format,
                    file_version=file_version,
                )
            # If we had to probe metadata first, reopen the resource for the
            # actual read. Some remote HDF5/fileobj stacks do not reliably
            # tolerate reusing the same handle across sniffing and full reads.
            if inferred_format:
                man.clear_cache()
            fiber_io = FiberIO.manager.get_fiberio(
                format=file_format, version=file_version
            )
            out = fiber_io.read(
                man,
                file_version=file_version,
                _provenance_source=source,
                time=time,
                distance=distance,
                _pre_cast=True,
                **kwargs,
            )
            # The reader's own spelling of its format, not the caller's:
            # `dc.read(path, "netcdf_cf")` and `dc.read(path, "NETCDF_CF")`
            # resolve to one FiberIO and must name one datum.
            return out


def scan_to_df(
    path: ScanInput | pd.DataFrame,
    file_format: str | None = None,
    file_version: str | None = None,
    ext: str | None = None,
    timestamp: float | None = None,
    progress: PROGRESS_LEVELS = "standard",
    exclude=("history",),
) -> pd.DataFrame:
    """
    Scan a path, return a dataframe of contents.

    The columns of the dataframe depend on the attributes and coordinates
    found in the data files.

    Parameters
    ----------
    path
        The path to the to file to scan
    file_format
        Format of the file. If not provided DASCore will try to determine it.
    file_version
        The version string of the file.
    ext
        The extensions to map.
    timestamp
        Minimum modification time.
    progress
        The type of progress bar to use. None disables progress bar and
        "basic" is best for low latency scenarios.
    exclude
        A sequence of column names to exclude in the final dataframe.

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> df = dc.scan_to_df("examples://terra15_das_1_trimmed.hdf5")
    """
    if isinstance(path, pd.DataFrame):
        return path
    if isinstance(path, Spool):
        return path.get_contents()
    info = scan(
        path=path,
        file_format=file_format,
        file_version=file_version,
        ext=ext,
        timestamp=timestamp,
        progress=progress,
    )
    records = []
    for item in info:
        records.append(item.flat_dump(exclude=exclude))
    df = pd.DataFrame(records)
    if "dims" in df.columns:
        df["dims"] = df["dims"].astype(str)
    return df


def _iterate_scan_inputs(patch_source, ext, mtime, include_directories=True, **kwargs):
    """Yield scan candidates."""
    for el in iterate(patch_source):
        el = resolve_example_uri(el)
        if isinstance(el, str | Path | UPath):
            path = (
                coerce_to_local_path(el) if is_local_path(el) else coerce_to_upath(el)
            )
            if path.exists():
                generator = _iter_filesystem(
                    path,
                    ext=ext,
                    timestamp=mtime,
                    include_directories=include_directories,
                )
                try:
                    candidate = next(generator)
                except StopIteration:
                    continue
                while True:
                    signal = yield candidate
                    try:
                        candidate = generator.send(signal)
                    except StopIteration:
                        break
                continue
        yield el


def _get_fiber_io_and_req_type(
    manager,
    file_format: str | None = None,
    file_version: str | None = None,
    fiber_io_hint=None,
):
    """
    Get the fiber IO for a patch source.

    Raises
    ------
    UnknownFileFormatError if no format is determinable from the
    patch_source

    """
    if not file_format or not file_version:
        file_format_, file_version_ = FiberIO.manager._get_format(
            path=manager,
            file_format=file_format,
            file_version=file_version,
            fiber_io_hint=fiber_io_hint,
        )
    else:
        # we need separate loop variables so this doesn't get assumed
        # to be the version/format in all subsequent values for the loop.
        file_format_, file_version_ = file_format, file_version
    fiber_io_hint = FiberIO.manager.get_fiberio(
        format=file_format_, version=file_version_
    )
    req_type = _required_resource_type(fiber_io_hint.get_metadata)
    resource = manager.get_resource(req_type)
    # this will get the required resource type to pass to scan.
    return fiber_io_hint, resource


def _count_generator(generator):
    """Estimate the number of updates needed."""
    # TODO: This is a but sloppy, need to think of a better way to do
    # this to avoid double iteration.
    # First get total number of possible update-able files
    entity_count = 0
    for _ in generator:
        entity_count += 1
    return entity_count


_MISSING_MODULE_PATTERN = re.compile(r"^(\S+) is not installed")


def _get_missing_install_name(exception: MissingOptionalDependencyError) -> str:
    """Get the installable package name from a missing dependency error."""
    if exception.install_name:
        return exception.install_name
    # Errors raised outside of optional_import (eg by a third party FiberIO)
    # identify the module, if at all, with the module name or the message form
    # optional_import used to use. Any other message could say anything, so
    # nothing is recommended for installation.
    if not (name := exception.name or ""):
        match = _MISSING_MODULE_PATTERN.match(exception.msg or "")
        name = match.group(1) if match else ""
    return _get_install_name(name)


def _handle_missing_optionals(output_count, optional_dep_dict):
    """
    Inform the user there are files that can be read but the proper
    dependencies are not installed.

    If there are other readable files that were found, raise a warning.
    Otherwise, raise a MissingOptionalDependencyError.
    """
    counts = ", ".join(
        f"{name or 'unknown'} ({count} files)"
        for name, count in sorted(optional_dep_dict.items())
    )
    # Unidentifiable packages can't be included in an install command.
    packages = [x for x in optional_dep_dict if x]
    install = f" {_get_install_message(packages)}" if packages else ""
    msg = (
        f"DASCore found files that can be read if additional packages are "
        f"installed. The needed packages and the found number of files are: "
        f"{counts}.{install}"
    )
    warn_or_raise(
        msg,
        exception=MissingOptionalDependencyError,
        warning=UserWarning,
        behavior="warn" if output_count else "raise",
    )


def _iter_scan_results(
    path: ScanInput,
    file_format: str | None = None,
    file_version: str | None = None,
    ext: str | None = None,
    timestamp: float | None = None,
    progress: PROGRESS_LEVELS | Progress = "standard",
    *,
    snap: bool = True,
) -> Generator[tuple[dc.Patch, int], None, None]:
    """
    Yield raw scan results with dispatcher-owned source information.

    Each result is tagged with the index of the input it came from, which
    is the only thing that tells a file's second patch apart from the same
    file scanned twice: both spell one source path and, for a format which
    names no patch within a file, one key.
    """
    output_count = 0
    input_index = -1
    fiber_io_hint: dict[str, FiberIO] = {}
    # A dict for keeping track of missing optional dependencies.
    missing_optional_deps = defaultdict(lambda: 0)
    # A one-shot iterator (e.g. a generator) can't survive both walks
    # below, so materialize it once up front (see #818). The cast just
    # keeps the element type ty loses when narrowing the union.
    if isinstance(path, Iterator):
        path = list(cast("Iterable[path_types | dc.Patch | IOResourceManager]", path))
    # Unfortunately, we have to iterate the scan candidates twice to get
    # an estimate for the progress bar length. Maybe there is a better way...
    _generator = _iterate_scan_inputs(
        path, ext=ext, mtime=timestamp, include_directories=False
    )
    length = _count_generator(_generator)
    generator = _iterate_scan_inputs(path, ext=ext, mtime=timestamp)
    # We want to avoid printing long object str reprs, so only print paths.
    resource_str = path if isinstance(path, str | Path | UPath) else ""
    tracker = track(
        generator,
        f"scan {resource_str}",
        progress=progress,
        length=length,
        min_length=20,
    )
    try:
        with remote_cache_scope("metadata"):
            for patch_source in tracker:
                input_index += 1
                if isinstance(patch_source, dc.Patch):
                    patch = patch_source
                    result = dc.Patch(
                        coords=patch.coords,
                        attrs=patch.attrs,
                        dtype=patch.dtype,
                        source=patch._source,
                    )
                    output_count += 1
                    yield result, input_index
                    continue
                with IOResourceManager(patch_source) as man:
                    try:
                        fiber_io, resource = _get_fiber_io_and_req_type(
                            man,
                            file_format=file_format,
                            file_version=file_version,
                            fiber_io_hint=fiber_io_hint,
                        )
                    except UnknownFiberFormatError:  # skip bad entities
                        continue
                    # Cache this fiber io to given preferential treatment next
                    # iteration. This speeds up the common case of many files
                    # with the same format.
                    fiber_io_hint[fiber_io.input_type] = fiber_io
                    # Special handling of directory FiberIOs.
                    if fiber_io.input_type == "directory":
                        # Directory fiber_io should send skip signal back to generator
                        # so that no files/sub directories are scanned.
                        generator.send("skip")
                        if not fiber_io._updated_after(resource, timestamp):
                            continue
                        # Directory FiberIO may need to know the time after which
                        # contents should be returned.
                        scan_kwargs = {"timestamp": timestamp, "_pre_cast": True}
                        if snap is not None:
                            scan_kwargs["snap"] = snap
                        source = fiber_io.scan(resource, **scan_kwargs)
                    else:
                        try:
                            scan_kwargs = {"_pre_cast": True}
                            if snap is not None:
                                scan_kwargs["snap"] = snap
                            source = fiber_io.scan(resource, **scan_kwargs)
                        except MissingOptionalDependencyError as ex:
                            missing_optional_deps[_get_missing_install_name(ex)] += 1
                            continue
                        # scan() is best-effort across many resources, so surface
                        # dependency/compatibility problems as warnings and keep
                        # scanning the remaining files.
                        except DependencyError as exc:
                            warnings.warn(str(exc), UserWarning, stacklevel=2)
                            continue
                        except RemoteCacheError:
                            raise
                        # This happens if the file is corrupt see #346.
                        except (
                            OSError,
                            InvalidFiberFileError,
                            ValueError,
                            TypeError,
                        ):
                            warnings.warn(f"Failed to scan {resource}", UserWarning)
                            continue
                    patches = [_validate_metadata(patch) for patch in source]
                    patches = _stamp_source_ids(
                        patches, fiber_io.name, fiber_io.version, man.source
                    )
                    for result in patches:
                        output_count += 1
                        yield result, input_index
    # Stop the progress display before propagating Ctrl+C.
    except KeyboardInterrupt:
        getattr(progress, "stop", lambda: None)()
        raise
    if missing_optional_deps:
        _handle_missing_optionals(output_count, missing_optional_deps)


def scan_payloads(
    path: ScanInput,
    file_format: str | None = None,
    file_version: str | None = None,
    ext: str | None = None,
    timestamp: float | None = None,
    progress: PROGRESS_LEVELS | Progress = "standard",
    snap: bool = True,
) -> list[dc.Patch]:
    """
    Scan a potential patch source and return full coordinate payloads.

    Parameters
    ----------
    path
        A resource containing fiber data.
    file_format
        File format. DASCore detects it when omitted. Only applies to path-like inputs.
    file_version
        File version. DASCore detects it when omitted. Only applies to path-like inputs.
    ext
        The extensions to map.
    timestamp
        Minimum modification time.
    progress
        The type of progress bar to use. None disables the progress bar.
    snap
        If True (the default), formats may represent stored sample times as an
        idealized uniform range. If False, returned coords represent stored
        coordinate values exactly when the format exposes them.

    Returns
    -------
    A list of data-less [`Patch`](`dascore.Patch`) objects with full coordinate
    managers, dtype, and private source provenance.

    Notes
    -----
    Scan payloads retain real coordinate arrays and can use substantially more
    memory than [`scan`](`dascore.scan`) summaries. Prefer scanning specific
    files and discard payloads promptly when probing many resources.
    """
    return [
        patch
        for patch, _ in _iter_scan_results(
            path=path,
            file_format=file_format,
            file_version=file_version,
            ext=ext,
            timestamp=timestamp,
            progress=progress,
            snap=snap,
        )
    ]


def scan(
    path: ScanInput,
    file_format: str | None = None,
    file_version: str | None = None,
    ext: str | None = None,
    timestamp: float | None = None,
    progress: PROGRESS_LEVELS | Progress = "standard",
) -> list[PatchSummary]:
    """
    Scan a potential patch source and return its patch summaries.

    Parameters
    ----------
    path
        A resource containing Fiber data.
    file_format
        File format. DASCore detects it when omitted. Only applies to path-like inputs.
    file_version
        File version. DASCore detects it when omitted. Only applies to path-like inputs.
    ext
        The extensions to map.
    timestamp
        Minimum modification time.
    progress
        Progress display. None disables it, ``"basic"`` suits low-latency
        operations, and a ``rich.progress.Progress`` subclass customizes it.

    Returns
    -------
    A list of [`PatchSummary`](`dascore.PatchSummary`) instances.

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> summary = dc.scan("examples://terra15_das_1_trimmed.hdf5")[0]

    See Also
    --------
    [`scan_payloads`](`dascore.scan_payloads`)
        Return full coordinate managers instead of envelope summaries.
    """
    return [
        patch.summary
        for patch, _ in _iter_scan_results(
            path=path,
            file_format=file_format,
            file_version=file_version,
            ext=ext,
            timestamp=timestamp,
            progress=progress,
        )
    ]


def get_format(
    path: path_types | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    fiber_io_hint: dict[str, FiberIO] | None = None,
    **kwargs,
) -> tuple[str, str]:
    """
    Return the file's format name and version.

    Parameters
    ----------
    path
        The path to the file.
    file_format
        The known file format.
    file_version
        The known file version.
    fiber_io_hint
        Mapping of input type to the last-used FiberIO, used as a detection hint.

    Returns
    -------
    A tuple of (file_format_name, version) both as strings.

    Raises
    ------
    dascore.exceptions.UnknownFiberFormat - Could not determine the fiber format.

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> file_format, file_version = dc.get_format("examples://prodml_2.1.h5")
    """
    path = resolve_example_uri(path)
    scope = get_remote_cache_scope()
    if scope == "read":
        return FiberIO.manager._get_format(
            path, file_format, file_version, fiber_io_hint, **kwargs
        )
    with remote_cache_scope("metadata"):
        out = FiberIO.manager._get_format(
            path, file_format, file_version, fiber_io_hint, **kwargs
        )
    return out


def is_directory_format(path) -> bool:
    """
    Return True if a directory is itself one FiberIO scan unit.

    A directory-format source (e.g. XMLBinary) is read as a whole rather
    than by traversing its members. This is the single definition of that
    condition; dc.scan's traversal skips such a directory's contents and
    the directory indexer treats it as one stat unit.
    """
    if not Path(path).is_dir():
        return False
    try:
        get_format(path)
    except (UnknownFiberFormatError, OSError):
        # An unreadable directory (e.g. PermissionError) is simply not a
        # scan unit; it should not abort a directory index traversal.
        return False
    return True


def _may_hold_gaps(spool) -> bool:
    """
    Return True when the spool can produce a patch with gapped coordinates.

    Live patches and plan-assembled outputs can carry a segmented
    coordinate, and so can a file read from a format which stores one
    (`segmented_write`); every other file read is contiguous, so a spool
    of those skips gap inspection rather than loading every patch.
    """
    if getattr(spool, "has_live_patches", False):
        return True
    catalog = getattr(spool, "_catalog", None)
    resolver = getattr(catalog, "resolver", None)
    if getattr(resolver, "plan_entries", dict)():
        return True
    df = spool.get_contents()
    sources = set(zip(df.get("source_format", ()), df.get("source_version", ())))
    manager = FiberIO.manager
    return any(
        manager.get_fiberio(format=name, version=version).segmented_write
        for name, version in sources
    )


def _maybe_split_gapped_patches(spool, fiber_io, split):
    """Handle patches whose dimensional coords contain gaps before writing."""
    # a destination which stores gaps needs no inspection, which would
    # otherwise load every patch of a file-backed spool at once
    if (fiber_io.segmented_write and not split) or not _may_hold_gaps(spool):
        return spool

    def _has_gaps(patch):
        coords = (patch.get_coord(x) for x in patch.dims)
        return any(isinstance(x, CoordSegmented) for x in coords)

    # Materialize once (cheap; patches are in memory) so gap detection and
    # splitting see the same patch sequence.
    contents = list(spool)
    gapped = [_has_gaps(x) for x in contents]
    if not any(gapped):
        return spool
    if not split:
        msg = (
            f"Format {fiber_io.name} cannot write patches whose dimensional "
            "coordinates contain gaps (segmented coordinates); its patches "
            "must be contiguous. Pass split=True to write each contiguous "
            "section as its own patch, or split explicitly with "
            "patch.split_gaps()."
        )
        raise ParameterError(msg)
    patches = []
    for patch, has_gaps in zip(contents, gapped, strict=True):
        patches.extend(patch.split_gaps() if has_gaps else [patch])
    if len(patches) > 1 and not fiber_io.multi_patch_write:
        msg = (
            f"Format {fiber_io.name} writes a single patch per file, so "
            "gapped patches cannot be split into it. Use patch.split_gaps() "
            "and write each patch to its own file."
        )
        raise ParameterError(msg)
    return dc.spool(patches)


# write hands back the path it was given, so the return follows the
# argument rather than collapsing to the union: a Path in, a Path out.
_PathT = TypeVar("_PathT", bound=path_types)


def write(
    patch_or_spool,
    path: _PathT,
    file_format: str,
    file_version: str | None = None,
    split: bool = False,
    **kwargs,
) -> _PathT:
    """
    Write a Patch or Spool to disk.

    Parameters
    ----------
    patch_or_spool
        The [`Patch`](`dascore.Patch`) or spool to write to disk.
    path
        The path to the file.
    file_format
        The string indicating the format to write.
    file_version
        Optionally specify the version of the file, else use the latest
        version for the format.
    split
        If True, patches whose dimensional coordinates contain gaps
        (segmented coordinates, e.g. from merging nearly-contiguous data)
        are split into contiguous patches before writing; this requires a
        format which supports multiple patches per file. If False (default)
        a format which stores gapped patches (DASDAE version 2) writes them
        whole and any other raises a
        [`ParameterError`](`dascore.exceptions.ParameterError`).

    Raises
    ------
    [`UnknownFiberFormatError`](`dascore.exceptions.UnknownFiberFormatError`)
        - Could not determine the fiber format.
    [`ParameterError`](`dascore.exceptions.ParameterError`)
        - The path is an ``examples://`` name, which is read-only.

    Examples
    --------
    >>> from pathlib import Path
    >>> import dascore as dc
    >>>
    >>> patch = dc.get_example_patch()
    >>> path = Path("output.h5")
    >>> _ = dc.write(patch, path, "dasdae")
    >>>
    >>> assert path.exists()
    >>> path.unlink()
    """
    # Example files are read-only; writing to one would land on top of the
    # downloader's cached copy. A manager is unwrapped first so wrapping the
    # uri is not a way around this.
    target = path.source if isinstance(path, IOResourceManager) else path
    if is_example_uri(target):
        msg = (
            f"Cannot write to {target}; examples:// names are read-only. "
            f"Give a path to write to instead."
        )
        raise ParameterError(msg)
    fiber_io = FiberIO.manager.get_fiberio(format=file_format, version=file_version)
    if not isinstance(patch_or_spool, dc.Spool):
        patch_or_spool = dc.spool([patch_or_spool])
    patch_or_spool = _maybe_split_gapped_patches(patch_or_spool, fiber_io, split)
    with IOResourceManager(path) as man:
        func = fiber_io.write
        required_type = _required_resource_type(func)
        resource = man.get_resource(required_type)
        func(patch_or_spool, resource, _pre_cast=True, **kwargs)
    return path
