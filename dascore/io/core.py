"""
Base functionality for reading, writing, determining file formats, and scanning
Das Data.
"""

from __future__ import annotations

import inspect
import warnings
from collections import defaultdict
from collections.abc import Generator, Mapping
from contextlib import suppress
from functools import cached_property, wraps
from numbers import Integral
from pathlib import Path
from threading import RLock
from typing import Any, Literal, NotRequired, TypedDict, get_type_hints

import numpy as np
import pandas as pd

import dascore as dc
from dascore.compat import Progress, UPath
from dascore.constants import (
    PROGRESS_LEVELS,
    PatchType,
    SpoolType,
    path_types,
    timeable_types,
)
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import CoordManager
from dascore.core.spool import Spool
from dascore.core.summary import PatchSummary, normalize_source_patch_id
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
from dascore.utils.io import IOResourceManager, get_handle_from_resource
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    _iter_filesystem,
    _locked,
    _reinit_after_fork,
    cached_method,
    iterate,
    warn_or_raise,
)
from dascore.utils.paths import coerce_to_local_path, coerce_to_upath, is_local_path
from dascore.utils.plugins import get_entry_point_loaders
from dascore.utils.progress import track
from dascore.utils.remote_io import get_remote_cache_scope, remote_cache_scope


class ScanPayload(TypedDict):
    """The structured payload contract returned by `FiberIO.scan()`."""

    attrs: PatchAttrs
    coords: CoordManager
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    source_patch_id: NotRequired[str]
    source_path: NotRequired[str | Path | UPath]
    source_format: NotRequired[str]
    source_version: NotRequired[str]


def _make_scan_payload(
    *,
    attrs: dc.PatchAttrs | Mapping[str, Any] | None,
    coords,
    dims=(),
    shape=(),
    dtype: str = "",
    source_patch_id: str = "",
) -> ScanPayload:
    """Build one normalized FiberIO scan payload."""
    return {
        "attrs": PatchAttrs.from_dict(attrs),
        "coords": coords,
        "dims": tuple(dims),
        "shape": tuple(shape),
        "dtype": str(dtype),
        "source_patch_id": normalize_source_patch_id(source_patch_id),
    }


_SCAN_PAYLOAD_REQUIRED = ("attrs", "coords", "dims", "shape", "dtype")


def _validate_scan_payload(result, require_coord_manager: bool = False):
    """
    Validate one FiberIO.scan() result against the ScanPayload contract.

    Shared by the `dc.scan` summary path and `dc.scan_payloads` so both
    public boundaries enforce the same requirements; only the payload
    API additionally requires `coords` to be a full CoordManager (the
    summary path also accepts already-collapsed coordinate mappings) and
    requires `dims` and `shape` to match it exactly.
    """
    if isinstance(result, dc.PatchAttrs):
        msg = (
            "DASCore no longer accepts PatchAttrs from FiberIO.scan(). "
            "Return a structured scan payload instead. "
            "See docs/contributing/new_format.qmd."
        )
        raise ValueError(msg)
    if not isinstance(result, Mapping):
        msg = (
            "FiberIO.scan() must return ScanPayload mappings; got "
            f"{type(result).__name__}. See docs/contributing/new_format.qmd."
        )
        raise TypeError(msg)
    missing = sorted(set(_SCAN_PAYLOAD_REQUIRED) - set(result))
    if missing:
        msg = (
            f"scan payload is missing required keys {missing}; a ScanPayload "
            "requires a mapping with `coords`, `attrs`, and `dtype` as well "
            "as `dims` and `shape`. See docs/contributing/new_format.qmd."
        )
        raise TypeError(msg)
    if require_coord_manager and not isinstance(result["coords"], CoordManager):
        msg = (
            "scan payload `coords` must be a CoordManager holding the full "
            f"coordinates; got {type(result['coords']).__name__}."
        )
        raise TypeError(msg)
    if not isinstance(result["coords"], CoordManager | Mapping):
        msg = "scan payload `coords` must be a CoordManager or coordinate mapping."
        raise TypeError(msg)
    attrs = result["attrs"]
    if not isinstance(attrs, PatchAttrs | Mapping):
        msg = "scan payload `attrs` must be PatchAttrs or an attribute mapping."
        raise TypeError(msg)
    try:
        PatchAttrs.from_dict(attrs)
    except (TypeError, ValueError) as exc:
        msg = "scan payload `attrs` contains invalid attribute values."
        raise TypeError(msg) from exc
    dims = result["dims"]
    valid_dims = (
        isinstance(dims, tuple)
        and all(isinstance(x, str) and x for x in dims)
        and len(set(dims)) == len(dims)
    )
    if not valid_dims:
        msg = "scan payload `dims` must be a tuple of unique, non-empty strings."
        raise TypeError(msg)
    shape = result["shape"]
    valid_shape = (
        isinstance(shape, tuple)
        and len(shape) == len(dims)
        and all(
            isinstance(x, Integral) and not isinstance(x, bool) and x >= 0
            for x in shape
        )
    )
    if not valid_shape:
        msg = "scan payload `shape` must contain one non-negative integer per dim."
        raise TypeError(msg)
    dtype = result["dtype"]
    if not isinstance(dtype, str):
        msg = "scan payload `dtype` must be a string."
        raise TypeError(msg)
    try:
        np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        msg = f"scan payload `dtype` is invalid: {dtype!r}."
        raise TypeError(msg) from exc
    optional_types = {
        "source_patch_id": str,
        "source_path": (str, Path, UPath),
        "source_format": str,
        "source_version": str,
    }
    for name, expected_type in optional_types.items():
        if name in result and not isinstance(result[name], expected_type):
            msg = f"scan payload `{name}` has an invalid type."
            raise TypeError(msg)
    if require_coord_manager:
        coords = result["coords"]
        if dims != coords.dims:
            msg = "scan payload `dims` must exactly match `coords.dims`."
            raise ValueError(msg)
        if shape != coords.shape:
            msg = "scan payload `shape` must exactly match `coords.shape`."
            raise ValueError(msg)
    return result


def _scan_payload_to_summary(
    payload: ScanPayload | Mapping[str, Any],
    *,
    source_path: str | Path | UPath | None = None,
    source_format: str | None = None,
    source_version: str | None = None,
    source_patch_id: str | None = None,
) -> PatchSummary:
    """Convert one structured FiberIO scan payload into a PatchSummary."""
    _validate_scan_payload(payload)
    coords = payload["coords"]
    if hasattr(coords, "to_summary_dict"):
        coords = coords.to_summary_dict()
    return PatchSummary(
        attrs=PatchAttrs.from_dict(payload["attrs"]),
        coords=coords,
        dims=tuple(payload.get("dims", ())),
        shape=tuple(payload.get("shape", ())),
        dtype=str(payload["dtype"]),
        source_path=source_path,
        source_format=source_format,
        source_version=source_version,
        source_patch_id=(
            normalize_source_patch_id(source_patch_id)
            or normalize_source_patch_id(payload.get("source_patch_id"))
        ),
    )


def _scan_result_to_summary(
    patch_summary: PatchSummary | ScanPayload | Mapping[str, Any],
    *,
    source_path: str | Path | UPath | None = None,
    source_format: str | None = None,
    source_version: str | None = None,
    source_patch_id: str | None = None,
) -> PatchSummary:
    """Convert scan metadata into a patch summary."""
    if isinstance(patch_summary, PatchSummary) and all(
        value in (None, "")
        for value in (source_path, source_format, source_version, source_patch_id)
    ):
        return patch_summary
    normalized_source_path = "" if source_path in (None, "") else source_path
    normalized_source_format = "" if source_format in (None, "") else source_format
    normalized_source_version = "" if source_version in (None, "") else source_version
    summary_source_patch_id = normalize_source_patch_id(source_patch_id)
    if isinstance(patch_summary, Mapping):
        return _scan_payload_to_summary(
            patch_summary,
            source_path=normalized_source_path,
            source_format=normalized_source_format,
            source_version=normalized_source_version,
            source_patch_id=summary_source_patch_id,
        )
    if isinstance(patch_summary, PatchSummary):
        return PatchSummary(
            attrs=patch_summary.attrs,
            coords=dict(patch_summary.coords),
            dims=tuple(patch_summary.dims),
            shape=tuple(patch_summary.shape),
            dtype=patch_summary.dtype,
            source_path=normalized_source_path or patch_summary.source_path,
            source_format=normalized_source_format or patch_summary.source_format,
            source_version=normalized_source_version or patch_summary.source_version,
            source_patch_id=summary_source_patch_id or patch_summary.source_patch_id,
        )
    if isinstance(patch_summary, dc.PatchAttrs):
        msg = (
            "DASCore no longer accepts PatchAttrs from FiberIO.scan(). "
            "Return a structured scan payload instead. "
            "See docs/contributing/new_format.qmd."
        )
        raise ValueError(msg)
    msg = (
        "_scan_result_to_summary only accepts PatchSummary or structured "
        "scan payload mappings. "
        f"Got {type(patch_summary).__name__}."
    )
    raise TypeError(msg)


def _patch_to_summary(
    patch: dc.Patch,
    *,
    source_path: str | Path | UPath | None = None,
    source_format: str | None = None,
    source_version: str | None = None,
) -> PatchSummary:
    """Convert a loaded patch into a summary tied to its source."""
    return _scan_result_to_summary(
        patch.summary,
        source_path=source_path or "",
        source_format=source_format,
        source_version=source_version,
    )


def _patch_to_scan_payload(patch: dc.Patch) -> ScanPayload:
    """Convert a loaded patch into one structured FiberIO scan payload."""
    return _make_scan_payload(
        attrs=patch.attrs,
        coords=patch.coords,
        dims=patch.dims,
        shape=patch.shape,
        dtype=str(np.dtype(patch.data.dtype)),
        source_patch_id=patch.attrs.get("_source_patch_id", ""),
    )


def _resolve_read_spool(spool, source_patch_id: object = "") -> dc.Patch:
    """
    Resolve one patch from a read result by source identity.

    Readers that consume source_patch_id may return the single matching
    patch without preserving that reload metadata on it; only trust that
    when the patch doesn't claim a different identity.
    """
    source_patch_id = normalize_source_patch_id(source_patch_id)
    if source_patch_id and len(spool) == 1:
        found = normalize_source_patch_id(spool[0].attrs.get("_source_patch_id", ""))
        if found == source_patch_id or (not found and not source_patch_id.isdigit()):
            return spool[0]
    return _select_patch_from_spool(spool, source_patch_id=source_patch_id)


def _select_patch_from_spool(spool, source_patch_id: object = "") -> dc.Patch:
    """Select one loaded patch from a spool using source identity."""
    if len(spool) == 0:
        # Iteration skips these with a warning, see #583.
        msg = (
            "No patch remained after applying load filters; the requested "
            "range may have trimmed it to nothing."
        )
        raise MissingPatchError(msg)
    if source_patch_id not in (None, ""):
        source_patch_id = str(source_patch_id)
        # Native source ids are preserved on patch attrs by their readers.
        matches = [
            patch
            for patch in spool
            if normalize_source_patch_id(patch.attrs.get("_source_patch_id", ""))
            == source_patch_id
        ]
        if len(matches) == 1:
            return matches[0]
        # Synthesized ids are positional within the full source read.
        try:
            index = int(source_patch_id)
        except (TypeError, ValueError):
            index = None
        if index is not None and 0 <= index < len(spool):
            return spool[index]
        if len(spool) == 1 and spool[0].get_patch_name() == source_patch_id:
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
        self._lookup_cache: dict[tuple, frozenset | tuple] = {}

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
        return pd.Series(get_entry_point_loaders("dascore.fiber_io"))

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
        key = ("input_type", input_type)
        if (cached := self._lookup_cache.get(key)) is None:
            if (out := self._fiber_io_by_input_type.get(input_type)) is None:
                out = set()
                for input_set in self._fiber_io_by_input_type.values():
                    out |= input_set
            cached = self._lookup_cache[key] = frozenset(out)
        return cached

    @_locked("_lock")
    def _get_prioritized_list(self, input_type="file") -> tuple[FiberIO, ...]:
        """Yield a prioritized list of fiber_ios."""
        key = ("prioritized", input_type)
        if (cached := self._lookup_cache.get(key)) is not None:
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
        self._lookup_cache[key] = out
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
        forma, ver = fiberio.name.upper(), fiberio.version
        id_tuple = (forma, ver)
        if id_tuple in self._fiber_io_name_ver:
            return
        self._loaded_eps.add(fiberio.name)
        for ext in iter(fiberio.preferred_extensions):
            self._extension_list.setdefault(ext, []).append(fiberio)
        self._format_version.setdefault(forma, {})[ver] = fiberio
        self._fiber_io_by_input_type.setdefault(fiberio.input_type, set()).add(fiberio)
        self._fiber_io_name_ver.add(id_tuple)
        # Snapshots derived from the registry are now stale.
        self._lookup_cache.clear()

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
        for fiber_io in iterator:
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
        path: str | Path | IOResourceManager,
        file_format: str | None = None,
        file_version: str | None = None,
        fiber_io_hint: dict[str, FiberIO] | None = None,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Return the name of the format contained in the file and version number.

        See [`dascore.io.core.get_format`](`dascore.io.core.get_format`)
        for docs.
        """
        with IOResourceManager(path) as man:
            path = man.source
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
                required_type = func._required_type
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
        except BaseException as e:  # get_format can't raise; must return false.
            # A handle created here must close even on failure, including on
            # KeyboardInterrupt; leaking a remote handle would leave garbage
            # collection paused for as long as the traceback is retained.
            if new_resource is not None and new_resource is not resource:
                with suppress(Exception):
                    getattr(new_resource, "close", lambda: None)()
            if not isinstance(e, Exception):
                raise
            if fun_name == "get_format":
                out = False
            else:
                raise e
        else:
            # if a new file handle was created we need to close it now. But it
            # shouldn't close any passed in, that should happen up the stack.
            if new_resource is not resource and hasattr(new_resource, "close"):
                new_resource.close()
        return out

    # attach the function and required type for later use
    _wrapper.func = func
    # subclasses of FIBERIO subclasses can wrap this twice, so we mark
    # it to avoid that scenario.
    _wrapper._type_caster_wrapped = True
    # also specify required type
    _wrapper._required_type = required_type

    return _wrapper


def _is_wrapped_func(func1, func2):
    """Small helper function to determine if func1 is func2, unwrapping decorators."""
    func = func1
    while hasattr(func, "func") or hasattr(func, "__func__"):
        func = getattr(func, "func", func)
        func = getattr(func, "__func__", func)
    return func is func2


class FiberIO:
    """
    An interface which adds support for a given filer format.

    This class should be subclassed when adding support for new formats.
    """

    name: str = ""
    version: str = ""
    preferred_extensions: tuple[str] = ()
    # Specifies if this fiber IO expects a directory or single file
    input_type: Literal["file", "directory"] = "file"
    # True when a single resource can hold more than one patch.
    multi_patch_write: bool = False

    manager = _FiberIOManager("dascore.fiber_io")

    # A dict of methods which should implement automatic type casting.
    # and the index of the parameter to type cast.
    _automatic_type_casters = FrozenDict(
        {
            "read": 1,
            "scan": 1,
            "write": 2,
            "get_format": 1,
        }
    )

    def read(self, resource, **kwargs) -> SpoolType:
        """
        Load data from a path.

        *kwargs should include support for selecting expected dimensions. For
        example, distance=(100, 200) would only read data with distance from
        100 to 200. Multi-patch formats may also accept `source_patch_id` to
        load one or more logical patches from the source.
        """
        msg = f"FiberIO: {self.name} has no read method"
        raise NotImplementedError(msg)

    def scan(self, resource, snap: bool = True, **kwargs) -> list[ScanPayload]:
        """
        Return patch-local metadata and exact coords for a resource.

        Each item in the returned list should be a `ScanPayload` dict with
        exact coords and attrs for one logical patch. Do not populate source
        metadata such as `path`, `file_format`, or `file_version`; DASCore
        attaches those in the higher-level `dc.scan(...)` pipeline.

        Multi-patch formats should set `source_patch_id` when needed so
        DASCore can reload the same logical patch later.

        Parameters
        ----------
        resource
            The resource to scan.
        snap
            If True (the default), formats may represent stored sample times
            as an idealized uniform range. If False, returned coords must
            represent stored coordinate values exactly. This is a documented
            no-op for formats whose coordinates are defined by start, step,
            and sample count metadata.
        """
        # default scan method reads in the file and returns required attributes
        # however, this can be very slow, so each parser should implement scan
        # when possible.
        read_params = inspect.signature(self.read).parameters
        read_kwargs = dict(kwargs)
        if "snap" in read_params:
            read_kwargs["snap"] = snap
        elif "snap_dims" in read_params:
            read_kwargs["snap_dims"] = snap
        try:
            spool = self.read(resource, **read_kwargs)
        except NotImplementedError:
            msg = f"FiberIO: {self.name} has no scan or read method"
            raise NotImplementedError(msg)
        return [_patch_to_scan_payload(pa) for pa in spool]

    def write(self, spool: SpoolType, resource, **kwargs):
        """Write the spool to a resource (eg path, stream, etc.)."""
        msg = f"FiberIO: {self.name} has no write method"
        raise NotImplementedError(msg)

    def get_format(self, resource, **kwargs) -> tuple[str, str] | bool:
        """
        Return a tuple of (format_name, version_numbers).

        This should only work if path is the supported file format, otherwise
        raise UnknownFiberError or return False.
        """
        msg = f"FiberIO: {self.name} has no get_version method"
        raise NotImplementedError(msg)

    @property
    def implements_read(self) -> bool:
        """Returns True if the subclass implements its own scan method else False."""
        return not _is_wrapped_func(self.read, FiberIO.read)

    @property
    def implements_write(self) -> bool:
        """Returns True if the subclass implements its own scan method else False."""
        return not _is_wrapped_func(self.write, FiberIO.write)

    @property
    def implements_scan(self) -> bool:
        """Returns True if the subclass implements its own scan method else False."""
        return not _is_wrapped_func(self.scan, FiberIO.scan)

    @property
    def implements_get_format(self) -> bool:
        """Return True if the subclass implements its own get_format method."""
        return not _is_wrapped_func(self.get_format, FiberIO.get_format)

    @classmethod
    def get_supported_io_table(cls):
        """Make a table of all the supported formats and the methods."""
        # load all the plugins, so we know about all the FiberIO classes
        FiberIO.manager.load_plugins()
        out = []
        # iterate the dict _format_version_items,
        # which has the form {format_name: {version_str: FiberIO}}
        for format_name, version_dict in FiberIO.manager._format_version.items():
            for version_name, fiberio in version_dict.items():
                format_info = {
                    "name": format_name,
                    "version": version_name,
                    "scan": fiberio.implements_scan,
                    "get_format": fiberio.implements_get_format,
                    "read": fiberio.implements_read,
                    "write": fiberio.implements_write,
                }
                out.append(format_info)
        return pd.DataFrame(out)

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
        manager: _FiberIOManager = cls.__mro__[1].manager
        manager.register_fiberio(cls())
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


def read(
    path: path_types | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    time: tuple[timeable_types | None, timeable_types | None] | None = None,
    distance: tuple[float | None, float | None] | None = None,
    **kwargs,
) -> SpoolType:
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
    Unlike [`spool`](`dascore.spool`) this function reads the entire file
    into memory.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> file_path = fetch("terra15_das_1_trimmed.hdf5")
    >>>
    >>> patch = dc.read(file_path)
    """
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
            required_type = fiber_io.read._required_type
            path = man.get_resource(required_type)
            out = fiber_io.read(
                path,
                file_version=file_version,
                time=time,
                distance=distance,
                _pre_cast=True,
                **kwargs,
            )
            # if resource has a seek go back to 0 so this stream can be re-used.
            getattr(path, "seek", lambda x: None)(0)
            return out


def scan_to_df(
    path: path_types | PatchType | SpoolType | IOResourceManager | pd.DataFrame,
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
    exclude
        A sequence of column names to exclude in the final dataframe.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> file_path = fetch("terra15_das_1_trimmed.hdf5")
    >>>
    >>> df = dc.scan_to_df(file_path)
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
    req_type = getattr(fiber_io_hint.scan, "_required_type", None)
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


def _handle_missing_optionals(output_count, optional_dep_dict):
    """
    Inform the user there are files that can be read but the proper
    dependencies are not installed.

    If there are other readable files that were found, raise a warning.
    Otherwise, raise a MissingOptionalDependencyError.
    """
    msg = (
        f"DASCore found files that can be read if additional packages are "
        f"installed. The needed packages and the found number of files are: "
        f"{dict(optional_dep_dict)}"
    )
    warn_or_raise(
        msg,
        exception=MissingOptionalDependencyError,
        warning=UserWarning,
        behavior="warn" if output_count else "raise",
    )


def _iter_scan_results(
    path: path_types | PatchType | SpoolType | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    ext: str | None = None,
    timestamp: float | None = None,
    progress: PROGRESS_LEVELS | Progress = "standard",
    *,
    snap: bool | None = None,
    payloads: bool = False,
) -> Generator[tuple[ScanPayload | PatchSummary, dict[str, Any]], None, None]:
    """Yield raw scan results with dispatcher-owned source information."""
    output_count = 0
    fiber_io_hint: dict[str, FiberIO] = {}
    # A dict for keeping track of missing optional dependencies.
    missing_optional_deps = defaultdict(lambda: 0)
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
                # Normalize direct patch inputs to summary objects.
                if isinstance(patch_source, dc.Patch):
                    if payloads:
                        summary = patch_source.summary
                        source_info = {
                            "source_path": _get_reloadable_source_path(
                                summary.source_path
                            ),
                            "source_format": summary.source_format,
                            "source_version": summary.source_version,
                        }
                        result = _patch_to_scan_payload(patch_source)
                    else:
                        source_info = {}
                        result = _patch_to_summary(
                            patch_source,
                            source_path=_get_reloadable_source_path(
                                patch_source.summary.source_path
                            ),
                        )
                    output_count += 1
                    yield result, source_info
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
                            missing_optional_deps[ex.msg.split(" ")[0]] += 1
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
                    source_path = _get_reloadable_source_path(
                        resource, fallback=man.source
                    )
                    source_info = {
                        "source_path": source_path,
                        "source_format": fiber_io.name,
                        "source_version": fiber_io.version,
                    }
                    for result in source:
                        output_count += 1
                        yield result, source_info
    # Ensure ctl + c exists scan.
    except KeyboardInterrupt:
        getattr(progress, "stop", lambda: None)()
        raise
    if missing_optional_deps:
        _handle_missing_optionals(output_count, missing_optional_deps)


def scan_payloads(
    path: path_types | PatchType | SpoolType | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    ext: str | None = None,
    timestamp: float | None = None,
    progress: PROGRESS_LEVELS | Progress = "standard",
    snap: bool = True,
) -> list[ScanPayload]:
    """
    Scan a potential patch source and return full coordinate payloads.

    Parameters
    ----------
    path
        A resource containing fiber data.
    file_format
        Format of the file. If not provided DASCore will try to determine it.
        Only applicable for path-like inputs.
    file_version
        Version of the file. If not provided DASCore will try to determine it.
        Only applicable for path-like inputs.
    ext
        The extensions to map.
    timestamp
        Time stamp indicating the minimum mtime.
    progress
        The type of progress bar to use. None disables the progress bar.
    snap
        If True (the default), formats may represent stored sample times as an
        idealized uniform range. If False, returned coords represent stored
        coordinate values exactly when the format exposes them.

    Returns
    -------
    A list of [`ScanPayload`](`dascore.io.core.ScanPayload`) dictionaries with
    full coordinate managers and source provenance.

    Notes
    -----
    Scan payloads retain real coordinate arrays and can use substantially more
    memory than [`scan`](`dascore.scan`) summaries. Prefer scanning specific
    files and discard payloads promptly when probing many resources.
    """
    out = []
    iterator = _iter_scan_results(
        path=path,
        file_format=file_format,
        file_version=file_version,
        ext=ext,
        timestamp=timestamp,
        progress=progress,
        snap=snap,
        payloads=True,
    )
    for result, source_info in iterator:
        _validate_scan_payload(result, require_coord_manager=True)
        payload = dict(result)
        payload["attrs"] = PatchAttrs.from_dict(payload["attrs"])
        payload.update(
            {
                "source_path": source_info.get("source_path") or "",
                "source_format": source_info.get("source_format") or "",
                "source_version": source_info.get("source_version") or "",
            }
        )
        out.append(payload)
    return out


def scan(
    path: path_types | PatchType | SpoolType | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    ext: str | None = None,
    timestamp: float | None = None,
    progress: PROGRESS_LEVELS | Progress = "standard",
) -> list[PatchSummary]:
    """
    Scan a potential patch source, return a list of patch summaries.

    Parameters
    ----------
    path
        A resource containing Fiber data.
    file_format
        Format of the file. If not provided DASCore will try to determine it.
        Only applicable for path-like inputs.
    file_version
        Version of the file. If not provided DASCore will try to determine it.
        Only applicable for path-like inputs.
    ext
        The extensions to map.
    timestamp
        Time stamp indicating the minimum mtime.
    progress
        The type of progress bar to use. None disables progress bar and
        "basic" is best for low latency scenarios. Can also accept a subclass
        of rich.progress.Progress.

    Returns
    -------
    A list of [`PatchSummary`](`dascore.PatchSummary`) instances.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> file_path = fetch("terra15_das_1_trimmed.hdf5")
    >>> summary = dc.scan(file_path)[0]

    See Also
    --------
    [`scan_payloads`](`dascore.scan_payloads`)
        Return full coordinate managers instead of envelope summaries.
    """
    out = []
    iterator = _iter_scan_results(
        path=path,
        file_format=file_format,
        file_version=file_version,
        ext=ext,
        timestamp=timestamp,
        progress=progress,
    )
    for result, source_info in iterator:
        out.append(_scan_result_to_summary(result, **source_info))
    return out


def get_format(
    path: path_types | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    fiber_io_hint: dict[str, FiberIO] | None = None,
    **kwargs,
) -> tuple[str, str]:
    """
    Return the name of the format contained in the file and version number.

    Parameters
    ----------
    path
        The path to the file.
    file_format
        The known file format.
    file_version
        The known file version.
    fiber_io_hint
        A dict of {input_type: fiber_io}. This is an optimization
        which assumes the last used fiberio (for a given input type)
        is likely to be the next one.

    Returns
    -------
    A tuple of (file_format_name, version) both as strings.

    Raises
    ------
    dascore.exceptions.UnknownFiberFormat - Could not determine the fiber format.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> file_path = fetch("prodml_2.1.h5")
    >>>
    >>> file_format, file_version = dc.get_format(file_path)
    """
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


def _resolves_assembled_patches(spool) -> bool:
    """
    Return True when the spool can produce patches that are not literal
    persisted file reads (live patches or plan-assembled outputs).

    Persisted patches are always contiguous, so purely file-backed
    spools skip gap inspection; plan resolvers can assemble several
    sources across a real gap into a segmented coordinate.
    """
    if getattr(spool, "has_live_patches", False):
        return True
    catalog = getattr(spool, "_catalog", None)
    resolver = getattr(catalog, "resolver", None)
    return bool(getattr(resolver, "plan_entries", dict)())


def _maybe_split_gapped_patches(spool, fiber_io, split):
    """Handle patches whose dimensional coords contain gaps before writing."""
    from dascore.core.coords import CoordSegmented

    # Gap inspection depends on what the spool resolves, not on where
    # its ultimate members live: only literal file reads are always
    # contiguous (gapped patches are never persisted).
    if not _resolves_assembled_patches(spool):
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
            "Cannot write patches whose dimensional coordinates contain "
            "gaps (segmented coordinates); a written patch must be "
            "contiguous. Pass split=True to write each contiguous section "
            "as its own patch, or split explicitly with patch.split_gaps()."
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


def write(
    patch_or_spool,
    path: path_types,
    file_format: str,
    file_version: str | None = None,
    split: bool = False,
    **kwargs,
) -> Path:
    """
    Write a Patch or Spool to disk.

    Parameters
    ----------
    path
        The path to the file.
    file_format
        The string indicating the format to write.
    file_version
        Optionally specify the version of the file, else use the latest
        version for the format.
    split
        A written patch must have contiguous (non-gapped) coordinates.
        If True, patches whose dimensional coordinates contain gaps
        (segmented coordinates, e.g. from merging nearly-contiguous data)
        are split into contiguous patches before writing; this requires a
        format which supports multiple patches per file. If False (default)
        such patches raise a
        [`ParameterError`](`dascore.exceptions.ParameterError`).

    Raises
    ------
    [`UnkownFiberFormatError`](`dascore.exceptions.UnknownFiberFormatError`)
        - Could not determine the fiber format.

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
    fiber_io = FiberIO.manager.get_fiberio(format=file_format, version=file_version)
    if not isinstance(patch_or_spool, dc.BaseSpool):
        patch_or_spool = dc.spool([patch_or_spool])
    patch_or_spool = _maybe_split_gapped_patches(patch_or_spool, fiber_io, split)
    with IOResourceManager(path) as man:
        func = fiber_io.write
        required_type = func._required_type
        resource = man.get_resource(required_type)
        func(patch_or_spool, resource, _pre_cast=True, **kwargs)
    return path
