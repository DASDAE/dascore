"""Utilities for session-scoped remote IO caching and fallback."""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from threading import RLock

from dascore.compat import UPath
from dascore.config import get_config
from dascore.constants import http_protocols
from dascore.exceptions import RemoteCacheError
from dascore.utils.misc import _reinit_after_fork
from dascore.utils.paths import coerce_to_upath, is_local_path, is_pathlike

_NO_RANGE_HTTP_PATTERNS = (
    "doesn't appear to support range requests",
    "only reading this file from the beginning is supported",
)
_NO_SIZE_HTTP_PATTERN = "cannot seek streaming http file"
_REMOTE_RESOURCE_CACHE: dict[str, UPath] = {}
# One lock per cached resource, so two threads never download the same
# file at once while unrelated downloads still run together. Entries are
# never removed; the dict is bounded by the resources a session touches.
_REMOTE_KEY_LOCKS: dict[tuple[str, Path], RLock] = {}
_REMOTE_CACHE_SCOPE: ContextVar[str] = ContextVar(
    "remote_cache_scope", default="default"
)

_gc_pause_lock = threading.Lock()
_gc_pause_depth = 0
_gc_was_enabled = False
_gc_collect_after = 0.0
_gc_pause_warned = False
# Seconds between safety-valve collections; a full collect is O(live objects),
# measured at 7-160 ms here, so it must not run on every remote open.
_GC_COLLECT_INTERVAL = 10.0


def _warn_gc_pause_once() -> None:
    """
    Say once per process that a remote read is pausing collection.

    The pause is process-global and invisible otherwise, so a program whose
    memory grows, or which finds ``gc.isenabled()`` False, has no way to
    connect either to a remote read. Warning once keeps a spool over many
    remote files from repeating it.
    """
    global _gc_pause_warned
    # Claimed under the lock so two openers cannot both warn.
    with _gc_pause_lock:
        if _gc_pause_warned or not get_config().warn_on_gc_pause:
            return
        _gc_pause_warned = True
    # Warned outside it: a filter turning this into an error must not
    # propagate while the lock is held.
    msg = (
        "Reading remote HDF5 pauses Python's automatic garbage collection "
        "until the last such handle closes, which avoids a deadlock between "
        "h5py's global lock and collection on fsspec's event-loop thread. "
        "Reference counting is unaffected, but cyclic garbage accumulates "
        "meanwhile. Set `warn_on_gc_pause=False` to silence this warning."
    )
    warnings.warn(msg, UserWarning, stacklevel=4)


def _claim_safety_collect() -> bool:
    """Return True when this caller wins the rate-limited safety collection."""
    global _gc_collect_after
    with _gc_pause_lock:
        now = time.monotonic()
        if now < _gc_collect_after:
            return False
        _gc_collect_after = now + _GC_COLLECT_INTERVAL
        return True


def pause_gc() -> None:
    """
    Pause automatic garbage collection for one remote read session.

    h5py holds its process-global lock while blocking on fsspec's event-loop
    thread for each remote fetch. An automatic collection on that thread which
    has to deallocate a dead h5py object needs the same lock, so the two
    threads deadlock. Disabling automatic collection for the handle's lifetime
    closes that window; reference counting still frees non-cyclic garbage.

    h5py documents this deadlock, and disabling collection as one of its two
    mitigations, under "Python file-like objects":
    https://docs.h5py.org/en/stable/high/file.html#python-file-like-objects
    Its other mitigation, avoiding reference cycles which keep h5py objects
    alive, is not available to us: the cycle can be anywhere in the process.

    Calls nest; every ``pause_gc`` needs one ``resume_gc``. ``_ManagedH5pyFile``
    pairs them with ``close``/``__del__``. An interrupted ``pause_gc`` still
    leaves the depth consistent with the pauses it took, so a caller which
    resumes on any failure stays balanced.

    The pause is process-global, so cyclic garbage from every thread
    accumulates until the last remote handle closes. A handle which is never
    closed keeps it paused; one which is dropped inside a reference cycle is
    recovered by the collection below, on the next remote open -- if the
    program makes one. Otherwise the pause outlives every remote read.
    """
    global _gc_pause_depth, _gc_was_enabled
    # Safety valve: finalize cyclic garbage, including a handle leaked by an
    # earlier session, so a stranded pause cannot disable collection forever.
    # It runs before this pause is taken, so an interrupt during the collect
    # cannot strand one, and outside the lock, since finalizing such a handle
    # calls resume_gc, which takes it.
    if _claim_safety_collect():
        gc.collect()
    with _gc_pause_lock:
        # The count must move even if ``disable`` is interrupted, or the depth
        # stops matching the live handles: it would never fall back to zero,
        # so no later pause would ever disable collection again.
        try:
            if _gc_pause_depth == 0:
                _gc_was_enabled = gc.isenabled()
                gc.disable()
        finally:
            _gc_pause_depth += 1
    # Last, so a warning filter which raises cannot escape before the pause is
    # accounted for. The caller resumes on any failure, and would otherwise
    # release a pause this call never took -- stranding another live handle.
    _warn_gc_pause_once()


def resume_gc() -> None:
    """Undo one ``pause_gc``; the last one restores the previous gc state."""
    global _gc_pause_depth
    with _gc_pause_lock:
        if _gc_pause_depth == 0:
            return
        # Enable first, for the mirror image of the reason pause counts first.
        if _gc_pause_depth == 1 and _gc_was_enabled:
            gc.enable()
        _gc_pause_depth -= 1


@_reinit_after_fork
def _reset_gc_pause_state():
    """
    Drop a pause inherited from a fork; no close in the child can undo it.

    Handles inherited by the child record the pid that paused for them, so
    closing one there cannot resume a pause this child never took.
    """
    global _gc_pause_depth, _gc_pause_lock, _gc_collect_after, _gc_pause_warned
    _gc_pause_lock = threading.Lock()
    if _gc_pause_depth and _gc_was_enabled:
        gc.enable()
    _gc_pause_depth = 0
    # Neither the parent's rate limit nor its warning describes this process;
    # a pool worker which pauses collection should say so itself.
    _gc_collect_after = 0.0
    _gc_pause_warned = False


@_reinit_after_fork
def _reinit_remote_cache_locks():
    """Install fresh download locks; see _reinit_after_fork."""
    global _REMOTE_KEY_LOCKS
    _REMOTE_KEY_LOCKS = {}


@contextmanager
def remote_cache_scope(scope: str):
    """Temporarily set the current remote-cache policy scope."""
    token = _REMOTE_CACHE_SCOPE.set(scope)
    try:
        yield
    finally:
        _REMOTE_CACHE_SCOPE.reset(token)


def get_remote_cache_scope() -> str:
    """Return the current remote-cache policy scope."""
    return _REMOTE_CACHE_SCOPE.get()


def get_remote_cache_path() -> Path:
    """Return the on-disk directory used for remote file caching."""
    path = get_config().remote_cache_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_remote_name(path: UPath) -> str:
    """Return a basename suitable for a cached local copy."""
    name = path.name or f"remote-file{path.suffix or '.tmp'}"
    return Path(name).name


def normalize_remote_id(path) -> str:
    """Return a stable identifier for a remote path."""
    resource = coerce_to_upath(path)
    storage_options = getattr(resource, "storage_options", None) or {}
    options_suffix = ""
    if storage_options:
        serialized = json.dumps(storage_options, sort_keys=True, default=str)
        options_suffix = f"#{sha256(serialized.encode()).hexdigest()}"
    remote_id = f"{resource}{options_suffix}"
    _REMOTE_RESOURCE_CACHE[remote_id] = resource
    return remote_id


def _normalize_cache_root(cache_root: Path | str) -> Path:
    """Return one normalized cache-root path."""
    return Path(cache_root).expanduser()


def _remote_cache_local_path(remote_id: str, cache_root: Path, resource: UPath) -> Path:
    """
    Return the local path one remote resource materializes to.

    The materializer and the cached-path probe must agree on this, or one
    would download a file the other cannot find.
    """
    digest = sha256(remote_id.encode()).hexdigest()
    return cache_root / digest / _safe_remote_name(resource)


def _redact_remote_resource(resource: UPath | str) -> str:
    """Return a minimally identifying label for remote-cache messages."""
    path = coerce_to_upath(resource)
    protocol = getattr(path, "protocol", "remote")
    name = path.name or f"remote-file{path.suffix or ''}"
    if isinstance(protocol, list | tuple):
        protocol = "+".join(str(x) for x in protocol)
    return f"{protocol}://.../{name}"


def _warn_remote_cache_download(resource: UPath, local_path: Path):
    """Warn that DASCore is downloading a remote file into the local cache."""
    scope = get_remote_cache_scope()
    if scope == "metadata":
        guidance = (
            "Set `warn_on_remote_cache=False` to silence this warning or "
            "`allow_remote_cache_for_metadata=False` to disallow metadata-time "
            "remote-file caching."
        )
    else:
        guidance = (
            "Set `warn_on_remote_cache=False` to silence this warning or "
            "`allow_remote_cache=False` to disallow local remote-file caching."
        )
    msg = (
        "Downloading remote file "
        f"{_redact_remote_resource(resource)} to local cache at {local_path}. "
        "This may be slow and consume local disk space. "
        f"{guidance}"
    )
    warnings.warn(msg, UserWarning, stacklevel=4)


def clear_remote_file_cache():
    """
    Remove all locally cached remote files and memoized paths.

    Clearing is a single-writer operation: it is not synchronized against
    materializations running at the same time, so run it while nothing
    else is reading remote files.
    """
    shutil.rmtree(get_remote_cache_path(), ignore_errors=True)
    get_remote_cache_path().mkdir(parents=True, exist_ok=True)
    _REMOTE_RESOURCE_CACHE.clear()
    _materialize_remote_file.cache_clear()


def _download_remote_file(path, local_path: Path):
    """Download a remote path into its cache location."""
    resource = coerce_to_upath(path)
    protocol = getattr(resource, "protocol", None)
    open_kwargs = {"block_size": 0} if protocol in http_protocols else {}
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=local_path.parent,
        prefix=f"{local_path.stem}.",
        suffix=f"{local_path.suffix}.part",
    )
    os.close(fd)
    tmp_path = Path(temp_name)
    try:
        with (
            resource.open("rb", **open_kwargs) as remote_fi,
            tmp_path.open("wb") as local_fi,
        ):
            while chunk := remote_fi.read(get_config().remote_download_block_size):
                local_fi.write(chunk)
        tmp_path.replace(local_path)
    finally:
        tmp_path.unlink(missing_ok=True)


@lru_cache
def _materialize_remote_file(remote_id: str, cache_root: Path) -> Path:
    """
    Materialize one remote resource to a stable local cache path.

    Callers wanting the same resource take turns on its download lock:
    the first downloads, the rest find the published file. Failures are
    not memoized, so the next caller retries the download.
    """
    resource = _REMOTE_RESOURCE_CACHE.get(remote_id)
    if resource is None:
        resource = coerce_to_upath(remote_id.split("#", maxsplit=1)[0])
    local_path = _remote_cache_local_path(remote_id, cache_root, resource)
    # setdefault is atomic, so every caller gets the same lock object.
    with _REMOTE_KEY_LOCKS.setdefault((remote_id, cache_root), RLock()):
        if not local_path.exists():
            _download_to_cache(resource, local_path)
    return local_path


def _download_to_cache(resource: UPath, local_path: Path) -> None:
    """Apply the remote-cache policy, then download the resource."""
    config = get_config()
    scope = get_remote_cache_scope()
    if scope == "metadata" and not config.allow_remote_cache_for_metadata:
        msg = (
            "Remote metadata access requires a local cached file for "
            f"{_redact_remote_resource(resource)}, "
            "but DASCore does not download remote files during "
            "`scan()` or public `get_format()` by default. Set "
            "`allow_remote_cache_for_metadata=True` to opt in to metadata-time "
            "remote caching."
        )
        raise RemoteCacheError(msg)
    if scope != "metadata" and not config.allow_remote_cache:
        msg = (
            f"Remote caching is disabled, but DASCore needs a local file for "
            f"{_redact_remote_resource(resource)}. "
            "Set `allow_remote_cache=True` to permit downloading "
            "remote files into the local cache."
        )
        raise RemoteCacheError(msg)
    if config.warn_on_remote_cache:
        _warn_remote_cache_download(resource, local_path)
    _download_remote_file(resource, local_path)


def ensure_local_file(resource) -> Path:
    """Return a stable local path for one resource for the current session."""
    if is_pathlike(resource) and is_local_path(resource):
        return Path(resource)
    if is_pathlike(resource):
        cache_root = _normalize_cache_root(get_remote_cache_path())
        remote_id = normalize_remote_id(resource)
        return _materialize_remote_file(remote_id, cache_root)
    name = getattr(resource, "name", None)
    if name and is_local_path(name):
        return Path(name)
    msg = f"Cannot ensure a local file for resource {resource!r}"
    raise TypeError(msg)


def _get_cached_local_file(resource) -> Path | None:
    """Return a cached local path without materializing a missing resource."""
    if not is_pathlike(resource) or is_local_path(resource):
        return None
    remote = coerce_to_upath(resource)
    cache_root = _normalize_cache_root(get_remote_cache_path())
    remote_id = normalize_remote_id(remote)
    local_path = _remote_cache_local_path(remote_id, cache_root, remote)
    return local_path if local_path.exists() else None


def get_local_handle(resource, opener):
    """Materialize a resource locally, then pass it to an opener."""
    return opener(ensure_local_file(resource))


def is_no_range_http_error(exc: Exception) -> bool:
    """Return True when an exception indicates no-range HTTP random access."""
    if not isinstance(exc, ValueError):
        return False
    message = str(exc).lower()
    # A server reporting no size streams instead of ranging; seeking such a
    # file needs the same local-file fallback as an outright no-range server.
    if _NO_SIZE_HTTP_PATTERN in message:
        return True
    return all(pattern in message for pattern in _NO_RANGE_HTTP_PATTERNS)


class _FallbackFileObj:
    """
    A seekable binary file adapter that starts remote and falls back to local.

    This private wrapper is used when DASCore wants to give a consumer such as h5py a
    normal file-like object for a remote resource without eagerly downloading
    the whole file first.

    Behavior
    --------
    - Opens the resource with ``remote_opener`` initially.
    - Proxies standard file operations like ``read``, ``readinto``, ``seek``,
      ``tell``, and ``close`` to the active handle.
    - If one proxied operation raises an exception matched by
      ``error_predicate``, the remote handle is abandoned and replaced with a
      local handle from ``local_opener``.
    - The current logical file position is preserved across that switch.
    - Once fallback happens, all later operations use the local handle.

    Why this exists
    ---------------
    Some remote backends work for simple sequential reads but fail when a
    library such as h5py performs the random-access pattern required to read
    HDF5 metadata. A common case is HTTP servers that do not support range
    requests well enough for seek-heavy reads. This wrapper lets DASCore stay
    remote-first when that works, while still recovering by materializing a
    local file only when needed.

    Notes
    -----
    This is not a general retry wrapper for arbitrary IO failures. It is meant
    for one known fallback condition where switching from remote access to a
    local cached file is safe and expected.

    Load-bearing invariant: references flow one way here - h5py objects may
    reference this wrapper and the fsspec handle, but nothing that crosses to
    fsspec's event-loop thread may ever reference an h5py object. Otherwise
    that thread could perform the final decref of an h5py object, whose
    C-level deallocation takes h5py's global lock, and deadlock against a
    caller blocked on the loop while holding that lock (``pause_gc`` only
    stops cyclic collection, not refcount-driven deallocation).
    """

    def __init__(self, remote_opener, local_opener, error_predicate):
        self._remote_opener = remote_opener
        self._local_opener = local_opener
        self._error_predicate = error_predicate
        self._closed = False
        self._using_local = False
        self._handle = self._remote_opener()
        self._pos = 0

    def _set_pos_from_handle(self, fallback=None):
        """Synchronize the tracked position with the wrapped handle."""
        try:
            self._pos = self._handle.tell()
        except Exception:
            if fallback is not None:
                self._pos = fallback

    def _switch_to_local(self):
        """Swap the backing handle to the cached local file."""
        if self._using_local:
            return
        old_handle = self._handle
        old_handle.close()
        # Reopen against the stable local artifact and continue from the same
        # logical file position the caller was already using.
        self._handle = self._local_opener()
        self._handle.seek(self._pos)
        self._using_local = True

    def _with_fallback(self, func, fallback_pos=None):
        """Run an operation and retry against the cached local file if needed."""
        try:
            return func()
        except Exception as exc:
            if not self._error_predicate(exc):
                raise
            # The first matching remote-read failure permanently moves this
            # wrapper onto the local file; later operations stay local.
            self._switch_to_local()
            result = func()
            self._set_pos_from_handle(fallback=fallback_pos)
            return result

    def read(self, size=-1):
        """Read bytes from the file-like object."""
        out = self._with_fallback(lambda: self._handle.read(size))
        self._set_pos_from_handle(
            fallback=self._pos + (len(out) if size != -1 and out is not None else 0)
        )
        return out

    def readinto(self, b):
        """Read bytes into a writable buffer."""
        out = self._with_fallback(lambda: self._handle.readinto(b))
        if out is not None:
            self._set_pos_from_handle(fallback=self._pos + out)
        return out

    def seek(self, offset, whence=0):
        """Move the file cursor, triggering fallback if random access fails."""
        out = self._with_fallback(lambda: self._handle.seek(offset, whence))
        self._set_pos_from_handle(fallback=out)
        return out

    def tell(self):
        """Return the current file position."""
        out = self._handle.tell()
        self._pos = out
        return out

    def close(self):
        """Close the backing handle."""
        if self._closed:
            return
        self._handle.close()
        self._closed = True

    @property
    def closed(self):
        """Return True if the wrapper has been closed."""
        return self._closed

    def seekable(self):
        """Indicate the wrapped handle is seekable."""
        return True

    def readable(self):
        """Indicate the wrapped handle is readable."""
        return True

    def writable(self):
        """Delegate writability if the wrapped handle exposes it."""
        writable = getattr(self._handle, "writable", None)
        if callable(writable):
            return writable()
        return False

    def flush(self):
        """Flush the backing handle."""
        return getattr(self._handle, "flush", lambda: None)()

    def __getattr__(self, item):
        """Delegate unknown attributes to the backing handle."""
        return getattr(self._handle, item)
