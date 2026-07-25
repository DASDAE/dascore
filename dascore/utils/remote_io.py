"""Utilities for session-scoped remote IO caching and fallback."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from threading import RLock, local

from dascore.compat import UPath
from dascore.config import get_config
from dascore.exceptions import RemoteCacheError
from dascore.utils.misc import _reinit_after_fork
from dascore.utils.paths import coerce_to_upath, is_local_path, is_pathlike

_HTTP_PROTOCOLS = {"http", "https"}
_NO_RANGE_HTTP_PATTERNS = (
    "doesn't appear to support range requests",
    "only reading this file from the beginning is supported",
)
_REMOTE_RESOURCE_CACHE: dict[str, UPath] = {}
# Resources already materialized this session, so the common case (the
# file is present) costs one dict lookup rather than a hash, a path
# build and a stat. Emptied by clear_remote_file_cache.
_REMOTE_MATERIALIZED: dict[tuple[str, Path], Path] = {}
# One lock per cached resource, so unrelated downloads still run at the
# same time. Entries are never removed: the dict is bounded by the number
# of distinct remote resources a session touches, and keeping them lets a
# cache clear hold every lock without racing new ones into existence.
_REMOTE_KEY_LOCKS: dict[tuple[str, Path], RLock] = {}
# Guards the two dicts above. Always acquired before a download lock and
# never while holding one.
_REMOTE_CACHE_LOCK = RLock()
# Counts this thread's in-progress materializations (see clear).
_REMOTE_CACHE_LOCAL = local()
_REMOTE_CACHE_SCOPE: ContextVar[str] = ContextVar(
    "remote_cache_scope", default="default"
)


@_reinit_after_fork
def _reinit_remote_cache_locks():
    """Install fresh remote-cache locks; see _reinit_after_fork."""
    global _REMOTE_CACHE_LOCK, _REMOTE_KEY_LOCKS, _REMOTE_CACHE_LOCAL
    _REMOTE_CACHE_LOCK = RLock()
    _REMOTE_KEY_LOCKS = {}
    _REMOTE_CACHE_LOCAL = local()


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
    with _REMOTE_CACHE_LOCK:
        _REMOTE_RESOURCE_CACHE[remote_id] = resource
    return remote_id


def _get_download_lock(key: tuple[str, Path]) -> RLock:
    """Return the download lock for one cached resource, creating it once."""
    with _REMOTE_CACHE_LOCK:
        if (lock := _REMOTE_KEY_LOCKS.get(key)) is None:
            lock = _REMOTE_KEY_LOCKS[key] = RLock()
        return lock


def _get_remote_cache_dir(remote_id: str) -> Path:  # pragma: no cover
    """Return the cache directory for a normalized remote identifier."""
    return get_remote_cache_path() / sha256(remote_id.encode()).hexdigest()


def _normalize_cache_root(cache_root: Path | str) -> Path:
    """Return one normalized cache-root path."""
    return Path(cache_root).expanduser()


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

    Clearing is a single-writer operation: it holds every download lock,
    so materializations already in flight finish before their artifacts
    are removed, and later ones download again.
    """
    if getattr(_REMOTE_CACHE_LOCAL, "materializing", 0):
        msg = "Cannot clear the remote file cache from inside a materialization."
        raise RuntimeError(msg)
    with _REMOTE_CACHE_LOCK, ExitStack() as stack:
        for lock in _REMOTE_KEY_LOCKS.values():
            stack.enter_context(lock)
        shutil.rmtree(get_remote_cache_path(), ignore_errors=True)
        get_remote_cache_path().mkdir(parents=True, exist_ok=True)
        _REMOTE_RESOURCE_CACHE.clear()
        _REMOTE_MATERIALIZED.clear()


def _download_remote_file(path, local_path: Path):
    """Download a remote path into its cache location."""
    resource = coerce_to_upath(path)
    protocol = getattr(resource, "protocol", None)
    open_kwargs = {"block_size": 0} if protocol in _HTTP_PROTOCOLS else {}
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


def _materialize_remote_file(remote_id: str, cache_root: Path) -> Path:
    """
    Materialize one remote resource to a stable local cache path.

    Callers wanting the same resource take turns on its download lock:
    the first downloads, the rest find the published file. A download
    which fails publishes nothing, so the next caller retries it.
    """
    key = (remote_id, cache_root)
    with _REMOTE_CACHE_LOCK:
        # Already materialized: skip hashing, path building and the stat.
        if (published := _REMOTE_MATERIALIZED.get(key)) is not None:
            return published
        resource = _REMOTE_RESOURCE_CACHE.get(remote_id)
    if resource is None:
        resource = coerce_to_upath(remote_id.split("#", maxsplit=1)[0])
    local_path = (
        cache_root
        / sha256(remote_id.encode()).hexdigest()
        / _safe_remote_name(resource)
    )
    # Record the depth so a cache clear attempted from inside a download
    # (eg from a warning handler) raises rather than deleting live state.
    depth = getattr(_REMOTE_CACHE_LOCAL, "materializing", 0)
    _REMOTE_CACHE_LOCAL.materializing = depth + 1
    try:
        with _get_download_lock(key):
            if not local_path.exists():
                _download_to_cache(resource, local_path)
    finally:
        _REMOTE_CACHE_LOCAL.materializing = depth
    # Only published paths are memoized, so a failed download is retried.
    with _REMOTE_CACHE_LOCK:
        _REMOTE_MATERIALIZED[key] = local_path
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
    local_path = (
        cache_root / sha256(remote_id.encode()).hexdigest() / _safe_remote_name(remote)
    )
    return local_path if local_path.exists() else None


def get_local_handle(resource, opener):
    """Materialize a resource locally, then pass it to an opener."""
    return opener(ensure_local_file(resource))


def is_no_range_http_error(exc: Exception) -> bool:
    """Return True when an exception indicates no-range HTTP random access."""
    message = str(exc).lower()
    return isinstance(exc, ValueError) and all(
        pattern in message for pattern in _NO_RANGE_HTTP_PATTERNS
    )


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
