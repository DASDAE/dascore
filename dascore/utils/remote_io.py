"""Utilities for session-scoped remote IO caching and fallback."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dascore.compat import UPath
from dascore.config import get_config
from dascore.exceptions import RemoteCacheError
from dascore.utils.paths import coerce_to_upath, is_local_path, is_pathlike

_HTTP_PROTOCOLS = {"http", "https"}
_NO_RANGE_HTTP_PATTERNS = (
    "doesn't appear to support range requests",
    "only reading this file from the beginning is supported",
)
_REMOTE_RESOURCE_CACHE: dict[str, UPath] = {}
_REMOTE_CACHE_SCOPE: ContextVar[str] = ContextVar(
    "remote_cache_scope", default="default"
)


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
    """Remove all locally cached remote files and memoized paths."""
    shutil.rmtree(get_remote_cache_path(), ignore_errors=True)
    get_remote_cache_path().mkdir(parents=True, exist_ok=True)
    _materialize_remote_file.cache_clear()
    _REMOTE_RESOURCE_CACHE.clear()


def _download_remote_file(path, local_path: Path):
    """Download a remote path into its cache location."""
    resource = coerce_to_upath(path)
    protocol = getattr(resource, "protocol", None)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=local_path.parent,
        prefix=f"{local_path.stem}.",
        suffix=f"{local_path.suffix}.part",
    )
    os.close(fd)
    tmp_path = Path(temp_name)
    try:
        if protocol in _HTTP_PROTOCOLS:
            # Use a direct blocking HTTP download here rather than
            # ``resource.open(...)``. The fallback path can be entered while an
            # active fsspec HTTP read is already in progress, and re-entering
            # that stack from inside the fallback can deadlock.
            headers = dict(getattr(resource, "storage_options", {}) or {})
            # Re-validate the actual URL string to ensure its scheme is safe
            url_string = str(resource)
            parsed_url = urlparse(url_string)
            if parsed_url.scheme not in _HTTP_PROTOCOLS:
                msg = (
                    f"URL scheme '{parsed_url.scheme}' is not allowed. "
                    f"Only {_HTTP_PROTOCOLS} are permitted."
                )
                raise ValueError(msg)
            request = Request(url_string, headers=headers)
            timeout = get_config().remote_download_timeout
            with (
                urlopen(request, timeout=timeout) as remote_fi,
                tmp_path.open("wb") as local_fi,
            ):
                while chunk := remote_fi.read(get_config().remote_download_block_size):
                    local_fi.write(chunk)
        else:
            with resource.open("rb") as remote_fi, tmp_path.open("wb") as local_fi:
                while chunk := remote_fi.read(get_config().remote_download_block_size):
                    local_fi.write(chunk)
        tmp_path.replace(local_path)
    finally:
        tmp_path.unlink(missing_ok=True)


@lru_cache
def _materialize_remote_file(  # pragma: no cover
    remote_id: str, cache_root: Path
) -> Path:
    """Materialize one remote resource to a stable local cache path."""
    resource = _REMOTE_RESOURCE_CACHE.get(remote_id)
    if resource is None:
        resource = coerce_to_upath(remote_id.split("#", maxsplit=1)[0])
    local_path = (
        cache_root
        / sha256(remote_id.encode()).hexdigest()
        / _safe_remote_name(resource)
    )
    if not local_path.exists():
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
    return local_path


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
        # Reopen against the stable local artifact and continue from the same
        # logical file position the caller was already using.
        self._handle = self._local_opener()
        self._handle.seek(self._pos)
        self._using_local = True
        old_handle.close()

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