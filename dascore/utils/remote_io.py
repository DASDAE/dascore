"""Utilities for session-scoped remote IO caching and fallback."""

from __future__ import annotations

import shutil
import tempfile
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from dascore.compat import UPath
from dascore.utils.paths import coerce_path, is_local_path, is_pathlike

_HTTP_PROTOCOLS = {"http", "https"}
_NO_RANGE_HTTP_PATTERNS = (
    "doesn't appear to support range requests",
    "only reading this file from the beginning is supported",
)


def get_remote_cache_path() -> Path:
    """Return the on-disk directory used for remote file caching."""
    path = Path(tempfile.gettempdir()) / "dascore-remote-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_remote_name(path: UPath) -> str:
    """Return a basename suitable for a cached local copy."""
    name = path.name or f"remote-file{path.suffix or '.tmp'}"
    return Path(name).name


def normalize_remote_id(path) -> str:
    """Return a stable identifier for a remote path."""
    return str(coerce_path(path))


def _get_remote_cache_dir(remote_id: str) -> Path:
    """Return the cache directory for a normalized remote identifier."""
    return get_remote_cache_path() / sha256(remote_id.encode()).hexdigest()


def clear_remote_file_cache():
    """Remove all locally cached remote files and memoized paths."""
    shutil.rmtree(get_remote_cache_path(), ignore_errors=True)
    get_remote_cache_path().mkdir(parents=True, exist_ok=True)
    _materialize_remote_file.cache_clear()


def _download_remote_file(path, local_path: Path):
    """Download a remote path into its cache location."""
    resource = coerce_path(path)
    protocol = getattr(resource, "protocol", None)
    open_kwargs = {"block_size": 0} if protocol in _HTTP_PROTOCOLS else {}
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(f"{local_path.suffix}.part")
    try:
        with resource.open("rb", **open_kwargs) as remote_fi, tmp_path.open(
            "wb"
        ) as local_fi:
            while chunk := remote_fi.read(8192):
                local_fi.write(chunk)
        tmp_path.replace(local_path)
    finally:
        tmp_path.unlink(missing_ok=True)


@lru_cache
def _materialize_remote_file(remote_id: str) -> Path:
    """Materialize one remote resource to a stable local cache path."""
    resource = coerce_path(remote_id)
    local_path = _get_remote_cache_dir(remote_id) / _safe_remote_name(resource)
    if not local_path.exists():
        _download_remote_file(resource, local_path)
    return local_path


def ensure_local_file(resource) -> Path:
    """Return a stable local path for one resource for the current session."""
    if is_pathlike(resource) and is_local_path(resource):
        return Path(resource)
    if is_pathlike(resource):
        return _materialize_remote_file(normalize_remote_id(resource))
    name = getattr(resource, "name", None)
    if name and is_local_path(name):
        return Path(name)
    msg = f"Cannot ensure a local file for resource {resource!r}"
    raise TypeError(msg)


def get_local_handle(resource, opener):
    """Materialize a resource locally, then pass it to an opener."""
    return opener(ensure_local_file(resource))


def is_no_range_http_error(exc: Exception) -> bool:
    """Return True when an exception indicates no-range HTTP random access."""
    message = str(exc).lower()
    return isinstance(exc, ValueError) and all(
        pattern in message for pattern in _NO_RANGE_HTTP_PATTERNS
    )


class FallbackFileObj:
    """A file-like object that switches from remote to local on one error."""

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
        """Move the file cursor."""
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
