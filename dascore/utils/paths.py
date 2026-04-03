"""Utilities for classifying DASCore path inputs."""

from __future__ import annotations

from pathlib import Path

from dascore.compat import UPath
from dascore.exceptions import InvalidSpoolError


def is_pathlike(resource) -> bool:
    """Return True if resource is supported path-like input."""
    return isinstance(resource, str | Path | UPath)


def coerce_to_upath(resource) -> UPath:
    """Return a UPath for path-like resources."""
    return resource if isinstance(resource, UPath) else UPath(resource)


def get_path_protocol(resource) -> str | None:
    """Return the normalized protocol for path-like resources."""
    if isinstance(resource, Path):
        return "file"
    if isinstance(resource, UPath):
        protocol = getattr(resource, "protocol", "file")
        return protocol or "file"
    if isinstance(resource, str):
        return "file" if "://" not in resource else coerce_to_upath(resource).protocol
    return None


def is_local_path(resource) -> bool:
    """Return True if resource refers to the local filesystem."""
    if not is_pathlike(resource):
        return False
    return get_path_protocol(resource) in {"", "file", "local"}


def requires_local_directory(resource, *, label: str):
    """Raise when directory operations are requested on non-local filesystems."""
    if is_pathlike(resource) and not is_local_path(resource):
        msg = f"{label} only supports local filesystem paths."
        raise InvalidSpoolError(msg)
