"""Utilities for classifying DASCore path inputs."""

from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.parse import unquote

from typing_extensions import TypeIs

from dascore.compat import UPath
from dascore.exceptions import InvalidSpoolError

# Hive's sentinel for a NULL partition value; means "no value", so the
# key is treated as absent rather than set to the sentinel string.
_HIVE_NULL = "__HIVE_DEFAULT_PARTITION__"
# Synthetic URI schemes for in-memory patch identities (see
# dascore.io.index.catalog); such paths dispatch to in-memory registries
# and are never treated as file names.
_MEMORY_SCHEMES = ("memorypatch://", "memory://")
# Synthetic URI scheme naming a file in the example data registry
# (dascore/data_registry.txt); resolved to a local path by
# dascore.utils.downloader.resolve_example_uri.
EXAMPLE_SCHEME = "examples://"


def is_pathlike(resource) -> TypeIs[str | Path | UPath]:
    """Return True if resource is supported path-like input."""
    return isinstance(resource, str | Path | UPath)


def quote_path(path: Path) -> str:
    """
    Name a file for an error message.

    Its container is included because a bare name is ambiguous across
    containers and the full path is noise the reader already knows.

    Examples
    --------
    >>> from pathlib import Path
    >>> from dascore.utils.paths import quote_path
    >>> Path(quote_path(Path("inventory/path@2020-01-01/coupling.csv"))).parts
    ('path@2020-01-01', 'coupling.csv')
    """
    return str(Path(path.parent.name) / path.name)


def is_memory_uri(path) -> bool:
    """
    Return True if a path is a synthetic in-memory patch identity.

    Matches the exact ``memorypatch://`` / ``memory://`` schemes rather
    than any string beginning with "memory", so a real file or directory
    named e.g. ``memory_notes.h5`` is not misclassified.
    """
    return str(path).startswith(_MEMORY_SCHEMES)


def is_example_uri(path) -> bool:
    """
    Return True if a path names a file in the example data registry.

    Matches the exact ``examples://`` scheme, so a real file named e.g.
    ``examples_notes.h5`` is not misclassified.

    Examples
    --------
    >>> from dascore.utils.paths import is_example_uri
    >>> is_example_uri("examples://terra15_das_1_trimmed.hdf5")
    True
    >>> is_example_uri("examples_notes.h5")
    False
    """
    return str(path).startswith(EXAMPLE_SCHEME)


def directory_writable(path) -> bool:
    """Return True if the directory is writable else False."""
    directory = Path(path)
    try:
        directory.mkdir(exist_ok=True, parents=True)
        with tempfile.NamedTemporaryFile(prefix="._dascore_write_test_", dir=directory):
            pass
    except OSError:
        return False
    return True


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


def coerce_to_local_path(resource) -> Path:
    """
    Return a plain `Path` for a local resource, stripping any URI scheme.

    ``Path("file:///tmp/x")`` keeps the scheme as a literal path segment, so
    local inputs carrying a ``file://`` (or ``local://``) scheme must be routed
    through ``UPath`` first to recover the real filesystem path. Plain paths
    take a fast path and skip that round-trip.
    """
    if isinstance(resource, Path):
        return resource
    if isinstance(resource, str) and "://" not in resource:
        return Path(resource)
    return Path(coerce_to_upath(resource).path)


def parse_hive_path_attrs(rel_posix: str) -> dict[str, str]:
    """
    Parse hive-style ``key=value`` pairs from a relative path.

    Only the directories containing the source participate, as in Hive
    partitioning; the source's own name — the last segment, whether that
    is a file or a directory-format unit — is opaque. A segment holds one
    pair (``acquisition_key=XX.R2D1..RAW/``) or several separated by
    ``__``, the same separator DASCore's default patch names use, as in
    ``cable=n__tag=raw/``.
    Keys and values are percent-decoded after splitting, so encoded
    ``=`` (``%3D``), ``__`` (``%5F%5F``), and ``.`` survive inside either
    part. When a key repeats, the deepest (then rightmost) one wins.
    Pairs with an empty key or value, or with Hive's NULL sentinel as
    the value, are skipped, as are segments/parts without ``=``.

    Parameters
    ----------
    rel_posix
        A POSIX-style path relative to the spool root (as stored in the
        index), e.g. ``"acquisition_key=XX.R2D1..RAW/file.h5"``.

    Examples
    --------
    >>> from dascore.utils.paths import parse_hive_path_attrs
    >>> parse_hive_path_attrs("acquisition_key=XX.R2D1..RAW/cable=n/file.h5")
    {'acquisition_key': 'XX.R2D1..RAW', 'cable': 'n'}
    """
    out: dict[str, str] = {}
    # The last segment names the source itself, so it is dropped: a file
    # name cannot be told from a partition value once an extension is in
    # play ("acquisition_key=XX.R2D1..RAW" ends in a plausible one).
    for segment in rel_posix.split("/")[:-1]:
        for part in segment.split("__"):
            key, sep, value = part.partition("=")
            if not sep:
                continue
            key, value = unquote(key), unquote(value)
            if not key or not value or value == _HIVE_NULL:
                continue
            out[key] = value
    return out


def requires_local_directory(resource, *, label: str):
    """Raise when directory operations are requested on non-local filesystems."""
    if is_pathlike(resource) and not is_local_path(resource):
        msg = f"{label} only supports local filesystem paths."
        raise InvalidSpoolError(msg)
