"""
Utilities related to working with file systems.

These include actual file systems or virtual ones.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Generator
from typing_extensions import Self

import fsspec


# Detect if the string has an associated protocol.
_PROTOCOL_DETECTION_REGEX = r"^([a-zA-Z][a-zA-Z0-9+.-]*):\/\/"



def get_fspath(obj):
    """

    """
    uri = get_uri(obj)
    fs = fsspec.open(uri)
    return fs


class FSPath:
    """
    A class that behaves like a pathlib.Path object.

    This helps smooth out some of the edges of fsspec.
    """
    fs: fsspec.AbstractFileSystem

    def __init__(self, obj):
        """

        """
        if isinstance(obj, FSPath):
            self.__dict__.update(obj.__dict__)
            return
        if isinstance(obj, fsspec.core.OpenFile):
            self._fs = obj.fs
            self._path = Path(obj.path)
        else:
            fs, path = fsspec.url_to_fs(obj)
            self._fs = fs
            self._path = Path(path)

    @classmethod
    def from_fs_path(cls, fs, path):
        out = cls.__new__(cls)
        out._fs = fs
        out._path = path
        return out

    @property
    def path(self) -> Path:
        """Get the pathlib object representing this item."""
        return self._path

    @property
    def parent(self) -> Path:
        """Get the pathlib object representing this item."""
        return self.from_fs_path(fs=self._fs, path=self._path.parent)

    def _full_name(self):
        """
        Return the full name.

        Ideally, this is a string that can be used to recreate the
        filesystem and path.
        """
        name = self._fs.unstrip_protocol(self._path)
        return name

    def exists(self):
        """Determine if the file exists."""
        return self._fs.exists(self._path)

    # --- Dunders

    def __truediv__(self, other: str) -> Self:
        """Enables division to add to string to Path."""
        return self.from_fs_path(fs=self._fs, path=self._path / other)


    def __repr__(self) -> str:
        return self._full_name()





def get_uri(obj) -> str:
    """
    Get the uri string of an object representing a file.

    Parameters
    ----------
    obj
        An object that represents a path to a resource.
    """
    if isinstance(obj, str):
        # Assume the string rep a local file.
        if not re.match(_PROTOCOL_DETECTION_REGEX, obj):
            obj = f"file://{obj}"
    elif hasattr(obj, "filename"):
        obj = f"file://{obj.filename}"
    elif isinstance(obj, Path):
        obj = f"file://{obj.absolute()}"
    elif hasattr(obj, "name"):
        obj = f"file://{obj.name}"
    elif isinstance(obj, fsspec.core.OpenFiles):
        obj = get_fspath(obj)
    if hasattr(obj, "full_name"):
        obj = obj.full_name
    return obj


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
