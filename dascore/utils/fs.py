"""
Utilities related to working with file systems.

These include actual file systems or virtual ones.
"""

from __future__ import annotations

import os
import re
from collections.abc import Generator, Iterable
from collections import deque
from pathlib import Path

import fsspec
from typing_extensions import Self, Literal

from dascore.utils.misc import iterate

# Detect if the string has an associated protocol.
_PROTOCOL_DETECTION_REGEX = r"^([a-zA-Z][a-zA-Z0-9+.-]*):\/\/"


def get_fspath(obj):
    """ """
    uri = get_uri(obj)
    fs = fsspec.open(uri)
    return fs


class FSPath:
    """
    A pathlib-like abstraction for handling multiple filesystems.

    This helps smooth out some of the edges of fsspec.

    Parameters
    ----------
    obj
        A

    """

    def __init__(self, obj):
        """ """
        if isinstance(obj, FSPath):
            self.__dict__.update(obj.__dict__)
            return
        elif isinstance(obj, fsspec.core.OpenFile):
            self._fs = obj.fs
            self._path = Path(obj.path)
        elif isinstance(obj, fsspec.spec.AbstractFileSystem):
            self._fs = obj
            self._path = Path("/")
        else:
            fs, path = fsspec.url_to_fs(obj)
            self._fs = fs
            self._path = Path(path)

    @classmethod
    def from_fs_path(cls, fs, path):
        """Create new FSPath from file system and path."""
        out = cls.__new__(cls)
        out._fs = fs
        out._path = path
        return out

    def from_path(self, path):
        """Create a new FSPath from the same file system and a new path."""
        out = self.__class__.__new__(self.__class__)
        out._fs = self._fs
        out._path = path
        return out

    @property
    def path(self) -> Path:
        """Get the pathlib object representing this item."""
        return self._path

    @property
    def is_local(self):
        return self._fs.protocol == ("file", "local")

    @property
    def parent(self) -> Path:
        """Get the pathlib object representing this item."""
        return self.from_fs_path(fs=self._fs, path=self._path.parent)

    @property
    def full_name(self):
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

    def glob(self, arg: str) -> Generator[Self, None, None]:
        """
        Glob search the contents of the file system/directory.
        """
        glob_str = str(self._path / arg)
        for obj in self._fs.glob(glob_str):
            yield self.from_path(obj)


    def get_ls_details(
            self,
            path_str: str | None=None,
            on_error: Literal["ignore", "raise"] | callable="ignore",
            **kwargs
    ):
        """
        Get the details of path contents.
        """
        path_str = str(self._path) if path_str is None else path_str
        listing = None
        try:
            listing = self._fs.ls(path_str, detail=True, **kwargs)
        except (FileNotFoundError, OSError) as e:
            if on_error == "raise":
                raise
            if callable(on_error):
                on_error(e)
        return listing

    def iter_contents(
            self,
            ext: str | None = None,
            timestamp: float | None = None,
            skip_hidden: bool = True,
            include_directories: bool = False,
            maxdepth: None | int = None,
            on_error: Literal["ignore", "raise"] | callable = "omit",
            _dir_deque=None,
            **kwargs,
    ):
        """
        Iterate over the contents of the file system.

        This implements a breadth-first search of the path's contents.

        Parameters
        ----------
        ext
            The extension of the files to include.
        timestamp
            The modified time of the files to include.
        skip_hidden
            Whether to skip hidden (starts with '.') files and directories.
        include_directories
            If True, also yield directory paths.
        maxdepth
            The maximum traversal depth.
        on_error
            The behavior when contents of a directory like thing aren't
            retrievable.
        kwargs
            Passed to filesystem ls call.
        """
        # A queue of directories to transverse through.
        _dir_deque = _dir_deque if _dir_deque is not None else deque()
        path_str = str(self._path)
        listing = self.get_ls_details(path_str, on_error, **kwargs)
        for info in iterate(listing):
            # Don't include self in the ls.
            if info['name'] == path_str:
                continue
            pathname = info["name"].rstrip("/")
            name = pathname.rsplit("/", 1)[-1]
            is_dir = info['type'] == "directory"
            mtime = info['mtime']
            good_ext = ext is None or name.endswith(ext)
            good_mtime = timestamp is None or mtime >= timestamp
            good_hidden = not skip_hidden or not name.startswith(".")
            # Handle files.
            if not is_dir and good_ext and good_hidden and good_mtime:
                yield self.from_path(info["name"])
            elif good_hidden:
                dirpath = self.from_path(info["name"])
                # If we are to also yield directories
                if include_directories:
                    signal = yield dirpath
                    # Here we bail out on this directory/contents.
                    if signal is not None and signal == "skip":
                        continue
                # Add the directory to the queue to be traversed.
                _dir_deque.append(dirpath)
        # Handle the directories that need to be traversed.
        while _dir_deque:
            next_fspath = _dir_deque.popleft()
            if maxdepth is not None and maxdepth <= 1:
                continue
            new_iter = next_fspath.iter_contents(
                ext=ext,
                timestamp=timestamp,
                skip_hidden=skip_hidden,
                include_directories=include_directories,
                maxdepth=maxdepth - 1 if maxdepth is not None else None,
                on_error=on_error,
                _dir_deque=_dir_deque,
            )
            yield from new_iter


    def __truediv__(self, other: str) -> Self:
        """Enables division to add to string to Path."""
        return self.from_fs_path(fs=self._fs, path=self._path / other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.full_name})"

    def __hash__(self) -> int:
        return hash(self.full_name)

    def __eq__(self, other: Self) -> bool:
        return self.full_name == other.full_name


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

