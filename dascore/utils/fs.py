"""
Utilities related to working with file systems.

These include actual file systems or virtual ones.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Literal

from upath import UPath

# Detect if the string has an associated protocol.


def get_ls_details(
    upath: UPath, on_error: Literal["ignore", "raise"] | callable = "ignore", **kwargs
):
    """
    Get the details of path contents.
    """
    listing = None
    fs = upath.fs
    try:
        listing = fs.ls(upath.path, detail=True, **kwargs)
    except (FileNotFoundError, OSError) as e:
        if on_error == "raise":
            raise
        if callable(on_error):
            on_error(e)
    return listing


def iter_path_contents(
    upath: str | Path | UPath,
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
    upath = UPath(upath)
    _dir_deque = _dir_deque if _dir_deque is not None else deque()
    path_str = str(upath.path)
    for info in get_ls_details(upath, on_error=on_error, **kwargs):
        # Don't include self in the ls.
        if info["name"] == path_str:
            continue
        pathname = info["name"].rstrip("/")
        name = pathname.rsplit("/", 1)[-1]
        is_dir = info["type"] == "directory"
        mtime = info.get("mtime", 0)
        good_ext = ext is None or name.endswith(ext)
        good_mtime = timestamp is None or mtime >= timestamp
        good_hidden = not skip_hidden or not name.startswith(".")
        # Handle files.
        if not is_dir and good_ext and good_hidden and good_mtime:
            yield upath / info["name"]
        elif good_hidden:
            dirpath = upath / info["name"]
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
        next_upath_directory = _dir_deque.popleft()
        if maxdepth is not None and maxdepth <= 1:
            continue
        new_iter = iter_path_contents(
            next_upath_directory,
            ext=ext,
            timestamp=timestamp,
            skip_hidden=skip_hidden,
            include_directories=include_directories,
            maxdepth=maxdepth - 1 if maxdepth is not None else None,
            on_error=on_error,
            _dir_deque=_dir_deque,
        )
        yield from new_iter


def get_fspath(obj):
    """ """
    uri = get_path(obj)
    fs = fsspec.open(uri)
    return fs


def get_path(obj) -> str:
    """
    Get the path of an object representing a file.

    Parameters
    ----------
    obj
        An object that represents a path to a resource.
    """
    attrs = ("filename", "name", "path", "full_name")
    for attr in attrs:
        if (out := getattr(obj, attr, None)) is not None:
            return out
    if isinstance(obj, str):
        return UPath(obj).absolute().path
    elif isinstance(obj, Path):
        obj = obj.absolute()
    return obj
