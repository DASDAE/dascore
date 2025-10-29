"""Utilities for basic IO tasks."""

from __future__ import annotations

import io
import os
import tempfile
import typing
from collections.abc import Generator, Iterable
from functools import cache
from inspect import isfunction, ismethod
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np

import dascore as dc
from dascore.compat import UPath
from dascore.constants import PatchType
from dascore.exceptions import InvalidFileHandlerError, PatchConversionError
from dascore.utils.misc import (
    cached_method,
    optional_import,
)
from dascore.utils.time import to_float

HANDLE_FUNCTIONS = {
    str: lambda x: str(x),
    Path: lambda x: UPath(x),
    UPath: lambda x: UPath(x) if not isinstance(x, UPath) else x,
}


RequiredType = typing.TypeVar("RequiredType")


def _is_remote_path(path):
    """Check if a path is a remote path (not local filesystem)."""
    # Check if it has a protocol that's not file or local
    protocol = getattr(path, "protocol", None)
    return protocol not in (None, "file", "")


def download_remote_to_temp(path):
    """
    Download a remote file to a temporary location.

    This is needed for libraries like pytables that require local file paths.

    Parameters
    ----------
    path : str | Path | UPath
        The remote path to download

    Returns
    -------
    Path
        Path to the downloaded temporary file
    """
    path_obj = UPath(path) if not isinstance(path, UPath) else path
    # Create temp file with same extension
    suffix = path_obj.suffix or ".tmp"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        # Download the file
        with path_obj.open("rb") as remote_file:
            with open(temp_path, "wb") as local_file:
                # Copy in chunks for memory efficiency
                chunk_size = 8192
                while chunk := remote_file.read(chunk_size):
                    local_file.write(chunk)
    finally:
        os.close(fd)

    return UPath(temp_path)


class BinaryReader(io.BytesIO):
    """Base file for binary things."""

    mode = "rb"
    reset_offset = True

    @classmethod
    def get_handle(cls, resource):
        """Get the handle object from various sources."""
        if isinstance(resource, cls | io.BufferedIOBase):
            if cls.reset_offset:
                resource.seek(0)  # reset byte offset
            return resource
        try:
            _maybe_make_parent_directory(resource)
            return open(resource, mode=cls.mode)
        except TypeError:
            msg = f"Couldn't get handle from {resource} using {cls}"
            raise InvalidFileHandlerError(msg)


class BinaryWriter(BinaryReader):
    """Dummy class for streams which write binary."""

    mode = "ab"
    reset_offset = False


@cache
def _get_required_type(required_type, arg_name=None):
    """Get the type hint for the first argument."""
    if required_type not in HANDLE_FUNCTIONS:
        # here we try to get the type from the function type hints
        # but we need to skip things that aren't functions
        is_func_y = isfunction(required_type) or ismethod(required_type)
        if not is_func_y or not (hints := get_type_hints(required_type)):
            return required_type
        arg_name = arg_name if arg_name is not None else next(iter(hints))
        required_type = hints.get(arg_name)
    return required_type


def get_handle_from_resource(uri, required_type):
    """
    Get a handle for a file of preferred type.

    return uri if required type is not specified or supported in either
    handle function or has a `get_handle` method.
    """
    if hasattr(required_type, "get_handle"):
        uri = required_type.get_handle(uri)
    elif required_type in HANDLE_FUNCTIONS:
        uri = HANDLE_FUNCTIONS[required_type](uri)
    return uri


class IOResourceManager:
    """A class for managing opening/closing files."""

    def __init__(self, source: Any):
        self._source = source
        self._cache = {}

    @property
    @cached_method
    def source(self):
        """Get the source of the IO manager."""
        source = self._source
        # this handles IO managers derived from other IO managers;
        # effectively, we need to go back to the original, non-io manager source
        while isinstance(source, self.__class__):
            source = source.source
        return source

    def get_resource(self, required_type: RequiredType) -> RequiredType:
        """Get the requested resource."""
        # no required type, just return source of manager.
        if required_type is None:
            return self.source
        # this is so the context managers can be nested and the child
        # context manager only calls to the parent. Then, the resources
        # get closed only after the original exists its context.
        if isinstance(self._source, self.__class__):
            return self._source.get_resource(required_type)
        required_type = _get_required_type(required_type)
        if required_type not in self._cache:
            out = get_handle_from_resource(self._source, required_type)
            self._cache[required_type] = out
        return self._cache[required_type]

    def close_all(self):
        """Close any open file handles."""
        for handle in self._cache.values():
            getattr(handle, "close", lambda: None)()

    def __enter__(self):
        """Entering context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Simply ensure all file handles are closed."""
        self.close_all()

    def __del__(self):
        self.close_all()


def patch_to_xarray(patch: PatchType):
    """Return a data array with patch contents."""
    xr = optional_import("xarray")
    attrs = dict(patch.attrs)
    patch_dims = patch.dims
    coords = {}
    for name, coord in patch.coords.coord_map.items():
        if coord._partial:
            continue
        dims = patch.coords.dim_map[name]
        coords[name] = (dims, coord.values)
    # Need to exclude non-coords
    return xr.DataArray(patch.data, attrs=attrs, dims=patch_dims, coords=coords)


def xarray_to_patch(data_array) -> PatchType:
    """Convert an xarray dataarray to a patch."""
    # this cant work if xarray isn't installed. This ensures it is.
    _ = optional_import("xarray")

    params = dict(
        coords={i: (x.dims, x.values) for i, x in data_array.coords.items()},
        attrs=dict(data_array.attrs.items()),
        dims=data_array.dims,
        data=data_array.data,
    )
    return dc.Patch(**params)


def patch_to_obspy(patch: PatchType):
    """
    Convert a patch to an ObsPy Stream.

    The patch must have a dimension named time.

    Parameters
    ----------
    patch
        The input patch object.
    """
    obspy = optional_import("obspy")

    def _check_patch(patch):
        """Ensure the patch can be converted to a stream else raise."""
        is_2d = len(patch.dims) == 2
        has_time = "time" in patch.dims
        if not has_time and is_2d:
            msg = "Can only convert 2d patches with a time dimension to stream."
            raise PatchConversionError(msg)

    def _get_time_stats(patch):
        """Get stats dict with time values."""
        coord = patch.get_coord("time")
        tmin = dc.to_datetime64(coord.min())
        tmax = dc.to_datetime64(coord.max())
        dt = np.timedelta64(1, "s") / coord.step

        time_stats = {
            "starttime": obspy.UTCDateTime(str(tmin)),
            "endtime": obspy.UTCDateTime(str(tmax)),
            "sampling_rate": to_float(dt),
        }
        return time_stats

    _check_patch(patch)
    # ensure time is last axis
    patch = patch.transpose(..., "time")
    other_dim = next(iter(set(patch.dims) - {"time"}))
    other_vals = patch.coords.get_array(other_dim)
    base_stats = _get_time_stats(patch)

    traces = []
    for data, other_val in zip(patch.data, other_vals):
        stats = patch.attrs.model_dump()
        stats.pop("coords")
        stats.update(base_stats)
        stats[other_dim] = other_val
        trace = obspy.Trace(data=data, header=stats)
        traces.append(trace)
    return obspy.Stream(traces)


def obspy_to_patch(stream, dim="distance") -> PatchType:
    """
    Convert an obspy stream to a patch.

    Each trace must have some common value in its stats dict which can be used
    to create a new dimension. Also, each trace must have the same data length.

    Parameters
    ----------
    stream
        The input ObsPy Stream object.
    dim
        The new dimension whose data is contained in the stats dict.
    """

    def _check_stream(stream):
        """Run simple checks on stream."""
        equal_len_data = len({len(x.data) for x in stream}) == 1
        has_dim = all([dim in x.stats for x in stream])
        if not (equal_len_data and has_dim):
            msg = (
                "Cannot convert stream without specified value in all stats "
                f"dicts {dim} or with traces of different lengths to a Patch"
            )
            raise PatchConversionError(msg)

    def _get_attrs(tr):
        """Get stats from one of the traces."""
        # these are mainly obspy-specific things.
        to_remove = {"starttime", "endtime", "sampling_rate", "delta", "npts", "calib"}
        attrs = {i: v for i, v in tr.stats.items() if i not in to_remove}
        return attrs

    if not len(stream):
        return dc.Patch()

    _check_stream(stream)
    data = []
    new_dim = []
    for tr in stream:
        data.append(tr.data)
        new_dim.append(tr.stats[dim])

    dims = (dim, "time")
    coords = {
        dim: ((dim,), np.asarray(new_dim)),
        "time": (("time",), dc.to_datetime64(tr.times("timestamp"))),
    }
    attrs = _get_attrs(tr)
    patch = dc.Patch(
        data=np.stack(data),
        dims=dims,
        attrs=attrs,
        coords=coords,
    )
    return patch


def _maybe_make_parent_directory(path):
    """Maybe make parent directories."""
    path = UPath(path) if not isinstance(path, UPath) else path
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    return


def _iter_filesystem(
    paths: str | Path | UPath | Iterable[str | Path | UPath],
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
    # Convert to UPath for universal handling
    try:
        path = UPath(paths) if not isinstance(paths, UPath) else paths
    except (TypeError, ValueError):
        # Multiple paths were passed
        for p in paths:
            yield from _iter_filesystem(
                p, ext, timestamp, skip_hidden, include_directories
            )
        return
    # handle returning directories if requested.
    if include_directories and path.is_dir():
        if not (skip_hidden and path.name.startswith(".")):
            signal = yield str(path)
            if signal is not None and signal == "skip":
                yield None
                return
    try:
        if path.is_dir():
            # Iterate directory contents
            for entry in path.iterdir():
                if entry.is_file() and (ext is None or entry.name.endswith(ext)):
                    if timestamp is None or entry.stat().st_mtime >= timestamp:
                        if not skip_hidden or not entry.name.startswith("."):
                            yield str(entry)
                elif entry.is_dir() and not (
                    skip_hidden and entry.name.startswith(".")
                ):
                    yield from _iter_filesystem(
                        entry,
                        ext=ext,
                        timestamp=timestamp,
                        skip_hidden=skip_hidden,
                        include_directories=include_directories,
                    )
        elif path.is_file():
            # A single file was passed, just return it
            yield str(path)
    except (OSError, PermissionError):
        # Can't read directory or file doesn't exist, skip it
        pass
