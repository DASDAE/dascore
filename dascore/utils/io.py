"""Utilities for basic IO tasks."""

from __future__ import annotations

import io
import typing
from contextlib import suppress
from functools import cache
from inspect import isfunction, ismethod
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np

import dascore as dc
from dascore.compat import UPath
from dascore.constants import PatchType
from dascore.exceptions import PatchConversionError
from dascore.utils.misc import (
    _maybe_make_parent_directory,
    cached_method,
    iterate,
    optional_import,
)
from dascore.utils.paths import coerce_to_upath, is_local_path, is_pathlike
from dascore.utils.remote_io import ensure_local_file as _ensure_local_file
from dascore.utils.remote_io import get_local_handle
from dascore.utils.time import to_float

HANDLE_FUNCTIONS = {
    Path: lambda x: Path(x),
    UPath: lambda x: coerce_to_upath(x),
}


RequiredType = typing.TypeVar("RequiredType")


def ensure_local_file(resource) -> Path:
    """Return a stable local path for one resource for the current session."""
    if isinstance(resource, IOResourceManager):
        resource = resource.source
    return _ensure_local_file(resource)


def _normalize_resource_identity(resource):
    """Normalize one pathlike input to a local Path or remote UPath."""
    if is_local_path(resource):
        return Path(resource)
    return coerce_to_upath(resource)


def _resolve_resource(resource, required_type):
    """Resolve resource to a form suitable for required_type."""
    # already have a resource thing of some kind; just pass through.
    if not is_pathlike(resource):
        return resource
    # Otherwise get Upath or Path, if Path ensure it is downloaded.
    resource = _normalize_resource_identity(resource)
    if isinstance(resource, Path):
        return resource
    if required_type is Path:
        return ensure_local_file(resource)
    return resource


def _annotate_handle_path(handle, resource):  # pragma: no cover
    """Attach lightweight source-path metadata to a remote handle when absent."""
    path_str = str(resource)
    # This is intentionally a small compatibility hack for readers that still
    # inspect handle.name/path metadata. Some remote text handles come back as
    # TextIOWrapper(name=None), and that name attribute may not be writable, so
    # we keep a private fallback for downstream format sniffers.
    for attr_name in ("_dascore_source_path",):
        with suppress(AttributeError, TypeError):
            setattr(handle, attr_name, path_str)
    if getattr(handle, "name", None) in (None, ""):
        with suppress(AttributeError, TypeError):
            setattr(handle, "name", path_str)
    return handle


def _normalize_source_patch_ids(source_patch_id) -> set[str]:
    """Coerce source patch identifiers into a deduplicated set of strings."""
    return {str(value) for value in iterate(source_patch_id) if value not in (None, "")}


class BinaryReader(io.BytesIO):
    """Base file for reading binary files."""

    mode = "rb"
    reset_offset = True

    @classmethod
    def get_handle(cls, resource):
        """Get the handle object from various sources."""
        if isinstance(resource, cls | io.BufferedIOBase):
            if cls.reset_offset:
                resource.seek(0)  # reset byte offset
            return resource
        if isinstance(resource, UPath):
            return _annotate_handle_path(resource.open(cls.mode), resource)
        try:
            _maybe_make_parent_directory(resource)
            return open(resource, mode=cls.mode)
        except TypeError:
            msg = f"Couldn't get handle from {resource} using {cls}"
            raise NotImplementedError(msg)


class LocalBinaryReader(BinaryReader):
    """A binary reader which first materializes remote resources locally."""

    @classmethod
    def get_handle(cls, resource):
        """Get the binary handle, materializing remote resources if needed."""
        if isinstance(resource, cls | io.BufferedIOBase):
            if cls.reset_offset:
                resource.seek(0)
            return resource
        return get_local_handle(resource, super().get_handle)


class BinaryWriter(BinaryReader):
    """Dummy class for streams which write binary."""

    mode = "wb"
    reset_offset = False


class TextReader(BinaryReader):
    """Base class for reading text files."""

    mode = "r"

    @classmethod
    def get_handle(cls, resource):
        """Get a text handle from a resource."""
        if isinstance(resource, cls | io.TextIOBase):
            if cls.reset_offset:
                resource.seek(0)
            return resource
        if isinstance(resource, UPath):
            return _annotate_handle_path(
                resource.open(cls.mode, encoding="utf-8"), resource
            )
        try:
            _maybe_make_parent_directory(resource)
            return open(resource, mode=cls.mode, encoding="utf-8")
        except TypeError:
            msg = f"Couldn't get handle from {resource} using {cls}"
            raise NotImplementedError(msg)


class TextWriter(BinaryWriter):
    """Base class for writing text files."""

    mode = "w"


class LocalPath:
    """A local path adapter for callsites that require a concrete filename."""

    @classmethod
    def get_handle(cls, resource):
        """Return a local path for the supplied resource."""
        return get_local_handle(resource, Path)


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
        return hints.get(arg_name)
    return required_type


def get_handle_from_resource(uri, required_type):
    """
    Get a handle for a file of preferred type.

    Return uri unchanged if required type is not specified or supported in
    either handle functions or has no `get_handle` method.
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
            source = _resolve_resource(self._source, required_type)
            out = get_handle_from_resource(source, required_type)
            self._cache[required_type] = out
        return self._cache[required_type]

    def close_all(self):
        """Close any open file handles."""
        for handle in self._cache.values():
            getattr(handle, "close", lambda: None)()

    def clear_cache(self):
        """Close and forget any cached resources so they can be reopened fresh."""
        self.close_all()
        self._cache.clear()

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
    # Omit None-valued attrs because xarray backends may reject them during
    # NetCDF serialization, while a missing attr round-trips cleanly.
    attrs = {
        key: value for key, value in dict(patch.attrs).items() if value is not None
    }
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
