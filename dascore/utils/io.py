"""Utilities for basic IO tasks."""

from __future__ import annotations

import io
import typing
from contextlib import suppress
from functools import cache
from inspect import isfunction, ismethod
from pathlib import Path
from threading import RLock
from typing import Any, Literal, get_type_hints

import numpy as np
import pandas as pd

import dascore as dc
from dascore.compat import UPath
from dascore.constants import PatchType, SpoolType
from dascore.exceptions import PatchConversionError
from dascore.utils.misc import (
    _maybe_make_parent_directory,
    iterate,
    optional_import,
)
from dascore.utils.paths import (
    coerce_to_local_path,
    coerce_to_upath,
    is_local_path,
    is_pathlike,
)
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
        return coerce_to_local_path(resource)
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


def _annotate_handle_path(handle, resource):
    """Attach lightweight source-path metadata to a remote handle when absent."""
    path_str = str(resource)
    # Compatibility hack for readers which inspect handle.name/path. Remote
    # text handles can come back as TextIOWrapper(name=None) whose name is
    # not writable, so keep a private fallback for format sniffers.
    for attr_name in ("_dascore_source_path",):
        with suppress(AttributeError, TypeError):
            setattr(handle, attr_name, path_str)
    if getattr(handle, "name", None) in (None, ""):
        with suppress(AttributeError, TypeError):
            setattr(handle, "name", path_str)
    return handle


def _normalize_source_patch_keys(source_patch_key) -> set[str]:
    """Coerce source patch identifiers into a deduplicated set of strings."""
    return {
        str(value) for value in iterate(source_patch_key) if value not in (None, "")
    }


def _read_file_header(path, length: int) -> bytes:
    """Return the first bytes from a file-like path or empty bytes on IO errors."""
    try:
        with open(path, "rb") as fi:
            return fi.read(length)
    except OSError:
        return b""


class BinaryReader(io.BytesIO):
    """Base file for reading binary files."""

    mode = "rb"
    reset_offset = True

    @classmethod
    def get_handle(cls, resource):
        """Get the handle object from various sources."""
        if isinstance(resource, (cls, io.BufferedIOBase)):
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
        if isinstance(resource, (cls, io.BufferedIOBase)):
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
        if isinstance(resource, (cls, io.TextIOBase)):
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


def release_handle(handle, abort: bool = False):
    """
    Release a file handle, closing it or discarding its uncommitted work.

    Only a few handles can ``abort``; a remote HDF5 writer does, because
    closing it uploads whatever was written so far. Everything else is
    closed, and a handle with no ``close`` needs no release at all.
    """
    if abort and hasattr(handle, "abort"):
        handle.abort()
    else:
        getattr(handle, "close", lambda: None)()


class IOResourceManager:
    """
    A class for managing opening/closing files.

    One manager serves one IO operation. Creating and closing its
    resources is synchronized, so concurrent callers share a single
    handle per type; a handle it hands back is not itself safe to use
    from several threads at once.
    """

    def __init__(self, source: Any):
        self._source = source
        self._cache = {}
        self._lock = RLock()

    @property
    def source(self):
        """Get the source of the IO manager."""
        # Not cached: the walk is a couple of isinstance checks, and
        # memoizing it into _cache would let close_all close the source.
        source = self._source
        # this handles IO managers derived from other IO managers;
        # effectively, we need to go back to the original, non-io manager source
        while isinstance(source, self.__class__):
            source = source.source
        return source

    def get_resource(self, required_type: RequiredType) -> RequiredType:
        """Get the requested resource, opening each handle exactly once."""
        # no required type, just return source of manager.
        if required_type is None:
            return self.source
        # this is so the context managers can be nested and the child
        # context manager only calls to the parent. Then, the resources
        # get closed only after the original exists its context.
        if isinstance(self._source, self.__class__):
            return self._source.get_resource(required_type)
        required_type = _get_required_type(required_type)
        with self._lock:
            if required_type not in self._cache:
                source = _resolve_resource(self._source, required_type)
                out = get_handle_from_resource(source, required_type)
                self._cache[required_type] = out
            return self._cache[required_type]

    def close_all(self, abort: bool = False):
        """
        Close any open file handles.

        With ``abort=True``, handles that support it discard uncommitted
        work (e.g. remote writers skip uploading a partial file). One
        handle failing must not skip cleanup of the others (remote handles
        resume garbage collection in close), so the first error is
        re-raised only after every handle was attempted. BaseException is
        caught for that reason too: a Ctrl-C mid-close would otherwise
        strand the GC pause of every handle after it.
        """
        first_exc = None
        with self._lock:
            for handle in self._cache.values():
                try:
                    release_handle(handle, abort=abort)
                except BaseException as exc:
                    first_exc = first_exc if first_exc is not None else exc
        if first_exc is not None:
            raise first_exc

    def clear_cache(self):
        """Close and forget any cached resources so they can be reopened fresh."""
        with self._lock:
            try:
                self.close_all()
            finally:
                self._cache.clear()

    def __enter__(self):
        """Entering context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close all handles; on error, abort uncommitted writes instead."""
        if exc_type is None:
            self.close_all()
            return
        try:
            self.close_all(abort=True)
        except Exception as cleanup_error:
            # A cleanup failure must not replace the error which caused it.
            exc_val.add_note(f"Aborting IO resources also failed: {cleanup_error!r}")

    def __del__(self):
        with suppress(Exception):
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


def xarray_to_patch(data_array) -> dc.Patch:
    """Convert an xarray dataarray to a patch."""
    # this cant work if xarray isn't installed. This ensures it is.
    _ = optional_import("xarray")

    return dc.Patch(
        coords={i: (x.dims, x.values) for i, x in data_array.coords.items()},
        attrs=dict(data_array.attrs.items()),
        dims=data_array.dims,
        data=data_array.data,
    )


def _xarray_block_count(low, high, step) -> int:
    """Sample count of an inclusive envelope, snapped to its step."""
    return int(np.round(to_float(high - low) / to_float(step))) + 1


def _load_xarray_block(catalog, row, dim, lims, dims, shape):
    """
    Load one dask block: a source patch trimmed to its member window.

    Runs at compute time, once per block; the trim hint limits what the
    reader loads and the select re-applies the window exactly.
    """
    patch = catalog.resolve_row(row, extra_trim={dim: lims}).select(**{dim: lims})
    if patch.dims != dims:
        patch = patch.transpose(*dims)
    if patch.shape != shape:
        msg = (
            f"Loaded block from '{row.get('source_path', '<memory>')}' has shape "
            f"{patch.shape}, but the spool index promised {shape}. The source "
            "file changed after indexing; run spool.update() and convert again."
        )
        raise PatchConversionError(msg)
    return patch.data


def _xarray_group_nodes(outputs, group_attrs):
    """Name a DataTree node per output group, by the spool's naming rule."""
    # The same rule which names a repr's tracks and a coverage plot's
    # lanes; a node name must also be a valid single path component.
    from dascore.utils.display import ACQUISITION_ATTR, group_names  # noqa: PLC0415

    if not group_attrs:
        frame = pd.DataFrame(index=[0])
        codes = pd.Series(0, index=outputs.index)
    else:
        codes = outputs.groupby(group_attrs, dropna=False, sort=True).ngroup()
        frame = (
            outputs[group_attrs]
            .assign(_code=codes)
            .drop_duplicates("_code")
            .sort_values("_code")
            .drop(columns="_code")
            .reset_index(drop=True)
        )
    fallback = ACQUISITION_ATTR if ACQUISITION_ATTR in frame.columns else None
    names = group_names(frame, fallback=fallback)
    if bad := [x for x in names if "/" in x]:
        msg = (
            f"Group name(s) {bad} contain '/', which cannot name a DataTree "
            "node. Pass different `group` attributes."
        )
        raise PatchConversionError(msg)
    return names, codes


def spool_to_xarray(
    spool: SpoolType,
    dim: str = "time",
    group: str | typing.Sequence[str] | None = None,
    tolerance=1.5,
    conflict: Literal["drop", "raise", "keep_first"] = "raise",
):
    """
    Convert a spool to a lazy, dask-backed xarray DataTree.

    Patches are partitioned exactly as [`chunk`](`dascore.Spool.chunk`)
    partitions them: one tree node per group of related patches, holding
    one child node per contiguous segment (``segment_0`` onward, ordered
    along ``dim``), each with a dask-backed ``data`` variable. Building
    the tree reads no patch data — every shape, dtype, and coordinate
    comes from the spool's metadata — and computing a selection loads
    only the source patches it touches.

    Parameters
    ----------
    spool
        The spool to convert.
    dim
        The dimension segments are merged along.
    group
        Attributes which partition patches into unrelated groups, exactly
        as `chunk` uses them. Defaults to the config's ``patch_kind_attrs``.
    tolerance
        The sampling tolerance deciding segment continuity, as in `chunk`.
    conflict
        How attribute conflicts within a segment resolve, as in `chunk`.

    Notes
    -----
    Requires ``xarray`` and ``dask``. Coordinates associated with a
    dimension (rather than defining one) are not carried into the tree.
    """
    xr = optional_import("xarray")
    dask = optional_import("dask")
    da = optional_import("dask.array")
    # function-level to avoid a circular import through dascore.core
    from dascore.core.coords import get_coord  # noqa: PLC0415
    from dascore.utils.chunk_plan import build_chunk_plan  # noqa: PLC0415

    base, working = spool._plan_frames(dim)
    if not len(working):
        return xr.DataTree()
    merge_kwargs: dict[str, Any] = {dim: None}
    plan = build_chunk_plan(
        working, tolerance=tolerance, conflict=conflict, group=group, **merge_kwargs
    )
    outputs, members = plan.outputs, plan.members
    group_attrs = list(plan.params["group"])
    names, codes = _xarray_group_nodes(outputs, group_attrs)
    sources = base.set_index("_patch_id")
    catalog = spool._catalog
    envelope_cols = {
        f"{d}_{end}"
        for dims_str in outputs["dims"].unique()
        for d in str(dims_str).split(",")
        for end in ("min", "max", "step")
    }
    tree = {}
    for code, sub in outputs.groupby(codes.to_numpy(), sort=True):
        node = f"/{names[code]}"
        first = sub.iloc[0]
        node_attrs = {key: first[key] for key in group_attrs if pd.notnull(first[key])}
        tree[node] = xr.Dataset(attrs=node_attrs)
        for segment, (_, out) in enumerate(sub.sort_values(f"{dim}_min").iterrows()):
            dims = tuple(str(out["dims"]).split(","))
            axis = dims.index(dim)
            if (dtype_str := out["_dtype"]) is None or pd.isnull(dtype_str):
                msg = (
                    "Cannot build a lazy array without a dtype in the spool "
                    "index; re-index the spool with spool.update()."
                )
                raise PatchConversionError(msg)
            dtype = np.dtype(dtype_str)
            step = out[f"{dim}_step"]
            mem = members[members["output_id"] == out["output_id"]]
            mem = mem.sort_values(f"{dim}_min")
            counts = [
                _xarray_block_count(m[f"{dim}_min"], m[f"{dim}_max"], step)
                for _, m in mem.iterrows()
            ]
            coords, sizes = {}, {}
            for d in dims:
                if d == dim:
                    # The member counts are authoritative, and the coordinate
                    # is snapped over the segment envelope exactly as chunk
                    # snaps a merged coordinate: min and max unchanged, step
                    # recomputed from the span.
                    n = sum(counts)
                    lo, hi = out[f"{dim}_min"], out[f"{dim}_max"]
                    span = hi - lo
                    if n > 1 and isinstance(span, pd.Timedelta | np.timedelta64):
                        nanos = pd.Timedelta(span).value
                        eff = np.timedelta64(int(np.round(nanos / (n - 1))), "ns")
                    elif n > 1:
                        eff = span / (n - 1)
                    else:
                        eff = step
                    coord = get_coord(min=lo, max=lo + n * eff, step=eff)
                    if len(coord) != n:
                        coord = coord.change_length(n)
                else:
                    d_step = out[f"{d}_step"]
                    coord = get_coord(
                        min=out[f"{d}_min"], max=out[f"{d}_max"] + d_step, step=d_step
                    )
                coords[d] = coord.values
                sizes[d] = len(coord)
            blocks = []
            for (_, m), count in zip(mem.iterrows(), counts, strict=True):
                shape = tuple(count if d == dim else sizes[d] for d in dims)
                lims = (m[f"{dim}_min"], m[f"{dim}_max"])
                row = sources.loc[m["_patch_id"]].to_dict()
                delayed = dask.delayed(_load_xarray_block)(
                    catalog, row, dim, lims, dims, shape
                )
                blocks.append(da.from_delayed(delayed, shape=shape, dtype=dtype))
            array = da.concatenate(blocks, axis=axis)
            attrs = {
                key: value
                for key, value in out.items()
                if not str(key).startswith("_")
                and key not in envelope_cols
                and key not in ("output_id", "dims")
                and pd.notnull(value)
            }
            data = xr.DataArray(array, dims=dims, coords=coords, attrs=attrs)
            tree[f"{node}/segment_{segment}"] = xr.Dataset({"data": data})
    return xr.DataTree.from_dict(tree)


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


def obspy_to_patch(stream, dim="distance") -> dc.Patch:
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
