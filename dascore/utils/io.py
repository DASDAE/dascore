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
from dascore.exceptions import ParameterError, PatchConversionError
from dascore.utils.downloader import resolve_example_uri
from dascore.utils.misc import (
    _maybe_make_parent_directory,
    iterate,
    optional_import,
)
from dascore.utils.paths import (
    coerce_to_local_path,
    coerce_to_upath,
    is_example_uri,
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
    # An examples:// name has no filesystem behind it, so it becomes a real
    # path before anything asks what protocol it speaks. It happens here,
    # when a handle is actually wanted, rather than when a manager is built:
    # constructing one must not reach the network, and the manager's source
    # must keep naming the uri so a write of it can still be refused.
    resource = resolve_example_uri(resource)
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


# Handle classes name the mode they open in. Everything else truncates or
# appends, so an example file must not be opened through one.
_READ_MODES = frozenset({"r", "rb"})


def _refuse_example_write(source, required_type) -> None:
    """Refuse a handle which would write over a downloaded example file."""
    if not is_example_uri(source):
        return
    # A type with no get_handle opens nothing: the resolved path is handed
    # through as it is. LocalPath has one, but it returns a filename rather
    # than an open file, so neither can truncate anything by itself.
    if not hasattr(required_type, "get_handle"):
        return
    if isinstance(required_type, type) and issubclass(required_type, LocalPath):
        return
    # Any other handle must say it opens to read. A missing mode is refused
    # rather than assumed harmless: every reader here declares one, so
    # silence is more likely an unknown writer than a safe default.
    if getattr(required_type, "mode", None) in _READ_MODES:
        return
    msg = (
        f"Cannot open {source} for writing; examples:// names are read-only. "
        f"Give a path to write to instead."
    )
    raise ParameterError(msg)


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
        _refuse_example_write(self._source, required_type)
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


def _np_scalar(value):
    """A relation-row scalar as numpy; pandas scalars make object coords."""
    if isinstance(value, pd.Timestamp):
        return value.to_datetime64()
    if isinstance(value, pd.Timedelta):
        return value.to_timedelta64()
    return value


def _envelope_coord(low, high, step, get_coord):
    """A dimension coordinate stated by its index envelope, either order."""
    low, high, step = _np_scalar(low), _np_scalar(high), _np_scalar(step)
    if pd.isnull(step) or to_float(step) == 0:
        if low == high:
            return get_coord(data=[low])
        msg = (
            f"Cannot size a lazy array: a dimension spanning {low} to "
            f"{high} records no sampling step in the spool index."
        )
        raise PatchConversionError(msg)
    if to_float(step) < 0:
        # A descending coordinate starts at its max; stop is exclusive.
        return get_coord(start=high, stop=low + step, step=step)
    return get_coord(min=low, max=high + step, step=step)


def _member_coord(low, high, step, env_low, env_high, get_coord, units=None):
    """
    The coordinate a member presents inside its trim window.

    The member's full coordinate is rebuilt from its envelope and trimmed
    by the coordinate's own select, so block sizes and sample labels
    follow exactly the rule loading follows, not a parallel rounding.
    Units ride along so a unit-bearing tolerance can be read against the
    merged coordinate, as chunk reads it.

    Also returns the window as half-open sample indices on the member's
    own grid — the form `FiberIO.read_array` takes.
    """
    low, high, step = _np_scalar(low), _np_scalar(high), _np_scalar(step)
    env_low, env_high = _np_scalar(env_low), _np_scalar(env_high)
    # Rows state units as strings; anything else (absent, null) is none.
    units = units if isinstance(units, str) and units else None
    # A single-sample merge dimension never reaches here: the planner
    # refuses to chunk a dimension it cannot order.
    if pd.isnull(step) or to_float(step) == 0:
        msg = (
            f"Cannot size a lazy block: a patch spanning {low} to {high} "
            "records no sampling step in the spool index."
        )
        raise PatchConversionError(msg)
    full = get_coord(min=env_low, max=env_high + step, step=step, units=units)
    coord, indexer = full.select((low, high))
    # plan invariant: a published member always presents at least a sample
    assert len(coord), "a plan member never presents an empty window"
    # a range select of an evenly sampled coordinate is a unit slice
    assert isinstance(indexer, slice), "range select yields a slice"
    start, stop, stride = indexer.indices(len(full))
    assert stride == 1, "range select never strides"
    return coord, (start, stop)


def _load_xarray_block(resolver, row, dim, lims, dims, shape, dtype, window=None):
    """
    Load one dask block: a source patch trimmed to its member window.

    Runs at compute time, once per block. When the row's format offers a
    `read_array` fast path, only the raw array for the sample window is
    read; otherwise member loading goes through the plan resolver — the
    same path a chunked spool loads through — so residuals, units, and
    nested plans are honored. The select re-applies the window exactly
    since read hints only reduce reading.
    """
    array = None
    stated = row.get("dims")
    src_dims = tuple(stated.split(",")) if isinstance(stated, str) else ()
    # The row must state the same dimensions the tree promises, or the
    # transpose below could not be built; disagreement means the patch
    # path, whose own errors say what is wrong.
    if window is not None and sorted(src_dims) == sorted(dims):
        array = resolver._load_member_array(row, {dim: window})
    if array is None:
        patch = resolver._load_member(row).select(**{dim: lims})
        if patch.dims != dims:
            patch = patch.transpose(*dims)
        array = patch.data
    elif src_dims != dims:
        array = np.transpose(array, [src_dims.index(d) for d in dims])
    if array.shape != shape:
        msg = (
            f"Loaded block from '{row.get('source_path', '<memory>')}' has shape "
            f"{array.shape}, but the spool index promised {shape}. The source "
            "file changed after indexing; run spool.update() and convert again."
        )
        raise PatchConversionError(msg)
    # A member narrower than the segment's combined dtype upcasts here so
    # the array holds the dtype its metadata states.
    return array.astype(dtype, copy=False)


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
    if bad := [x for x in names if "/" in x or x in ("", ".", "..")]:
        msg = (
            f"Group name(s) {bad} cannot name a DataTree node (a name may "
            "not contain '/' or be '.' or '..'). Pass different `group` "
            "attributes."
        )
        raise PatchConversionError(msg)
    # A literal value can collide with a generated fallback name (a group
    # tagged "group 0" beside an untagged one); a shared node path would
    # silently overwrite arrays.
    if len(set(names)) != len(names):
        dupes = sorted({x for x in names if names.count(x) > 1})
        msg = (
            f"Group name(s) {dupes} name more than one group of patches. "
            "Pass `group` attributes which tell the groups apart."
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
    one child node per merged output (``segment_0`` onward, ordered along
    ``dim``), each with a dask-backed ``data`` variable. Node names
    follow the spool's own naming rule — the one its repr's tracks and a
    coverage plot's lanes use. Building the tree reads no patch data —
    every shape, dtype, and coordinate comes from the spool's metadata —
    and computing a selection loads only the source patches it touches.

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
        The continuity tolerance deciding when a gap splits segments, as
        in `chunk` (not the sampling-step grouping tolerance).
    conflict
        How attribute conflicts within a segment resolve, as in `chunk`.

    Notes
    -----
    Requires ``xarray`` and ``dask``. Coordinates associated with a
    dimension (rather than defining one) are not carried into the tree,
    and coordinates are rebuilt from their indexed envelopes, whose
    numeric values are floats — an integer-valued dimension coordinate
    comes back as floats.

    A spool with pending value-range selections cannot be converted: the
    catalog states such bounds as candidacy rather than sample positions,
    so the arrays cannot be sized without reading. Convert first and
    select on the tree (e.g. ``sel(time=...)``), or select with
    ``samples=True`` on dimensions, which stays exact. Pending inventory
    enrichment is likewise refused, since the tree would omit the
    enriched attributes.
    """
    xr = optional_import("xarray")
    dask = optional_import("dask")
    da = optional_import("dask.array")
    # function-level to avoid circular imports through the package root
    from dascore.core.coords import concat_coords, get_coord  # noqa: PLC0415
    from dascore.io.index.planned import PlanResolver, derived_catalog  # noqa: PLC0415
    from dascore.units import carries_units  # noqa: PLC0415
    from dascore.utils.chunk_plan import (  # noqa: PLC0415
        _normalize_chunk_units,
        build_chunk_plan,
    )
    from dascore.utils.misc import get_middle_value  # noqa: PLC0415

    source_rows, working = spool._plan_frames(dim)
    if not len(working):
        return xr.DataTree()
    if spool._enrich_kwargs:
        msg = (
            "Cannot convert a spool with pending inventory enrichment: the "
            "tree would omit the enriched attributes. Convert the spool "
            "before enriching it."
        )
        raise PatchConversionError(msg)
    all_dims = {
        d for dims_str in working["dims"].dropna() for d in str(dims_str).split(",")
    }
    for selected, samples in spool._catalog.residuals:
        # A samples selection on a dimension adjusts that dimension's
        # envelopes exactly; anything else changes what loads in ways the
        # envelopes do not state.
        if not selected or (samples and set(selected) <= all_dims):
            continue
        kind = (
            "sample selections on associated coordinates"
            if samples
            else ("value selections")
        )
        msg = (
            f"Cannot convert a spool with pending {kind} (on "
            f"{sorted(selected)}): such bounds are candidacy, not sample "
            "positions, so the lazy arrays cannot be sized. Convert "
            "first and select on the tree, or select dimensions with "
            "samples=True."
        )
        raise PatchConversionError(msg)
    steps = working.get(f"{dim}_step")
    if steps is not None and (to_float(steps.values) < 0).any():
        msg = (
            f"A descending '{dim}' coordinate is not supported by "
            f"to_xarray; sort the patches along {dim} first."
        )
        raise PatchConversionError(msg)
    chunk_kwargs: dict[str, Any] = {dim: None}
    plan = build_chunk_plan(
        working, tolerance=tolerance, conflict=conflict, group=group, **chunk_kwargs
    )
    outputs = plan.outputs
    group_attrs = list(plan.params["group"])
    names, codes = _xarray_group_nodes(outputs, group_attrs)
    # The same derivation chunk performs: its resolver is what knows how
    # to load one member row, whatever kind of spool this is.
    catalog = derived_catalog(
        source_rows=source_rows,
        plan=plan,
        parent=spool._catalog,
        merge_kwargs={
            "conflict": conflict,
            "snap_coords": True,
            "tolerance": plan.params["tolerance"],
        },
        mode="chunk",
        origin_path=spool.spool_path,
    )
    resolver = catalog.resolver
    assert isinstance(resolver, PlanResolver)  # a chunk derivation always is
    # plan.members and the resolver's member_rows are the same rows in the
    # same order; the former keeps _patch_id (for the source grid), the
    # latter is what the resolver loads from. Verify the invariant, since
    # derived_catalog does not promise it to other callers.
    member_rows = resolver.member_rows.reset_index(drop=True)
    members = plan.members.reset_index(drop=True)
    check_cols = ["output_id", f"{dim}_min", f"{dim}_max"]
    assert members[check_cols].equals(member_rows[check_cols])
    # Member grids in the plan's normalized units, so trims and envelopes
    # speak the same unit; the plan normalized the same frame identically.
    norm = _normalize_chunk_units(working, dim).set_index("_patch_id")
    # A working row which is itself a trim (a collapsed plan's member)
    # states a trimmed envelope, so a sample window measured against it
    # is not anchored on the file's grid; such members load by value.
    if "_modified" in norm.columns:
        modified = members["_patch_id"].map(norm["_modified"]).fillna(True).astype(bool)
    else:
        modified = pd.Series(False, index=members.index)
    members = members.assign(
        _pos=np.arange(len(members)),
        _env_low=members["_patch_id"].map(norm[f"{dim}_min"]),
        _env_high=members["_patch_id"].map(norm[f"{dim}_max"]),
        _env_anchored=~modified,
    )
    # One shared graph node for the resolver, not a copy inside every
    # block task — it can hold live patches or a large member table.
    resolver_ref = dask.delayed(resolver, pure=True)
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
            if (dtype_str := out["_dtype"]) is None or not str(dtype_str):
                msg = (
                    "Cannot build a lazy array without a dtype in the spool "
                    "index; re-index the spool with spool.update()."
                )
                raise PatchConversionError(msg)
            dtype = np.dtype(dtype_str)
            mem = members[members["output_id"] == out["output_id"]]
            mem = mem.sort_values(f"{dim}_min")
            # Each member's block is sized by its own coordinate: the same
            # select which trims the block at load also counts it here,
            # and its indexer is the sample window a data-only read takes.
            member_coords, member_windows = [], []
            for _, m in mem.iterrows():
                coord, window = _member_coord(
                    m[f"{dim}_min"],
                    m[f"{dim}_max"],
                    m[f"{dim}_step"],
                    m["_env_low"],
                    m["_env_high"],
                    get_coord,
                    units=m.get(f"_{dim}_units"),
                )
                member_coords.append(coord)
                member_windows.append(window)
            coords, sizes = {}, {}
            for d in dims:
                if d == dim:
                    # The same construction chunk merges by: concatenate
                    # the member coordinates truth-preservingly, then
                    # absorb sub-tolerance seams. A seam beyond tolerance
                    # stays segmented here exactly as it does there.
                    merged = concat_coords(*member_coords)
                    if carries_units(tolerance):
                        merged = merged.simplify(tolerance)
                    else:
                        mem_steps = [
                            abs(c.step)
                            for c in member_coords
                            if getattr(c, "step", None) is not None
                            and not pd.isnull(c.step)
                        ]
                        if mem_steps:
                            merged = merged.simplify(
                                tolerance * get_middle_value(mem_steps)
                            )
                    coord = merged
                else:
                    coord = _envelope_coord(
                        out[f"{d}_min"], out[f"{d}_max"], out[f"{d}_step"], get_coord
                    )
                coords[d] = coord.values
                sizes[d] = len(coord)
            blocks = []
            zipped = zip(member_coords, member_windows, mem.iterrows(), strict=True)
            for member_coord, window, (_, m) in zipped:
                count = len(member_coord)
                shape = tuple(count if d == dim else sizes[d] for d in dims)
                lims = (m[f"{dim}_min"], m[f"{dim}_max"])
                row = member_rows.iloc[int(m["_pos"])].to_dict()
                delayed = dask.delayed(_load_xarray_block)(
                    resolver_ref,
                    row,
                    dim,
                    lims,
                    dims,
                    shape,
                    dtype,
                    window if m["_env_anchored"] else None,
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
