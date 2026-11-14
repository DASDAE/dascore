"""Zarr IO as CF-on-zarr, read and written through xarray."""

from __future__ import annotations

import re

import numpy as np

import dascore as dc
from dascore.compat import UPath
from dascore.constants import snap_type, windows_type
from dascore.core.coords import get_coord
from dascore.core.source import ArraySource
from dascore.exceptions import (
    CoordError,
    InvalidFiberFileError,
    MissingOptionalDependencyError,
    ParameterError,
    PatchAttributeError,
)
from dascore.io import FiberIO
from dascore.io.dasdae.utils import grid_fields, read_grid_fields
from dascore.io.netcdf.utils import (
    COORD_PREFIX,
    dataset_to_patch_meta,
    get_xarray_data_var_name,
    payload_datasets,
    read_dataset_array,
    spool_to_cf_dataset,
    stated_range,
)
from dascore.io.utils import resolve_keyed_source, windows_to_slices
from dascore.utils.io import staged_path
from dascore.utils.misc import optional_import, suppress_warnings
from dascore.utils.patch import (
    _unique_patch_names,
    check_data_units,
    check_kind,
    get_patch_names,
)

# The file which marks a directory as a zarr group, by zarr format.
_MARKERS = {"3": "zarr.json", "2": ".zgroup"}
# The attribute of a dimension's coordinate array listing the half-open
# [start, stop) positions along it which hold data, where not all do.
SPANS_ATTR = "_dascore_spans"
# The key of one span of a payload: "<payload key>[start:stop]".
_SPAN_KEY = re.compile(r"^(?P<key>.+)\[(?P<start>\d+):(?P<stop>\d+)\]$")


def _open_group(path: UPath, zarr_format: int | None):
    """Open a store's root group with zarr, from consolidated metadata if any."""
    zarr = optional_import("zarr")
    with suppress_warnings(UserWarning, message="Consolidated metadata"):
        return zarr.open_group(path, mode="r", zarr_format=zarr_format)


def _open_zarr(path: UPath, zarr_format: int | None, group: str, node):
    """
    Open a store (or one group) as a lazy dataset.

    No indexes are built, so a windowed read touches only the chunks it
    needs. A CF time is decoded, reading its first and last chunk, unless
    its grid is described beside it (see `COORD_PREFIX`); such labels are
    never read. ``node`` is the group open with zarr, which states that.

    A child group is read from its own metadata, not the root's
    consolidated copy, which xarray would parse again for every group.
    Naming the format (None sniffs it) saves probing for the other format's
    metadata files, each a request on a remote store.
    """
    xr = optional_import("xarray")

    def _open(undecoded):
        # A store without consolidated metadata still opens, only slower.
        with suppress_warnings(RuntimeWarning, message="Failed to open Zarr store"):
            return xr.open_dataset(
                path,
                engine="zarr",
                group=group or None,
                consolidated=False if group else None,
                chunks=None,
                cache=False,
                create_default_indexes=False,
                zarr_format=zarr_format,
                decode_times=undecoded,
                decode_timedelta=undecoded,
            )

    ranged = {name for name, array in node.arrays() if stated_range(array)}
    dataset = _open(dict.fromkeys(ranged, False))
    # Consolidated metadata older than the group's own (an append not yet
    # consolidated) can describe labels which no longer match: decode them.
    variables = dataset.variables
    if stale := {
        x for x in ranged if x in variables and not stated_range(variables[x])
    }:
        dataset.close()
        dataset = _open(dict.fromkeys(ranged - stale, False))
    return dataset


# Encoding keys which lay out storage without changing the labels read back;
# codecs (filters, compressors, serializers) may be lossy.
_LAYOUT_KEYS = frozenset(("chunks", "shards"))


def _range_fields(coord) -> dict | None:
    """
    Return the fields describing a coordinate's grid (see `COORD_PREFIX`).

    None where it has no grid (segments, labels, partial), or where JSON (no
    numpy scalar types) leaves the description unable to rebuild the labels
    exactly; those labels are read instead.
    """
    if (fields := grid_fields(coord)) is None:
        return None
    fields = {
        k: v.item() if isinstance(v, np.generic) else v for k, v in fields.items()
    }
    back = read_grid_fields(fields, coord.units)
    # Whole ticks survive JSON exactly; float labels are compared one by one.
    exact = "step_numerator" in fields
    same = exact or np.array_equal(back.values, coord.values)
    return fields if same and back.dtype == coord.dtype and back == coord else None


def coord_attrs(coord) -> dict:
    """
    Return the attributes stating a coordinate's grid on a zarr array.

    Set on a coordinate array, they describe its labels, which a scan then
    never reads; the array may have no chunks written at all. The values
    are JSON-safe. If the array's length is later changed, its attributes
    must be rewritten, since a stated length unlike the array's is taken
    as stale and the labels are read instead.

    Parameters
    ----------
    coord
        A coordinate of one grid.

    Raises
    ------
    ParameterError
        If the coordinate is not one grid JSON describes exactly.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.core.coords import get_coord
    >>> from dascore.io.zarr import coord_attrs
    >>> start, step = np.datetime64(0, "ns"), np.timedelta64(1, "s")
    >>> time = get_coord(start=start, step=step, shape=(10,))
    >>> attrs = coord_attrs(time)  # e.g. zarr_array.attrs.update(attrs)
    """
    if (fields := _range_fields(coord)) is None:
        msg = f"Cannot state {coord} as a grid; its labels must be stored."
        raise ParameterError(msg)
    return {COORD_PREFIX + k: v for k, v in fields.items()}


def _cf_dataset(spool, encoding: dict | None):
    """
    Return the only patch of a spool as a CF dataset, describing its grids.

    The labels are written too, so any CF tool reads them. A coordinate
    whose ``encoding`` changes the labels stored (a dtype, units or
    scaling) is not described, so its stored labels are read.
    """
    patches = [spool] if isinstance(spool, dc.Patch) else list(spool)
    if not patches[0].dims:  # it would read back as nothing
        raise ParameterError("A patch with no dimensions cannot be written to zarr.")
    dataset = spool_to_cf_dataset(patches)
    for name, coord in patches[0].coords.coord_map.items():
        if set((encoding or {}).get(name, ())) - _LAYOUT_KEYS:
            continue
        if (fields := _range_fields(coord)) is not None:
            dataset[name].attrs.update({COORD_PREFIX + k: v for k, v in fields.items()})
    return dataset


def _payloads(dataset) -> dict:
    """
    Map each payload of an open dataset to a dataset holding it alone.

    ``data``, else the only variable, is the payload; otherwise each
    variable sharing the largest one's dimensions is, as NetCDF reads them
    (see `payload_datasets`). Coordinates, the auxiliary coordinates a
    ``coordinates`` attribute names, CF ``bounds`` tables and scalars never
    are.
    """
    try:
        payloads = payload_datasets(dataset)
    except ValueError:
        return {}
    # a lone bounds table passes the only-variable rule
    bounds = {x.attrs.get("bounds") for x in dataset.variables.values()}
    return {k: v for k, v in payloads.items() if v[k].dims and k not in bounds}


def _stated_spans(dataset, dims, where: str) -> tuple[str, list] | None:
    """
    Return the dimension whose coordinate states spans, and the spans.

    None where no dimension states them (the payload is one patch).
    """
    variables = dataset.variables
    stated = {
        dim: variables[dim].attrs[SPANS_ATTR]
        for dim in dims
        if dim in variables and SPANS_ATTR in variables[dim].attrs
    }
    if not stated:
        return None
    if len(stated) > 1:
        msg = (
            f"{where}: only one dimension may state {SPANS_ATTR}, not {sorted(stated)}."
        )
        raise InvalidFiberFileError(msg)
    ((dim, raw),) = stated.items()
    try:
        spans = [(int(start), int(stop)) for start, stop in raw]
    except (TypeError, ValueError):
        spans = None
    ends = [x for span in spans or () for x in span]
    size = dataset.sizes[dim]
    if (
        spans is None
        or ends != sorted(ends)
        or any(a >= b for a, b in spans)
        or (ends and (ends[0] < 0 or ends[-1] > size))
    ):
        msg = (
            f"{where}: {SPANS_ATTR} of {dim!r} must be sorted, non-overlapping "
            f"[start, stop) positions within its {size} samples, not {raw!r}."
        )
        raise InvalidFiberFileError(msg)
    return dim, spans


def _span_metas(meta: dc.PatchMeta, dataset, key: str, where: str) -> list:
    """Split a payload's description into one per span it states."""
    if (stated := _stated_spans(dataset, meta.dims, f"{where} {key!r}")) is None:
        return [meta]
    dim, spans = stated
    return [
        dc.PatchMeta(
            attrs=meta.attrs,
            coords=meta.coords.select(samples=True, **{dim: (start, stop)})[0],
            dims=meta.dims,
            dtype=meta.dtype,
            source=ArraySource(key=f"{key}[{start}:{stop}]"),
        )
        for start, stop in spans
    ]


def _patch_keys(group: str, payloads: dict) -> dict:
    """
    Key each payload of a group by its path.

    A group's only payload is keyed by the group's path instead (the root's
    by its own name), as stores DASCore writes have always been read.
    """
    if len(payloads) == 1 and group:
        return {group: next(iter(payloads.values()))}
    return {f"{group}/{k}".lstrip("/"): v for k, v in payloads.items()}


def _groups(root):
    """Yield ``(path, group)`` for a store's root and every group below it."""
    zarr = optional_import("zarr")
    yield "", root
    members = root.members(max_depth=None)
    groups = (x for x in members if isinstance(x[1], zarr.Group))
    yield from sorted(groups, key=lambda x: x[0])


def _open_payload_group(path: UPath, zarr_format: int | None, node):
    """Open one group as a dataset, or return None if xarray finds no payload."""
    if not any(True for _ in node.array_keys()):
        return None  # e.g. a group of groups, never opened by xarray
    try:
        return _open_zarr(path, zarr_format, node.path, node)
    except KeyError as exc:  # raw zarr arrays with no dimension names
        if any(_dimension_names(x) for _, x in node.arrays()):
            msg = f"{path} group {node.path!r} mixes arrays with and without dims."
            raise InvalidFiberFileError(msg) from exc
        return None


def _dimension_names(array) -> list | None:
    """Return the dimension names a zarr array states (format 3 or 2)."""
    names = getattr(array.metadata, "dimension_names", None)
    return names or array.attrs.get("_ARRAY_DIMENSIONS")


def _patches(path: UPath, zarr_format: int | None):
    """
    Yield ``(key, dataset holding its payload alone)`` for each patch.

    Every group at any depth is a path to its payloads; attributes are the
    payload's own, never inherited from the groups above it.
    """
    for group, node in _groups(_open_group(path, zarr_format)):
        if (dataset := _open_payload_group(path, zarr_format, node)) is None:
            continue
        with dataset:
            yield from _patch_keys(group, _payloads(dataset)).items()


def _span_windows(payload, span, windows, where: str) -> tuple[slice, ...]:
    """Return windows into one span of a payload as windows into the payload."""
    name = get_xarray_data_var_name(payload)
    dims, shape = payload[name].dims, list(payload[name].shape)
    stated = _stated_spans(payload, dims, where)
    if stated is None or tuple(span) not in stated[1]:
        msg = f"No span {list(span)} stated for '{name}' in {where}."
        raise PatchAttributeError(msg)
    axis = dims.index(stated[0])
    shape[axis] = span[1] - span[0]
    slices = list(windows_to_slices(windows, shape))
    within = slices[axis]
    slices[axis] = slice(within.start + span[0], within.stop + span[0])
    return tuple(slices)


def _payload_group(root, key: str, where: str):
    """Return the group holding the payload a patch key names."""
    zarr = optional_import("zarr")
    try:  # a key names a group holding one payload, else a payload array
        node = root[key]
    except KeyError as exc:
        raise PatchAttributeError(f"No patch named '{key}' in {where}.") from exc
    if isinstance(node, zarr.Group):
        return node
    group = key.rpartition("/")[0]
    return root[group] if group else root


def _group_names(spool) -> list[str]:
    """Name a group for each patch, as `get_patch_names` names them."""
    names = get_patch_names(spool).str.replace("/", "_", regex=False)
    return list(names.mask(names == "", "patch"))  # "" would name the root


def _add_groups(spool, resource: UPath, zarr_format: int, encoding):
    """
    Add each patch to a store as a new group, its name not yet taken.

    Each group is built beside the store and moved in, so a killed write
    leaves only a hidden sibling of the store, which scans skip.
    """
    taken = list(_open_group(resource, zarr_format).keys())
    names = _unique_patch_names([*taken, *_group_names(spool)])[len(taken) :]
    fs = resource.fs
    for name, patch in zip(names, spool, strict=True):
        staging = resource.parent / f".{resource.name}.{name}.partial"
        try:
            dataset = _cf_dataset(patch, encoding)
            dataset.to_zarr(
                staging,
                mode="w",
                consolidated=False,
                zarr_format=zarr_format,
                encoding=encoding,
            )
            target = resource / name
            try:
                fs.mv(staging.path, target.path, recursive=True)
            except BaseException:  # a move may copy part of the group first
                if target.exists():
                    fs.rm(target.path, recursive=True)
                raise
        finally:
            if staging.exists():
                fs.rm(staging.path, recursive=True)


def _joined(meta: dc.PatchMeta, patch, dim: str):
    """
    Return the coordinate a stored patch and ``patch`` make along ``dim``.

    None unless ``patch`` continues it there on its grid with nothing
    between: the same kind, dimensions, dtype and other coordinates.
    """
    if meta.dims != patch.dims or np.dtype(meta.dtype) != patch.data.dtype:
        return None
    if not check_kind(meta, patch, "ignore", strict=True):
        return None
    if not check_data_units(meta, patch, "ignore"):
        return None
    old, new = meta.coords.coord_map, patch.coords.coord_map
    # those along dim (refused later, as not extended) differ, as dim does
    across = [x for x, dims in patch.coords.dim_map.items() if dim not in dims]
    if set(old) != set(new) or any(old[x] != new[x] for x in across):
        return None
    if not (old[dim].evenly_sampled and new[dim].evenly_sampled):
        return None
    try:
        joined = get_coord(segments=(old[dim], new[dim]))
    except CoordError:  # overlapping, or out of order
        return None
    # segments are joined in their own order, but the new samples go last
    first = joined[:1].values, old[dim][:1].values
    if not np.array_equal(*first) or joined.step != old[dim].step:
        return None
    return joined if joined.evenly_sampled else None


def _encoded(values: np.ndarray, array) -> np.ndarray:
    """
    Encode labels as a stored coordinate array holds them (CF times).

    Labels continuing a stored grid fit the units its labels were stored
    in; were they not to, xarray would quietly choose finer units, which
    the stored ones would not state.
    """
    if values.dtype.kind not in "mM":
        encoded = back = values.astype(array.dtype)
    else:
        xr = optional_import("xarray")
        time = values.dtype.kind == "M"
        coder = xr.coders.CFDatetimeCoder() if time else xr.coders.CFTimedeltaCoder()
        attrs = dict(array.attrs)
        encoding = {x: attrs[x] for x in ("units", "calendar") if x in attrs}
        encoding["dtype"] = array.dtype
        variable = xr.Variable(("x",), values, encoding=encoding)
        encoded = coder.encode(variable, "x").values
        back = coder.decode(xr.Variable(("x",), encoded, attrs=attrs), "x").values
    assert np.array_equal(back, values), f"{array.path!r} cannot hold the labels"
    return encoded


def _grown(array, shape: tuple[int, ...]):
    """
    Return ``array`` with a larger shape, which writes chunks, not metadata.

    Chunks written through it lie outside the stored shape, unseen until
    the array is resized.
    """
    zarr = optional_import("zarr")
    metadata = array.metadata.update_shape(shape)
    return zarr.Array(zarr.AsyncArray(metadata, array.store_path))


def _extend(patch, resource: UPath, zarr_format: int, dim: str) -> bool:
    """
    Extend the stored patch ``patch`` continues along ``dim``, if one does.

    Return False where none does (a gap, or another kind), so the caller
    adds it as a new patch. Several it could continue, or one it cannot
    extend whole, raise: those are errors, not gaps.

    The new chunks are written first, then the labels' shape and
    description, then the payload's shape, so a reader never sees a shape
    whose samples are not written. The caller consolidates last.
    """
    where = str(resource)
    if dim not in patch.dims:
        raise ParameterError(f"The patch has no dimension {dim!r} to append along.")
    joins = {}
    for key, dataset in _patches(resource, zarr_format):
        (meta,) = dataset_to_patch_meta(dataset, key=key)
        if (joined := _joined(meta, patch, dim)) is not None:
            joins[key] = joined
    if not joins:
        return False
    if len(joins) > 1:
        msg = (
            f"Several patches in {where} ({sorted(joins)}) end where this patch "
            f"starts along {dim!r}; which to extend is ambiguous."
        )
        raise ParameterError(msg)
    along = [x for x, dims in patch.coords.dim_map.items() if dim in dims]
    if along != [dim]:
        msg = f"Only {dim!r} itself may run along {dim!r} to append; not {along}."
        raise NotImplementedError(msg)
    ((key, joined),) = joins.items()
    zarr = optional_import("zarr")
    with suppress_warnings(UserWarning, message="Consolidated metadata"):
        root = zarr.open_group(
            resource, mode="r+", zarr_format=zarr_format, use_consolidated=False
        )
    node = _payload_group(root, key, where)
    dataset = _open_payload_group(resource, zarr_format, node)
    assert dataset is not None, "a scanned patch's group holds its payload"
    with dataset:
        name = next(iter(_payloads(dataset)))  # siblings are refused below
        variables = dataset.variables.items()
        others = sorted(
            x for x, v in variables if dim in v.dims and x not in (name, dim)
        )
    if others:
        msg = f"{key!r} shares {dim!r} with {others}; alone, they would fall short."
        raise NotImplementedError(msg)
    payload, labels = node[name], node[dim]
    scaled = {"scale_factor", "add_offset"}
    stored = (payload, labels)
    if payload.dtype != patch.data.dtype or any(scaled & set(x.attrs) for x in stored):
        msg = f"{key!r} is stored encoded (scaled, or as another dtype); not extended."
        raise NotImplementedError(msg)
    axis, start = patch.dims.index(dim), labels.shape[0]
    stop = start + patch.shape[axis]
    shape = tuple(stop if i == axis else x for i, x in enumerate(payload.shape))
    region = tuple(
        slice(start, stop) if i == axis else slice(None) for i in range(len(shape))
    )
    encoded = _encoded(patch.get_array(dim), labels)
    fields = _range_fields(joined)  # anything which may fail, before writing
    _grown(labels, (stop,))[start:stop] = encoded
    _grown(payload, shape)[region] = patch.data
    # A description of another length than the labels is not believed, so
    # a reader between these two writes reads the labels.
    labels.resize((stop,))
    kept = {k: v for k, v in labels.attrs.items() if not k.startswith(COORD_PREFIX)}
    described = {COORD_PREFIX + k: v for k, v in (fields or {}).items()}
    labels.attrs.put(kept | described)  # one write: the new length, or none
    payload.resize(shape)
    return True


def _consolidate(resource: UPath, zarr_format: int):
    """Consolidate a store's metadata, renewing what its index watches."""
    zarr = optional_import("zarr")
    with suppress_warnings(UserWarning, message="Consolidated metadata"):
        zarr.consolidate_metadata(resource, zarr_format=zarr_format)


def _marked_zarr_format(path: UPath) -> str | None:
    """Return the zarr format a directory's marker files name, else None."""
    return next((v for v, name in _MARKERS.items() if (path / name).is_file()), None)


def _store_format(path: UPath) -> int | None:
    """Return the zarr format a store's marker names, else None (sniff it)."""
    version = _marked_zarr_format(path)
    return None if version is None else int(version)


class ZarrV3(FiberIO):
    """Zarr IO (zarr format 3); each payload in any group is a patch."""

    name = "ZARR"
    version = "3"
    input_type = "directory"
    preferred_extensions = ("zarr",)
    multi_patch_write = True

    def unit_members(self, path) -> list | None:
        """
        Return a store's root metadata files, which every dascore write renews.

        Chunks rewritten in place, or groups and arrays added or edited
        without consolidating, go unseen: stating every group's metadata
        files would cost a store of thousands of arrays as many requests.
        """
        if _marked_zarr_format(path) is None:
            return None
        names = ("zarr.json", ".zgroup", ".zmetadata", ".zattrs")
        return [x for x in (path / n for n in names) if x.is_file()]

    def get_version(self, resource: UPath, **kwargs) -> str | None:
        """Return the zarr format of a zarr store with a dimensioned payload."""
        # Markers first: this is probed on every directory a scan meets.
        version = _marked_zarr_format(resource)
        if version is None:
            return None
        try:
            for _ in _patches(resource, int(version)):
                return version
        except MissingOptionalDependencyError:
            # Claimed, so reading names the missing package and a scan
            # does not walk into the store's chunks.
            return version
        return None

    def get_metadata(
        self, resource: UPath, *, snap: snap_type = ()
    ) -> list[dc.PatchMeta]:
        """Describe each patch from its metadata and coordinates."""
        where = str(resource)
        return [
            span
            for key, dataset in _patches(resource, _store_format(resource))
            for meta in dataset_to_patch_meta(dataset, snap, key=key)
            for span in _span_metas(meta, dataset, key, where)
        ]

    def read_array(
        self, resource: UPath, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Read a window of one patch; only the chunks it touches are read."""
        where, zarr_format = str(resource), _store_format(resource)
        span = None
        if match := _SPAN_KEY.match(key):  # one span of a payload
            key, span = match["key"], (int(match["start"]), int(match["stop"]))
        if not key:  # every group is opened only when no key names a patch
            keys = [x for x, _ in _patches(resource, zarr_format)]
            key = resolve_keyed_source(dict(zip(keys, keys)), key, where)
        node = _payload_group(_open_group(resource, zarr_format), key, where)
        if (dataset := _open_payload_group(resource, zarr_format, node)) is None:
            raise PatchAttributeError(f"No patch named '{key}' in {where}.")
        with dataset:
            patches = _patch_keys(node.path, _payloads(dataset))
            payload = resolve_keyed_source(patches, key, where)
            if span is not None:
                windows = _span_windows(payload, span, windows, where)
            return read_dataset_array(payload, windows, "", where)

    def write(
        self,
        spool,
        resource: UPath,
        encoding: dict | None = None,
        append: bool | str = False,
        **kwargs,
    ):
        """
        Write a spool to a zarr store, replacing any store there by default.

        One patch is written at the store root. Several are written one
        group each, named by `get_patch_names` with "/" replaced by "_" and
        repeats suffixed ``__1``, ``__2``, ... as DASDAE does; one patch is
        held in memory at a time.

        The store is built beside ``resource`` and moved into place once
        complete, so a failed write leaves the previous store intact. On
        object storage the move is a copy, so it is not atomic.

        Parameters
        ----------
        encoding
            Passed to xarray's ``Dataset.to_zarr``, e.g.
            ``{"data": {"chunks": (100, 1000), "shards": (100, 10000)}}``;
            ``shards`` needs zarr format 3. Not used to extend a patch, whose
            arrays keep their own.
        append
            False (the default) replaces the store. True adds to it, in its
            own zarr format and without rewriting it: each patch becomes a
            new group, named as above and not yet taken. A dimension name,
            such as ``"time"``, also extends: a patch which continues a
            stored patch along it (the same kind, data units, dimensions,
            dtype and other coordinates, its labels starting one step after
            the stored ones end) extends that patch, its chunks written
            before its shape so readers never see a part written; a patch
            continuing none (after a gap, or of another kind) is added as a
            new group. One which could continue several, or which DASCore
            cannot extend whole, raises. With no store at the path, a store
            is written, as for False.

        Notes
        -----
        Adding ends by consolidating the store's metadata, which DASCore
        readers list groups from and a spool's index watches, so
        `Spool.update` sees the new content and a reader sees none of it
        before then.
        """
        if not isinstance(append, bool | str) or append == "":
            msg = f"append must be True, False or a dimension name; got {append!r}."
            raise ParameterError(msg)
        zarr = optional_import("zarr")  # xarray is required by the conversion
        spool = dc.spool([spool]) if isinstance(spool, dc.Patch) else spool
        if not len(spool):
            raise ValueError("Cannot write empty spool")
        if resource.exists() and not _marked_zarr_format(resource):
            if resource.is_file() or any(resource.iterdir()):
                msg = f"{resource} exists and is not a zarr store; not replacing it."
                raise FileExistsError(msg)
        if append and (stored := _store_format(resource)) is not None:
            if append is True:
                _add_groups(spool, resource, stored, encoding)
                _consolidate(resource, stored)
                return
            for patch in spool:
                if not _extend(patch, resource, stored, append):
                    _add_groups(dc.spool([patch]), resource, stored, encoding)
                _consolidate(resource, stored)  # the next patch sees this one
            return
        names = _group_names(spool)
        zarr_format = int(self.version)
        options = dict(zarr_format=zarr_format, encoding=encoding)
        # zarr warns that consolidated metadata is outside the format 3
        # spec; it saves a reader one metadata request per array.
        with (
            staged_path(resource) as staging,
            suppress_warnings(UserWarning, message="Consolidated metadata"),
        ):
            if len(names) <= 1:
                dataset = _cf_dataset(spool, encoding)
                dataset.to_zarr(staging, mode="w", consolidated=True, **options)
            else:
                zarr.open_group(staging, mode="w", zarr_format=zarr_format)
                names = _unique_patch_names(names)
                for name, patch in zip(names, spool, strict=True):
                    dataset = _cf_dataset(patch, encoding)
                    dataset.to_zarr(
                        staging, group=name, mode="w", consolidated=False, **options
                    )
                zarr.consolidate_metadata(staging, zarr_format=zarr_format)


class ZarrV2(ZarrV3):
    """Zarr IO (zarr format 2)."""

    version = "2"
