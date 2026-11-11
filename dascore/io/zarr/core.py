"""Zarr IO as CF-on-zarr, read and written through xarray."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.compat import UPath
from dascore.constants import snap_type, windows_type
from dascore.exceptions import MissingOptionalDependencyError, PatchAttributeError
from dascore.io import FiberIO
from dascore.io.dasdae.utils import grid_fields, read_grid_fields
from dascore.io.netcdf.utils import (
    COORD_PREFIX,
    dataset_to_patch_meta,
    payload_datasets,
    read_dataset_array,
    spool_to_cf_dataset,
    stated_range,
)
from dascore.io.utils import resolve_keyed_source
from dascore.utils.io import staged_path
from dascore.utils.misc import optional_import, suppress_warnings
from dascore.utils.patch import _unique_patch_names, get_patch_names

# The file which marks a directory as a zarr group, by zarr format.
_MARKERS = {"3": "zarr.json", "2": ".zgroup"}


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

    None where it has no grid (holes, jitter, partial), or where JSON (no
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


def _cf_dataset(spool, encoding: dict | None):
    """
    Return the only patch of a spool as a CF dataset, describing its grids.

    The labels are written too, so any CF tool reads them. A coordinate
    whose ``encoding`` changes the labels stored (a dtype, units or
    scaling) is not described, so its stored labels are read.
    """
    patches = [spool] if isinstance(spool, dc.Patch) else list(spool)
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
    except KeyError:  # raw zarr arrays with no dimension names
        return None


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
        return [
            meta
            for key, dataset in _patches(resource, _store_format(resource))
            for meta in dataset_to_patch_meta(dataset, snap, key=key)
        ]

    def read_array(
        self, resource: UPath, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Read a window of one patch; only the chunks it touches are read."""
        where, zarr_format = str(resource), _store_format(resource)
        if not key:  # every group is opened only when no key names a patch
            keys = [x for x, _ in _patches(resource, zarr_format)]
            key = resolve_keyed_source(dict(zip(keys, keys)), key, where)
        root = _open_group(resource, zarr_format)
        zarr = optional_import("zarr")
        try:  # a key names a group holding one payload, else a payload array
            node = root[key]
        except KeyError as exc:
            raise PatchAttributeError(f"No patch named '{key}' in {where}.") from exc
        if not isinstance(node, zarr.Group):
            group = key.rpartition("/")[0]
            node = root[group] if group else root
        if (dataset := _open_payload_group(resource, zarr_format, node)) is None:
            raise PatchAttributeError(f"No patch named '{key}' in {where}.")
        with dataset:
            patches = _patch_keys(node.path, _payloads(dataset))
            payload = resolve_keyed_source(patches, key, where)
            return read_dataset_array(payload, windows, "", where)

    def write(self, spool, resource: UPath, encoding: dict | None = None, **kwargs):
        """
        Write a spool to a zarr store, replacing any store there.

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
            ``shards`` needs zarr format 3.
        """
        zarr = optional_import("zarr")  # xarray is required by the conversion
        spool = dc.spool([spool]) if isinstance(spool, dc.Patch) else spool
        names = get_patch_names(spool).str.replace("/", "_", regex=False)
        names = names.mask(names == "", "patch")  # "" would name the root
        if resource.exists() and not _marked_zarr_format(resource):
            if resource.is_file() or any(resource.iterdir()):
                msg = f"{resource} exists and is not a zarr store; not replacing it."
                raise FileExistsError(msg)
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
