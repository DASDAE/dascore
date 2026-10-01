"""Zarr IO as CF-on-zarr, read and written through xarray."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.compat import UPath
from dascore.constants import snap_type, windows_type
from dascore.exceptions import MissingOptionalDependencyError, PatchAttributeError
from dascore.io import FiberIO
from dascore.io.netcdf.utils import (
    dataset_to_patch_meta,
    get_xarray_data_var_name,
    read_dataset_array,
    spool_to_cf_dataset,
)
from dascore.io.utils import resolve_keyed_source
from dascore.utils.io import staged_path
from dascore.utils.misc import optional_import, suppress_warnings
from dascore.utils.patch import _unique_patch_names, get_patch_names

# The file which marks a directory as a zarr group, by zarr format.
_MARKERS = {"3": "zarr.json", "2": ".zgroup"}


def _open_zarr(path: UPath, group: str | None = None):
    """Open a store (or one group) as a lazy dataset; only coordinates are read."""
    xr = optional_import("xarray")
    optional_import("zarr")
    # A store without consolidated metadata still opens, only slower.
    with suppress_warnings(RuntimeWarning, message="Failed to open Zarr store"):
        return xr.open_dataset(
            path, engine="zarr", group=group, chunks=None, cache=False
        )


def _has_payload(dataset) -> bool:
    """Return True if an open dataset holds a dimensioned payload."""
    try:
        name = get_xarray_data_var_name(dataset)
    except ValueError:
        return False
    return bool(dataset[name].dims)


def _patch_datasets(path: UPath):
    """
    Yield ``(group, open dataset)`` for each patch of a store.

    A root payload is the one patch (group None) and child groups are then
    ignored; otherwise each child group holding a payload is a patch.
    """
    with _open_zarr(path) as root:
        if _has_payload(root):
            yield None, root
            return
    zarr = optional_import("zarr")
    with suppress_warnings(UserWarning, message="Consolidated metadata"):
        names = sorted(zarr.open_group(path, mode="r").group_keys())
    for name in names:
        try:
            dataset = _open_zarr(path, name)
        except KeyError:  # raw zarr arrays with no dimension names
            continue
        with dataset:
            if _has_payload(dataset):
                yield name, dataset


def _marked_zarr_format(path: UPath) -> str | None:
    """Return the zarr format a directory's marker files name, else None."""
    return next((v for v, name in _MARKERS.items() if (path / name).is_file()), None)


class ZarrV3(FiberIO):
    """Zarr IO (zarr format 3); the payload is ``data``, else the only variable."""

    name = "ZARR"
    version = "3"
    input_type = "directory"
    preferred_extensions = ("zarr",)
    multi_patch_write = True

    def unit_members(self, path) -> list | None:
        """
        Return a store's root metadata files, which every dascore write renews.

        Chunks rewritten in place, or array metadata edited without
        consolidating, go unseen.
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
            for _ in _patch_datasets(resource):
                return version
        except MissingOptionalDependencyError:
            # Claimed, so reading names the missing package and a scan
            # does not walk into the store's chunks.
            return version
        return None

    def get_metadata(
        self, resource: UPath, *, snap: snap_type = True
    ) -> list[dc.PatchMeta]:
        """Describe each patch from its metadata and coordinates."""
        return [
            meta
            for group, dataset in _patch_datasets(resource)
            for meta in dataset_to_patch_meta(dataset, snap, key=group)
        ]

    def read_array(
        self, resource: UPath, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Read a window of one patch; only the chunks it touches are read."""
        where = str(resource)
        with _open_zarr(resource) as root:
            if _has_payload(root):
                return read_dataset_array(root, windows, key, where)
        if not key:  # groups are listed only when no key names one
            groups = [x for x, _ in _patch_datasets(resource)]
            key = resolve_keyed_source(dict(zip(groups, groups)), key, where)
        try:
            dataset = _open_zarr(resource, key)
        except (KeyError, FileNotFoundError) as exc:
            msg = f"No patch named '{key}' in {where}."
            raise PatchAttributeError(msg) from exc
        with dataset:
            return read_dataset_array(dataset, windows, "", where)

    def write(self, spool, resource: UPath, encoding: dict | None = None, **kwargs):
        """
        Write a spool to a zarr store, replacing any store there.

        One patch is written at the store root. Several are written one
        group each, named by `get_patch_names` with "/" replaced by "_" and
        repeats suffixed ``__1``, ``__2``, ... as DASDAE does; one patch is
        held in memory at a time.

        The store is built beside ``resource`` and moved into place once
        complete, so a failed write leaves the previous store intact.

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
                dataset = spool_to_cf_dataset(spool)
                dataset.to_zarr(staging, mode="w", consolidated=True, **options)
            else:
                zarr.open_group(staging, mode="w", zarr_format=zarr_format)
                names = _unique_patch_names(names)
                for name, patch in zip(names, spool, strict=True):
                    dataset = spool_to_cf_dataset(patch)
                    dataset.to_zarr(
                        staging, group=name, mode="w", consolidated=False, **options
                    )
                zarr.consolidate_metadata(staging, zarr_format=zarr_format)


class ZarrV2(ZarrV3):
    """Zarr IO (zarr format 2)."""

    version = "2"
