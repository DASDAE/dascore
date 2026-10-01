"""Zarr IO as CF-on-zarr, read and written through xarray."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.compat import UPath
from dascore.constants import snap_type, windows_type
from dascore.exceptions import MissingOptionalDependencyError
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
from dascore.utils.patch import _unique_patch_names

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


def _patch_groups(path: UPath) -> list[str]:
    """Return a store's child groups, one per patch; empty for a root payload."""
    zarr = optional_import("zarr")
    with suppress_warnings(UserWarning, message="Consolidated metadata"):
        return sorted(zarr.open_group(str(path), mode="r").group_keys())


def _marked_zarr_format(path: UPath) -> str | None:
    """Return the zarr format a directory's marker files name, else None."""
    return next((v for v, name in _MARKERS.items() if (path / name).is_file()), None)


class ZarrV3(FiberIO):
    """
    Zarr IO (zarr format 3); the payload is ``data``, else the only variable.

    A single patch is stored at the store root, several as one group each,
    named as DASDAE names its patch groups.
    """

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
            groups = _patch_groups(resource)
            dataset = _open_zarr(resource, groups[0] if groups else None)
        except MissingOptionalDependencyError:
            # Claimed, so reading names the missing package and a scan
            # does not walk into the store's chunks.
            return version
        with dataset:
            name = get_xarray_data_var_name(dataset)
            return version if dataset[name].dims else None

    def get_metadata(
        self, resource: UPath, *, snap: snap_type = True
    ) -> list[dc.PatchMeta]:
        """Describe each patch from its metadata and coordinates."""
        out = []
        for group in _patch_groups(resource) or [None]:
            with _open_zarr(resource, group) as dataset:
                out.extend(dataset_to_patch_meta(dataset, snap, key=group))
        return out

    def read_array(
        self, resource: UPath, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Read a window of one patch; only the chunks it touches are read."""
        where = str(resource)
        group = None
        if groups := _patch_groups(resource):
            group = resolve_keyed_source({x: x for x in groups}, key, where)
            key = ""  # the group holds one payload
        with _open_zarr(resource, group) as dataset:
            return read_dataset_array(dataset, windows, key, where)

    def write(self, spool, resource: UPath, encoding: dict | None = None, **kwargs):
        """
        Write a spool to a zarr store, replacing any store there.

        One patch is written at the store root; several are written one
        group each, named by `get_patch_names` with repeats suffixed.

        The store is built beside ``resource`` and moved into place once
        complete, so a failed write leaves the previous store intact.

        Parameters
        ----------
        encoding
            Passed to xarray's ``Dataset.to_zarr``, e.g.
            ``{"data": {"chunks": (100, 1000), "shards": (100, 10000)}}``;
            ``shards`` needs zarr format 3. Applied to every patch's group.
        """
        xr = optional_import("xarray")
        optional_import("zarr")
        patches = [spool] if isinstance(spool, dc.Patch) else list(spool)
        if len(patches) > 1:
            names = _unique_patch_names(patches)
            datasets = (spool_to_cf_dataset(x) for x in patches)
            data = xr.DataTree.from_dict(dict(zip(names, datasets, strict=True)))
            if encoding:
                encoding = {f"/{x}": encoding for x in names}
        else:
            data = spool_to_cf_dataset(patches)
        if resource.exists() and not _marked_zarr_format(resource):
            if resource.is_file() or any(resource.iterdir()):
                msg = f"{resource} exists and is not a zarr store; not replacing it."
                raise FileExistsError(msg)
        # zarr warns that consolidated metadata is outside the format 3
        # spec; it saves a reader one metadata request per array.
        with (
            staged_path(resource) as staging,
            suppress_warnings(UserWarning, message="Consolidated metadata"),
        ):
            data.to_zarr(
                staging,
                mode="w",
                zarr_format=int(self.version),
                consolidated=True,
                encoding=encoding,
            )


class ZarrV2(ZarrV3):
    """Zarr IO (zarr format 2)."""

    version = "2"
