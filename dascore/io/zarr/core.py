"""Zarr IO as CF-on-zarr, read and written through xarray."""

from __future__ import annotations

import importlib.util

import numpy as np

import dascore as dc
from dascore.compat import UPath
from dascore.constants import snap_type, windows_type
from dascore.io import FiberIO
from dascore.io.netcdf.utils import (
    dataset_to_patch_meta,
    get_xarray_data_var_name,
    read_dataset_array,
    spool_to_cf_dataset,
)
from dascore.utils.misc import optional_import, suppress_warnings

# The file which marks a directory as a zarr group, by zarr format.
_MARKERS = {"3": "zarr.json", "2": ".zgroup"}


def _open_zarr(path: UPath):
    """Open a zarr store as a lazy xarray dataset; no chunk is read."""
    xr = optional_import("xarray")
    optional_import("zarr")
    # A store without consolidated metadata still opens, only slower.
    with suppress_warnings(RuntimeWarning, message="Failed to open Zarr store"):
        return xr.open_dataset(path, engine="zarr", chunks=None, cache=False)


def _marked_zarr_format(path: UPath) -> str | None:
    """Return the zarr format a directory's marker files name, else None."""
    return next((v for v, name in _MARKERS.items() if (path / name).is_file()), None)


def _has_zarr_deps() -> bool:
    """Return True when xarray and zarr are both importable."""
    return all(importlib.util.find_spec(x) for x in ("xarray", "zarr"))


class ZarrV3(FiberIO):
    """Zarr IO (zarr format 3); the payload is the ``data`` variable."""

    name = "ZARR"
    version = "3"
    input_type = "directory"
    preferred_extensions = ("zarr",)

    def get_version(self, resource: UPath, **kwargs) -> str | None:
        """Return the zarr format when the directory holds a dascore-readable store."""
        # Markers first: this is probed on every directory a scan meets.
        version = _marked_zarr_format(resource)
        if version is None or not _has_zarr_deps():
            return None
        with _open_zarr(resource) as dataset:
            name = get_xarray_data_var_name(dataset)
            return version if dataset[name].dims else None

    def get_metadata(
        self, resource: UPath, *, snap: snap_type = True
    ) -> list[dc.PatchMeta]:
        """Describe the store's payload from its metadata and coordinates."""
        with _open_zarr(resource) as dataset:
            return dataset_to_patch_meta(dataset, snap)

    def read_array(
        self, resource: UPath, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Read a window of the payload; only the chunks it touches are read."""
        with _open_zarr(resource) as dataset:
            return read_dataset_array(dataset, windows, key, str(resource))

    def write(self, spool, resource: UPath, encoding: dict | None = None, **kwargs):
        """
        Write a single-patch spool to a zarr store through xarray.

        Parameters
        ----------
        encoding
            Passed to xarray's ``Dataset.to_zarr``, e.g.
            ``{"data": {"chunks": (100, 1000), "shards": (100, 10000)}}``.
        """
        optional_import("zarr")  # xarray is required by the conversion
        dataset = spool_to_cf_dataset(spool, "Zarr")
        # Consolidated metadata lets a reader open the store in one request.
        with suppress_warnings(UserWarning, message="Consolidated metadata"):
            dataset.to_zarr(
                resource,
                mode="w",
                zarr_format=int(self.version),
                consolidated=True,
                encoding=encoding,
            )


class ZarrV2(ZarrV3):
    """Zarr IO (zarr format 2)."""

    version = "2"
