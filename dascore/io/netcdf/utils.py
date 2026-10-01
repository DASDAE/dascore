"""Helpers for the xarray-based NetCDF and Zarr readers and writers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import h5py
import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.core.source import ArraySource
from dascore.io.utils import resolve_keyed_source, should_snap, windows_to_slices
from dascore.units import get_quantity_str
from dascore.xarray import patch_to_xarray

XDAS_PAYLOAD_VARIABLE = "__values__"
XDAS_MAPPINGS = ("coordinate_interpolation", "coordinate_sampling")
XDAS_TILING = "__tiling__"


def get_xarray_data_var_name(dataset) -> str:
    """Return the main xarray data variable name."""
    if "data" in dataset.data_vars:
        return "data"
    if len(dataset.data_vars) == 1:
        return next(iter(dataset.data_vars))
    msg = "No suitable data variable found in the dataset"
    raise ValueError(msg)


def parse_cf_version(cf_version: str) -> tuple[int, int]:
    """Parse a CF version string into comparable major/minor integers."""
    parts = cf_version.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    return major, minor


class _HasAttrs(Protocol):
    """Anything carrying HDF5-style attrs.

    The two checks below only read `attrs`, and they are handed the managed
    handle a FiberIO caster produces rather than an `h5py.File` proper.
    """

    @property
    def attrs(self) -> Mapping[str, Any]: ...


def is_netcdf4_file(h5file: _HasAttrs) -> bool:
    """Return True when an HDF5 file exposes strong NetCDF/CF markers."""
    try:
        if "_NCProperties" in h5file.attrs:
            return True
        conventions = h5file.attrs.get("Conventions", "")
        if isinstance(conventions, bytes):
            conventions = conventions.decode("utf-8", errors="ignore")
        return bool(conventions and "CF" in conventions)
    except (AttributeError, KeyError):
        return False


def get_cf_version(h5file: _HasAttrs) -> str | None:
    """Extract the CF convention version string from a NetCDF file."""
    conventions = h5file.attrs.get("Conventions", "")
    if isinstance(conventions, bytes):
        conventions = conventions.decode("utf-8", errors="ignore")
    if "CF-" in conventions:
        return conventions.split("CF-", 1)[1].split()[0].rstrip(",;")
    if conventions.startswith("CF "):
        return conventions.split()[1].rstrip(",;")
    return None


def _is_xdas_marked(name, node):
    """Return True for a dataset only this layout writes, else None."""
    if isinstance(node, h5py.Dataset) and (
        name.rsplit("/", 1)[-1] == XDAS_PAYLOAD_VARIABLE
        or any(x in node.attrs for x in (*XDAS_MAPPINGS, XDAS_TILING))
    ):
        return True
    return None  # visititems stops at the first value which is not None


def is_xdas_file(h5) -> bool:
    """Return True if a NetCDF-4 file holds an XDAS tie-point or unnamed signal."""
    return is_netcdf4_file(h5) and bool(h5.visititems(_is_xdas_marked))


def _get_dim_coord(h5file, coord_name: str, coord_len: int) -> np.ndarray:
    """Return one dimension coordinate for a coord-less payload variable."""
    if coord_name in h5file:
        return h5file[coord_name][:]
    return np.arange(coord_len)


def get_scan_coord(coord, snap=True):
    """Return a coordinate for scanning; snap only controls exactness."""
    values = coord.values
    if np.ndim(values) != 1:
        return values
    units = coord.attrs.get("units")
    return dc.core.get_coord(data=values, units=units, snap=snap)


def dataset_to_patch_meta(
    dataset, snap: snap_type = True, key: str | None = None
) -> list[dc.PatchMeta]:
    """Describe an open dataset's payload without reading it; ``key`` sets its key."""
    data_var_name = get_xarray_data_var_name(dataset)
    data_array = dataset[data_var_name]
    dims, shape = data_array.dims, data_array.shape
    coords = {
        name: (coord.dims, get_scan_coord(coord, snap=should_snap(snap, name)))
        for name, coord in data_array.coords.items()
    }
    # A dimension the dataset gives no coordinate is rebuilt on its own.
    for dim, size in zip(dims, shape, strict=True):
        coords.setdefault(dim, (dim, _get_dim_coord(dataset, dim, size)))
    meta = dc.PatchMeta(
        attrs=dict(data_array.attrs),
        coords=dc.get_coord_manager(coords=coords, dims=dims),
        dims=dims,
        dtype=str(data_array.dtype),
        source=ArraySource(key=key or data_var_name),
    )
    return [meta]


def read_dataset_array(dataset, windows: windows_type, key: str, where: str):
    """
    Slice the payload variable of an open xarray dataset.

    The selection goes through xarray rather than the stored array so CF
    decoding (scaling, offsets, fill values) applies exactly as in `read`.
    """
    data_var_name = get_xarray_data_var_name(dataset)
    resolve_keyed_source({data_var_name: data_var_name}, key, where=where)
    data_array = dataset[data_var_name]
    return data_array[windows_to_slices(windows, data_array.shape)].to_numpy()


def spool_to_cf_dataset(spool):
    """Return the only patch of a spool as a CF dataset with a ``data`` payload."""
    patches = [spool] if isinstance(spool, dc.Patch) else list(spool)
    if len(patches) == 0:
        msg = "Cannot write empty spool"
        raise ValueError(msg)
    if len(patches) > 1:
        msg = "Multi-patch spools not yet supported for this format"
        raise NotImplementedError(msg)
    dataset = patch_to_xarray(patches[0]).rename("data").to_dataset()
    attrs = dataset["data"].attrs
    if "data_units" in attrs:  # a pint quantity; stores hold its string
        attrs["data_units"] = get_quantity_str(attrs["data_units"])
    dataset.attrs["Conventions"] = "CF-1.8"
    return dataset
