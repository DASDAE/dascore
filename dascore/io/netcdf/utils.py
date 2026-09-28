"""Helpers for the xarray-based NetCDF and Zarr readers and writers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.core.source import ArraySource
from dascore.io.utils import resolve_keyed_source, should_snap, windows_to_slices
from dascore.units import get_quantity_str
from dascore.xarray import patch_to_xarray

XDAS_PAYLOAD_VARIABLE = "__values__"


def get_xarray_data_var_name(dataset) -> str | None:
    """Return the main xarray data variable name."""
    if "data" in dataset.data_vars:
        return "data"
    # XDAS-style files can surface the primary payload under a None key while
    # exposing coordinate helper arrays as additional data variables.
    if None in dataset.data_vars:
        return None
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


def _get_tie_point_coord(h5file, coord_name: str, coord_len: int) -> np.ndarray | None:
    """Decode one XDAS-style tie-point coordinate array."""
    values_name = f"{coord_name}_values"
    indices_name = f"{coord_name}_indices"
    if values_name not in h5file:
        return None
    values_var = h5file[values_name]
    values = values_var[:]
    if indices_name in h5file:
        indices = h5file[indices_name][:]
        if len(values) >= 2 and len(indices) >= 2:
            sample_index = np.arange(coord_len, dtype=np.float64)
            if np.issubdtype(np.asarray(values).dtype, np.datetime64):
                value_ns = values.astype("datetime64[ns]").astype(np.int64)
                values = np.interp(sample_index, indices, value_ns).astype(np.int64)
                values = values.astype("datetime64[ns]")
            else:
                values = np.interp(sample_index, indices, values)
    return values


def _get_dim_coord(h5file, coord_name: str, coord_len: int) -> np.ndarray:
    """Return one dimension coordinate for a coord-less payload variable."""
    tied_values = _get_tie_point_coord(h5file, coord_name, coord_len)
    if tied_values is not None:
        return tied_values
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


def _get_patch_key(data_var_name):
    """Normalize the selected xarray payload name to a patch id."""
    return XDAS_PAYLOAD_VARIABLE if data_var_name is None else data_var_name


def dataset_to_patch_meta(dataset, snap: snap_type = True) -> list[dc.PatchMeta]:
    """Describe the payload of an open xarray dataset without reading it."""
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
        source=ArraySource(key=_get_patch_key(data_var_name)),
    )
    return [meta]


def read_dataset_array(dataset, windows: windows_type, key: str, where: str):
    """
    Slice the payload variable of an open xarray dataset.

    The selection goes through xarray rather than the stored array so CF
    decoding (scaling, offsets, fill values) applies exactly as in `read`.
    """
    data_var_name = get_xarray_data_var_name(dataset)
    patch_key = _get_patch_key(data_var_name)
    resolve_keyed_source({patch_key: data_var_name}, key, where=where)
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
