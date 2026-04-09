"""NetCDF helper functions for DASCore IO."""

from __future__ import annotations

import h5py
import numpy as np

import dascore as dc

XDAS_PAYLOAD_VARIABLE = "__values__"


def get_xarray_data_var_name(dataset) -> str:
    """Return the main xarray data variable name."""
    if "data" in dataset.data_vars:
        return "data"
    # XDAS-style files can surface the primary payload under a None key while
    # exposing coordinate helper arrays as additional data variables.
    if None in dataset.data_vars:
        return None
    if len(dataset.data_vars) == 1:
        return next(iter(dataset.data_vars))
    msg = "No suitable data variable found in NetCDF file"
    raise ValueError(msg)


def parse_cf_version(cf_version: str) -> tuple[int, int]:
    """Parse a CF version string into comparable major/minor integers."""
    parts = cf_version.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    return major, minor


def is_netcdf4_file(h5file: h5py.File) -> bool:
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


def get_cf_version(h5file: h5py.File) -> str | None:
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


def get_coord_manager_for_coordless_data_var(
    h5file, dims: tuple[str, ...], shape: tuple[int, ...]
):
    """Build dimension coordinates for payloads xarray exposes without coords."""
    coords = {
        dim: _get_dim_coord(h5file, dim, size)
        for dim, size in zip(dims, shape, strict=True)
    }
    return dc.get_coord_manager(coords=coords, dims=dims)
