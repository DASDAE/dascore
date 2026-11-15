"""Helpers for the xarray-based NetCDF and Zarr readers and writers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.core.coords import get_coord
from dascore.core.source import ArraySource
from dascore.exceptions import InvalidFiberFileError
from dascore.io.dasdae.utils import read_grid_fields
from dascore.io.utils import (
    resolve_keyed_source,
    should_snap,
    stored_coord,
    windows_to_slices,
    written_step,
)
from dascore.utils.time import to_int
from dascore.xarray import patch_to_xarray
from dascore.xarray.patch import _cf_attrs, _cf_units

XDAS_PAYLOAD_VARIABLE = "__values__"
# The attribute holding the step of labels which are not one grid, so they
# read back as their segments (in nanoseconds for times).
STEP_ATTR = "dascore_step"
# The prefix of the attributes describing labels which are one grid, one per
# field of a DASDAE ``CoordRange`` node (``_dascore_coord_start``, ...), so a
# scan rebuilds them without reading them.
COORD_PREFIX = "_dascore_coord_"


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


def get_shared_payload_names(dataset) -> list[str]:
    """
    Return the payloads of a dataset with no single payload variable.

    They are the data variables on the dimensions of the largest variable
    of the highest rank, in any order. Scalars, lower-rank helpers, smaller
    variables on other dimensions and CF ``bounds`` tables are left out.
    """
    bounds = {x.attrs.get("bounds") for x in dataset.variables.values()}
    variables = {k: v for k, v in dataset.data_vars.items() if k not in bounds}
    if not (rank := max((x.ndim for x in variables.values()), default=0)):
        msg = "No suitable data variable found in the dataset"
        raise ValueError(msg)
    tops = [x for x in variables.values() if x.ndim == rank]
    dims = set(max(tops, key=lambda x: x.size).dims)
    return [name for name, var in variables.items() if set(var.dims) == dims]


def payload_datasets(dataset) -> dict:
    """Map each payload's name to a dataset holding it as the only payload."""
    try:
        return {get_xarray_data_var_name(dataset): dataset}
    except ValueError:
        names = get_shared_payload_names(dataset)
    # Siblings differ only by name, so it tags them unless their own tags
    # already tell them apart; otherwise chunking would merge them.
    tags = {x: dataset[x].attrs.get("tag") or x for x in names}
    if len(set(tags.values())) < len(tags):
        tags = {x: x for x in names}
    out = {}
    for name in names:
        sub = dataset[[name]]
        if len(names) > 1:
            sub[name] = sub[name].assign_attrs(tag=tags[name])
        out[name] = sub
    return out


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


def stated_range(variable) -> dict | None:
    """
    Return the grid a variable's (xarray's or zarr's) attrs describe, else None.

    A description of another length than the labels is stale (they were
    appended to by another tool) and is ignored, so the labels are read.
    """
    attrs = variable.attrs.items()
    size = len(COORD_PREFIX)
    fields = {k[size:]: v for k, v in attrs if k.startswith(COORD_PREFIX)}
    if not fields or (fields.get("length"),) != tuple(variable.shape):
        return None
    return fields


def get_scan_coord(coord, snap: snap_type = False, where=""):
    """Return a coordinate for scanning, on a grid if `snap` names it."""
    if (fields := stated_range(coord)) is not None:
        # Already a grid, so snapping changes nothing. A time's "units" are
        # its CF encoding, left undecoded so its labels are never read.
        time_like = np.dtype(fields["dtype"]).kind in "mM"
        units = None if time_like else _cf_units(coord.attrs.get("units"), coord.name)
        return read_grid_fields(fields, units)
    values = coord.values
    if values.dtype == object:
        # e.g. times on a non-standard calendar, decoded as cftime objects
        msg = f"Coordinate {coord.name!r} holds objects DASCore cannot store."
        raise InvalidFiberFileError(msg)
    if np.ndim(values) != 1:
        return values
    units = _cf_units(coord.attrs.get("units"), coord.name)
    step = coord.attrs.get(STEP_ATTR)
    if step is not None and not should_snap(snap, coord.name):
        # labels written with their step: their segments read back
        if values.dtype.kind in "mM":
            step = np.timedelta64(int(step), "ns")
        return get_coord(data=values, step=step, units=units)
    return stored_coord(values, snap, name=coord.name, where=where, units=units)


def _get_patch_key(data_var_name):
    """Normalize the selected xarray payload name to a patch id."""
    return XDAS_PAYLOAD_VARIABLE if data_var_name is None else data_var_name


def dataset_to_patch_meta(
    dataset, snap: snap_type = (), key: str | None = None
) -> list[dc.PatchMeta]:
    """Describe an open dataset's payload without reading it; ``key`` sets its key."""
    data_var_name = get_xarray_data_var_name(dataset)
    data_array = dataset[data_var_name]
    dims, shape = data_array.dims, data_array.shape
    where = dataset.encoding.get("source", "")
    coords = {
        name: (coord.dims, get_scan_coord(coord, snap if name in dims else (), where))
        for name, coord in data_array.coords.items()
    }
    # A dimension the dataset gives no coordinate is rebuilt on its own:
    # from XDAS tie points, else sample numbers; those of two or more are a
    # range never built as an array (a store allocated ahead of its data
    # can hold billions), and shorter ones keep the labels they always had.
    for dim, size in zip(dims, shape, strict=True):
        if dim in coords:
            continue
        values = _get_tie_point_coord(dataset, dim, size)
        if values is None and size > 1:
            coords[dim] = (dim, get_coord(start=0, stop=size, step=1))
            continue
        values = np.arange(size) if values is None else values
        coords[dim] = (dim, stored_coord(values, snap, name=dim, where=where))
    meta = dc.PatchMeta(
        attrs=_cf_attrs(data_array.attrs, data_var_name),
        coords=dc.get_coord_manager(coords=coords, dims=dims),
        dims=dims,
        dtype=str(data_array.dtype),
        source=ArraySource(key=key or _get_patch_key(data_var_name)),
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
    patch = patches[0]
    dataset = patch_to_xarray(patch).rename("data").to_dataset()
    why = "NetCDF and Zarr, which store labels with one step"
    for name, coord in patch.coords.coord_map.items():
        if (step := written_step(name, coord, why)) is not None:
            timed = isinstance(step, np.timedelta64)
            dataset[name].attrs[STEP_ATTR] = to_int(step) if timed else step
    dataset.attrs["Conventions"] = "CF-1.8"
    return dataset
