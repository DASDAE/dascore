"""NetCDF helper functions for DASCore IO."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import pairwise
from typing import Any, Protocol

import numpy as np

import dascore as dc
from dascore.core.coords import BaseCoord, NumericND, concat_tables
from dascore.exceptions import CoordError

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
    msg = "No suitable data variable found in NetCDF file"
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


def _get_tie_point_coord(h5file, coord_name: str, coord_len: int) -> BaseCoord | None:
    """Decode XDAS ties as grid runs without expanding the sample axis."""
    values_name = f"{coord_name}_values"
    indices_name = f"{coord_name}_indices"
    if values_name not in h5file:
        return None
    values_var = h5file[values_name]
    values = np.asarray(values_var[:])
    if indices_name in h5file:
        indices = np.asarray(h5file[indices_name][:])
        if len(values) >= 2 and len(indices) >= 2:
            if (
                len(indices) != len(values)
                or indices.dtype.kind not in "iu"
                or np.any(np.diff(indices.astype(np.int64)) <= 0)
            ):
                raise CoordError(
                    "XDAS tie indices must be distinct increasing integers "
                    "matching the values."
                )
            temporal = values.dtype.kind in "Mm"
            if temporal:
                values = (
                    dc.to_datetime64(values)
                    if values.dtype.kind == "M"
                    else dc.to_timedelta64(values)
                )
            runs = []
            zero = Fraction(0) if temporal else 0.0
            if indices[0] > 0:
                start = values[0] if temporal else float(values[0])
                runs.append(
                    NumericND.from_run(start, zero, min(int(indices[0]), coord_len))
                )
            for i, (left, right) in enumerate(pairwise(indices)):
                length = int(right) - int(left)
                if temporal:
                    ticks = int(values[i + 1].view("i8")) - int(values[i].view("i8"))
                    step = Fraction(ticks, length * 1_000_000_000)
                    start = values[i]
                else:
                    start = float(values[i])
                    step = (float(values[i + 1]) - start) / length
                # Each shared tie belongs to the following run. Only the
                # last interval owns its right endpoint.
                lo = max(0, int(left))
                hi = min(coord_len, int(right) + (i == len(indices) - 2))
                if hi > lo:
                    run = NumericND.from_run(start, step, length + 1)
                    runs.append(run._sliced(lo - int(left), 1, hi - lo))
            tail = max(int(indices[-1]) + 1, 0)
            if tail < coord_len:
                start = values[-1] if temporal else float(values[-1])
                runs.append(NumericND.from_run(start, zero, coord_len - tail))
            if not runs:
                return dc.get_coord(data=values[:0])
            return concat_tables(*runs)
    return dc.get_coord(data=values)


def _get_dim_coord(h5file, coord_name: str, coord_len: int) -> BaseCoord | np.ndarray:
    """Return one dimension coordinate for a coord-less payload variable."""
    tied_values = _get_tie_point_coord(h5file, coord_name, coord_len)
    if tied_values is not None:
        return tied_values
    if coord_name in h5file:
        return h5file[coord_name][:]
    return dc.get_coord(start=0, step=1, shape=(coord_len,))


def get_coord_manager_for_coordless_data_var(
    h5file, dims: tuple[str, ...], shape: tuple[int, ...]
):
    """Build dimension coordinates for payloads xarray exposes without coords."""
    coords = {
        dim: _get_dim_coord(h5file, dim, size)
        for dim, size in zip(dims, shape, strict=True)
    }
    return dc.get_coord_manager(coords=coords, dims=dims)
