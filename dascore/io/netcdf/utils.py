"""Utilities for NetCDF IO with CF conventions support.

See https://cfconventions.org/ for the Climate and Forecast metadata standard.
"""

from __future__ import annotations

import datetime
from functools import cache
from typing import TYPE_CHECKING

import h5py
import numpy as np

import dascore as dc
from dascore.exceptions import MissingOptionalDependencyError
from dascore.utils.misc import optional_import
from dascore.utils.time import to_datetime64, to_float

if TYPE_CHECKING:
    from dascore.core.attrs import PatchAttrs
    from dascore.core.coordmanager import CoordManager

# CF-compliant time reference (Unix epoch is standard)
CF_TIME_REFERENCE = "seconds since 1970-01-01 00:00:00"
CF_CALENDAR = "proleptic_gregorian"
XDAS_PAYLOAD_VARIABLE = "__values__"

# CF standard names for DAS data types
CF_STANDARD_NAMES = {
    "strain": "strain",
    "strain_rate": "strain_rate",
    "velocity": "velocity",
    "acceleration": "acceleration",
    "temperature": "air_temperature",
    "pressure": "air_pressure",
    "acoustic": "acoustic_signal",
}


def coord_attrs(name: str, coord) -> dict[str, str]:
    """Return CF-ish coordinate attrs for xarray-backed NetCDF output."""
    # Keep time explicit because CF consumers treat it as a special semantic
    # axis, not just another coordinate with datetime-like values.
    if name == "time":
        return {
            "standard_name": "time",
            "long_name": "Time",
            "axis": "T",
        }
    return {
        "long_name": name.replace("_", " ").title(),
        "standard_name": name.lower(),
        "units": str(coord.units.units) if coord.units else ("m" if "depth" in name else "1"),
    }


def get_xarray_data_var_name(dataset) -> str:
    """Return the main xarray data variable name."""
    if "data" in dataset.data_vars:
        return "data"
    # XDAS-style files can surface the primary payload under a None key while
    # exposing coord helper arrays (for example *_indices/*_values) as data vars.
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


@cache
def get_xarray_engine(on_missing: str = "raise") -> str | None:
    """Return the preferred xarray engine for NetCDF-4 files."""
    # Map importable module names onto the engine strings xarray expects.
    module_to_engine = {
        "netCDF4": "netcdf4",
        "h5netcdf": "h5netcdf",
    }
    for module_name, engine_name in module_to_engine.items():
        mod = optional_import(module_name, on_missing="ignore")
        if mod is not None:
            return engine_name
    # If no backend available ignore or raise.
    if on_missing == "ignore":
        return None
    msg = (
        "Either netCDF4 or h5netcdf is required for NetCDF-4 read/write "
        "functionality."
    )
    raise MissingOptionalDependencyError(msg)


def iter_written_aux_coords(patch: dc.Patch):
    """Yield names of auxiliary coordinates serialized to NetCDF."""
    for name, coord in patch.coords.coord_map.items():
        if coord._partial or name in patch.dims:
            continue
        yield name


def is_netcdf4_file(h5file: h5py.File) -> bool:
    """
    Check if an HDF5 file follows NetCDF-4 conventions.

    Parameters
    ----------
    h5file
        Open h5py.File object

    Returns
    -------
    bool
        True if file appears to be NetCDF-4 format

    Notes
    -----
    NetCDF-4 files are identified by:
    - _NCProperties attribute (NetCDF-4 specific)
    - Conventions attribute starting with "CF-"
    """
    try:
        # Check for NetCDF-specific markers
        if "_NCProperties" in h5file.attrs:
            return True

        # Check for Conventions attribute indicating CF compliance
        conventions = h5file.attrs.get("Conventions", "")
        if isinstance(conventions, bytes):
            conventions = conventions.decode("utf-8", errors="ignore")
        if conventions and "CF" in conventions:
            return True

        return False

    except (AttributeError, KeyError):
        return False


def get_cf_version(h5file: h5py.File) -> str | None:
    """
    Extract CF convention version from NetCDF file.

    Parameters
    ----------
    h5file
        Open h5py.File object

    Returns
    -------
    str | None
        CF version string (e.g., "1.8") or None if not found
    """
    conventions = h5file.attrs.get("Conventions", "")
    if isinstance(conventions, bytes):
        conventions = conventions.decode("utf-8", errors="ignore")

    # Handle various CF convention formats
    if "CF-" in conventions:
        # Format: "CF-1.8" or "CF-1.8, ACDD-1.3"
        parts = conventions.split("CF-", 1)[1].split()[0].rstrip(",;")
        return parts
    elif conventions.startswith("CF "):
        # Format: "CF 1.8"
        return conventions.split()[1].rstrip(",;")

    return None


def datetime64_to_cf_time(dt_array: np.ndarray) -> np.ndarray:
    """Convert datetime64 array to CF time (seconds since 1970-01-01)."""
    return to_float(dt_array)


def cf_time_to_datetime64(time_array: np.ndarray, units: str) -> np.ndarray:
    """Convert CF time array with units string to datetime64[ns].

    Parameters
    ----------
    time_array
        Numeric time values.
    units
        CF units string, e.g. "seconds since 1970-01-01 00:00:00".
    """
    if " since " not in units:
        msg = f"Invalid CF time units format: {units}"
        raise ValueError(msg)
    time_unit, ref_str = units.split(" since ", 1)
    time_unit = time_unit.strip().lower()
    unit_to_seconds = {
        "days": 86400.0,
        "hours": 3600.0,
        "minutes": 60.0,
        "seconds": 1.0,
        "milliseconds": 1e-3,
        "microseconds": 1e-6,
    }
    if time_unit not in unit_to_seconds:
        msg = f"Unsupported time unit: {time_unit}"
        raise ValueError(msg)
    # Convert to seconds since 1970 then to datetime64
    ref_epoch = to_float(to_datetime64(ref_str.strip()))
    seconds_from_epoch = (
        np.asarray(time_array, dtype=np.float64) * unit_to_seconds[time_unit]
        + ref_epoch
    )
    return to_datetime64(seconds_from_epoch)


def get_cf_data_attrs(data_type: str = "acoustic_signal") -> dict[str, str | float]:
    """
    Get CF-compliant attributes for DAS data variable.

    Parameters
    ----------
    data_type
        Type of DAS data

    Returns
    -------
    dict
        CF-compliant data attributes
    """
    attrs = {
        "long_name": "Distributed Acoustic Sensing data",
        "_FillValue": np.nan,
    }

    # Map data type to CF standard name
    data_type_lower = data_type.lower() if data_type else "acoustic"

    # Check for exact match first
    if data_type_lower in CF_STANDARD_NAMES:
        attrs["standard_name"] = CF_STANDARD_NAMES[data_type_lower]
    else:
        # Check for partial matches
        for key, std_name in CF_STANDARD_NAMES.items():
            if key in data_type_lower:
                attrs["standard_name"] = std_name
                break
        else:
            # Default to generic acoustic signal
            attrs["standard_name"] = "acoustic_signal"

    # Add units based on data type
    if "strain_rate" in data_type_lower:
        attrs["units"] = "1/s"
        attrs["long_name"] = "Strain rate"
    elif "strain" in data_type_lower:
        attrs["units"] = "1"
        attrs["long_name"] = "Strain"
    elif "velocity" in data_type_lower:
        attrs["units"] = "m/s"
        attrs["long_name"] = "Velocity"
    elif "temperature" in data_type_lower:
        attrs["units"] = "K"
        attrs["long_name"] = "Temperature"
    elif "pressure" in data_type_lower:
        attrs["units"] = "Pa"
        attrs["long_name"] = "Pressure"
    else:
        # Generic units for acoustic data
        attrs["units"] = "1"

    return attrs


def get_cf_global_attrs(
    patch_attrs: PatchAttrs, cf_version: str = "1.8"
) -> dict[str, str]:
    """
    Get CF-compliant global attributes from PatchAttrs.

    Parameters
    ----------
    patch_attrs
        DASCore patch attributes
    cf_version
        CF convention version

    Returns
    -------
    dict
        CF-compliant global attributes
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    attrs = {
        "Conventions": f"CF-{cf_version}",
        "title": "DAS data from DASCore",
        "source": f"DASCore v{dc.__version__}",
        "history": f"{now.isoformat()}: Created by DASCore",
        "references": "https://dascore.org",
        "comment": "Distributed Acoustic Sensing data",
        "date_created": now.isoformat(),
    }

    # Add optional attributes from patch
    if patch_attrs.station:
        attrs["station"] = patch_attrs.station
    if patch_attrs.network:
        attrs["network"] = patch_attrs.network
    if patch_attrs.instrument_id:
        attrs["instrument"] = patch_attrs.instrument_id
    if patch_attrs.acquisition_id:
        attrs["acquisition"] = patch_attrs.acquisition_id
    if patch_attrs.tag:
        attrs["tag"] = patch_attrs.tag

    # Add data category/type info
    if patch_attrs.data_category:
        attrs["data_category"] = patch_attrs.data_category
    if hasattr(patch_attrs, "category") and patch_attrs.category:
        attrs["category"] = patch_attrs.category
    if patch_attrs.data_type:
        attrs["data_type"] = patch_attrs.data_type

    # Add processing history if available
    if hasattr(patch_attrs, "history") and patch_attrs.history:
        history_str = " | ".join(str(h) for h in patch_attrs.history)
        attrs["processing_history"] = history_str

    return attrs


def extract_patch_attrs_from_netcdf(h5file: h5py.File) -> dict:
    """
    Extract patch attributes from NetCDF global attributes.

    Parameters
    ----------
    h5file
        Open h5py.File object

    Returns
    -------
    dict
        Dictionary of patch attributes
    """
    attrs = {}

    # Map CF global attributes to patch attributes
    attr_mapping = {
        "station": "station",
        "network": "network",
        "instrument": "instrument_id",
        "acquisition": "acquisition_id",
        "tag": "tag",
        "data_category": "data_category",
        "category": "category",
    }

    for cf_name, patch_name in attr_mapping.items():
        if cf_name in h5file.attrs:
            value = h5file.attrs[cf_name]
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            if value:  # Only add non-empty values
                attrs[patch_name] = value
    for cf_name in ("data_type", "source_data_type"):
        if cf_name not in h5file.attrs:
            continue
        value = h5file.attrs[cf_name]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if value:
            attrs.setdefault("data_type", value)

    return attrs


def _handle_time_interpolation(
    h5file: h5py.File, coord_name: str, coord_data: np.ndarray
) -> np.ndarray | None:
    """
    Handle coordinate interpolation for time coordinates.

    Parameters
    ----------
    h5file
        Open h5py.File object
    coord_name
        Name of the coordinate (e.g., "time")
    coord_data
        Raw coordinate data array

    Returns
    -------
    np.ndarray | None
        Converted datetime64 array if interpolation data found, None otherwise
    """
    # Compatibility path for xdas-produced NetCDF files that store time as CF
    # tie points in `<coord>_values` plus optional `<coord>_indices`. This is
    # not generic CF behavior and does not require an xdas dependency.
    # Look for time_values with proper CF units
    time_values_name = f"{coord_name}_values"
    if time_values_name not in h5file:
        return None

    time_values_var = h5file[time_values_name]
    units = time_values_var.attrs.get("units", "")
    if isinstance(units, bytes):
        units = units.decode("utf-8", errors="ignore")

    if "since" not in units:
        return None

    # Get the time values and interpolate to full coordinate array
    time_values = time_values_var[:]
    time_indices_name = f"{coord_name}_indices"

    if time_indices_name in h5file:
        # Use indices for interpolation
        time_indices = h5file[time_indices_name][:]
        if len(time_values) >= 2 and len(time_indices) >= 2:
            # Linear interpolation from tie points to full coordinate
            interpolated_values = np.interp(coord_data, time_indices, time_values)
        else:
            # Fall back to using time_values directly
            interpolated_values = time_values
    else:
        # Use time_values directly
        interpolated_values = time_values

    # Convert to datetime64
    try:
        return cf_time_to_datetime64(interpolated_values, units)
    except (ValueError, KeyError):
        return None


def read_netcdf_coordinates(
    h5file: h5py.File, data_var_name: str | None = None
) -> CoordManager:
    """Read coordinate information from a NetCDF file into a CoordManager."""
    coords = {}
    coord_order = []

    # Use provided data variable name, or discover it
    main_data_var = data_var_name or find_main_data_variable(h5file)
    expected_shape = None
    if main_data_var and main_data_var in h5file:
        data_var = h5file[main_data_var]
        expected_shape = data_var.shape

        # Try to get dimension order from DIMENSION_LIST
        if "DIMENSION_LIST" in data_var.attrs:
            dim_list = data_var.attrs["DIMENSION_LIST"]
            for ref_array in dim_list:
                try:
                    ref = ref_array[0]
                    dim_scale = h5file[ref]
                    dim_name = dim_scale.name.strip("/")
                    coord_order.append(dim_name)
                except (IndexError, KeyError, TypeError):
                    # If we can't resolve reference, we'll fall back to discovery
                    pass

    # Collect dimension-scale coordinates
    dim_coords = {}
    for name, dataset in h5file.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        if not (dataset.is_scale or "NAME" in dataset.attrs):
            continue
        # Skip auxiliaries that don't match any data dimension size
        if expected_shape and dataset.shape[0] not in expected_shape:
            continue
        if any(
            skip in name.lower()
            for skip in ("_points", "_indices", "_values", "_interpolation")
        ):
            continue
        data = dataset[:]
        if name == "time" or dataset.attrs.get("axis") == "T":
            units = dataset.attrs.get("units", "")
            if isinstance(units, bytes):
                units = units.decode("utf-8", errors="ignore")
            if "since" in units:
                try:
                    data = cf_time_to_datetime64(data, units)
                    dim_coords["time"] = data
                    continue
                except (ValueError, KeyError):
                    pass
            time_coord = _handle_time_interpolation(h5file, name, data)
            if time_coord is not None:
                dim_coords["time"] = time_coord
                continue
        dim_coords[name] = data

    # Build coords dict preserving dimension order from DIMENSION_LIST
    coords = {}
    if coord_order:
        for dim_name in coord_order:
            if dim_name in dim_coords:
                coords[dim_name] = dim_coords[dim_name]
        for name, data in dim_coords.items():
            if name not in coords:
                coords[name] = data
    else:
        coords = dim_coords

    # Collect non-dimension coordinates stored with _DASCORE_DIMS attribute
    if main_data_var and main_data_var in h5file:
        coord_attr = h5file[main_data_var].attrs.get("coordinates", "")
        if isinstance(coord_attr, bytes):
            coord_attr = coord_attr.decode("utf-8", errors="ignore")
        for name in coord_attr.split():
            if name in coords or name not in h5file:
                continue
            dataset = h5file[name]
            if not isinstance(dataset, h5py.Dataset):
                continue
            dims_attr = dataset.attrs.get("_DASCORE_DIMS", "")
            if isinstance(dims_attr, bytes):
                dims_attr = dims_attr.decode("utf-8", errors="ignore")
            if dims_attr:
                associated_dims = tuple(dims_attr.split(","))
                coords[name] = (associated_dims, dataset[:])

    return dc.core.coordmanager.get_coord_manager(coords)


def validate_cf_compliance(h5file: h5py.File) -> list[str]:
    """
    Validate CF compliance and return list of issues.

    Parameters
    ----------
    h5file
        Open h5py.File object

    Returns
    -------
    list[str]
        List of CF compliance issues found
    """
    issues = []

    # Check for required global attributes
    if "Conventions" not in h5file.attrs:
        issues.append("Missing required 'Conventions' global attribute")

    # Check coordinate variables
    for name, dataset in h5file.items():
        if isinstance(dataset, h5py.Dataset) and dataset.is_scale:
            # Check for required coordinate attributes
            if "units" not in dataset.attrs:
                issues.append(f"Coordinate '{name}' missing 'units' attribute")

            # Check time coordinate
            if name == "time" or dataset.attrs.get("standard_name") == "time":
                units = dataset.attrs.get("units", "")
                if not units or "since" not in str(units):
                    issues.append(f"Time coordinate '{name}' has invalid units")

    # Check data variables
    for name, dataset in h5file.items():
        if isinstance(dataset, h5py.Dataset) and not dataset.is_scale:
            # Check for required data variable attributes
            if "units" not in dataset.attrs:
                issues.append(f"Data variable '{name}' missing 'units' attribute")
            if "long_name" not in dataset.attrs:
                issues.append(f"Data variable '{name}' missing 'long_name' attribute")

    return issues


def find_main_data_variable(h5file: h5py.File) -> str | None:
    """
    Find the main data variable in the NetCDF file.

    Looks for 2D+ datasets that are not dimension scales.
    Prioritizes variables with standard names suggesting DAS data.

    Parameters
    ----------
    h5file
        Open h5py.File object

    Returns
    -------
    str | None
        Name of the main data variable, or None if not found
    """
    priority_names = [
        "data",
        "acoustic_data",
        "das_data",
        "strain",
        "strain_rate",
        "velocity",
        "amplitude",
    ]

    candidates = []

    for name, item in h5file.items():
        if not _is_data_variable_candidate(item):
            continue

        # Check for priority matches first
        if name in priority_names or _has_priority_standard_name(item, priority_names):
            return name

        candidates.append(name)

    return candidates[0] if candidates else None


def _is_data_variable_candidate(item) -> bool:
    """Check if item is a candidate for main data variable."""
    return isinstance(item, h5py.Dataset) and not item.is_scale and item.ndim >= 2


def _has_priority_standard_name(item, priority_names: list[str]) -> bool:
    """Check if dataset has a standard_name matching priority names."""
    std_name = str(item.attrs.get("standard_name", "")).lower()
    return any(name in std_name for name in priority_names)
