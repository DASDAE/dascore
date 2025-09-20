"""Utilities for NetCDF IO with CF conventions support."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

import dascore as dc

if TYPE_CHECKING:
    from dascore.core.attrs import PatchAttrs
    from dascore.core.coordmanager import CoordManager

# CF-compliant time reference (Unix epoch is standard)
CF_TIME_REFERENCE = "seconds since 1970-01-01 00:00:00"
CF_CALENDAR = "proleptic_gregorian"

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
    - Presence of dimension scales with DIMENSION_LIST
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

        # Look for dimension scales (NetCDF-4 specific)
        for dataset in h5file.values():
            if isinstance(dataset, h5py.Dataset):
                if "DIMENSION_LIST" in dataset.attrs:
                    return True
                # Also check if it's a dimension scale
                if dataset.is_scale:
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
        # Format: "CF-1.8"
        parts = conventions.split("CF-", 1)[1].split()[0]
        return parts
    elif conventions.startswith("CF "):
        # Format: "CF 1.8"
        return conventions.split()[1]

    return None


def create_dimension_scale(
    h5group: h5py.Group,
    name: str,
    data: np.ndarray,
    cf_attrs: dict[str, str] | None = None,
) -> h5py.Dataset:
    """
    Create a NetCDF-4 compatible dimension scale with CF attributes.

    Parameters
    ----------
    h5group
        HDF5 group to create dimension in
    name
        Name of the dimension
    data
        Data array for the coordinate
    cf_attrs
        CF-compliant attributes for the coordinate

    Returns
    -------
    h5py.Dataset
        Created dimension scale dataset
    """
    # Create dataset
    dataset = h5group.create_dataset(name, data=data)

    # Make it a dimension scale (NetCDF-4 requirement)
    dataset.make_scale(name)

    # Add CF attributes if provided
    if cf_attrs:
        for key, value in cf_attrs.items():
            if value is not None:
                dataset.attrs[key] = value

    # Add NetCDF dimension attributes
    dataset.attrs["NAME"] = name
    dataset.attrs["_Netcdf4Dimid"] = len(h5group) - 1

    return dataset


def datetime64_to_cf_time(dt_array: np.ndarray) -> np.ndarray:
    """
    Convert numpy datetime64 array to CF-compliant time coordinates.

    Parameters
    ----------
    dt_array
        Array of datetime64 values

    Returns
    -------
    np.ndarray
        Array of seconds since CF reference time (1970-01-01)

    Notes
    -----
    CF conventions typically use "seconds since" a reference time.
    The Unix epoch (1970-01-01) is a common choice.
    """
    # Ensure datetime64[ns] precision
    if dt_array.dtype != "datetime64[ns]":
        dt_array = dt_array.astype("datetime64[ns]")

    # Convert to pandas for easier manipulation
    dt_series = pd.to_datetime(dt_array)

    # Calculate seconds since epoch
    reference = pd.Timestamp("1970-01-01", tz=None)
    seconds = (dt_series - reference).total_seconds()

    return seconds.values


def cf_time_to_datetime64(time_array: np.ndarray, units: str) -> np.ndarray:
    """
    Convert CF-compliant time coordinates to numpy datetime64.

    Parameters
    ----------
    time_array
        Array of time values
    units
        CF time units string (e.g., "seconds since 1970-01-01 00:00:00")

    Returns
    -------
    np.ndarray
        Array of datetime64[ns] values

    Raises
    ------
    ValueError
        If units format is not recognized
    """
    # Parse units string
    if " since " not in units:
        msg = f"Invalid CF time units format: {units}"
        raise ValueError(msg)

    # Extract time unit and reference
    time_unit, ref_str = units.split(" since ", 1)
    time_unit = time_unit.strip().lower()

    # Map CF time units to pandas units
    unit_map = {
        "days": "D",
        "hours": "h",
        "minutes": "min",
        "seconds": "s",
        "milliseconds": "ms",
        "microseconds": "us",
    }

    if time_unit not in unit_map:
        msg = f"Unsupported time unit: {time_unit}"
        raise ValueError(msg)

    # Parse reference time
    reference = pd.Timestamp(ref_str.strip())

    # Convert to datetime64
    deltas = pd.to_timedelta(time_array, unit=unit_map[time_unit])
    result = reference + deltas

    return result.values.astype("datetime64[ns]")


def get_cf_time_attrs(coord_name: str = "time") -> dict[str, str]:
    """
    Get CF-compliant attributes for time coordinate.

    Parameters
    ----------
    coord_name
        Name of the coordinate

    Returns
    -------
    dict
        CF-compliant time attributes
    """
    return {
        "standard_name": "time",
        "long_name": "Time",
        "units": CF_TIME_REFERENCE,
        "calendar": CF_CALENDAR,
        "axis": "T",
    }


def get_cf_distance_attrs(coord_name: str = "distance") -> dict[str, str]:
    """
    Get CF-compliant attributes for distance coordinate.

    Parameters
    ----------
    coord_name
        Name of the coordinate

    Returns
    -------
    dict
        CF-compliant distance attributes
    """
    attrs = {
        "long_name": "Distance along fiber",
        "units": "m",
        "axis": "X",
    }

    # Add standard name if it's a recognized CF standard name
    if coord_name.lower() in ["distance", "range"]:
        attrs["standard_name"] = "distance"

    return attrs


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

    # Add valid range for data quality
    attrs["valid_min"] = -1e10
    attrs["valid_max"] = 1e10

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
        "institution": patch_attrs.network or "Unknown",
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
        "data_type": "data_type",
        "data_category": "data_category",
        "source_data_type": "data_type",  # Fallback
    }

    for cf_name, patch_name in attr_mapping.items():
        if cf_name in h5file.attrs:
            value = h5file.attrs[cf_name]
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            if value:  # Only add non-empty values
                attrs[patch_name] = value

    # Set format info
    attrs["file_format"] = "NETCDF_CF"
    cf_version = get_cf_version(h5file)
    if cf_version:
        attrs["file_version"] = cf_version

    return attrs


def _handle_time_interpolation(h5file: h5py.File, coord_name: str, coord_data: np.ndarray) -> np.ndarray | None:
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


def read_netcdf_coordinates(h5file: h5py.File) -> CoordManager:
    """
    Read coordinate information from NetCDF file.

    Parameters
    ----------
    h5file
        Open h5py.File object

    Returns
    -------
    CoordManager
        Coordinate manager with loaded coordinates

    Notes
    -----
    This function looks for dimension scales in the NetCDF file
    and converts them to DASCore coordinates. It preserves the
    dimension order as specified in the NetCDF file.
    """
    coords = {}
    coord_order = []

    # First, find the main data variable to understand expected dimensions
    main_data_var = find_main_data_variable(h5file)
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
                except Exception:
                    # If we can't resolve reference, we'll fall back to discovery
                    pass

    # Look for coordinate variables (datasets that are dimension scales)
    all_coords = {}
    for name, dataset in h5file.items():
        if isinstance(dataset, h5py.Dataset):
            # Check if it's a dimension scale
            if dataset.is_scale or "NAME" in dataset.attrs:
                # Skip auxiliary coordinates that don't match main data dimensions
                if expected_shape and dataset.shape[0] not in expected_shape:
                    continue

                # Skip coordinates that look like interpolation auxiliaries
                if any(skip_word in name.lower() for skip_word in
                       ["_points", "_indices", "_values", "_interpolation"]):
                    continue

                data = dataset[:]

                # Handle time coordinates specially
                if name == "time" or dataset.attrs.get("axis") == "T":
                    units = dataset.attrs.get("units", "")
                    if isinstance(units, bytes):
                        units = units.decode("utf-8", errors="ignore")

                    if "since" in units:
                        # Convert CF time to datetime64
                        try:
                            data = cf_time_to_datetime64(data, units)
                            all_coords["time"] = data
                        except (ValueError, KeyError):
                            # If conversion fails, use as-is
                            all_coords[name] = data
                    else:
                        # Check for coordinate interpolation case
                        time_coord = _handle_time_interpolation(h5file, name, data)
                        if time_coord is not None:
                            all_coords["time"] = time_coord
                        else:
                            # Use as-is (may be index-based)
                            all_coords[name] = data
                else:
                    # Regular coordinate
                    all_coords[name] = data

    # Build coords dict in the correct order
    if coord_order:
        # Use the discovered dimension order
        for dim_name in coord_order:
            if dim_name in all_coords:
                coords[dim_name] = all_coords[dim_name]
        # Add any remaining coordinates
        for name, data in all_coords.items():
            if name not in coords:
                coords[name] = data
    else:
        # Fallback to discovered coordinates
        coords = all_coords

    # Create coordinate manager
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
        if _has_priority_standard_name(item, priority_names):
            return name

        candidates.append(name)

    return candidates[0] if candidates else None


def _is_data_variable_candidate(item) -> bool:
    """Check if item is a candidate for main data variable."""
    return isinstance(item, h5py.Dataset) and not item.is_scale and item.ndim >= 2


def _has_priority_standard_name(item, priority_names: list[str]) -> bool:
    """Check if dataset has a standard_name matching priority names."""
    std_name = str(item.attrs.get("standard_name", "")).lower()
    return any(pn in std_name for pn in priority_names)
