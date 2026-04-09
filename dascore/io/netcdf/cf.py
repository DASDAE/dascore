"""CF metadata helpers for NetCDF IO.

See https://cfconventions.org/ for the Climate and Forecast metadata standard.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np

import dascore as dc
from dascore.utils.time import to_datetime64, to_float

if TYPE_CHECKING:
    from dascore.core.attrs import PatchAttrs

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


def datetime64_to_cf_time(dt_array: np.ndarray) -> np.ndarray:
    """Convert datetime64 array to CF time (seconds since 1970-01-01)."""
    return to_float(dt_array)


def cf_time_to_datetime64(time_array: np.ndarray, units: str) -> np.ndarray:
    """Convert CF time array with units string to datetime64[ns]."""
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
    ref_epoch = to_float(to_datetime64(ref_str.strip()))
    seconds_from_epoch = (
        np.asarray(time_array, dtype=np.float64) * unit_to_seconds[time_unit]
        + ref_epoch
    )
    return to_datetime64(seconds_from_epoch)


def get_cf_data_attrs(data_type: str = "acoustic_signal") -> dict[str, str | float]:
    """Get CF-compliant attributes for a DAS data variable."""
    attrs = {
        "long_name": "Distributed Acoustic Sensing data",
        "_FillValue": np.nan,
    }
    data_type_lower = data_type.lower() if data_type else "acoustic"
    if data_type_lower in CF_STANDARD_NAMES:
        attrs["standard_name"] = CF_STANDARD_NAMES[data_type_lower]
    else:
        for key, std_name in CF_STANDARD_NAMES.items():
            if key in data_type_lower:
                attrs["standard_name"] = std_name
                break
        else:
            attrs["standard_name"] = "acoustic_signal"
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
        attrs["units"] = "1"
    return attrs


def get_cf_global_attrs(
    patch_attrs: PatchAttrs, cf_version: str = "1.8"
) -> dict[str, str]:
    """Get CF-compliant global attributes from PatchAttrs."""
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
    if patch_attrs.data_category:
        attrs["data_category"] = patch_attrs.data_category
    if hasattr(patch_attrs, "category") and patch_attrs.category:
        attrs["category"] = patch_attrs.category
    if patch_attrs.data_type:
        attrs["data_type"] = patch_attrs.data_type
    if hasattr(patch_attrs, "history") and patch_attrs.history:
        attrs["processing_history"] = " | ".join(str(h) for h in patch_attrs.history)
    return attrs
