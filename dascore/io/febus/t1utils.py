"""Utilities for reading Febus T1 data"""

from __future__ import annotations

import dascore as dc
from dascore import get_coord_manager
from dascore.io.utils import (
    drop_blank_attrs,
    get_exact_coord,
    get_gridded_coord,
    should_snap,
)
from dascore.utils.hdf5 import H5Reader
from dascore.utils.misc import maybe_get_items

_DATA = "Data"


def _is_t1_file(fi: H5Reader) -> bool:
    """Minimal fingerprint check — T1 files always have these datasets."""
    required = {"Temperature", "Distance", "DistanceSignal", "Time"}
    present = set(fi.get(_DATA, {}).keys())
    return required.issubset(present)


def _get_distance_coord(fi, snap=True):
    """
    Get the distances from the T1 file.

    The interrogator fixes the spatial sampling, so the snapped path puts the
    stored values back on the grid they restate.
    """
    dist = fi["Data/Distance"][()]
    if should_snap(snap, "distance"):
        return get_gridded_coord(dist, units="m")
    return get_exact_coord(dist, units="m")


def _get_time_coord(fi, snap=True):
    """Get the times from the T1 file"""
    ts = fi["Data/Time"][()].squeeze()
    times = (ts * 1e9).astype("datetime64[ns]")
    if should_snap(snap, "time"):
        return dc.get_coord(values=times, units="s")
    return get_exact_coord(times, units="s")


def _get_coords(fi, snap=True) -> dc.CoordManager:
    """Return the T1 coord manager."""
    time_coord = _get_time_coord(fi, snap=snap)
    distance_coord = _get_distance_coord(fi, snap=snap)
    dims = ("time", "distance")
    return get_coord_manager(
        {"time": time_coord, "distance": distance_coord},
        dims=dims,
    )


# Attrs the format itself fixes. manufacturer and model are asserted by
# format detection, not read from the header: a file this reader claims
# is a Febus T1. The header states no maker or model of its own.
_T1_ATTRS: dict[str, str] = {
    "data_type": "temperature",
    "data_units": "°C",
    "data_category": "DTS",
    "interrogator.manufacturer": "FEBUS",
    "interrogator.model": "T1",
}

# Root attrs naming the unit: device_name is the host (eg "ft1-24090217"),
# device the kind of instrument it ran as (eg "DTS").
_T1_ROOT_ATTRS = {
    "device_name": "interrogator.name",
    "device": "interrogator.instrument_type",
}


def _get_t1_attrs(fi: H5Reader) -> dict[str, str]:
    """Return the fixed T1 attrs plus the interrogator the file names."""
    named = maybe_get_items(fi.attrs, _T1_ROOT_ATTRS)
    return _T1_ATTRS | drop_blank_attrs(named, _T1_ROOT_ATTRS.values())


def _scan_t1(fi: H5Reader, snap=True):
    """Get the coordinates and attributes for a T1 data patch"""
    coords = _get_coords(fi, snap=snap)
    return dc.Patch(
        attrs=_get_t1_attrs(fi),
        coords=coords,
        dtype=str(fi["Data/Temperature"].dtype),
    )
