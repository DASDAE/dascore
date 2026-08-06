"""Utilities for Uptech Sensing HDF5 files."""

from __future__ import annotations

import dascore as dc
from dascore.core import get_coord, get_coord_manager

_DATASET = "Acquisition/StrainRate"
_TIME = "Acquisition/Time"
_ATTRS = frozenset(
    {
        "acquisition_frequency",
        "fiber_length",
        "gauge_length",
        "sampling_interval",
        "spatial_resolution",
    }
)


def _is_uptech(resource) -> bool:
    """Return whether an HDF5 resource has the Uptech export layout."""
    try:
        data, time = resource[_DATASET], resource[_TIME]
    except KeyError:
        return False
    return (
        data.ndim == 2
        and time.ndim == 1
        and len(time) == data.shape[0]
        and _ATTRS.issubset(data.attrs)
    )


def _get_coords(resource):
    """Build time and distance coordinates."""
    data = resource[_DATASET]
    time = get_coord(data=dc.to_datetime64(resource[_TIME][:]))
    distance = get_coord(
        start=0,
        step=float(data.attrs["sampling_interval"]),
        shape=(data.shape[1],),
        units="m",
    )
    return get_coord_manager(
        {"time": time, "distance": distance}, dims=("time", "distance")
    )


def _get_attrs(resource, coords=None, extras=None):
    """Build patch attributes from signal metadata."""
    data = resource[_DATASET]
    attrs = {
        "coords": coords or _get_coords(resource),
        "data_type": "strain_rate",
        "gauge_length": float(data.attrs["gauge_length"]),
        "gauge_length_units": "m",
        "spatial_resolution": float(data.attrs["spatial_resolution"]),
        "spatial_resolution_units": "m",
        "sampling_frequency": float(data.attrs["acquisition_frequency"]),
    }
    attrs.update(extras or {})
    return dc.PatchAttrs(**attrs)
