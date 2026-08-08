"""Utilities for Uptech Sensing HDF5 files."""

from __future__ import annotations

import numpy as np

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
        # Uptech metadata is expected on the signal dataset, not its group.
        and _ATTRS.issubset(data.attrs)
    )


def _get_time(resource):
    """Return validated Uptech time values as datetime64 coordinates."""
    data = resource[_DATASET]
    values = np.asarray(resource[_TIME][:], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Uptech acquisition time contains non-finite values.")
    frequency = float(data.attrs["acquisition_frequency"])
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Uptech acquisition_frequency must be finite and positive.")
    if len(values) > 1:
        steps = np.diff(values)
        if np.any(steps <= 0) or not np.allclose(
            steps, steps[0], rtol=1e-4, atol=1e-9
        ):
            raise ValueError("Uptech acquisition time must be uniformly increasing.")
        if not np.isclose(steps.mean() * frequency, 1, rtol=1e-3):
            raise ValueError(
                "Uptech acquisition time disagrees with acquisition_frequency."
            )
    return dc.to_datetime64(values)


def _get_coords(resource):
    """Build time and distance coordinates."""
    data = resource[_DATASET]
    time = get_coord(data=_get_time(resource))
    # Uptech's sampling interval is the spatial channel pitch. The
    # spatial resolution is the sensing resolution and may be different.
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
        "data_units": "1/s",
        "fiber_length": float(data.attrs["fiber_length"]),
        "fiber_length_units": "m",
        "gauge_length": float(data.attrs["gauge_length"]),
        "gauge_length_units": "m",
        "spatial_resolution": float(data.attrs["spatial_resolution"]),
        "spatial_resolution_units": "m",
        "sampling_frequency": float(data.attrs["acquisition_frequency"]),
    }
    attrs.update(extras or {})
    return dc.PatchAttrs(**attrs)
