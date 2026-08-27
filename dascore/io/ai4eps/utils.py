"""Utility functions for the AI4EPS event format."""

from __future__ import annotations

import warnings

import pandas as pd

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.io.utils import build_patches
from dascore.units import get_quantity
from dascore.utils.hdf5 import h5_matches_structure
from dascore.utils.misc import _maybe_unpack, maybe_get_items, unbyte

# Structure ("dataset.attr" addresses) which must all be present for a file
# to be considered AI4EPS format.
_REQUIRED_STRUCTURE = (
    "data.begin_time",
    "data.end_time",
    "data.dt_s",
    "data.dx_m",
    "data.unit",
    "data.event_id",
)

# Maps event attrs stored in the file to their PatchAttrs names.
_EVENT_ATTR_MAP = {
    "event_id": "event_id",
    "magnitude": "magnitude",
    "magnitude_type": "magnitude_type",
    "latitude": "event_latitude",
    "longitude": "event_longitude",
    "depth_km": "event_depth_km",
}


def _is_ai4eps(resource) -> bool:
    """Return True if the resource looks like an AI4EPS event file."""
    if not h5_matches_structure(resource, _REQUIRED_STRUCTURE):
        return False
    return len(getattr(resource["data"], "shape", None) or ()) == 2


def _safe_units(raw_units):
    """Return the unit string if parseable, else None (don't crash scans)."""
    try:
        get_quantity(raw_units)
    except Exception:
        return None
    return raw_units


def _to_utc_datetime64(value):
    """Convert an ISO time string (possibly tz-aware) to naive UTC datetime64."""
    # to_datetime64 converts tz-aware pd.Timestamps to their naive UTC value.
    return dc.to_datetime64(pd.Timestamp(unbyte(_maybe_unpack(value))))


def _get_coords(dataset):
    """Build the coordinate manager from the data dataset's attrs."""
    n_distance, n_time = dataset.shape
    attrs = dataset.attrs
    time = get_coord(
        start=_to_utc_datetime64(attrs["begin_time"]),
        step=dc.to_timedelta64(float(_maybe_unpack(attrs["dt_s"]))),
        shape=(n_time,),
    )
    # end_time is redundant (begin_time + n * dt_s); warn if the file
    # disagrees with itself so bad timing doesn't pass silently.
    end_time = _to_utc_datetime64(attrs["end_time"])
    expected_stop = time.min() + n_time * time.step
    if abs(end_time - expected_stop) > time.step / 2:
        warnings.warn(
            f"AI4EPS file end_time ({end_time}) is inconsistent with "
            f"begin_time + n_samples * dt_s ({expected_stop}); using the latter.",
            UserWarning,
            stacklevel=2,
        )
    distance = get_coord(
        start=0,
        step=float(_maybe_unpack(attrs["dx_m"])),
        shape=(n_distance,),
        units="m",
    )
    return get_coord_manager(
        {"time": time, "distance": distance}, dims=("distance", "time")
    )


def _get_attrs_dict(dataset) -> dict:
    """Extract patch attributes from the data dataset."""
    attrs = dataset.attrs
    out = {
        "data_category": "DAS",
        "data_units": _safe_units(unbyte(_maybe_unpack(attrs["unit"]))),
    }
    out.update(
        maybe_get_items(attrs, _EVENT_ATTR_MAP, unpack_names=set(_EVENT_ATTR_MAP))
    )
    if "event_id" in out:
        out["event_id"] = str(out["event_id"])
    if "event_time" in attrs:
        out["event_time"] = _to_utc_datetime64(attrs["event_time"])
    return out


def _get_patches(resource, time=None, distance=None, attr_cls=dc.PatchAttrs):
    """Read patches from an AI4EPS file, optionally trimming coords."""
    data = resource["data"]
    return build_patches(
        _get_coords(data),
        data,
        _get_attrs_dict(data),
        attr_cls=attr_cls,
        selection={"time": time, "distance": distance},
    )
