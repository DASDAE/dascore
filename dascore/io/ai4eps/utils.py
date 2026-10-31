"""Utility functions for the AI4EPS event format."""

from __future__ import annotations

import warnings

import pandas as pd

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.io.utils import parse_units
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
    # Exports use either the last sample or the exclusive stop as end_time.
    # Both describe the same samples; only warn on an actual disagreement.
    end_time = _to_utc_datetime64(attrs["end_time"])
    expected_stop = time.min() + n_time * time.step
    expected_last = expected_stop - time.step
    if (
        min(abs(end_time - expected_last), abs(end_time - expected_stop))
        > time.step / 2
    ):
        warnings.warn(
            f"AI4EPS file end_time ({end_time}) is inconsistent with "
            f"the last sample ({expected_last}) and exclusive stop "
            f"({expected_stop}); using begin_time, sample count and dt_s.",
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
        "data_units": parse_units(unbyte(_maybe_unpack(attrs["unit"]))),
    }
    out.update(
        maybe_get_items(attrs, _EVENT_ATTR_MAP, unpack_names=set(_EVENT_ATTR_MAP))
    )
    if "event_id" in out:
        out["event_id"] = str(out["event_id"])
    if "event_time" in attrs:
        out["event_time"] = _to_utc_datetime64(attrs["event_time"])
    return out
