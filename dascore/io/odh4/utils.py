"""Utility functions for the OptaSense ODH4 format."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.core import get_coord, get_coord_manager
from dascore.exceptions import InvalidFiberFileError
from dascore.io.utils import build_patches
from dascore.units import get_quantity
from dascore.utils.misc import maybe_get_items

# Root attributes which must all be present for a file to be considered
# ODH4 format. The names are informal, so the full set is required to
# avoid claiming unrelated files.
_REQUIRED_ATTRS = frozenset(
    (
        "GL m",
        "channel spacing m",
        "channel_start",
        "channel_end",
        "starttime",
        "endtime",
        "sampling rate Hz",
        "scale factor to strain",
        "raw_data_units",
    )
)

# The raw_data_units attr is a description, not a parseable unit string.
_UNIT_MAP = {"phase shift in radians": "radians"}


def _is_odh4(resource) -> bool:
    """Return True if the resource looks like an ODH4 file."""
    dataset = resource.get("raw_data")
    if dataset is None or len(getattr(dataset, "shape", None) or ()) != 2:
        return False
    return _REQUIRED_ATTRS.issubset(set(resource.attrs))


def _read_attrs(resource) -> dict:
    """Read the required root attrs in one pass (unbyte + unpack 0-d)."""
    identity_map = {name: name for name in _REQUIRED_ATTRS}
    return maybe_get_items(
        resource.attrs, identity_map, unpack_names=set(_REQUIRED_ATTRS)
    )


def _safe_units(raw_units):
    """Map the unit description to a parseable unit string, or None."""
    mapped = _UNIT_MAP.get(raw_units, raw_units)
    try:
        get_quantity(mapped)
    except Exception:
        return None
    return mapped


def _get_coords(attrs, shape):
    """
    Build the coordinate manager from the root attrs.

    The file uses exclusive-bound conventions: endtime - starttime
    == n_time / rate and channel_end - channel_start == n_channel. A file
    whose attrs disagree with its data shape beyond slack for an
    inclusive-bound writer raises (the data may be transposed or the
    metadata untrustworthy), rather than silently producing wrong
    coordinates.
    """
    n_channel, n_time = shape
    try:
        rate = float(attrs["sampling rate Hz"])
    except (TypeError, ValueError) as exc:
        msg = "ODH4 file has an unreadable sampling rate attr."
        raise InvalidFiberFileError(msg) from exc
    if not np.isfinite(rate) or rate <= 0:
        msg = f"ODH4 file has an unusable sampling rate ({rate})."
        raise InvalidFiberFileError(msg)
    time_start = dc.to_datetime64(attrs["starttime"])
    time_end = dc.to_datetime64(attrs["endtime"])
    step_from_rate = dc.to_timedelta64(1 / rate)
    # Deriving the step from the time span avoids accumulating the
    # nanosecond truncation of 1/rate over long files. An inclusive-bound
    # writer spans (n - 1) samples rather than n; pick whichever
    # convention the file is closer to so the step stays exact.
    span = time_end - time_start
    denominator = n_time
    if n_time > 1 and abs(span - (n_time - 1) * step_from_rate) < abs(
        span - n_time * step_from_rate
    ):
        denominator = n_time - 1
    step_from_span = span / denominator if denominator else step_from_rate
    channel_range = int(attrs["channel_end"]) - int(attrs["channel_start"])
    time_consistent = abs(step_from_span - step_from_rate) <= 0.01 * step_from_rate
    channel_consistent = channel_range in (n_channel, n_channel - 1)
    if not (time_consistent and channel_consistent):
        msg = (
            "ODH4 file attrs (endtime/channel_end) are inconsistent with the "
            "raw_data shape and sampling metadata; refusing to guess "
            "coordinates (the data may be transposed or corrupt)."
        )
        raise InvalidFiberFileError(msg)
    time = get_coord(start=time_start, step=step_from_span, shape=(n_time,))
    # Distance along the fiber: channel index times channel spacing.
    dx = float(attrs["channel spacing m"])
    channel_start = int(attrs["channel_start"])
    distance = get_coord(
        start=channel_start * dx, step=dx, shape=(n_channel,), units="m"
    )
    # Also carry the interrogator channel numbers so they stay correct
    # under distance trimming.
    channel = get_coord(data=np.arange(channel_start, channel_start + n_channel))
    return get_coord_manager(
        {"time": time, "distance": distance, "channel": ("distance", channel)},
        dims=("distance", "time"),
    )


def _get_attrs_dict(attrs) -> dict:
    """Extract patch attributes from the root attrs."""
    return {
        "data_category": "DAS",
        "data_units": _safe_units(attrs["raw_data_units"]),
        "gauge_length": float(attrs["GL m"]),
        "scale_factor_to_strain": float(attrs["scale factor to strain"]),
    }


def _get_patches(resource, time=None, distance=None, attr_cls=dc.PatchAttrs):
    """Read patches from an ODH4 file, optionally trimming coords."""
    file_attrs = _read_attrs(resource)
    data = resource["raw_data"]
    return build_patches(
        _get_coords(file_attrs, data.shape),
        data,
        _get_attrs_dict(file_attrs),
        attr_cls=attr_cls,
        selection={"time": time, "distance": distance},
    )
