"""Utilities for terra15."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.io.utils import stored_coord
from dascore.utils.misc import maybe_get_items
from dascore.utils.time import to_datetime64

# --- Getting format/version


def _get_terra15_version_str(hdf_fi) -> str:
    """Return the version string for terra15 file."""
    # define a few root attrs that act as a "fingerprint" for terra15 files
    expected_attrs = {
        "acoustic_bandwidth_end",
        "amplifier_incoming_current",
        "file_start_computer_time",
        "file_version",
    }
    root_attrs = hdf_fi.attrs
    if not set(root_attrs).issuperset(expected_attrs):
        return ""
    return str(root_attrs["file_version"])


# --- Getting File summaries


def _get_time_node(data_node):
    """
    Get the time node from data.

    This will prefer GPS time but gets posix time if it is missing.
    """
    try:
        time = data_node["gps_time"]
    except (IndexError, KeyError):
        time = data_node["posix_time"]
    return time


def _get_written_count(time) -> int:
    """
    How many leading rows of the time node the file has written.

    An unfinished file ends in rows whose time is still zero, which only
    reading the whole node can count; a finished one ascends to its end.
    """
    count = len(time)
    if count > 1 and time[0] > time[-1]:
        return int(np.count_nonzero(time[:] > 0))
    return count


def _get_version_data_node(root):
    """Get the version, time, and data node from terra15 file."""
    version = str(root.attrs["file_version"])
    if version == "4":
        data_type = root.attrs["data_product"]
        data_node = root[data_type]
    elif version in {"5", "6"}:
        data_node = root["data_product"]
    else:
        raise NotImplementedError("Unknown Terra15 version")
    return version, data_node


def _scan_terra15(h5_fi, data_node, *, snap=True) -> list[dc.PatchMeta]:
    """Scan a terra15 file, return metadata."""
    out = _get_default_attrs(h5_fi.attrs)
    coords = {
        "time": _get_time_coord(data_node, snap=snap),
        "distance": _get_distance_coord(h5_fi),
    }
    coord_manager = get_coord_manager(coords=coords, dims=tuple(coords))
    return [
        dc.PatchMeta(
            attrs=PatchAttrs.from_dict(out),
            coords=coord_manager,
            dtype=str(data_node["data"].dtype),
        )
    ]


# --- Reading patch


def _get_default_attrs(root_node_attrs):
    """
    Return the required/default attributes which can be fetched from attributes.

    Note: missing time, distance absolute ranges. Downstream functions should handle
    this.
    """
    out = {}
    _root_attrs = {
        "data_product": "data_type",
        "serial_number": "interrogator.serial_number",
        "data_product_units": "data_units",
        "pulse_rate": "pulse_rate",
        "pulse_length": "pulse_length",
        "gauge_length": "gauge_length",
    }
    out.update(maybe_get_items(root_node_attrs, _root_attrs))
    return out


def _get_time_coord(data_node, snap=True):
    """
    Get the time coordinate of the written rows.

    Per-sample GPS times jitter, even backwards (#71), so the snapped grid
    spans the first and last written times (#72); the stated `dt_computer`
    is the computer clock's, which can be off by percent.
    """
    time = _get_time_node(data_node)
    return stored_coord(
        time,
        snap,
        name="time",
        where=data_node.file,
        decode=to_datetime64,
        count=_get_written_count(time),
        units="s",
    )


def _get_distance_coord(root):
    """Get the distance coordinate."""
    # TODO: At least for the v4 test file, sensing_range_start, sensing_range_stop,
    # nx, and dx are not consistent, meaning d_min + dx * nx != d_max
    # so I just used this method. We need to look more into this.
    attrs = getattr(root, "attrs", root)
    start, step = attrs["sensing_range_start"], attrs["dx"]
    stop = attrs["sensing_range_start"] + (attrs["nx"] - 1) * attrs["dx"]
    return get_coord(start=start, stop=stop + step, step=step, units="m")
