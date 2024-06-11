"""Utilities for terra15."""

from __future__ import annotations

import dascore as dc
from dascore.constants import timeable_types
from dascore.core import Patch
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.utils.misc import maybe_get_items
from dascore.utils.time import to_datetime64, to_timedelta64

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


def _get_scanned_time_info(data_node):
    """Get the min, max, len, and dt from time array."""
    time = _get_time_node(data_node)
    t_len = len(time)
    # first try fast path by tacking first/last of time
    tmin, tmax = time[0], time[-1]
    # This doesn't work if an incomplete datablock exists at the end of
    # the file. In this case we need to read/filter time array (slower).
    if tmin > tmax:
        time = time[:]
        time_filtered = time[time > 0]
        t_len = len(time_filtered)
        tmin, tmax = time_filtered[0], time_filtered[-1]
    # surprisingly, using gps time column, dt is much different than dt
    # reported in data attrs so we calculate it this way.
    dt = (tmax - tmin) / (t_len - 1)
    starttime = to_datetime64(tmin)
    time_step = to_timedelta64(dt)
    endtime = starttime + time_step * (t_len - 1)
    return starttime, endtime, t_len, time_step


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


def _scan_terra15(h5_fi, data_node, extras=None):
    """Scan a terra15 file, return metadata."""
    out = extras
    out.update(_get_default_attrs(h5_fi.attrs))
    coords = {
        "time": _get_time_coord(data_node, snap_dims=True),
        "distance": _get_distance_coord(h5_fi),
    }
    out["coords"] = coords
    return [dc.PatchAttrs(**out)]


# --- Reading patch


def _get_raw_time_coord(data_node):
    """Read the time from the data node and return it."""
    time = _get_time_node(data_node)[:]
    return get_coord(data=to_datetime64(time))


def _read_terra15(
    pyfi,
    time: tuple[timeable_types, timeable_types] | None = None,
    distance: tuple[float, float] | None = None,
    snap_dims: bool = True,
) -> Patch:
    """
    Read a terra15 file.

    Notes
    -----
    The time array is complicated. There is GPS time and Posix time included
    in the file. In version 0.0.6 and less of dascore we just used gps time.
    However, sometimes this results in subsequent samples having a time before
    the previous sample (time did not increase monotonically).

    So now, we use the first GPS sample, the last sample, and length
    to determine the dt (new in dascore>0.0.11).
    """
    _, data_node = _get_version_data_node(pyfi)
    time_coord_ = _get_time_coord(data_node, snap_dims)
    time_coord, time_slice = time_coord_.select(time)
    time_len = len(time_coord)
    # get data and sliced distance coord
    dist_coord = _get_distance_coord(pyfi)
    dist_coord, dist_slice = dist_coord.select(distance)
    _data = data_node["data"]
    # checks for incomplete data blocks
    if _data.shape[0] > time_len:
        new_start = time_slice.start or 0
        t_stop = time_slice.stop
        new_stop = new_start + time_len if t_stop is None else t_stop
        time_slice = slice(new_start, new_stop)
    data = data_node["data"][time_slice, dist_slice]
    coords = get_coord_manager(
        {"time": time_coord, "distance": dist_coord},
        dims=("time", "distance"),
    )
    dims = ("time", "distance")
    attrs = _get_default_attrs(pyfi.attrs)
    attrs["coords"] = coords
    return Patch(data=data, coords=coords, attrs=attrs, dims=dims)


def _get_default_attrs(root_node_attrs):
    """
    Return the required/default attributes which can be fetched from attributes.

    Note: missing time, distance absolute ranges. Downstream functions should handle
    this.
    """
    out = {}
    _root_attrs = {
        "data_product": "data_type",
        "serial_number": "instrument_id",
        "data_product_units": "data_units",
        "pulse_rate": "pulse_rate",
        "pulse_length": "pulse_length",
        "gauge_length": "gauge_length",
    }
    out.update(maybe_get_items(root_node_attrs, _root_attrs))
    return out


def _get_time_coord(data_node, snap_dims=True):
    """Get the time coordinate."""
    t_min, t_max, time_len, d_time = _get_scanned_time_info(data_node)
    if snap_dims:
        kwargs = dict(start=t_min, stop=t_max + d_time, step=d_time, units="s")
        time_coord = get_coord(**kwargs)
    else:
        time_coord = _get_raw_time_coord(data_node)
    return time_coord


def _get_distance_coord(root):
    """Get the distance coordinate."""
    # TODO: At least for the v4 test file, sensing_range_start, sensing_range_stop,
    # nx, and dx are not consistent, meaning d_min + dx * nx != d_max
    # so I just used this method. We need to look more into this.
    attrs = getattr(root, "attrs", root)
    start, step = attrs["sensing_range_start"], attrs["dx"]
    stop = attrs["sensing_range_start"] + (attrs["nx"] - 1) * attrs["dx"]
    return get_coord(start=start, stop=stop + step, step=step, units="m")
