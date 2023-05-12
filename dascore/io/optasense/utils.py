"""Utilities for terra15."""

from typing import Optional

import numpy as np

from dascore.constants import timeable_types
from dascore.core import Patch
from dascore.core.schema import PatchFileSummary
from dascore.utils.misc import get_slice
from dascore.utils.time import datetime_to_float, to_datetime64

# --- Getting format/version


def _get_terra15_version_str(hdf_fi) -> str:
    """
    Return the version string for terra15 file.
    """

    # define a few root attrs that act as a "fingerprint" for terra15 files
    expected_attrs = [
        "acoustic_bandwidth_end",
        "amplifier_incoming_current",
        "file_start_computer_time",
        "file_version",
    ]
    root_attrs = hdf_fi.root._v_attrs
    is_terra15 = all([hasattr(root_attrs, x) for x in expected_attrs])
    if not is_terra15:
        return ""
    return str(root_attrs.file_version)


# --- Getting File summaries


def _get_scanned_time_min_max(data_node):
    """Get the min/max time from time array."""
    time = data_node["gps_time"]
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
    return tmin, tmax, t_len


def _get_extra_scan_attrs(self, file_version, path, data_node):
    """Get the extra attributes that go into summary information."""
    tmin, tmax, _ = _get_scanned_time_min_max(data_node)
    out = {
        "time_min": to_datetime64(tmin),
        "time_max": to_datetime64(tmax),
        "path": path,
        "file_format": self.name,
        "file_version": str(file_version),
    }
    return out


def _get_version_data_node(root):
    """Get the version, time, and data node from terra15 file."""
    version = str(root._v_attrs.file_version)
    if version == "4":
        data_type = root._v_attrs.data_product
        data_node = root[data_type]
    elif version == "5":
        data_node = root["data_product"]
    else:
        raise NotImplementedError("Unknown Terra15 version")
    return version, data_node


def _scan_terra15(self, fi, path):
    """Scan a terra15 file, return metadata."""
    root = fi.root
    root_attrs = fi.root._v_attrs
    version, data_node = _get_version_data_node(root)
    out = _get_default_attrs(data_node.data.attrs, root_attrs)
    out.update(_get_extra_scan_attrs(self, version, path, data_node))
    return [PatchFileSummary.parse_obj(out)]


#
# --- Reading patch


def _get_start_stop(time_len, time_lims, file_tmin, dt):
    """Get the start/stop index along time axis."""
    # sst start index
    tmin = time_lims[0] or file_tmin
    tmax = time_lims[1] or dt * (time_len - 1) + file_tmin
    start_ind = int(np.round((tmin - file_tmin) / dt))
    stop_ind = int(np.round((tmax - file_tmin) / dt)) + 1
    # enforce upper limit on time end index.
    if stop_ind > time_len:
        stop_ind = time_len
    assert 0 <= start_ind < stop_ind
    return start_ind, stop_ind


def _get_dar_attrs(data_node, root, tar, dar):
    """Get the attributes for the terra15 data array (loaded)"""
    attrs = _get_default_attrs(data_node.data.attrs, root._v_attrs)
    attrs["time_min"] = tar.min()
    attrs["time_max"] = tar.max()
    attrs["distance_min"] = dar.min()
    attrs["distance_max"] = dar.max()
    return attrs


def _read_terra15(
    root,
    time: Optional[tuple[timeable_types, timeable_types]] = None,
    distance: Optional[tuple[float, float]] = None,
) -> Patch:
    """
    Read a terra15 file.

    Notes
    -----
    The time array is complicated. There is GPS time and Posix time included
    in the file. In version 0.0.6 and less of dascore we just used gps time.
    However, sometimes this results in subsequent samples having a time before
    the previous sample (time did not increase monotonically).

    So now, we use the first GPS sample, then the reported dt to calculate
    time array. The Terra15 folks suggested this is probably the best way to
    do it.
    """
    # get time array
    time_lims = tuple(
        datetime_to_float(x) if x is not None else None
        for x in (time if time is not None else (None, None))
    )
    _, data_node = _get_version_data_node(root)
    file_t_min, file_t_max, time_len = _get_scanned_time_min_max(data_node)
    # surprisingly, using gps time column, dt is much different than dt
    # reported in data attrs!, use GPS time here.
    dt = (file_t_max - file_t_min) / (time_len - 1)
    # get the start and stop along the time axis
    start_ind, stop_ind = _get_start_stop(time_len, time_lims, file_t_min, dt)
    req_t_min = file_t_min if start_ind == 0 else file_t_min + dt * start_ind
    # account for files that might not be full, adjust requested max time
    stop_ind = min(stop_ind, time_len)
    assert stop_ind > start_ind
    req_t_max = (
        time_lims[-1] if stop_ind < time_len else file_t_min + (stop_ind - 1) * dt
    )
    assert req_t_max > req_t_min
    # calculate time array, convert to datetime64
    t_float = file_t_min + np.arange(start_ind, stop_ind) * dt
    time_ar = to_datetime64(t_float)
    time_inds = (start_ind, stop_ind)
    # get data and sliced distance coord
    dist_ar = _get_distance_array(root)
    dslice = get_slice(dist_ar, distance)
    dist_ar_trimmed = dist_ar[dslice]
    data = data_node.data[slice(*time_inds), dslice]
    coords = {"time": time_ar, "distance": dist_ar_trimmed}
    dims = ("time", "distance")
    attrs = _get_dar_attrs(data_node, root, time_ar, dist_ar_trimmed)
    return Patch(data=data, coords=coords, attrs=attrs, dims=dims)


def _get_default_attrs(data_node_attrs, root_node_attrs):
    """
    Return the required/default attributes which can be fetched from attributes.

    Note: missing time, distance absolute ranges. Downstream functions should handle
    this.
    """
    out = dict(dims="time, distance")
    _root_attrs = {
        "data_product": "data_type",
        "dx": "d_distance",
        "serial_number": "instrument_id",
        "sensing_range_start": "distance_min",
        "sensing_range_end": "distance_max",
        "data_product_units": "data_units",
    }
    for treble_name, out_name in _root_attrs.items():
        out[out_name] = getattr(root_node_attrs, treble_name)

    out["d_time"] = data_node_attrs.dT

    return out


def _get_distance_array(root):
    """Get the distance (along fiber) array."""
    # TODO: At least for the v4 test file, sensing_range_start, sensing_range_stop,
    # nx, and dx are not consistent, meaning d_min + dx * nx != d_max
    # so I just used this method. We need to look more into this.
    attrs = root._v_attrs
    dist = (np.arange(attrs.nx) * attrs.dx) + attrs.sensing_range_start
    return dist
