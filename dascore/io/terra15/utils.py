"""
Utilities for terra15.
"""

from typing import Optional

import numpy as np

from dascore.constants import timeable_types
from dascore.core import Patch
from dascore.core.schema import PatchFileSummary
from dascore.utils.misc import get_slice
from dascore.utils.time import to_datetime64

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


def _get_scanned_time_min_max(time):
    """Get the min/max time from time array."""
    # first try fast path by tacking first/last of time
    tmin, tmax = time[0], time[-1]
    # This doesn't work if an incomplete datablock exists at the end of
    # the file. In this case we need to read/filter time array (slower).
    if tmin > tmax:
        time = time[:]
        time_filtered = time[time > 0]
        tmin, tmax = np.min(time_filtered), np.max(time_filtered)
    return tmin, tmax


def _get_extra_scan_attrs(self, file_version, path, time):
    """Get the extra attributes that go into summary information."""
    tmin, tmax = _get_scanned_time_min_max(time)
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
        raise NotImplementedError("Unkown Terra15 version")
    return version, data_node


def _scan_terra15(self, fi, path):
    """Scan a terra15 file, return metadata."""
    root = fi.root
    root_attrs = fi.root._v_attrs
    version, data_node = _get_version_data_node(root)
    out = _get_default_attrs(data_node.data.attrs, root_attrs)
    time = data_node["gps_time"]
    out.update(_get_extra_scan_attrs(self, version, path, time))
    return [PatchFileSummary.parse_obj(out)]


#
# --- Reading patch


def _read_terra15(
    root,
    time: Optional[tuple[timeable_types, timeable_types]] = None,
    distance: Optional[tuple[float, float]] = None,
) -> Patch:
    """
    Read a terra15 file.
    """
    # get time array
    time_lims = tuple(
        to_datetime64(x) if x is not None else None
        for x in (time if time is not None else (None, None))
    )
    _, data_node = _get_version_data_node(root)
    time_ar = to_datetime64(data_node["gps_time"].read())
    dist_ar = _get_distance_array(root)
    data, tar, dar = _get_data(time_lims, distance, time_ar, dist_ar, data_node)
    _coords = {"time": tar, "distance": dar}
    attrs = _get_default_attrs(data_node.data.attrs, root._v_attrs)
    attrs["time_min"] = tar.min()
    attrs["time_max"] = tar.max()
    attrs["distance_min"] = dar.min()
    attrs["distance_max"] = dar.max()
    return Patch(data=data, coords=_coords, attrs=attrs)


def _get_data(time, distance, time_array, dist_array, data_node):
    """
    Get the data array. Slice based on input and check for 0 blocks. Also
    return sliced coordinates.
    """
    # need to handle empty data blocks. This happens when data is stopped
    # recording before the pre-allocated file is filled.
    if time_array[-1] < time_array[0]:
        time = (time[0], time_array.max())
    tslice = get_slice(time_array, time)
    dslice = get_slice(dist_array, distance)
    return data_node.data[tslice, dslice], time_array[tslice], dist_array[dslice]


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
    # nx, and dx are not consistent, so I just used this method. We need to
    # look more into this.
    attrs = root._v_attrs
    dist = (np.arange(attrs.nx) * attrs.dx) + attrs.sensing_range_start
    return dist
