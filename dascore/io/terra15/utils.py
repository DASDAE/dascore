"""
Utilities for terra15.
"""

import numpy as np
import tables as tb

from dascore.utils.misc import get_slice
from dascore.utils.time import to_datetime64


def _get_version_str(hdf_fi) -> str:
    """Return True if the hdf5 file is version 2."""
    frame_groups = ["frame_id", "posix_frame_time", "gps_frame_time"]
    root = hdf_fi.root
    try:
        _ = [root[x] for x in frame_groups]
    except (KeyError, tb.NoSuchNodeError):
        return ""
    file_version = str(root._v_attrs.file_version)
    return file_version


def _get_time_array(fi, data_type):
    """Get the time array for the file."""
    time_stamps = fi.root[data_type]["gps_time"].read()
    time_array = to_datetime64(time_stamps)
    return time_array


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
    return data_node[tslice, dslice], time_array[tslice], dist_array[dslice]


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


def _get_distance_array(fi):
    """Get the distance (along fiber) array."""
    # Note: At least for the test file, sensing_range_start, sensing_range_stop,
    # nx, and dx are not consistent so I just used this method. We need to
    # look more into this.
    attrs = fi.root._v_attrs
    dist = (np.arange(attrs.nx) * attrs.dx) + attrs.sensing_range_start
    return dist
