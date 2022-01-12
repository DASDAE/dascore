"""
IO module for reading Terra15 DAS data.
"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import tables as tb

from dascore.constants import timeable_types
from dascore.core import Patch, Stream
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


def _is_terra15(path: Union[str, Path]) -> Union[tuple[str, str], bool]:
    """
    Return True if file contains terra15 version 2 data else False.

    Parameters
    ----------
    path
        A path to the file which may contain terra15 data.
    """
    try:
        with tb.open_file(path, "r") as fi:
            version_str = _get_version_str(fi)
            if version_str:
                return ("TERRA15", version_str)
    except (tb.HDF5ExtError, OSError, IndexError, KeyError, tb.NoSuchNodeError):
        return False


def _node_attr_to_dict(attr_set):
    """
    Convert a node attribute set to a dictionary.
    """
    # TODO There should be a cleaner way to do this but I haven't yet found it.
    hdf5_attrs = {"CLASS", "CARRAY", "TITLE", "VERSION"}
    public_attrs = [
        x
        for x in dir(attr_set)
        if not x.startswith("_")
        and x not in hdf5_attrs  # exclude hdf attrs not specific to terra15
    ]
    out = {x: getattr(attr_set, x) for x in public_attrs}
    return out


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


def _scan_terra15(path: Union[str, Path]) -> List[dict]:
    """
    Scan a terra15 v2 file, return summary information about the file's contents.
    """
    with tb.open_file(path) as fi:
        root_attrs = fi.root._v_attrs
        data_type = root_attrs.data_product
        data_node = fi.root[data_type]["data"]
        out = _get_default_attrs(data_node.attrs, root_attrs)
        # add time
        time = fi.root[data_type]["gps_time"]
        out["time_min"] = to_datetime64(np.min(time))
        out["time_max"] = to_datetime64(np.max(time))
        return [out]


def _read_terra15(
    path: Union[str, Path],
    time: Optional[tuple[timeable_types, timeable_types]] = None,
    distance: Optional[tuple[float, float]] = None,
) -> Stream:
    """
    Read a terra15 file, return a DataArray.

    See
    """

    # TODO need to create h5 file decorator to avoid too many open/close files.

    def _get_time_array(fi):
        """Get the time array for the file."""
        time_stamps = fi.root[data_type]["gps_time"].read()
        time_array = to_datetime64(time_stamps)
        return time_array

    def _get_distance_array(fi):
        """Get the distance (along fiber) array."""
        # Note: At least for the test file, sensing_range_start, sensing_range_stop,
        # nx, and dx are not consistent so I just used this method. We need to
        # look more into this.
        attrs = fi.root._v_attrs
        dist = (np.arange(attrs.nx) * attrs.dx) + attrs.sensing_range_start
        return dist

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

    with tb.open_file(path) as fi:
        # get time arra
        if time is None:
            time = (None, None)
        time = tuple(to_datetime64(x) for x in time)
        # get name of data group and use it to fetch data node
        data_type = fi.root._v_attrs.data_product
        data_node = fi.root[data_type]["data"]
        # get time and distance
        time_ar = _get_time_array(fi)
        dist_ar = _get_distance_array(fi)
        data, tar, dar = _get_data(time, distance, time_ar, dist_ar, data_node)
        _coords = {"time": tar, "distance": dar}
        attrs = _get_default_attrs(data_node.attrs, fi.root._v_attrs)
        attrs["time_min"] = tar.min()
        attrs["time_max"] = tar.max()
        attrs["distance_min"] = dar.min()
        attrs["distance_max"] = dar.max()
        # get slices of data and read
        patch = Patch(data=data, coords=_coords, attrs=attrs)
        return Stream([patch])
