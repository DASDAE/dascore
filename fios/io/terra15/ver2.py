"""
IO module for reading Terra15 DAS data.
"""
from pathlib import Path
from typing import List, Union

import numpy as np
import tables as tb

from fios.core import Patch, Stream
from fios.utils.time import to_datetime64, to_timedelta64


def _is_version_two(root):
    """Return True if the hdf5 file is version 2."""
    frame_groups = ["frame_id", "posix_frame_time", "gps_frame_time"]
    try:
        _ = [root[x] for x in frame_groups]
    except (KeyError, tb.NoSuchNodeError):
        return False
    return True


def _is_terra15_v2(path: Union[str, Path]) -> bool:
    """
    Return True if file contains terra15 version 2 data else False.

    Parameters
    ----------
    path
        A path to the file which may contain terra15 data.
    """
    try:
        with tb.open_file(path, "r") as fi:
            return _is_version_two(fi.root)
    except (tb.HDF5ExtError, OSError, IndexError, KeyError, tb.NoSuchNodeError):
        return False


def _get_node_attrs(node):
    """Hacky func to get attrs from hdf5 node, there should be a better way..."""
    attrs = node.attrs
    hdf5_attrs = {"CLASS", "CARRAY", "TITLE", "VERSION"}
    public_attrs = [
        x
        for x in dir(attrs)
        if not x.startswith("_")
        and x not in hdf5_attrs  # exclude hdf attrs not specific to terra15
    ]
    out = {x: getattr(attrs, x) for x in public_attrs}
    # add extra columns
    out["data_type"] = "velocity"
    out["data_category"] = "DAS"
    out["d_time"] = to_timedelta64(out["dT"])
    out["d_distance"] = out["dx"]

    return out


def _scan_terra15_v2(path: Union[str, Path]) -> List[dict]:
    """
    Scan a terra15 v2 file, return summary information about the file's contents.
    """
    with tb.open_file(path) as fi:
        data_node = fi.root["velocity"]["data"]
        time = fi.root["velocity"]["gps_time"]
        t1 = time[0]
        t2 = time[-1]
        # this happens when acquisition stops mid block
        if t2 == 0.0:  # need to read whole array and find las non-zero block
            time_array = time.read()
            no_zero = time_array[time_array > 0.0]
            t2 = no_zero[-1]

        data_shape = data_node.shape
        attrs = _get_node_attrs(data_node)
        out = dict(
            time_min=to_datetime64(t1),
            time_max=to_datetime64(t2),
            distance_min=0,
            distance_max=data_shape[1] * attrs["d_distance"],
            id=attrs["recorder_id"],
            nt=attrs.pop("nT"),
            dt=attrs.pop("dT"),
            category="das",
            data_type="velocity",
        )
        return [out]


def _read_terra15_v2(
    path: Union[str, Path],
    start_time=None,
    end_time=None,
    start_distance=None,
    end_distance=None,
    format="velocity",
) -> Stream:
    """
    Read a terra15 file, return a DataArray.

    See
    """
    with tb.open_file(path) as fi:
        data_node = fi.root["velocity"]["data"]
        data = data_node.read()
        time_stamps = fi.root["velocity"]["gps_time"].read()
        time = to_datetime64(time_stamps)
        # determine if any empty blocks are in this dataset and remove if so.
        if time_stamps[-1] == 0:
            max_time_ind = np.argmax(time_stamps > 0)
            time = time[:max_time_ind]
            data = data[:max_time_ind, :]
        channel_number = np.arange(data.shape[1])
        attrs = _get_node_attrs(data_node)
        distance = attrs["d_distance"] * channel_number
        coords = {"time": time, "distance": distance}
        patch = Patch(data=data, coords=coords, attrs=attrs)
        return Stream([patch])

        # out = create_das_array(
        #     data,
        #     time=time,
        #     distance=distance,
        #     attrs=attrs,
        #     sample_lenth=attrs["sample_length"],
        #     sample_time=attrs["sample_time"],
        #     datatype="velocity",
        # )
        # return Stream([out])
