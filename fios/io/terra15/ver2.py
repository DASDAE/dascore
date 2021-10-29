"""
IO module for reading Terra15 DAS data.
"""
from pathlib import Path
from typing import Union, List

import numpy as np
import tables as tb

from fios.core import Stream, Trace2D
from fios.utils.time import to_datetime64


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
    return out


def _scan_terra15_v2(path: Union[str, Path]) -> List[dict]:
    """
    Scan a terra15 v2 file, return summary information about the file's contents.
    """
    with tb.open_file(path) as fi:
        data_node = fi.root["velocity"]["data"]
        # data = data_node.read()
        time_stamps = fi.root["velocity"]["gps_time"].read()
        data_shape = data_node.shape
        attrs = _get_node_attrs(data_node)
        out = dict(
            time=(to_datetime64(time_stamps.min()), to_datetime64(time_stamps.max())),
            distance=(0, data_shape[1] * attrs["dx"]),
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
        channel_number = np.arange(data.shape[1])
        attrs = _get_node_attrs(data_node)
        distance = attrs["dx"] * channel_number
        coords = {"time": time, "distance": distance}
        trace2d = Trace2D(data=data, coords=coords, attrs=attrs)
        return Stream([trace2d])

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
