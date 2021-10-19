"""
IO module for reading Terra15 DAS data.
"""
from typing import Union
from pathlib import Path

import tables as tb
import numpy as np

from dfs.utils.time import to_datetime64, to_timedelta64
from dfs.core import create_das_array, DataArray, Stream


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
    except (tb.HDF5ExtError, OSError, IndexError, KeyError, tb.NoSuchNodeError) as e:
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
    # add required DAS data
    out["sample_length"] = out["dx"]
    out["sample_time"] = to_timedelta64(out["dT"])
    return out


def _scan_terra15_v2(
    path: Union[str, Path],
) -> dict:
    """
    Scan a terra15 v2 file, return summary information about the file's contents.
    """


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
        assert _is_version_two(fi.root)
        data_node = fi.root["velocity"]["data"]
        data = data_node.read()
        time_stamps = fi.root["velocity"]["gps_time"].read()
        time = to_datetime64(time_stamps)
        channel_number = np.arange(data.shape[1])
        attrs = _get_node_attrs(data_node)
        distance = attrs["sample_length"] * channel_number
        out = create_das_array(
            data,
            time=time,
            distance=distance,
            attrs=attrs,
            sample_lenth=attrs["sample_length"],
            sample_time=attrs["sample_time"],
            datatype="velocity",
        )
        return Stream([out])
