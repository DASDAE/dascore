"""
IO module for reading Terra15 DAS data.
"""
from typing import Union
from pathlib import Path

import tables as tb
import xarray as xr
import numpy as np

from dfs.utils.time import to_datetime64, to_timedelta64
from dfs.core import create_array


def is_version_two(root):
    """Return True if the hdf5 file is version 2."""
    frame_groups = ["frame_id", "posix_frame_time", "gps_frame_time"]
    try:
        _ = [root[x] for x in frame_groups]
    except (KeyError, tb.NoSuchNodeError):
        return False
    return True


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
    out["sampling_length"] = out["dx"]
    out["sampling_time"] = to_timedelta64(out["dT"])
    return out


def read_terra15(
    path: Union[str, Path],
    starttime=None,
    endtime=None,
    format="velocity",
) -> xr.DataArray:
    """
    Read a terra15 file, return a DataArray.

    See
    """
    with tb.open_file(path) as fi:
        assert is_version_two(fi.root)
        data_node = fi.root["velocity"]["data"]
        data = data_node.read()
        time_stamps = fi.root["velocity"]["gps_time"].read()
        time = to_datetime64(time_stamps)
        channel_number = np.arange(data.shape[1])
        attrs = _get_node_attrs(data_node)
        return create_array(data, time=time, channel=channel_number, attrs=attrs)
