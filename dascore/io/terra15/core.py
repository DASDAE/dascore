"""
IO module for reading Terra15 DAS data.
"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import tables as tb

from dascore.constants import timeable_types
from dascore.core import MemorySpool, Patch
from dascore.core.schema import PatchFileSummary
from dascore.io.core import FiberIO
from dascore.utils.time import to_datetime64

from .utils import (
    _get_data,
    _get_default_attrs,
    _get_distance_array,
    _get_time_array,
    _get_version_str,
)


class Terra15Formatter(FiberIO):
    """
    Support for Terra15 data format.
    """

    name = "TERRA15"
    preferred_extensions = ("hdf5", "hf")

    def get_format(self, path: Union[str, Path]) -> Union[tuple[str, str], bool]:
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

    def scan(self, path: Union[str, Path]) -> List[PatchFileSummary]:
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
            # first try fast path by tacking first/last of time
            tmin, tmax = time[0], time[-1]
            # This doesn't work if an incomplete datablock exists at the end of
            # the file. In this case we need to read/filter time array (slower).
            if tmin > tmax:
                time = time[:]
                time_filtered = time[time > 0]
                tmin, tmax = np.min(time_filtered), np.max(time_filtered)
            out["time_min"] = to_datetime64(tmin)
            out["time_max"] = to_datetime64(tmax)
            out["path"] = path
            out["format"] = self.name
            return [PatchFileSummary.parse_obj(out)]

    def read(
        self,
        path: Union[str, Path],
        time: Optional[tuple[timeable_types, timeable_types]] = None,
        distance: Optional[tuple[float, float]] = None,
        **kwargs
    ) -> MemorySpool:
        """
        Read a terra15 file, return a DataArray.

        See
        """

        # TODO need to create h5 file decorator to avoid too many open/close files.
        with tb.open_file(path) as fi:
            # get time arra
            if time is None:
                time = (None, None)
            time = tuple(to_datetime64(x) for x in time)
            # get name of data group and use it to fetch data node
            data_type = fi.root._v_attrs.data_product
            data_node = fi.root[data_type]["data"]
            # get time and distance
            time_ar = _get_time_array(fi, data_type)
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
            return MemorySpool([patch])
