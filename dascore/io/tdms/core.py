"""
IO module for reading Silixa's TDMS DAS data format.
"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from dascore.constants import timeable_types
from dascore.core import MemorySpool, Patch
from dascore.core.schema import PatchFileSummary
from dascore.io.core import FiberIO
from dascore.utils.time import to_datetime64

from .utils import (
    _get_data,
    _get_data_node,
    _get_default_attrs,
    _get_distance_array,
    _get_time_array,
    _get_version_str,
)


class TDMSFormatterV4713(FiberIO):
    """
    Support for Silixa data format (tdms).
    """

    name = "TDMS"
    version = "4713"
    preferred_extensions = ("tdms",)
    LEAD_IN_LENGTH = 28

    def get_format(self, path: Union[str, Path]) -> Union[tuple[str, str], bool]:
        """
        Return a tuple of (TDMS, version) if TDMS else False.

        Parameters
        ----------
        path
            A path to the file which may contain silixa data.
        """

        try:
            with open(path, "rb") as tdms_file:
                version_str = _get_version_str(tdms_file)
                if version_str:
                    return "TDMS", version_str
                else:
                    return False
        except Exception:  # noqa
            return False

    def scan(self, path: Union[str, Path]) -> List[PatchFileSummary]:
        """
        Scan a silixa tdms file, return summary information about the file's contents.
        """
        with open(path, "rb") as tdms_file:
            out = _get_default_attrs(tdms_file)
            out["path"] = path
            out["file_format"] = self.name
            out["file_version"] = self.version
            return [PatchFileSummary(**out)]

    def read(
        self,
        path: Union[str, Path],
        time: Optional[tuple[timeable_types, timeable_types]] = None,
        distance: Optional[tuple[float, float]] = None,
        **kwargs
    ) -> MemorySpool:
        """
        Read a silixa tdms file, return a DataArray.

        """

        with open(path, "rb") as tdms_file:
            # get time array. If an input isn't provided for time we return everything
            if time is None:
                time = (None, None)
            time = tuple(to_datetime64(x) for x in time)

            # get all data, total amount of samples and associated attributes
            data_node, channel_length, attrs = _get_data_node(
                tdms_file, LEAD_IN_LENGTH=28
            )
            # time_max scanned in attributes is updated after reading data
            attrs["time_max"] = attrs["time_min"] + np.timedelta64(
                int(1000 * channel_length * attrs["d_time"]), "ms"
            )
            # get time and distance array.
            time_ar = _get_time_array(tdms_file=None, attrs=attrs)
            dist_ar = _get_distance_array(tdms_file=None, attrs=attrs)

            # Get data in distance and time requested and assoociated time
            # and distance arrays
            data, tar, dar = _get_data(time, distance, time_ar, dist_ar, data_node)
            _coords = {"time": tar, "distance": dar}
            attrs = _get_default_attrs(tdms_file, attrs)

            # Update time and distance attributes to match requested parameters
            attrs["time_min"] = tar.min()
            attrs["time_max"] = tar.max()
            attrs["distance_min"] = dar.min()
            attrs["distance_max"] = dar.max()
            # get slices of data and read
            patch = Patch(data=data, coords=_coords, attrs=attrs)
            return MemorySpool([patch])
