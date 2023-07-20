"""
IO module for reading Silixa's TDMS DAS data format.
"""
from typing import List, Optional, Union

import dascore as dc
from dascore.constants import timeable_types
from dascore.core import Patch
from dascore.core.schema import PatchFileSummary
from dascore.io import BinaryReader, FiberIO

from .utils import _get_data, _get_data_node, _get_default_attrs, _get_version_str


class TDMSFormatterV4713(FiberIO):
    """
    Support for Silixa data format (tdms).
    """

    name = "TDMS"
    version = "4713"
    preferred_extensions = ("tdms",)
    LEAD_IN_LENGTH = 28

    def get_format(self, stream: BinaryReader) -> Union[tuple[str, str], bool]:
        """
        Return a tuple of (TDMS, version) if TDMS else False.

        Parameters
        ----------
        stream
            A path to the file which may contain silixa data.
        """

        try:
            version_str = _get_version_str(stream)
            if version_str:
                return "TDMS", version_str
            else:
                return False
        except Exception:  # noqa
            return False

    def scan(self, resource: BinaryReader) -> List[PatchFileSummary]:
        """
        Scan a silixa tdms file, return summary information about the file's contents.
        """
        out = _get_default_attrs(resource)
        out["path"] = getattr(resource, "name", "")
        out["file_format"] = self.name
        out["file_version"] = self.version
        return [PatchFileSummary(**out)]

    def read(
        self,
        resource: BinaryReader,
        time: Optional[tuple[timeable_types, timeable_types]] = None,
        distance: Optional[tuple[float, float]] = None,
        **kwargs
    ) -> dc.BaseSpool:
        """
        Read a silixa tdms file, return a DataArray.
        """
        # get all data, total amount of samples and associated attributes
        data_node, channel_length, attrs_full = _get_data_node(
            resource, LEAD_IN_LENGTH=28
        )
        attrs = _get_default_attrs(resource, attrs_full)
        # get time and distance coordinates.
        time_coord = attrs_full["_time_coord"]
        dist_coord = attrs_full["_distance_coord"]
        # Get data in distance and time requested and associated time
        # and distance arrays
        data, tar, dar = _get_data(time, distance, time_coord, dist_coord, data_node)
        dims = ("time", "distance")
        _coords = {"time": tar, "distance": dar}
        # Update time and distance attributes to match requested parameters
        patch = Patch(data=data, coords=_coords, attrs=attrs, dims=dims)
        return dc.spool(patch)
