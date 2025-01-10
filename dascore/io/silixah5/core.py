"""
Core modules for Silixa H5 support.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
import dascore.io.silixah5.utils as util
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader
from dascore.utils.misc import get_path


class SilixaPatchAttrs(dc.PatchAttrs):
    """Patch Attributes for Silixa hdf5 format."""

    gauge_length: float = np.nan
    gauge_length_units: str = "m"
    pulse_width: float = np.nan
    pulse_width_units: str = "ns"


class SilixaH5V1(FiberIO):
    """Support for Silixa hdf5 format."""

    name = "Silixa_H5"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def _get_attr_coords(self, resource):
        """Get attributes and coordinates of patch in file."""
        info, coords = util._get_attr_dict(resource)
        info["path"] = get_path(resource)
        info["format_name"] = self.name
        info["format_version"] = self.version
        return SilixaPatchAttrs(**info), coords

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """
        Return name and version string if Silixa hdf5 else False.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        version_str = util._get_version_string(resource, self.version)
        if version_str:
            return self.name, version_str

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchSummary]:
        """Scan a Silixa HDF5 file, return summary information on the contents."""
        attrs, coords = self._get_attr_coords(resource)
        data = resource["Acoustic"]
        summary = dc.PatchSummary(data=data, attrs=attrs, coords=coords)
        return [summary]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a single file with Silixa H5 data inside."""
        attrs, coords = self._get_attr_coords(resource)
        data = resource["Acoustic"]
        if time is not None or distance is not None:
            coords, data = coords.select(array=data, time=time, distance=distance)
        patch = dc.Patch(data=data[:], coords=coords, attrs=attrs)
        return dc.spool([patch])
