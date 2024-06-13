"""
Core modules for Silixa H5 support.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader

from .utils import _get_attr, _get_patch, _get_version_string


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

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """
        Return name and version string if Silixa hdf5 else False.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        version_str = _get_version_string(resource, self.version)
        if version_str:
            return self.name, version_str

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan a Silixa HDF5 file, return summary information on the contents."""
        file_version = _get_version_string(resource, self.version)
        extras = {
            "path": resource.filename,
            "file_format": self.name,
            "file_version": str(file_version),
        }
        attrs = _get_attr(resource, SilixaPatchAttrs, extras=extras)
        return [attrs]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a single file with Silixa H5 data inside."""
        patches = _get_patch(
            resource, time=time, distance=distance, attr_cls=SilixaPatchAttrs
        )
        return dc.spool(patches)
