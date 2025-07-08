"""
Core modules for AP sensing support.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader

from .utils import _get_attrs_dict, _get_patch, _get_version_string


class APSensingPatchAttrs(dc.PatchAttrs):
    """Patch Attributes for AP sensing."""

    gauge_length: float = np.nan
    radians_to_nano_strain: float = np.nan


class APSensingV10(FiberIO):
    """Support for APSensing V 10."""

    name = "APSensing"
    preferred_extensions = ("hdf5", "h5")
    version = "10"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """
        Return format name and version string if AP sensing, else False.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        version_str = _get_version_string(resource)
        if version_str:
            return self.name, version_str

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan an AP sensing file, return summary info about the contents."""
        file_version = _get_version_string(resource)
        extras = {
            "path": resource.filename,
            "file_format": self.name,
            "file_version": str(file_version),
        }
        attrs = _get_attrs_dict(resource)
        attrs.update(extras)
        return [APSensingPatchAttrs(**attrs)]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a single file with APSensing data inside."""
        patches = _get_patch(
            resource, time=time, distance=distance, attr_cls=APSensingPatchAttrs
        )
        return dc.spool(patches)
