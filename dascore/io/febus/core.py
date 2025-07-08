"""
IO module for reading Febus data.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader
from dascore.utils.models import UTF8Str

from .utils import (
    _get_febus_version_str,
    _read_febus,
    _yield_attrs_coords,
)


class FebusPatchAttrs(dc.PatchAttrs):
    """
    Patch attrs for febus.

    Attributes
    ----------
    source
        The source designation
    zone
        The zone designations
    """

    gauge_length: float = np.nan
    gauge_length_units: str = "m"
    pulse_width: float = np.nan
    pulse_width_units: str = "m"

    group: str = ""
    source: str = ""
    zone: str = ""

    folog_a1_software_version: UTF8Str = ""


class Febus2(FiberIO):
    """Support for Febus V 2.

    This should cover all versions 2.* of the format (maybe).
    """

    name = "febus"
    preferred_extensions = ("hdf5", "h5")
    version = "2"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """
        Return True if file contains febus version 8 data else False.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        version_str = _get_febus_version_str(resource)
        if version_str:
            return self.name, version_str

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan a febus file, return summary information about the file's contents."""
        out = []
        file_version = _get_febus_version_str(resource)
        extras = {
            "path": resource.filename,
            "file_format": self.name,
            "file_version": str(file_version),
        }
        for attr, cm, _ in _yield_attrs_coords(resource):
            attr["coords"] = cm.to_summary_dict()
            attr.update(dict(extras))
            out.append(FebusPatchAttrs(**attr))
        return out

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a febus spool of patches."""
        patches = _read_febus(
            resource, time=time, distance=distance, attr_cls=FebusPatchAttrs
        )
        return dc.spool(patches)


class Febus1(Febus2):
    """Support for Febus V 1.

    This is here to support legacy febus (eg pubdas Valencia)
    """

    version = "1"
