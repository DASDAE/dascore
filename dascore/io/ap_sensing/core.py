"""
Core modules for AP sensing support.
"""

from __future__ import annotations

from typing import Literal

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.models import FiniteFloat
from dascore.utils.hdf5 import H5Reader

from .utils import _get_attrs_dict, _get_coords, _get_patches, _get_version_string


class APSensingPatchAttrs(dc.PatchAttrs):
    """Patch Attributes for AP sensing."""

    gauge_length: FiniteFloat | None = None
    radians_to_nano_strain: FiniteFloat | None = None


class APSensingV10(FiberIO):
    """Support for APSensing V 10."""

    name = "APSensing"
    preferred_extensions = ("hdf5", "h5")
    version = "10"

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """
        Return format name and version string if AP sensing, else False.

        Parameters
        ----------
        resource
            An open h5 file which may contain AP sensing data.
        """
        version_str = _get_version_string(resource)
        if version_str:
            return self.name, version_str
        return False

    def scan(self, resource: H5Reader, **kwargs) -> list[ScanPayload]:
        """Scan an AP sensing file, return summary info about the contents."""
        coords = _get_coords(resource)
        attrs = APSensingPatchAttrs.model_validate(_get_attrs_dict(resource))
        return [
            make_scan_payload(
                attrs=attrs, coords=coords, dtype=str(resource["DAS"].dtype)
            )
        ]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a single file with APSensing data inside."""
        patches = _get_patches(
            resource, time=time, distance=distance, attr_cls=APSensingPatchAttrs
        )
        return dc.spool(patches)
