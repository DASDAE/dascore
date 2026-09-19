"""
Core modules for AP sensing support.
"""

from __future__ import annotations

import dascore as dc
from dascore.constants import snap_type
from dascore.io import ArraySource, FiberIO, H5ArrayMixin
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader

from .utils import _get_attrs_dict, _get_coords, _get_version_string


class APSensingPatchAttrs(dc.PatchAttrs):
    """Patch Attributes for AP sensing."""

    gauge_length: OptionalFiniteFloat = None
    radians_to_nano_strain: OptionalFiniteFloat = None


class APSensingV10(H5ArrayMixin, FiberIO):
    """Support for APSensing V 10."""

    name = "APSensing"
    preferred_extensions = ("hdf5", "h5")
    version = "10"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        version_str = _get_version_string(resource)
        if version_str:
            return version_str
        return None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.PatchMeta]:
        """Scan an AP sensing file, return summary info about the contents."""
        coords = _get_coords(resource)
        attrs = APSensingPatchAttrs.model_validate(_get_attrs_dict(resource))
        return [
            dc.PatchMeta(
                attrs=attrs,
                coords=coords,
                dtype=str(resource["DAS"].dtype),
                source=ArraySource(key=resource["DAS"].name),
            )
        ]
