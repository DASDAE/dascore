"""
Core modules for reading GDR data.

GDR files do not specify the GDR version directly. Instead, they use versions
from other standards for the metadata and raw data. These can be found in the
overview attributes MetadataStandard and RawDataStandard.
"""

from __future__ import annotations

import dascore as dc
from dascore.constants import snap_type
from dascore.io import ArraySource, FiberIO, H5ArrayMixin
from dascore.io.gdr.utils_das import _get_attrs_coords_and_data, _get_version
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader


class GDRPatchAttrs(dc.PatchAttrs):
    """Patch attrs for GDR files."""

    gauge_length: OptionalFiniteFloat
    project_number: str = ""


class GDR_V1(H5ArrayMixin, FiberIO):  # noqa
    """
    Support for GDR version 1.
    """

    name = "GDR_DAS"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        return _match[1] if (_match := _get_version(resource)) else None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.PatchMeta]:
        """Get the attributes of a resource belong to this type."""
        attrs, cm, data = _get_attrs_coords_and_data(resource, snap)
        return [
            dc.PatchMeta(
                attrs=GDRPatchAttrs.from_dict(attrs),
                coords=cm,
                dtype=str(data.dtype),
                source=ArraySource(key=data.name),
            )
        ]
