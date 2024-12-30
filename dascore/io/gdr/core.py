"""
Core modules for reading GDR data.

GDR files do not specify the GDR version directly. Instead, they use versions
from other standards for the metadata and raw data. These can be found in the
overview attributes MetadataStandard and RawDataStandard.
"""

from __future__ import annotations

import dascore as dc
from dascore.constants import SpoolType
from dascore.io import FiberIO
from dascore.io.gdr.utils_das import (
    _get_attrs_coords_and_data,
    _get_version,
    _maybe_trim_data,
)
from dascore.utils.hdf5 import H5Reader


class GDRPatchAttrs(dc.PatchAttrs):
    """Patch attrs for GDR files."""

    gauge_length: float
    gauge_length_units: str
    project_number: str = ""


class GDR_V1(FiberIO):  # noqa
    """
    Support for GDR version 1.
    """

    name = "GDR_DAS"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """Determine if the resource belongs to this format."""
        return _get_version(resource)

    def read(self, resource: H5Reader, snap=True, **kwargs) -> SpoolType:
        """
        Read a resource belonging to this format.

        Parameters
        ----------
        resource
            The open h5 object.
        snap
            If True, snap each coordinate to be evenly sampled.
        **kwargs
            Passed to filtering coordinates.
        """
        attr_dict, cm, data = _get_attrs_coords_and_data(resource, snap=snap)
        if kwargs:
            cm, data = _maybe_trim_data(cm, data, **kwargs)
        attrs = GDRPatchAttrs(**attr_dict)
        patch = dc.Patch(coords=cm, data=data[:], attrs=attrs)
        return dc.spool([patch])

    def scan(self, resource: H5Reader, snap=True, **kwargs) -> list[dc.PatchAttrs]:
        """Get the attributes of a resource belong to this type."""
        attrs, cm, data = _get_attrs_coords_and_data(resource, snap)
        attrs["coords"] = cm.to_summary_dict()
        attrs["path"] = resource.filename
        attrs["file_format"] = self.name
        attrs["file_version"] = self.version
        return [dc.PatchAttrs(**attrs)]
