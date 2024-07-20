"""
Core modules for reading Neubrex (Forge) dss/dts data.
"""

from __future__ import annotations

import dascore as dc
from dascore.constants import SpoolType
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader

from .utils import _get_attrs_coords_and_data, _is_neubrex, _maybe_trim_data


class NeubrexPatchAttrs(dc.PatchAttrs):
    """Patch attrs for Neubrex files."""

    api: str | None = None
    filed_name: str = ""
    well_id: str = ""
    well_name: str = ""
    well_bore_id: str = ""


class NeubrexV1(FiberIO):
    """Support for bare-bones h5 format."""

    name = "Neubrex"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """Determine if is simple h5 format."""
        if _is_neubrex(resource):
            return self.name, self.version
        return False

    def read(self, resource: H5Reader, snap=True, **kwargs) -> SpoolType:
        """
        Read a simple h5 file.

        Parameters
        ----------
        resource
            The open h5 object.
        snap
            If True, snap each coordinate to be evenly sampled.
        **kwargs
            Passed to filtering coordinates.
        """
        attr_dict, cm, data = _get_attrs_coords_and_data(resource, snap)
        if kwargs:
            cm, data = _maybe_trim_data(cm, data, **kwargs)
        attrs = NeubrexPatchAttrs(**attr_dict)
        patch = dc.Patch(coords=cm, data=data[:], attrs=attrs)
        return dc.spool([patch])

    def scan(self, resource: H5Reader, snap=True, **kwargs) -> list[dc.PatchAttrs]:
        """Get the attributes of a h5simple file."""
        attrs, cm, data = _get_attrs_coords_and_data(resource, snap)
        attrs["coords"] = cm.to_summary_dict()
        attrs["path"] = resource.filename
        attrs["file_format"] = self.name
        attrs["file_version"] = self.version
        return [dc.PatchAttrs(**attrs)]
