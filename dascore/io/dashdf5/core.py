"""IO module for DAS-HDF5 support."""
from __future__ import annotations

from dascore.io import FiberIO, HDF5Reader

from .utils import _get_das_hdf_version


class DASHDF5V1(FiberIO):
    """Support for DAS-HDF5 1.0."""

    name = "DAS-HDF5"
    preferred_extensions = ("hdf5", "h5")
    version = "1.0"

    def get_format(self, resource: HDF5Reader) -> tuple[str, str] | bool:
        """Determine if is simple h5 format."""
        version = _get_das_hdf_version(resource)
        if version:
            return self.name, self.version

        return False

    #
    # def read(self, resource: HDF5Reader, snap=True, **kwargs) -> SpoolType:
    #     """
    #     Read a simple h5 file.
    #
    #     Parameters
    #     ----------
    #     resource
    #         The open h5 object.
    #     snap
    #         If True, snap each coordinate to be evenly sampled.
    #     **kwargs
    #         Passed to filtering coordinates.
    #     """
    #     attrs, cm, data = _get_attrs_coords_and_data(resource, snap, self)
    #     new_cm, new_data = _maybe_trim_data(cm, data, kwargs)
    #     patch = dc.Patch(coords=new_cm, data=new_data[:], attrs=attrs)
    #     return dc.spool([patch])
    #
    # def scan(self, resource: HDF5Reader, snap=True) -> list[dc.PatchAttrs]:
    #     """Get the attributes of a h5simple file."""
    #     attrs, cm, data = _get_attrs_coords_and_data(resource, snap, self)
    #     attrs["coords"] = cm.to_summary_dict()
    #     attrs["path"] = resource.filename
    #     return [dc.PatchAttrs(**attrs)]
