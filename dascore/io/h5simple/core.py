"""IO module for reading simple h5 data."""

from __future__ import annotations

import dascore as dc
from dascore.constants import SpoolType
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader, PyTablesReader

from .utils import _get_attrs_coords_and_data, _is_h5simple, _maybe_trim_data


class H5Simple(FiberIO):
    """Support for bare-bones h5 format."""

    name = "H5Simple"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """Determine if is simple h5 format."""
        if _is_h5simple(resource):
            return self.name, self.version
        return False

    def read(self, resource: PyTablesReader, snap=True, **kwargs) -> SpoolType:
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
        attrs, cm, data = _get_attrs_coords_and_data(resource, snap, self)
        new_cm, new_data = _maybe_trim_data(cm, data, kwargs)
        patch = dc.Patch(coords=new_cm, data=new_data[:], attrs=attrs)
        return dc.spool([patch])

    def scan(
        self, resource: PyTablesReader, snap=True, **kwargs
    ) -> list[dc.PatchAttrs]:
        """Get the attributes of a h5simple file."""
        attrs, cm, data = _get_attrs_coords_and_data(resource, snap, self)
        attrs["coords"] = cm.to_summary_dict()
        attrs["path"] = resource.filename
        return [dc.PatchAttrs(**attrs)]
