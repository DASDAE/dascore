"""IO module for reading simple h5 data."""

from __future__ import annotations

from typing import Literal

import dascore as dc
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.io.utils import build_patches
from dascore.utils.hdf5 import H5Reader

from .utils import _get_attrs_coords_and_data, _is_h5simple


class H5Simple(FiberIO):
    """Support for bare-bones h5 format."""

    name = "H5Simple"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """Determine if is simple h5 format."""
        if _is_h5simple(resource):
            return self.name, self.version
        return False

    def read(self, resource: H5Reader, snap=True, **kwargs) -> dc.BaseSpool:
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
        attrs, cm, data = _get_attrs_coords_and_data(resource, snap)
        return dc.spool(build_patches(cm, data, attrs, selection=kwargs))

    def scan(self, resource: H5Reader, snap=True, **kwargs) -> list[ScanPayload]:
        """Get the attributes of a h5simple file."""
        attrs, cm, data = _get_attrs_coords_and_data(resource, snap)
        attrs.pop("file_format", None)
        attrs.pop("file_version", None)
        attrs = dc.PatchAttrs.from_dict(attrs)
        return [make_scan_payload(attrs=attrs, coords=cm, dtype=str(data.dtype))]
