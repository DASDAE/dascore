"""IO module for reading DASHDF5 (CF convention) data."""

from __future__ import annotations

import dascore as dc
from dascore.constants import snap_type
from dascore.io import ArraySource, FiberIO, H5ArrayMixin
from dascore.utils.hdf5 import H5Reader

from .utils import (
    _get_cf_attrs,
    _get_cf_coords,
    _get_cf_version_str,
)


class DASHDF5(H5ArrayMixin, FiberIO):
    """IO Support for DASHDF5 which uses CF version 1.7."""

    name = "DASHDF5"
    preferred_extensions = ("hdf5", "h5")
    version = "1.0"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        version_str = _get_cf_version_str(resource)
        if version_str:
            return version_str
        return None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.PatchMeta]:
        """Get metadata from file."""
        coords = _get_cf_coords(resource, snap=snap)
        attrs = _get_cf_attrs(resource, coords)
        return [
            dc.PatchMeta(
                attrs=attrs,
                coords=coords,
                dtype=str(resource["das"].dtype),
                source=ArraySource(key=resource["das"].name),
            )
        ]
