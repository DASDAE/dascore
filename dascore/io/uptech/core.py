"""Reader for HDF5 files exported by Uptech Sensing interrogators."""

from __future__ import annotations

import dascore as dc
from dascore.constants import snap_type
from dascore.io import ArraySource, FiberIO, H5ArrayMixin
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader

from .utils import _DATASET, _get_attrs_dict, _get_coords, _is_uptech


class UptechPatchAttrs(dc.PatchAttrs):
    """Patch attrs for Uptech Sensing files. Lengths are in meters."""

    fiber_length: OptionalFiniteFloat = None
    gauge_length: OptionalFiniteFloat = None
    spatial_resolution: OptionalFiniteFloat = None


class UptechH5V1(H5ArrayMixin, FiberIO):
    """Support Uptech Sensing AS1000 HDF5 exports."""

    name = "Uptech_H5"
    version = "1"
    preferred_extensions = ("hdf5", "h5")

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        return self.version if _is_uptech(resource) else None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.PatchMeta]:
        """Extract metadata without reading the signal array."""
        attrs = UptechPatchAttrs.model_validate(_get_attrs_dict(resource))
        return [
            dc.PatchMeta(
                attrs=attrs,
                coords=_get_coords(resource, snap=snap),
                dtype=str(resource[_DATASET].dtype),
                source=ArraySource(key=resource[_DATASET].name),
            )
        ]
