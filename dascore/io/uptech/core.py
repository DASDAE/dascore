"""Reader for HDF5 files exported by Uptech Sensing interrogators."""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.io.utils import build_patches, slice_dataset
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader
from dascore.utils.misc import raise_on_extra_kwargs

from .utils import _DATASET, _get_attrs_dict, _get_coords, _is_uptech


class UptechPatchAttrs(dc.PatchAttrs):
    """Patch attrs for Uptech Sensing files. Lengths are in meters."""

    fiber_length: OptionalFiniteFloat = None
    gauge_length: OptionalFiniteFloat = None
    spatial_resolution: OptionalFiniteFloat = None


class UptechH5V1(FiberIO):
    """Support Uptech Sensing AS1000 HDF5 exports."""

    name = "Uptech_H5"
    version = "1"
    preferred_extensions = ("hdf5", "h5")

    def get_format(
        self, resource: H5Reader, **kwargs
    ) -> tuple[str, str] | Literal[False]:
        """Return the format and version when resource is an Uptech file."""
        return (self.name, self.version) if _is_uptech(resource) else False

    def scan(self, resource: H5Reader, **kwargs) -> list[ScanPayload]:
        """Extract metadata without reading the signal array."""
        attrs = UptechPatchAttrs.model_validate(_get_attrs_dict(resource))
        return [
            make_scan_payload(
                attrs=attrs,
                coords=_get_coords(resource),
                dtype=str(resource[_DATASET].dtype),
            )
        ]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.Spool:
        """Read an Uptech HDF5 file, optionally selecting time and distance."""
        patches = build_patches(
            _get_coords(resource),
            resource[_DATASET],
            _get_attrs_dict(resource),
            attr_cls=UptechPatchAttrs,
            selection={"time": time, "distance": distance},
        )
        return dc.spool(patches)

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], **kwargs
    ) -> np.ndarray:
        """Slice the ``Acquisition/StrainRate`` dataset directly."""
        raise_on_extra_kwargs(kwargs, "windows")
        return slice_dataset(resource[_DATASET], ("time", "distance"), windows)
