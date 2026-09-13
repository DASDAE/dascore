"""
Core modules for reading GDR data.

GDR files do not specify the GDR version directly. Instead, they use versions
from other standards for the metadata and raw data. These can be found in the
overview attributes MetadataStandard and RawDataStandard.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.io.gdr.utils_das import _get_attrs_coords_and_data, _get_version
from dascore.io.utils import build_patches, slice_dataset
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader
from dascore.utils.misc import raise_on_extra_kwargs

from .utils_das import _get_dims


class GDRPatchAttrs(dc.PatchAttrs):
    """Patch attrs for GDR files."""

    gauge_length: OptionalFiniteFloat
    project_number: str = ""


class GDR_V1(FiberIO):  # noqa
    """
    Support for GDR version 1.
    """

    name = "GDR_DAS"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """Determine if the resource belongs to this format."""
        return _get_version(resource)

    def read(
        self, resource: H5Reader, snap=True, time=None, distance=None, **kwargs
    ) -> dc.Spool:
        """
        Read a resource belonging to this format.

        Parameters
        ----------
        resource
            The open h5 object.
        snap
            If True, snap each coordinate to be evenly sampled.
        time
            An optional tuple for filtering time.
        distance
            An optional tuple for filtering distance.
        """
        attr_dict, cm, data = _get_attrs_coords_and_data(resource, snap=snap)
        patches = build_patches(
            cm,
            data,
            attr_dict,
            attr_cls=GDRPatchAttrs,
            selection={"time": time, "distance": distance},
        )
        return dc.spool(patches)

    def read_array(
        self,
        resource: H5Reader,
        windows: dict[str, tuple[int, int]],
        snap: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Slice the ``DasRawData/RawData`` dataset directly.
        """
        raise_on_extra_kwargs(kwargs, "windows and snap")
        return slice_dataset(
            resource["DasRawData/RawData"],
            _get_dims(resource["DasRawData/RawData"]),
            windows,
        )

    def scan(self, resource: H5Reader, snap=True, **kwargs) -> list[ScanPayload]:
        """Get the attributes of a resource belong to this type."""
        attrs, cm, data = _get_attrs_coords_and_data(resource, snap)
        return [
            make_scan_payload(
                attrs=GDRPatchAttrs.from_dict(attrs), coords=cm, dtype=str(data.dtype)
            )
        ]
