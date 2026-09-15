"""
Core modules for reading GDR data.

GDR files do not specify the GDR version directly. Instead, they use versions
from other standards for the metadata and raw data. These can be found in the
overview attributes MetadataStandard and RawDataStandard.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type
from dascore.io import FiberIO
from dascore.io.gdr.utils_das import _get_attrs_coords_and_data, _get_version
from dascore.io.utils import slice_dataset
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader

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

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        return _match[1] if (_match := _get_version(resource)) else None

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Slice the ``DasRawData/RawData`` dataset directly.
        """
        return slice_dataset(
            resource["DasRawData/RawData"],
            _get_dims(resource["DasRawData/RawData"]),
            windows,
        )

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Get the attributes of a resource belong to this type."""
        attrs, cm, data = _get_attrs_coords_and_data(resource, snap)
        return [
            dc.Patch(
                attrs=GDRPatchAttrs.from_dict(attrs), coords=cm, dtype=str(data.dtype)
            )
        ]
