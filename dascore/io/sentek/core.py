"""IO module for reading Sentek's DAS data format."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type
from dascore.io.core import FiberIO
from dascore.io.utils import slice_dataset
from dascore.utils.io import BinaryReader, LocalBinaryReader

from .utils import _get_patch_attrs, _get_version


class SentekV5(FiberIO):
    """Support for Sentek Instrument data format."""

    name = "sentek"
    version = "5"
    preferred_extensions = ("das",)

    def read_array(
        self,
        resource: LocalBinaryReader,
        windows: dict[str, tuple[int, int]],
        key: str = "",
    ) -> np.ndarray:
        """Decode the stored layout and select in the metadata's dimension order."""
        _, coords, offsets = _get_patch_attrs(resource)
        resource.seek(offsets[0])
        data = np.fromfile(resource, dtype=np.float32, count=offsets[1] * offsets[2])
        data = data.reshape((offsets[1], offsets[2])).T
        return slice_dataset(data, coords.dims, windows)

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        return _match[1] if (_match := _get_version(resource)) else None

    def get_metadata(
        self, resource: BinaryReader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Extract metadata from sentek file."""
        attrs, coords, _ = _get_patch_attrs(resource)
        return [dc.Patch(attrs=attrs, coords=coords, dtype=str(np.dtype(np.float32)))]
