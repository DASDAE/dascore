"""IO module for reading Sentek's DAS data format."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.io import BinaryReader
from dascore.io.core import FiberIO

from .utils import _get_patch_attrs, _get_version


class SentekV5(FiberIO):
    """Support for Sentek Instrument data format."""

    name = "sentek"
    version = "5"
    preferred_extensions = ("das",)

    def read(
        self,
        resource: BinaryReader,
        time=None,
        distance=None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a Sentek das file, return a DataArray."""
        attrs, coords, offsets = _get_patch_attrs(resource)
        resource.seek(offsets[0])
        array = np.fromfile(resource, dtype=np.float32, count=offsets[1] * offsets[2])
        array = np.reshape(array, (offsets[1], offsets[2])).T
        patch = dc.Patch(data=array, attrs=attrs, coords=coords, dims=coords.dims)
        # Note: we are being a bit sloppy here in that selecting on time/distance
        # doesn't actually affect how much data is read from the binary file. This
        # is probably ok though since Sentek files tend to be quite small.
        return dc.spool(patch).select(time=time, distance=distance)

    def get_format(self, resource: BinaryReader, **kwargs) -> tuple[str, str] | bool:
        """Auto detect sentek format."""
        return _get_version(resource)

    def scan(self, resource: BinaryReader, **kwargs):
        """Extract metadata from sentek file."""
        extras = {
            "file_format": self.name,
            "file_version": self.version,
            "path": resource.name,
        }

        return [_get_patch_attrs(resource, extras=extras)[0]]
