"""IO module for reading Sentek's DAS data format."""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.io.core import FiberIO, ScanPayload, make_scan_payload
from dascore.utils.io import BinaryReader, LocalBinaryReader

from .utils import _get_patch_attrs, _get_version


class SentekV5(FiberIO):
    """Support for Sentek Instrument data format."""

    name = "sentek"
    version = "5"
    preferred_extensions = ("das",)

    def read(
        self,
        resource: LocalBinaryReader,
        time=None,
        distance=None,
        **kwargs,
    ) -> dc.Spool:
        """Read a Sentek das file, return a DataArray."""
        attrs, coords, offsets = _get_patch_attrs(resource)
        resource.seek(offsets[0])
        array = np.fromfile(resource, dtype=np.float32, count=offsets[1] * offsets[2])
        array = np.reshape(array, (offsets[1], offsets[2])).T
        patch = dc.Patch(data=array, attrs=attrs, coords=coords, dims=coords.dims)
        # Note: we are being a bit sloppy here in that selecting on
        # time/distance doesn't actually affect how much data is read from
        # the binary file. This is probably ok though since Sentek files
        # tend to be quite small.
        # The bare function: a patch read under a bound carries what a
        # whole read carries, and the spool records the trim it asked for.
        patch = patch.select.raw_function(patch, time=time, distance=distance)
        return dc.spool([patch]) if patch.data.size else dc.spool([])

    def get_format(
        self,
        resource: BinaryReader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """Auto detect sentek format."""
        return _get_version(resource)

    def scan(self, resource: BinaryReader, **kwargs) -> list[ScanPayload]:
        """Extract metadata from sentek file."""
        attrs, coords, _ = _get_patch_attrs(resource)
        return [
            make_scan_payload(
                attrs=attrs,
                coords=coords,
                dtype=str(np.dtype(np.float32)),
            )
        ]
