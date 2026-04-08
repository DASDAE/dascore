"""
Core module for reading Sintela protobuf format.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.core.summary import PatchSummary
from dascore.io import FiberIO
from dascore.utils.io import BinaryReader

from .utils import get_supported_family_tag, read_payload, scan_payload


class SintelaProtobufV1(FiberIO):
    """IO class for Sintela protobuf MTLV recordings."""

    name = "Sintela_Protobuf"
    preferred_extensions = ("pb",)
    version = "1"

    def get_format(self, resource: BinaryReader, **kwargs) -> tuple[str, str] | bool:
        """Return the format/version tuple if the file is Sintela protobuf."""
        tag = get_supported_family_tag(resource)
        return (self.name, self.version) if tag else False

    def scan(self, resource: BinaryReader, **kwargs) -> list[PatchSummary]:
        """Scan a Sintela protobuf recording."""
        return scan_payload(resource)

    def read(self, resource: BinaryReader, **kwargs) -> dc.BaseSpool:
        """Read a Sintela protobuf recording into a spool."""
        data, coords, attrs = read_payload(resource)
        selectors = {name: kwargs[name] for name in coords.dims if name in kwargs}
        if selectors:
            coords, data = coords.select(data, **selectors)
        if not np.size(data):
            return dc.spool([])
        return dc.spool([dc.Patch(data=data, coords=coords, attrs=attrs)])
