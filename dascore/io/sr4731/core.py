"""
Core module for reading SR-4731 OTDR SOR files.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type
from dascore.io import BinaryReader, FiberIO
from dascore.io.utils import slice_dataset

from .utils import SR4731PatchAttrs, _get_format, _get_patch_attrs, _parse_sor


class SR4731V200(FiberIO):
    """Support for version 200 SR-4731 SOR files."""

    name = "SR4731"
    version = "200"
    preferred_extensions = ("sor",)

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        return (
            _match[1]
            if (_match := _get_format(resource, self.name, self.version))
            else None
        )

    def get_metadata(
        self, resource: BinaryReader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Scan an SR-4731 SOR file."""
        attrs = _get_patch_attrs(resource, SR4731PatchAttrs)
        return [attrs]

    def read_array(
        self, resource: BinaryReader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Decode the trace and select its time/distance sample window."""
        parsed = _parse_sor(resource, load_samples=True)
        data = parsed["data_points"]["samples"][np.newaxis, :]
        return slice_dataset(data, ("time", "distance"), windows)
