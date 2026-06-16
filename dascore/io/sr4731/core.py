"""
Core module for reading SR-4731 OTDR SOR files.
"""

from __future__ import annotations

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import BinaryReader, FiberIO

from .utils import SR4731PatchAttrs, get_format, get_patch_attrs, get_patches


class SR4731V200(FiberIO):
    """Support for version 200 SR-4731 SOR files."""

    name = "SR4731"
    version = "200"
    preferred_extensions = ("sor",)

    def get_format(self, resource: BinaryReader, **kwargs) -> tuple[str, str] | bool:
        """Return format and version if the file is a supported SR-4731 file."""
        return get_format(resource, self.name, self.version)

    def scan(self, resource: BinaryReader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan an SR-4731 SOR file."""
        extras = {
            "path": getattr(resource, "name", ""),
            "file_format": self.name,
            "file_version": self.version,
        }
        attrs = get_patch_attrs(resource, SR4731PatchAttrs, extras=extras)
        return [attrs]

    def read(
        self,
        resource: BinaryReader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read an SR-4731 SOR file."""
        extras = {
            "path": getattr(resource, "name", ""),
            "file_format": self.name,
            "file_version": self.version,
        }
        patches = get_patches(
            resource,
            time=time,
            distance=distance,
            attr_class=SR4731PatchAttrs,
            extras=extras,
        )
        return dc.spool(patches)
