"""
Core module for reading SR-4731 OTDR SOR files.
"""

from __future__ import annotations

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import BinaryReader, FiberIO

from .utils import SR4731PatchAttrs, _get_format, _get_patch_attrs, _get_patches


class SR4731V200(FiberIO):
    """Support for version 200 SR-4731 SOR files."""

    name = "SR4731"
    version = "200"
    preferred_extensions = ("sor",)

    def _get_extras(self, resource) -> dict[str, str]:
        """Return DASCore IO provenance metadata."""
        return {
            "path": getattr(resource, "name", ""),
            "file_format": self.name,
            "file_version": self.version,
        }

    def get_format(self, resource: BinaryReader, **kwargs) -> tuple[str, str] | bool:
        """Return format and version if the file is a supported SR-4731 file."""
        return _get_format(resource, self.name, self.version)

    def scan(self, resource: BinaryReader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan an SR-4731 SOR file."""
        attrs = _get_patch_attrs(
            resource,
            SR4731PatchAttrs,
            extras=self._get_extras(resource),
        )
        return [attrs]

    def read(
        self,
        resource: BinaryReader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read an SR-4731 SOR file."""
        patches = _get_patches(
            resource,
            time=time,
            distance=distance,
            attr_class=SR4731PatchAttrs,
            extras=self._get_extras(resource),
        )
        return dc.spool(patches)
