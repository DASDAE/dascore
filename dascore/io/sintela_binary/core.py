"""
Core module for reading Sintela binary format.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.io import BinaryReader

from .utils import (
    _HEADER_SIZES,
    SYNC_WORD,
    _get_attrs_coords_header,
    _get_patch,
    _read_base_header,
)


class SintelaPatchAttrs(dc.PatchAttrs):
    """Patch Attributes for Sintela binary format."""

    gauge_length: float = np.nan
    gauge_length_units: str = "m"


class SintelaBinaryV3(FiberIO):
    """Version 3 of Sintela's binary format."""

    name = "Sintela_Binary"
    preferred_extensions = ("raw",)
    version = "3"

    def get_format(self, resource: BinaryReader, **kwargs) -> tuple[str, str] | bool:
        """
        Return name and version string or False.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        resource.seek(0)
        base = _read_base_header(resource)
        sync = base["sync_word"]
        version = str(base["version"])
        size = base["header_size"]
        expected_size = _HEADER_SIZES.get(version, 0)
        if sync == SYNC_WORD and version == self.version and size == expected_size:
            return self.name, version
        return False

    def scan(self, resource: BinaryReader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan a file, return summary information on the contents."""
        extras = {
            "path": resource.name,
            "file_format": self.name,
            "file_version": self.version,
        }
        attrs, _, _ = _get_attrs_coords_header(
            resource, SintelaPatchAttrs, extras=extras
        )

        return [attrs]

    def read(
        self,
        resource: BinaryReader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a single Sintela binary file."""
        patch = _get_patch(
            resource, time=time, distance=distance, attr_class=SintelaPatchAttrs
        )
        return dc.spool(patch)
