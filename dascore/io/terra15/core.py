"""IO module for reading Terra15 DAS data."""

from __future__ import annotations

import dascore as dc
from dascore.constants import timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader

from .utils import (
    _get_terra15_version_str,
    _get_version_data_node,
    _read_terra15,
    _scan_terra15,
)


class Terra15FormatterV4(FiberIO):
    """Support for Terra15 data format, version 4."""

    name = "TERRA15"
    preferred_extensions = ("hdf5", "h5")
    version = "4"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """
        Return True if file contains terra15 version 2 data else False.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        version_str = _get_terra15_version_str(resource)
        if version_str:
            return (self.name, version_str)

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan a terra15 v2 file, return summary information."""
        version, data_node = _get_version_data_node(resource)
        extras = {
            "path": resource.filename,
            "file_format": self.name,
            "file_version": str(version),
        }
        return _scan_terra15(resource, data_node, extras)

    def read(
        self,
        resource: H5Reader,
        time: tuple[timeable_types, timeable_types] | None = None,
        distance: tuple[float, float] | None = None,
        snap_dims: bool = True,
        **kwargs,
    ) -> dc.BaseSpool:
        """
        Read a terra15 file.

        Parameters
        ----------
        resource
            The path to the file.
        time
            A tuple for filtering time.
        distance
            A tuple for filtering distance.
        snap_dims
            If True, ensure the coordinates are evenly sampled monotonic.
            This will cause some loss in precision but it is usually
            negligible.
        """
        patch = _read_terra15(resource, time, distance, snap_dims=snap_dims)
        return dc.spool(patch)


class Terra15FormatterV5(Terra15FormatterV4):
    """Support for Terra15 data format, version 5."""

    version = "5"


class Terra15FormatterV6(Terra15FormatterV4):
    """Support for Terra15 data format, version 5."""

    version = "6"
