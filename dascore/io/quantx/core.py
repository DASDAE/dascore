"""
IO module for reading Optasense DAS data.
"""
from __future__ import annotations

import dascore as dc
from dascore.constants import timeable_types
from dascore.core.schema import PatchFileSummary
from dascore.io import FiberIO, HDF5Reader

from .utils import _get_qunatx_version_str, _read_quantx, _scan_quantx


class QuantXV2(FiberIO):
    """
    Support for Optasense data format, version 2.
    """

    name = "QUANTX"
    preferred_extensions = ("hdf5", "h5")
    version = "2.0"

    def get_format(self, resource: HDF5Reader) -> tuple[str, str] | bool:
        """
        Return True if file contains Optasense version 2 data else False.

        Parameters
        ----------
        resource
            A path to the file which may contain Optasense data.
        """
        version_str = _get_qunatx_version_str(resource)
        if version_str:
            return (self.name, version_str)

    def scan(self, resource: HDF5Reader) -> list[PatchFileSummary]:
        """
        Scan an Optasense v2 file, return summary information about the file's contents.
        """
        return _scan_quantx(self, resource)

    def read(
        self,
        resource: HDF5Reader,
        time: tuple[timeable_types, timeable_types] | None = None,
        distance: tuple[float, float] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """
        Read an Optasense file.
        """
        patch = _read_quantx(resource.get_node("/Acquisition"), time, distance)
        return dc.spool(patch)


if __name__ == "__main__":
    spool = dc.spool("myfile.hdf5")
