"""
IO module for reading prodML data.
"""

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.core.schema import PatchFileSummary
from dascore.io import FiberIO, HDF5Reader

from .utils import _get_prodml_attrs, _get_prodml_version_str, _read_prodml


class ProdMLV2_0(FiberIO):
    """
    Support for ProdML V 2.0.
    """

    name = "PRODML"
    preferred_extensions = ("hdf5", "h5")
    version = "2.0"

    def get_format(self, resource: HDF5Reader) -> tuple[str, str] | bool:
        """
        Return True if file contains terra15 version 2 data else False.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        version_str = _get_prodml_version_str(resource)
        if version_str:
            return (self.name, version_str)

    def scan(self, resource: HDF5Reader) -> list[PatchFileSummary]:
        """
        Scan a prodml file, return summary information about the file's contents.
        """
        file_version = _get_prodml_version_str(resource)
        extras = {
            "path": resource.filename,
            "file_format": self.name,
            "file_version": str(file_version),
        }
        return _get_prodml_attrs(resource, extras=extras, cls=PatchFileSummary)

    def read(
        self,
        resource: HDF5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs
    ) -> dc.BaseSpool:
        """
        Read a ProdML file.
        """
        patches = _read_prodml(resource, time=time, distance=distance)
        return dc.spool(patches)


class ProdMLV2_1(ProdMLV2_0):
    """
    Support for ProdML V 2.1.
    """

    version = "2.1"
