"""
IO module for reading prodML data.
"""
from pathlib import Path
from typing import List, Optional, Union

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.core.schema import PatchFileSummary
from dascore.io.core import FiberIO
from dascore.utils.hdf5 import HDF5ExtError, NoSuchNodeError, open_hdf5_file

from .utils import _get_prodml_attrs, _get_prodml_version_str, _read_prodml


class ProdMLV2_0(FiberIO):
    """
    Support for ProdML V 2.0.
    """

    name = "PRODML"
    preferred_extensions = ("hdf5", "h5")
    version = "2.0"

    def get_format(self, path: Union[str, Path]) -> Union[tuple[str, str], bool]:
        """
        Return True if file contains terra15 version 2 data else False.

        Parameters
        ----------
        path
            A path to the file which may contain terra15 data.
        """
        try:
            with open_hdf5_file(path, "r") as fi:
                version_str = _get_prodml_version_str(fi)
                if version_str:
                    return (self.name, version_str)
        except (HDF5ExtError, OSError, IndexError, KeyError, NoSuchNodeError):
            return False

    def scan(self, path: Union[str, Path]) -> List[PatchFileSummary]:
        """
        Scan a prodml file, return summary information about the file's contents.
        """
        with open_hdf5_file(path) as fi:
            file_version = _get_prodml_version_str(fi)
            extras = {
                "path": path,
                "file_format": self.name,
                "file_version": str(file_version),
            }
            return _get_prodml_attrs(fi, extras=extras, cls=PatchFileSummary)

    def read(
        self,
        path: Union[str, Path],
        time: Optional[tuple[opt_timeable_types, opt_timeable_types]] = None,
        distance: Optional[tuple[Optional[float], Optional[float]]] = None,
        **kwargs
    ) -> dc.BaseSpool:
        """
        Read a ProdML file.
        """
        kwargs = {}
        if time is not None:
            kwargs["time"] = time
        if distance is not None:
            kwargs["distance"] = distance
        with open_hdf5_file(path) as fi:
            patches = _read_prodml(fi, **kwargs)
        return dc.spool(patches)


class ProdMLV2_1(ProdMLV2_0):
    """
    Support for ProdML V 2.1.
    """

    version = "2.1"
