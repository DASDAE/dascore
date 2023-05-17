"""
IO module for reading Optasense DAS data.
"""
from pathlib import Path
from typing import List, Optional, Union

import dascore as dc
from dascore.constants import timeable_types
from dascore.core.schema import PatchFileSummary
from dascore.io.core import FiberIO
from dascore.utils.hdf5 import HDF5ExtError, NoSuchNodeError, open_hdf5_file

from .utils import _get_qunatx_version_str, _read_quantx, _scan_quantx


class QuantXV2(FiberIO):
    """
    Support for Optasense data format, version 2.
    """

    name = "QUANTX"
    preferred_extensions = ("hdf5", "h5")
    version = "2.0"

    def get_format(self, path: Union[str, Path]) -> Union[tuple[str, str], bool]:
        """
        Return True if file contains Optasense version 2 data else False.

        Parameters
        ----------
        path
            A path to the file which may contain Optasense data.
        """
        try:
            with open_hdf5_file(path, "r") as fi:
                version_str = _get_qunatx_version_str(fi)
                if version_str:
                    return (self.name, version_str)
        except (HDF5ExtError, OSError, IndexError, KeyError, NoSuchNodeError):
            return False

    def scan(self, path: Union[str, Path]) -> List[PatchFileSummary]:
        """
        Scan an Optasense v2 file, return summary information about the file's contents.
        """
        with open_hdf5_file(path) as fi:
            return _scan_quantx(self, fi, path)

    def read(
        self,
        path: Union[str, Path],
        time: Optional[tuple[timeable_types, timeable_types]] = None,
        distance: Optional[tuple[float, float]] = None,
        **kwargs
    ) -> dc.BaseSpool:
        """
        Read an Optasense file.
        """
        # TODO need to create h5 file decorator to avoid too many open/close files.
        with open_hdf5_file(path) as fi:
            patch = _read_quantx(fi.get_node("/Acquisition"), time, distance)
        return dc.spool(patch)
