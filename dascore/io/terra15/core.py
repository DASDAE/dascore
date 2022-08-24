"""
IO module for reading Terra15 DAS data.
"""
from pathlib import Path
from typing import List, Optional, Union

from dascore.constants import timeable_types
from dascore.core import MemorySpool
from dascore.core.schema import PatchFileSummary
from dascore.io.core import FiberIO
from dascore.utils.hdf5 import HDF5ExtError, NoSuchNodeError, open_hdf5_file

from .utils import _get_terra15_version_str, _read_terra15, _scan_terra15


class Terra15FormatterV4(FiberIO):
    """
    Support for Terra15 data format, version 4.
    """

    name = "TERRA15"
    preferred_extensions = ("hdf5", "h5")
    version = "4"

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
                version_str = _get_terra15_version_str(fi)
                if version_str:
                    return ("TERRA15", version_str)
        except (HDF5ExtError, OSError, IndexError, KeyError, NoSuchNodeError):
            return False

    def scan(self, path: Union[str, Path]) -> List[PatchFileSummary]:
        """
        Scan a terra15 v2 file, return summary information about the file's contents.
        """
        with open_hdf5_file(path) as fi:
            return _scan_terra15(self, fi, path)

    def read(
        self,
        path: Union[str, Path],
        time: Optional[tuple[timeable_types, timeable_types]] = None,
        distance: Optional[tuple[float, float]] = None,
        **kwargs
    ) -> MemorySpool:
        """
        Read a terra15 file.
        """

        # TODO need to create h5 file decorator to avoid too many open/close files.
        with open_hdf5_file(path) as fi:
            patch = _read_terra15(fi.root, time, distance)
        return MemorySpool([patch])


class Terra15FormatterV5(Terra15FormatterV4):
    """
    Support for Terra15 data format, version 5.
    """

    version = "5"
