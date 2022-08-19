"""
Base functionality for reading, writing, determining file formats, and scanning
Das Data.
"""
import os.path
from abc import ABC
from pathlib import Path
from typing import List, Optional, Union

import dascore
from dascore.constants import SpoolType, timeable_types
from dascore.core.schema import PatchFileSummary
from dascore.exceptions import InvalidFileFormatter, UnknownFiberFormat
from dascore.utils.docs import compose_docstring
from dascore.utils.plugin import FiberIOManager

# ------------- Protocol for File Format support

_IO_INSTANCES = FiberIOManager("dascore.plugin.fiber_io")


class FiberIO(ABC):
    """
    An interface which adds support for a given filer format.

    This class should be subclassed when adding support for new Patch/Spool
    formats.
    """

    name: str = ""
    preferred_extensions: tuple[str] = ()

    def read(self, path, **kwargs) -> SpoolType:
        """
        Load data from a path.

        *kwargs should include support for selecting expected dimensions. For
        example, distance=(100, 200) would only read data with distance from
        100 to 200.
        """
        msg = f"FileFormatter: {self.name} has no read method"
        raise NotImplementedError(msg)

    def scan(self, path) -> List[PatchFileSummary]:
        """
        Returns a list of summary info for patches contained in file.
        """
        # default scan method reads in the file and returns required attributes
        # however, this can be very slow, so each parser should implement scan
        # when possible.
        try:
            spool = self.read(path)
        except NotImplementedError:
            msg = f"FileFormatter: {self.name} has no scan or read method"
            raise NotImplementedError(msg)
        out = []
        for pa in spool:
            info = dict(pa.attrs)
            info["file_format"] = self.name
            info["path"] = str(path)
            out.append(PatchFileSummary.parse_obj(info))
        return out

    def write(self, spool: SpoolType, path: Union[str, Path]):
        """
        Write the spool to disk
        """
        msg = f"FileFormatter: {self.name} has no write method"
        raise NotImplementedError(msg)

    def get_format(self, path) -> Union[tuple[str, str], bool]:
        """
        Return a tuple of (format_name, version_numbers).

        This only works if path is supported, otherwise raise UnknownFiberError
        or return False.
        """
        msg = f"FileFormatter: {self.name} has no get_version method"
        raise NotImplementedError(msg)

    def __init_subclass__(cls, **kwargs):
        """
        Hook for registering subclasses.
        """
        # check that the subclass is valid
        if not cls.name:
            msg = "You must specify the file format with the name field."
            raise InvalidFileFormatter(msg)
        # register formatter
        _IO_INSTANCES[cls.name.upper()] = cls()


def read(
    path: Union[str, Path],
    file_format: Optional[str] = None,
    version: Optional[str] = None,
    time: Optional[tuple[Optional[timeable_types], Optional[timeable_types]]] = None,
    distance: Optional[tuple[Optional[float], Optional[float]]] = None,
    **kwargs,
) -> SpoolType:
    """
    Read a fiber file.

    Parameters
    ----------
    path
        A path to the file to read.
    file_format
        A string indicating the file format. If not provided dascore will
        try to estimate the format.
    version
        An optional string indicating the format version.
    time
        An optional tuple of time ranges.
    distance
        An optional tuple of distances.
    *kwargs
        All kwargs are passed to the format-specific read functions.
    """
    if not file_format:
        file_format = get_format(path)[0].upper()
    formatter = _IO_INSTANCES[file_format.upper()]
    return formatter.read(path, version=version, time=time, distance=distance, **kwargs)


@compose_docstring(fields=list(PatchFileSummary.__annotations__))
def scan(
    path: Union[Path, str],
    file_format: Optional[str] = None,
) -> List[PatchFileSummary]:
    """
    Scan a file, return the summary dictionary.

    Parameters
    ----------
    path
        The path the to file to scan
    file_format
        Format of the file. If not provided DASCore will try to determine it.

    Notes
    -----
    The summary dictionaries contain the following fields:
        {fields}
    """
    # dispatch to file format handlers
    if file_format is None:
        file_format = get_format(path)[0]
    out = _IO_INSTANCES[file_format].scan(path)
    return out


def get_format(path: Union[str, Path]) -> (str, str):
    """
    Return the name of the format contained in the file and version number.

    Parameters
    ----------
    path
        The path to the file.

    Returns
    -------
    A tuple of (file_format_name, version) both as strings.

    Raises
    ------
    dascore.exceptions.UnknownFiberFormat - Could not determine the fiber format.

    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")
    for name, formatter in _IO_INSTANCES.items():
        try:
            format = formatter.get_format(path)
        except Exception:  # NOQA
            continue
        if format:
            return format
    else:
        msg = f"Could not determine file format of {path}"
        raise UnknownFiberFormat(msg)


def write(patch_or_spool, path: Union[str, Path], file_format: str, **kwargs):
    """
    Write a Patch or Stream to disk.

    Parameters
    ----------
    path
        The path to the file.
    file_format
        The string indicating the format to write.

    Raises
    ------
    dascore.exceptions.UnknownFiberFormat - Could not determine the fiber format.

    """
    formatter = _IO_INSTANCES[file_format.upper()]
    if not isinstance(patch_or_spool, dascore.MemorySpool):
        patch_or_spool = dascore.MemorySpool([patch_or_spool])
    formatter.write(patch_or_spool, path, **kwargs)
