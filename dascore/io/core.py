"""
Base functionality for reading, writing, determining file formats, and scanning
Das Data.
"""
import os.path
from abc import ABC
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Union

import pkg_resources

import dascore
import dascore as dc
from dascore.constants import SpoolType, timeable_types
from dascore.core.schema import PatchFileSummary
from dascore.exceptions import DASCoreError, InvalidFileFormatter, UnknownFiberFormat
from dascore.utils.docs import compose_docstring
from dascore.utils.hdf5 import HDF5ExtError


class _FiberIOManager:
    """
    A structure for intelligently storing, loading, and return FiberIO objects.

    This should only be used in conjunction with `FiberIO`.
    """

    def __init__(self, entry_point: str):
        self._entry_point = entry_point
        self._loaded_eps: dict[str, "dc.io.core.FiberIO"] = {}
        self._format_version = defaultdict(dict)
        self._extention_list = defaultdict(list)

    @cached_property
    def _eps(self):
        """
        Get the unlaoded entry points registered to this domain into a dict of
        {name: ep}
        """
        out = {
            ep.name: ep.load
            for ep in pkg_resources.iter_entry_points(self._entry_point)
        }
        return out

    def __iter__(self):
        names = sorted(set(self._eps) | set(self._loaded_eps))
        for name in names:
            yield name

    def items(self):
        """return items and content."""
        for name in sorted(set(self._eps) | set(self._loaded_eps)):
            yield name, self[name]

    def __getitem__(self, item):
        if item in self._eps or item in self._loaded_eps:
            if item not in self._loaded_eps:  # load unloaded entry points
                self._eps[item]()
                assert item in self._loaded_eps
            return self._loaded_eps[item]
        else:
            known_formats = set(self._loaded_eps) | set(self._eps)
            msg = (
                f"File format {item} is unknown to DASCore. Known formats "
                f"are: [{', '.join(sorted(known_formats))}]"
            )
            raise UnknownFiberFormat(msg)

    def __setitem__(self, key, value: "dc.io.core.FiberIO"):
        """Set the loaded (instances of) formatters."""
        self._loaded_eps[key] = value
        self._format_version[key][value.file_version] = value
        self._extention_list[key].append(value)

    def get_formatter_list(
        self,
        file_format: Optional[str] = None,
        file_version: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        """
        Get a list of formatters appropriate to the specified inputs.

        The list is sorted in likelihoold of the formatter being correct. For
        example, if file format is specified but file_version is not, all
        formatters for the format will be returned with the newest versions
        first in the list.

        If neither version or format are specified but extension is all formatters
        specifying the extension will be first in the list, sorted by format name
        and format version.

        If nothing is specified, all formatters will be returned sorted by name
        and version.

        Parameters
        ----------
        file_format
            The format string indicating the format name
        file_version
            The version string of the format
        extension
            The extension of the file.
        """
        if file_format is not None:

            if file_version is not None:
                pass


# ------------- Protocol for File Format support

# _Manager = _FiberIOManager("dascore.plugin.fiber_io")


class FiberIO(ABC):
    """
    An interface which adds support for a given filer format.

    This class should be subclassed when adding support for new formats.
    """

    name: str = ""
    preferred_extensions: tuple[str] = ()
    file_version = None
    _manager = _FiberIOManager("dascore.plugin.fiber_io")

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

        This should only work if path is the supported file format, otherwise
        raise UnknownFiberError or return False.
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
        manager = getattr(cls.__mro__[1], "_manager")
        manager[cls.name.upper()] = cls()

    @classmethod
    @compose_docstring(doc_str=_FiberIOManager.get_formatter_list.__doc__)
    def get_formater_list(
        cls,
        file_format=None,
        file_version=None,
        extensions=None,
    ):
        """
        {doc_str}
        """
        return cls._manager.get_formatter_list(
            file_format=file_format,
            file_version=file_version,
            extension=extensions,
        )


def read(
    path: Union[str, Path],
    file_format: Optional[str] = None,
    file_version: Optional[str] = None,
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
    file_version
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
    formatter = FiberIO._manager[file_format.upper()]
    return formatter.read(
        path, file_version=file_version, time=time, distance=distance, **kwargs
    )


@compose_docstring(fields=list(PatchFileSummary.__annotations__))
def scan(
    path: Union[Path, str],
    file_format: Optional[str] = None,
    file_version: Optional[str] = None,
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
    out = FiberIO._manager[file_format].scan(path)
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
    for name, formatter in FiberIO._manager.items():
        try:
            format = formatter.get_format(path)
        except (ValueError, TypeError, HDF5ExtError, NotImplementedError, DASCoreError):
            continue
        if format:
            return format
    else:
        msg = f"Could not determine file format of {path}"
        raise UnknownFiberFormat(msg)


def write(
    patch_or_spool,
    path: Union[str, Path],
    file_format: str,
    file_version: Optional[str] = None,
    **kwargs,
):
    """
    Write a Patch or Spool to disk.

    Parameters
    ----------
    path
        The path to the file.
    file_format
        The string indicating the format to write.
    file_version
        Optionally specify the version of the file, else use the latest
        version.

    Raises
    ------
    dascore.exceptions.UnknownFiberFormat - Could not determine the fiber format.
    """
    formatter = FiberIO._manager[file_format.upper()]
    if not isinstance(patch_or_spool, dascore.MemorySpool):
        patch_or_spool = dascore.MemorySpool([patch_or_spool])
    formatter.write(patch_or_spool, path, **kwargs)
