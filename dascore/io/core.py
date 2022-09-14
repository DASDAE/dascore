"""
Base functionality for reading, writing, determining file formats, and scanning
Das Data.
"""
import os.path
from abc import ABC
from collections import defaultdict
from functools import cache, cached_property
from importlib.metadata import entry_points
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

import dascore
from dascore.constants import PatchType, SpoolType, timeable_types
from dascore.core.schema import PatchFileSummary
from dascore.exceptions import (
    DASCoreError,
    InvalidFiberFile,
    InvalidFileFormatter,
    UnknownFiberFormat,
)
from dascore.utils.docs import compose_docstring
from dascore.utils.hdf5 import HDF5ExtError
from dascore.utils.misc import suppress_warnings
from dascore.utils.patch import scan_patches
from dascore.utils.pd import list_ser_to_str


class _FiberIOManager:
    """
    A structure for intelligently storing, loading, and return FiberIO objects.

    This should only be used in conjunction with `FiberIO`.
    """

    def __init__(self, entry_point: str):
        self._entry_point = entry_point
        self._loaded_eps: set[str] = set()
        self._format_version = defaultdict(dict)
        self._extension_list = defaultdict(list)

    @cached_property
    def _eps(self):
        """
        Get the unlaoded entry points registered to this domain into a dict of
        {name: ep}
        """
        # TODO remove warning suppression and switch to select when 3.9 is dropped
        # see https://docs.python.org/3/library/importlib.metadata.html#entry-points
        with suppress_warnings(DeprecationWarning):
            out = {ep.name: ep.load for ep in entry_points()[self._entry_point]}
        return pd.Series(out)

    @cached_property
    def known_formats(self):
        """Return names of known formats."""
        formats = self._eps.index.str.split("__").str[0]
        return set(formats) | set(self._format_version)

    @property
    def unloaded_formats(self):
        """Return names of known formats."""
        return sorted(self.known_formats - set(self._format_version))

    @cached_property
    def _prioritized_list(self):
        """Yield a prioritized list of formatters."""
        # must load all plugins before getting list
        self.load_plugins()
        priority_formatters = []
        second_class_formatters = []
        for format_name in self.known_formats:
            unsorted = self._format_version[format_name]
            keys = sorted(unsorted, reverse=True)
            formatters = [unsorted[key] for key in keys]
            priority_formatters.append(formatters[0])
            if len(formatters) > 1:
                second_class_formatters.extend(formatters[1:])
        return tuple(priority_formatters + second_class_formatters)

    @cache
    def load_plugins(self, format: Optional[str] = None):
        """Load plugin for specific format or ensure all formats are loaded"""
        if format is not None and format in self._format_version:
            return  # already loaded
        if not (unloaded := self.unloaded_formats):
            return
        formats = {format} if format is not None else unloaded
        # load one, or all, formats
        for form in formats:
            for eps in self._eps.loc[self._eps.index.str.startswith(form)]:
                self.register_fiberio(eps()())
        # The selected format(s) should now be loaded
        assert set(formats).isdisjoint(self.unloaded_formats)

    def register_fiberio(self, fiberio: "FiberIO"):
        """Register a new fiber IO to manage."""
        forma, ver = fiberio.name.upper(), fiberio.version
        self._loaded_eps.add(fiberio.name)
        for ext in iter(fiberio.preferred_extensions):
            self._extension_list[ext].append(fiberio)
        self._format_version[forma][ver] = fiberio

    def get_fiberio(
        self,
        format: Optional[str] = None,
        version: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        """
        Return the most likely formatter for given inputs.

        If no such formatter exists, raise UnknownFiberFormat error.

        Parameters
        ----------
        format
            The format string indicating the format name
        version
            The version string of the format
        extension
            The extension of the file.
        """
        iterator = self.yield_fiberio(
            format=format,
            version=version,
            extension=extension,
        )
        for formatter in iterator:
            return formatter
        msg = (
            f"No fiberio instance found for format: {format} "
            f"version: {version} and extension: {extension}"
        )
        raise UnknownFiberFormat(msg)

    def yield_fiberio(
        self,
        format: Optional[str] = None,
        version: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        """
        Yields fiber IO object based on input priorities.

        The order is sorted in likelihood of the formatter being correct. For
        example, if file format is specified but file_version is not, all
        formatters for the format will be yielded with the newest versions
        first in the list.

        If neither version nor format are specified but extension is all formatters
        specifying the extension will be first in the list, sorted by format name
        and format version.

        If nothing is specified, all formatters will be returned starting with
        the newest (the highest version) of each formatter, followed by older
        versions.

        Parameters
        ----------
        format
            The format string indicating the format name
        version
            The version string of the format
        extension
            The extension of the file.
        """
        # TODO replace this with concise pattern matching once 3.9 is dropped
        if version and not format:
            msg = "Providing only a version is not sufficient to determine format"
            raise UnknownFiberFormat(msg)
        if format is not None:
            self.load_plugins(format)
            yield from self._yield_format_version(format, version)
        elif extension is not None:
            yield from self._yield_extensions(extension)
        else:
            yield from self._prioritized_list

    def _yield_format_version(self, format, version):
        """Yield file format/version prioritized formatters."""
        if format is not None:
            format = format.upper()
            self.load_plugins(format)
            formatters = self._format_version.get(format, None)
            # no format found
            if not formatters:
                format_list = list(self.known_formats)
                msg = f"Unknown format {format}, " f"known formats are {format_list}"
                raise UnknownFiberFormat(msg)
            # a version is specified
            if version:
                formatter = formatters.get(version, None)
                if formatter is None:
                    msg = (
                        f"Format {format} has no verion: {version} "
                        f"known versions of this format are: {list(formatters)}"
                    )
                    raise UnknownFiberFormat(msg)
                yield formatter
                return
            # reverse sort formatters and yield latest version first.
            for formatter in dict(sorted(formatters.items(), reverse=True)).values():
                yield formatter
            return

    def _yield_extensions(self, extension):
        """generator to get formatter prioritized by preferred extensions."""
        has_yielded = set()
        self.load_plugins()
        for formatter in self._extension_list[extension]:
            yield formatter
            has_yielded.add(formatter)
        for formatter in self._prioritized_list:
            if formatter not in has_yielded:
                yield formatter


# ------------- Protocol for File Format support


class FiberIO(ABC):
    """
    An interface which adds support for a given filer format.

    This class should be subclassed when adding support for new formats.
    """

    name: str = ""
    version: str = ""
    preferred_extensions: tuple[str] = ()
    manager = _FiberIOManager("dascore.fiber_io")

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

    @property
    def implements_scan(self) -> bool:
        """
        Returns True if the subclass implements its own scan method else False.
        """
        return self.scan.__func__ is not FiberIO.scan

    @property
    def implements_get_format(self) -> bool:
        """Return True if the subclass implements its own get_format method."""
        return self.get_format.__func__ is not FiberIO.get_format

    def __hash__(self):
        """FiberIO instances should be uniquely defined by (format, version)"""
        return hash((self.name, self.version))

    def __init_subclass__(cls, **kwargs):
        """
        Hook for registering subclasses.
        """
        # check that the subclass is valid
        if not cls.name:
            msg = "You must specify the file format with the name field."
            raise InvalidFileFormatter(msg)
        # register formatter
        manager: _FiberIOManager = getattr(cls.__mro__[1], "manager")
        manager.register_fiberio(cls())


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
    if not file_format or not file_version:
        file_format, file_version = get_format(
            path,
            file_format=file_format,
            file_version=file_version,
        )
    formatter = FiberIO.manager.get_fiberio(file_format, file_version)
    return formatter.read(
        path, file_version=file_version, time=time, distance=distance, **kwargs
    )


@compose_docstring(fields=list(PatchFileSummary.__fields__))
def scan_to_df(
    path: Union[Path, str, PatchType, SpoolType],
    file_format: Optional[str] = None,
    file_version: Optional[str] = None,
    ignore: bool = False,
) -> pd.DataFrame:
    """
    Scan a path, return a dataframe of contents.

    Parameters
    ----------
    path
        The path the to file to scan
    file_format
        Format of the file. If not provided DASCore will try to determine it.
    ignore
        If True, ignore non-DAS files by returning an empty list, else raise
        UnknownFiberFormat if unreadable file encountered.

    Returns
    -------
    Return a dataframe with columns:
        {fields}
    """
    info = scan(
        path_or_spool=path,
        file_format=file_format,
        file_version=file_version,
        ignore=ignore,
    )
    df = pd.DataFrame([dict(x) for x in info]).assign(
        dims=lambda x: list_ser_to_str(x["dims"])
    )
    return df


@compose_docstring(fields=list(PatchFileSummary.__annotations__))
def scan(
    path_or_spool: Union[Path, str, PatchType, SpoolType],
    file_format: Optional[str] = None,
    file_version: Optional[str] = None,
    ignore: bool = False,
) -> List[PatchFileSummary]:
    """
    Scan a file, return the summary dictionary.

    Parameters
    ----------
    path_or_spool
        The path the to file to scan
    file_format
        Format of the file. If not provided DASCore will try to determine it.
        Only applicable for path-like inputs.
    file_version
        Version of the file. If not provided DASCore will try to determine it.
        Only applicable for path-like inputs.
    ignore
        If True, ignore non-DAS files by returning an empty list, else raise
        UnknownFiberFormat if unreadable file encountered.

    Notes
    -----
    The summary dictionaries contain the following fields:
        {fields}
    """
    if isinstance(path_or_spool, (str, Path)):
        return _scan_from_path(
            path_or_spool,
            file_format=file_format,
            file_version=file_version,
            ignore=ignore,
        )
    return scan_patches(path_or_spool)


def _scan_from_path(
    path: Union[Path, str, PatchType, SpoolType],
    file_format: Optional[str] = None,
    file_version: Optional[str] = None,
    ignore: bool = False,
):
    """Scan from a single path."""
    if not os.path.exists(path) or os.path.isdir(path):
        msg = f"{path} does not exist or is a directory"
        raise InvalidFiberFile(msg)
    # dispatch to file format handlers
    if not file_format or not file_version:
        try:
            file_format, file_version = get_format(
                path,
                file_format=file_format,
                file_version=file_version,
            )
        except UnknownFiberFormat as e:
            if ignore:
                return []
            else:
                raise e

    formatter = FiberIO.manager.get_fiberio(file_format, file_version)
    out = formatter.scan(path)
    return out


def get_format(
    path: Union[str, Path],
    file_format: Optional[str] = None,
    file_version: Optional[str] = None,
) -> tuple[str, str]:
    """
    Return the name of the format contained in the file and version number.

    Parameters
    ----------
    path
        The path to the file.
    file_format
        The known file format.
    file_version
        The known file version.


    Returns
    -------
    A tuple of (file_format_name, version) both as strings.

    Raises
    ------
    dascore.exceptions.UnknownFiberFormat - Could not determine the fiber format.

    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")
    ext = Path(path).suffix or None
    iterator = FiberIO.manager.yield_fiberio(file_format, file_version, extension=ext)
    for formatter in iterator:
        try:
            format_version = formatter.get_format(path)
        except (ValueError, TypeError, HDF5ExtError, NotImplementedError, DASCoreError):
            continue
        if format_version:
            return format_version
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
    formatter = FiberIO.manager.get_fiberio(file_format, file_version)
    if not isinstance(patch_or_spool, dascore.MemorySpool):
        patch_or_spool = dascore.MemorySpool([patch_or_spool])
    formatter.write(patch_or_spool, path, **kwargs)
