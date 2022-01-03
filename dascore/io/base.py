"""
Base functionality for reading, writting, determining file formats, and scanning
Das Data.
"""
import os.path
from pathlib import Path
from typing import List, Optional, Union

import dascore
from dascore.constants import PatchSummaryDict, StreamType, timeable_types
from dascore.exceptions import UnknownFiberFormat
from dascore.utils.plugin import PluginManager

# ----------------- load plugins

IS_FORMAT_PLUGINS = PluginManager("dascore.plugin.is_format")
READ_PLUGINS = PluginManager("dascore.plugin.read")
SCAN_PLUGGINS = PluginManager("dascore.plugin.scan")
WRITE_PLUGGINS = PluginManager("dascore.plugin.write")


def read(
    path: Union[str, Path],
    format: Optional[str] = None,
    time: Optional[tuple[Optional[timeable_types], Optional[timeable_types]]] = None,
    distance: Optional[tuple[Optional[float], Optional[float]]] = None,
    **kwargs,
) -> StreamType:
    """
    Read a fiber file.
    """
    if format is None:
        format = get_format(path)[0].upper()
    func = READ_PLUGINS[format]
    return func(
        path,
        time=time,
        distance=distance,
        **kwargs,
    )


def scan_file(
    path_or_obj: Union[Path, str, "dascore.Patch", "dascore.Stream"],
    format=None,
) -> List[PatchSummaryDict]:
    """Scan a file, return the summary dictionary."""
    # dispatch to file format handlers
    if format is None:
        format = get_format(path_or_obj)[0]
    func = SCAN_PLUGGINS[format]
    return func(path_or_obj)


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
    for name, func in IS_FORMAT_PLUGINS.items():
        if out := func(path):
            return out
    else:
        msg = f"Could not determine file format of {path}"
        raise UnknownFiberFormat(msg)


def write(patch_or_stream, path: Union[str, Path], format: str):
    """
    Write a Patch or Stream to disk.

    Parameters
    ----------
    path
        The path to the file.
    format
        The string indicating the format to write.

    Raises
    ------
    dascore.exceptions.UnknownFiberFormat - Could not determine the fiber format.


    """
    if format.upper() not in WRITE_PLUGGINS:
        msg = f"Unknown write format {format}"
        raise UnknownFiberFormat(msg)
    func = WRITE_PLUGGINS[format.upper()]
    func(patch_or_stream, path)
