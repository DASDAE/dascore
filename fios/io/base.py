"""
Base functionality for reading, writting, determining file formats, and scanning
Das Data.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from fios import Stream
from fios.exceptions import UnknownFiberFormat
from fios.utils.plugin import PluginManager

# ----------------- load plugins

IS_FORMAT_PLUGINS = PluginManager("fios.plugin.is_format")
READ_PLUGINS = PluginManager("fios.plugin.read")
SCAN_PLUGGINS = PluginManager("fios.plugin.scan")
WRITE_PLUGGINS = PluginManager("fios.plugin.write")


def read(
    path: Union[str, Path],
    format: Optional[str] = None,
    start_time: Optional[np.datetime64] = None,
    end_time: Optional[np.datetime64] = None,
    start_distance: Optional[float] = None,
    end_distance: Optional[float] = None,
    **kwargs,
) -> Stream:
    """
    Read a fiber file.
    """
    if format is None:
        format = get_format(path)
    func = READ_PLUGINS[format]
    return func(
        path,
        start_time=start_time,
        end_time=end_time,
        start_distance=start_distance,
        end_distance=end_distance,
        **kwargs,
    )


def scan(path: Union[Path, str], format=None) -> dict:
    """Scan a file, return the summary dictionary."""
    if format is None:
        format = get_format(path)
    func = SCAN_PLUGGINS[format]
    return func(path)


def get_format(path: Union[str, Path]) -> str:
    """
    Return the name of the format contained in the file.

    Parameters
    ----------
    path
        The path to the file.

    Raises
    ------
    fios.exceptions.UnknownFiberFormat - Could not determine the fiber format.
    """
    for name, func in IS_FORMAT_PLUGINS.items():
        if func(path):
            return name
    else:
        msg = f"Could not determine file format of {path}"
        raise UnknownFiberFormat(msg)
