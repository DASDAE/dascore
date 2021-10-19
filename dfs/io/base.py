"""
Base functionality for reading, writting, determining file formats, and scanning
Das Data.
"""

from typing import Union, Optional
from pathlib import Path

import numpy as np
import xarray as xr

from dfs import Stream
from dfs.utils.plugin import PluginManager
from dfs.exceptions import UnknownFiberFormat


# ----------------- load plugins

IS_FORMAT_PLUGINS = PluginManager("dfs.plugin.is_format")
READ_PLUGINS = PluginManager("dfs.plugin.read")
SCAN_PLUGGINS = PluginManager("dfs.plugin.scan")
WRITE_PLUGGINS = PluginManager("dfs.plugin.write")


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
        **kwargs
    )


def scan(*args, **kwargs):
    pass


def get_format(path: Union[str, Path]) -> str:
    """
    Return the name of the format contained in the file.

    Parameters
    ----------
    path
        The path to the file.

    Raises
    ------
    dfs.exceptions.UknownFiberFormat - Could not determine the fiber format.
    """
    for name, func in IS_FORMAT_PLUGINS.items():
        if func(path):
            return name
    else:
        msg = f"Could not determine file format of {path}"
        raise UnknownFiberFormat(msg)


def write(*args, **kwargs):
    pass
