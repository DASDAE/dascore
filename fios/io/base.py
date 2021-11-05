"""
Base functionality for reading, writting, determining file formats, and scanning
Das Data.
"""
import itertools
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

import fios
from fios import Stream
from fios.constants import DEFAULT_ATTRS, PatchSummaryDict
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


def _scan_patch(patch) -> PatchSummaryDict:
    """Scan the patch, return summary information."""
    attrs = patch.attrs
    out = {i: attrs.get(i, DEFAULT_ATTRS[i]) for i in DEFAULT_ATTRS}
    return out


def scan(
    path_or_obj: Union[Path, str, "fios.Patch", "fios.Stream"],
    format=None,
) -> List[PatchSummaryDict]:
    """Scan a file, return the summary dictionary."""
    # handle summarize loaded objects
    if isinstance(path_or_obj, (fios.Stream)):
        return list(itertools.chain(*[scan(pa) for pa in path_or_obj]))
    elif isinstance(path_or_obj, fios.Patch):
        return [_scan_patch(path_or_obj)]
    # dispatch to file format handlers
    if format is None:
        format = get_format(path_or_obj)
    func = SCAN_PLUGGINS[format]
    return func(path_or_obj)


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
