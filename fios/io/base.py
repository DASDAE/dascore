"""
Base functionality for reading, writting, determining file formats, and scanning
Das Data.
"""
import os.path
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

import fios
from fios.constants import (DEFAULT_PATCH_ATTRS, PatchSummaryDict, PatchType,
                            StreamType)
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
) -> StreamType:
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


def scan_patches(
    patch: Union[PatchType, Sequence[PatchType]]
) -> List[PatchSummaryDict]:
    """
    Scan a sequence of patches and return a list of summary dicts.

    Parameters
    ----------
    patch
        A single patch or a sequence of patches.
    """
    if isinstance(patch, fios.Patch):
        patch = [patch]  # make sure we have an iterable
    out = []
    for pa in patch:
        attrs = pa.attrs
        summary = {i: attrs.get(i, DEFAULT_PATCH_ATTRS[i]) for i in DEFAULT_PATCH_ATTRS}
        out.append(summary)
    return out


def scan_file(
    path_or_obj: Union[Path, str, "fios.Patch", "fios.Stream"],
    format=None,
) -> List[PatchSummaryDict]:
    """Scan a file, return the summary dictionary."""
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")
    for name, func in IS_FORMAT_PLUGINS.items():
        if func(path):
            return name
    else:
        msg = f"Could not determine file format of {path}"
        raise UnknownFiberFormat(msg)
