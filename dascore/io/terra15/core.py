"""IO module for reading Terra15 DAS data."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset
from dascore.utils.hdf5 import H5Reader

from .utils import (
    _get_distance_coord,
    _get_scanned_time_info,
    _get_terra15_version_str,
    _get_version_data_node,
    _scan_terra15,
)


def _resolve_snap(snap, snap_dims, default=True):
    """
    Read the option Terra15 spells two ways.

    `scan` calls it ``snap`` and `read` calls it ``snap_dims``; a caller
    may forward either, so both are taken and ``snap`` wins.
    """
    if snap is not None:
        return snap
    return default if snap_dims is None else snap_dims


class Terra15FormatterV4(FiberIO):
    """Support for Terra15 data format, version 4."""

    name = "TERRA15"
    preferred_extensions = ("hdf5", "h5")
    version = "4"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        version_str = _get_terra15_version_str(resource)
        if version_str:
            return version_str
        return None

    def get_metadata(self, resource: H5Reader, *, snap: bool = True) -> list[dc.Patch]:
        """Scan a terra15 v2 file, return summary information."""
        _version, data_node = _get_version_data_node(resource)
        return _scan_terra15(resource, data_node, snap=snap)

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Slice the data node directly.

        Besides the requested block, only the time node's ends and the
        header are read (the whole time node for an unfinished file, to
        count the rows it actually wrote). Unwritten trailing rows are
        always excluded. ``snap`` and ``snap_dims`` are accepted for
        consistency with `read`; timestamp regularization does not affect
        positional array windows.
        """
        _, data_node = _get_version_data_node(resource)
        data = data_node["data"]
        _, _, time_len, _ = _get_scanned_time_info(data_node)
        shape = (time_len, len(_get_distance_coord(resource)))
        return slice_dataset(data, ("time", "distance"), windows, shape)


class Terra15FormatterV5(Terra15FormatterV4):
    """Support for Terra15 data format, version 5."""

    version = "5"


class Terra15FormatterV6(Terra15FormatterV4):
    """Support for Terra15 data format, version 5."""

    version = "6"
