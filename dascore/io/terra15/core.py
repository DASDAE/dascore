"""IO module for reading Terra15 DAS data."""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.constants import timeable_types
from dascore.io import FiberIO, ScanPayload
from dascore.io.utils import slice_dataset
from dascore.utils.hdf5 import H5Reader
from dascore.utils.misc import raise_on_extra_kwargs

from .utils import (
    _get_distance_coord,
    _get_scanned_time_info,
    _get_terra15_version_str,
    _get_version_data_node,
    _read_terra15,
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

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """
        Return True if file contains terra15 version 2 data else False.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        version_str = _get_terra15_version_str(resource)
        if version_str:
            return (self.name, version_str)
        return False

    def scan(
        self, resource: H5Reader, snap: bool = True, **kwargs
    ) -> list[ScanPayload]:
        """Scan a terra15 v2 file, return summary information."""
        _version, data_node = _get_version_data_node(resource)
        return _scan_terra15(resource, data_node, snap=snap)

    def read(
        self,
        resource: H5Reader,
        time: tuple[timeable_types, timeable_types] | None = None,
        distance: tuple[float, float] | None = None,
        snap_dims: bool | None = None,
        snap: bool | None = None,
        **kwargs,
    ) -> dc.Spool:
        """
        Read a terra15 file.

        Parameters
        ----------
        resource
            The path to the file.
        time
            A tuple for filtering time.
        distance
            A tuple for filtering distance.
        snap_dims
            If True, ensure the coordinates are evenly sampled monotonic.
            This will cause some loss in precision but it is usually
            negligible.
        snap
            The name `scan` gives ``snap_dims``, so a caller can forward
            what it gave `scan`; it wins when both are given.
        """
        snap_dims = _resolve_snap(snap, snap_dims)
        patch = _read_terra15(resource, time, distance, snap_dims=snap_dims)
        if not patch.data.size:
            return dc.spool([])
        return dc.spool(patch)

    def read_array(
        self,
        resource: H5Reader,
        windows: dict[str, tuple[int, int]],
        snap: bool | None = None,
        snap_dims: bool | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Slice the data node directly.

        Besides the requested block, only the time node's ends and the
        header are read (the whole time node for an unfinished file, to
        count the rows it actually wrote). ``snap`` selects the grid, and
        unlike other formats it decides how many rows there are: snapped,
        they stop at the last written sample; raw, every stored row
        counts, as the raw time coordinate does. ``read`` calls the same
        option ``snap_dims``, so both spellings are taken, and ``snap``
        wins when both are given, as it does in `read`.
        """
        raise_on_extra_kwargs(kwargs, "windows, snap and snap_dims")
        snap = _resolve_snap(snap, snap_dims)
        _, data_node = _get_version_data_node(resource)
        data = data_node["data"]
        time_len = data.shape[0]
        if snap:
            _, _, time_len, _ = _get_scanned_time_info(data_node)
        shape = (time_len, len(_get_distance_coord(resource)))
        return slice_dataset(data, ("time", "distance"), windows, shape)


class Terra15FormatterV5(Terra15FormatterV4):
    """Support for Terra15 data format, version 5."""

    version = "5"


class Terra15FormatterV6(Terra15FormatterV4):
    """Support for Terra15 data format, version 5."""

    version = "6"
