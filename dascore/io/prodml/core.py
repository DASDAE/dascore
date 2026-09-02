"""IO module for reading prodML data."""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.io.utils import slice_dataset
from dascore.utils.misc import raise_on_extra_kwargs

from ...utils.hdf5 import H5Reader, H5Writer
from .utils import (
    _get_data_node,
    _get_prodml_version_str,
    _read_prodml,
    _write_prodml,
    _yield_prodml_attrs_coords,
)


class ProdMLV2_0(FiberIO):  # noqa
    """Support for ProdML V 2.0."""

    name = "PRODML"
    preferred_extensions = ("hdf5", "h5")
    version = "2.0"

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """
        Return True if file contains prodML version 2 data else False.

        Parameters
        ----------
        resource
            A path to the file which may contain prodML data.
        """
        version_str = _get_prodml_version_str(resource)
        if version_str:
            return (self.name, version_str)
        return False

    def scan(
        self, resource: H5Reader, snap: bool = True, **kwargs
    ) -> list[ScanPayload]:
        """Scan a prodml file, return summary information about the file's contents."""
        out: list[ScanPayload] = []
        for attr, coords, source_patch_key in _yield_prodml_attrs_coords(
            resource, snap=snap
        ):
            attrs = attr.update(_source_patch_key=source_patch_key)
            out.append(
                make_scan_payload(
                    attrs=attrs,
                    coords=coords,
                    dtype=attrs.get("dtype", ""),
                    source_patch_key=source_patch_key,
                )
            )
        return out

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        source_patch_key=(),
        **kwargs,
    ) -> dc.Spool:
        """Read a ProdML file."""
        patches = _read_prodml(
            resource,
            time=time,
            distance=distance,
            source_patch_key=source_patch_key,
        )
        return dc.spool(patches)

    def read_array(
        self,
        resource: H5Reader,
        windows: dict[str, tuple[int, int]],
        source_patch_key="",
        **kwargs,
    ) -> np.ndarray:
        """
        Slice one acquisition node's data array directly.

        ``source_patch_key`` is the node name `scan` reports (for example
        ``Raw[0]`` or ``FbeData[0]``); a file holding several nodes needs
        one.
        """
        raise_on_extra_kwargs(kwargs, "windows and source_patch_key")
        dataset, dims = _get_data_node(resource, source_patch_key)
        return slice_dataset(dataset, dims, windows)


class ProdMLV2_1(ProdMLV2_0):  # noqa
    """Support for ProdML V 2.1."""

    version = "2.1"

    def write(self, spool: dc.Patch | dc.Spool, resource: H5Writer, **kwargs) -> None:
        """Write one raw Patch to a standalone ProdML HDF5 file."""
        _write_prodml(spool, resource)
