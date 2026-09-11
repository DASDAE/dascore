"""IO module for reading prodML data."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.core.source import PatchSource
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset

from ...utils.hdf5 import H5Reader, H5Writer
from .utils import (
    _get_data_node,
    _get_prodml_version_str,
    _write_prodml,
    _yield_prodml_attrs_coords,
)


class ProdMLV2_0(FiberIO):  # noqa
    """Support for ProdML V 2.0."""

    name = "PRODML"
    preferred_extensions = ("hdf5", "h5")
    version = "2.0"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        version_str = _get_prodml_version_str(resource)
        if version_str:
            return version_str
        return None

    def get_metadata(self, resource: H5Reader, *, snap: bool = True) -> list[dc.Patch]:
        """Scan a prodml file, return summary information about the file's contents."""
        out: list[dc.Patch] = []
        for attr, coords, source_patch_key in _yield_prodml_attrs_coords(
            resource, snap=snap
        ):
            attrs = attr
            out.append(
                dc.Patch(
                    attrs=attrs,
                    coords=coords,
                    dtype=attrs.get("dtype", ""),
                    source=PatchSource(key=source_patch_key),
                )
            )
        return out

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Slice one acquisition node's data array directly.

        ``source_patch_key`` is the node name `scan` reports (for example
        ``Raw[0]`` or ``FbeData[0]``); a file holding several nodes needs
        one.
        """
        dataset, dims = _get_data_node(resource, key)
        return slice_dataset(dataset, dims, windows)


class ProdMLV2_1(ProdMLV2_0):  # noqa
    """Support for ProdML V 2.1."""

    version = "2.1"

    def write(self, spool: dc.Patch | dc.Spool, resource: H5Writer, **kwargs) -> None:
        """Write one raw Patch to a standalone ProdML HDF5 file."""
        _write_prodml(spool, resource)
