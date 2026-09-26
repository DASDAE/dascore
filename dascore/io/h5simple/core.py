"""IO module for reading simple h5 data."""

from __future__ import annotations

from functools import partial

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset
from dascore.utils.hdf5 import H5Reader

from .utils import (
    DATA_ARRAY_NAMES,
    _get_attrs_coords_and_data,
    _is_h5simple,
)


class H5Simple(FiberIO):
    """Support for bare-bones h5 format."""

    name = "H5Simple"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        if _is_h5simple(resource):
            return self.version
        return None

    def read_array(
        self, resource: H5Reader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """
        Slice the data node directly.

        A dataset key takes precedence over candidate names ("raw" and "data").
        """
        return self._prepare_array_reader(resource, key=key)(windows)

    def _prepare_array_reader(self, resource: H5Reader, *, key: str = ""):
        """Find the data node once, for any number of windows."""
        if key and key in resource and hasattr(node := resource[key], "shape"):
            return partial(slice_dataset, node)
        nodes = [
            node
            for name in DATA_ARRAY_NAMES
            if name in resource and hasattr(node := resource[name], "shape")
        ]
        assert len(nodes) == 1, f"{resource} doesn't have exactly one data node."
        data_node = nodes[0]
        return partial(slice_dataset, data_node)

    def _prepare_read(self, manager, snap):
        """Reuse the data node found while reading metadata."""
        patches, data = self._metadata_and_data(manager.get_resource(H5Reader), snap)

        def load(requests):
            for windows, key in requests:
                yield slice_dataset(data, windows)

        return patches, load

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.PatchMeta]:
        """Get the attributes of a h5simple file."""
        return self._metadata_and_data(resource, snap)[0]

    @staticmethod
    def _metadata_and_data(resource, snap):
        """Parse metadata and retain its data node for a subsequent read."""
        attrs, cm, data = _get_attrs_coords_and_data(resource, snap)
        attrs = dc.PatchAttrs.from_dict(attrs)
        return [dc.PatchMeta(attrs=attrs, coords=cm, dtype=str(data.dtype))], data
