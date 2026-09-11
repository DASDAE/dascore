"""IO module for reading simple h5 data."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset
from dascore.utils.hdf5 import H5Reader

from .utils import (
    _get_attrs_coords_and_data,
    _get_dims_and_data,
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
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Slice the data node directly.

        The dimensions come from the file's ``dims`` attribute, or from
        which node's length matches each axis, exactly as `read` resolves
        them. No coordinate values are read either way, and the axis no
        node accounts for is named without building its index.
        """
        dims, data_node = _get_dims_and_data(resource)
        return slice_dataset(data_node, dims, windows)

    def get_metadata(self, resource: H5Reader, *, snap: bool = True) -> list[dc.Patch]:
        """Get the attributes of a h5simple file."""
        attrs, cm, data = _get_attrs_coords_and_data(resource, snap)
        attrs = dc.PatchAttrs.from_dict(attrs)
        return [dc.Patch(attrs=attrs, coords=cm, dtype=str(data.dtype))]
