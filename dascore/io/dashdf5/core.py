"""IO module for reading DASHDF5 (CF convention) data."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset
from dascore.utils.hdf5 import H5Reader

from .utils import (
    _get_cf_attrs,
    _get_cf_coords,
    _get_cf_dims,
    _get_cf_version_str,
)


class DASHDF5(FiberIO):
    """IO Support for DASHDF5 which uses CF version 1.7."""

    name = "DASHDF5"
    preferred_extensions = ("hdf5", "h5")
    version = "1.0"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        version_str = _get_cf_version_str(resource)
        if version_str:
            return version_str
        return None

    def get_metadata(self, resource: H5Reader, *, snap: bool = True) -> list[dc.Patch]:
        """Get metadata from file."""
        coords = _get_cf_coords(resource, snap=snap)
        attrs = _get_cf_attrs(resource, coords)
        return [dc.Patch(attrs=attrs, coords=coords, dtype=str(resource["das"].dtype))]

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Slice the ``das`` dataset directly.

        The dimension order is the one the dataset's shape implies, which
        is what `scan` reports.
        """
        return slice_dataset(resource["das"], _get_cf_dims(resource), windows)
