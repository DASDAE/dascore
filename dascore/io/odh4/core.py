"""Core modules for reading OptaSense ODH4 data."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader

from .utils import _get_attrs_dict, _get_coords, _is_odh4, _read_attrs


class ODH4PatchAttrs(dc.PatchAttrs):
    """Patch attributes for ODH4 files."""

    gauge_length: OptionalFiniteFloat = None
    scale_factor_to_strain: OptionalFiniteFloat = None


class ODH4V1(FiberIO):
    """
    Support for the OptaSense ODH4 HDF5 format.

    Files hold one "raw_data" dataset of shape (channel, time); the root
    attrs carry start/end times, sampling rate, channel range and spacing,
    gauge length, units, and the scale factor to strain. Used e.g. by the
    UW-Madison SURF deployment in the PubDAS Global DAS Month dataset.
    """

    name = "ODH4"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        if _is_odh4(resource):
            return self.version
        return None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Scan an ODH4 file, return summary info about the contents."""
        file_attrs = _read_attrs(resource)
        coords = _get_coords(file_attrs, resource["raw_data"].shape)
        attrs = ODH4PatchAttrs.model_validate(_get_attrs_dict(file_attrs))
        return [
            dc.Patch(attrs=attrs, coords=coords, dtype=str(resource["raw_data"].dtype))
        ]

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Slice the ``raw_data`` dataset directly."""
        return slice_dataset(resource["raw_data"], ("distance", "time"), windows)
