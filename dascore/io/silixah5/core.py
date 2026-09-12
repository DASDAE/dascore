"""
Core modules for Silixa H5 support.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader

from .utils import (
    _get_attr,
    _get_carina_attr,
    _get_carina_version_string,
    _get_version_string,
)


class SilixaPatchAttrs(dc.PatchAttrs):
    """Patch Attributes for Silixa hdf5 format."""

    gauge_length: OptionalFiniteFloat = None
    pulse_width: OptionalFiniteFloat = None


class SilixaH5V1(FiberIO):
    """Support for Silixa hdf5 format."""

    name = "Silixa_H5"
    preferred_extensions = ("hdf5", "h5")
    version = "1"
    # Hooks the Carina (netCDF-shell) variant overrides.
    _data_name = "Acoustic"
    _version_check = staticmethod(_get_version_string)
    _attr_getter = staticmethod(_get_attr)

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        version_str = self._version_check(resource, self.version)
        if version_str:
            return version_str
        return None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Scan a Silixa HDF5 file, return summary information on the contents."""
        attrs, coords = self._attr_getter(resource, SilixaPatchAttrs)
        return [
            dc.Patch(
                attrs=attrs, coords=coords, dtype=str(resource[self._data_name].dtype)
            )
        ]

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Slice the version's data dataset (``Acoustic`` or ``Fiber``)."""
        return slice_dataset(resource[self._data_name], ("time", "distance"), windows)


class SilixaH5V2(SilixaH5V1):
    """
    Support for the Silixa hdf5 format, Carina netCDF-shell variant.

    These files (e.g. the INGV Mt Etna deployment in the PubDAS Global
    DAS Month dataset) are written through a netCDF library: the Silixa
    attrs sit on the file root instead of an "Acoustic" dataset, samples
    live in a "Fiber" int16 dataset of shape (time, channel), and a
    "ChannelMap" dataset places each stored column on the physical
    fiber. The netCDF coordinate variables in the file are empty or
    zeroed, so coordinates derive from the root attrs (StartTime,
    Samplerate, Start Distance, SpatialResolution). Data are raw
    interrogator counts, so no data units are set.
    """

    version = "2"
    _data_name = "Fiber"
    _version_check = staticmethod(_get_carina_version_string)
    _attr_getter = staticmethod(_get_carina_attr)
