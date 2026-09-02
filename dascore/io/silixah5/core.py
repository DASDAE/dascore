"""
Core modules for Silixa H5 support.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.io.utils import slice_dataset
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader
from dascore.utils.misc import raise_on_extra_kwargs

from .utils import (
    _get_attr,
    _get_carina_attr,
    _get_carina_patches,
    _get_carina_version_string,
    _get_patches,
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
    _patch_getter = staticmethod(_get_patches)

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """
        Return name and version string if Silixa hdf5 else False.

        Parameters
        ----------
        resource
            An open h5 file which may contain Silixa data.
        """
        version_str = self._version_check(resource, self.version)
        if version_str:
            return self.name, version_str
        return False

    def scan(self, resource: H5Reader, **kwargs) -> list[ScanPayload]:
        """Scan a Silixa HDF5 file, return summary information on the contents."""
        attrs, coords = self._attr_getter(resource, SilixaPatchAttrs)
        return [
            make_scan_payload(
                attrs=attrs, coords=coords, dtype=str(resource[self._data_name].dtype)
            )
        ]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.Spool:
        """Read a single file with Silixa H5 data inside."""
        patches = self._patch_getter(
            resource, time=time, distance=distance, attr_cls=SilixaPatchAttrs
        )
        return dc.spool(patches)

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], **kwargs
    ) -> np.ndarray:
        """Slice the ``Acoustic`` dataset directly.

        Version 2 files hold the array under ``Fiber``; ``_data_name`` says which.
        """
        raise_on_extra_kwargs(kwargs, "windows")
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
    _patch_getter = staticmethod(_get_carina_patches)
