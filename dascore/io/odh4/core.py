"""Core modules for reading OptaSense ODH4 data."""

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

from .utils import _get_attrs_dict, _get_coords, _get_patches, _is_odh4, _read_attrs


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

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """
        Return format name and version if resource is an ODH4 file.

        Parameters
        ----------
        resource
            An open h5 file which might contain ODH4 data.
        """
        if _is_odh4(resource):
            return self.name, self.version
        return False

    def scan(self, resource: H5Reader, **kwargs) -> list[ScanPayload]:
        """Scan an ODH4 file, return summary info about the contents."""
        file_attrs = _read_attrs(resource)
        coords = _get_coords(file_attrs, resource["raw_data"].shape)
        attrs = ODH4PatchAttrs.model_validate(_get_attrs_dict(file_attrs))
        return [
            make_scan_payload(
                attrs=attrs, coords=coords, dtype=str(resource["raw_data"].dtype)
            )
        ]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.Spool:
        """Read an ODH4 file into a spool."""
        patches = _get_patches(
            resource, time=time, distance=distance, attr_cls=ODH4PatchAttrs
        )
        return dc.spool(patches)

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], **kwargs
    ) -> np.ndarray:
        """Slice the ``raw_data`` dataset directly."""
        raise_on_extra_kwargs(kwargs, "windows")
        return slice_dataset(resource["raw_data"], ("distance", "time"), windows)
