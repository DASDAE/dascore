"""IO module for reading prodML data."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader
from dascore.utils.models import UnitQuantity, UTF8Str

from .utils import _get_cf_attrs, _get_cf_coords, _get_cf_version_str


class ProdMLPatchAttrs(dc.PatchAttrs):
    """Patch attrs for ProdML."""

    pulse_width: float = np.nan
    pulse_width_units: UnitQuantity | None = None
    gauge_length: float = np.nan
    gauge_length_units: UnitQuantity | None = None
    schema_version: UTF8Str = ""


class DASHDF5(FiberIO):
    """IO Support for DASHDF5 which uses CF version 1.7."""

    name = "DASHDF5"
    preferred_extensions = ("hdf5", "h5")
    version = "1.0"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """
        Return True if file contains terra15 version 2 data else False.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        version_str = _get_cf_version_str(resource)
        if version_str:
            return self.name, version_str

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Get metadata from file."""
        coords = _get_cf_coords(resource)
        extras = {
            "path": resource.filename,
            "file_format": self.name,
            "file_version": str(self.version),
        }
        attrs = _get_cf_attrs(resource, coords, extras=extras)
        return [attrs]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        channel: tuple[float | None, float | None] | None = None,
        **kwargs,
    ):
        """Read a CF file and return a Patch."""
        coords = _get_cf_coords(resource)
        coords_new, data = coords.select(
            array=resource["das"],
            time=time,
            channel=channel,
        )
        attrs = _get_cf_attrs(resource, coords_new)
        patch = dc.Patch(
            data=data, attrs=attrs, coords=coords_new, dims=coords_new.dims
        )
        return dc.spool(patch)
