"""IO module for reading prodML data."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.models import UnitQuantity, UTF8Str

from ...utils.hdf5 import H5Reader
from .utils import _get_prodml_attrs, _get_prodml_version_str, _read_prodml


class ProdMLPatchAttrs(dc.PatchAttrs):
    """Patch attrs for ProdML."""

    pulse_width: float = np.nan
    pulse_width_units: UnitQuantity | None = None
    gauge_length: float = np.nan
    gauge_length_units: UnitQuantity | None = None
    schema_version: UTF8Str = ""


class ProdMLV2_0(FiberIO):  # noqa
    """Support for ProdML V 2.0."""

    name = "PRODML"
    preferred_extensions = ("hdf5", "h5")
    version = "2.0"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
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

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan a prodml file, return summary information about the file's contents."""
        file_version = _get_prodml_version_str(resource)
        extras = {
            "path": resource.filename,
            "file_format": self.name,
            "file_version": str(file_version),
        }
        attrs = _get_prodml_attrs(resource, extras=extras)
        return [ProdMLPatchAttrs(**x) for x in attrs]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a ProdML file."""
        patches = _read_prodml(
            resource, time=time, distance=distance, attr_cls=ProdMLPatchAttrs
        )
        return dc.spool(patches)


class ProdMLV2_1(ProdMLV2_0):  # noqa
    """Support for ProdML V 2.1."""

    version = "2.1"
