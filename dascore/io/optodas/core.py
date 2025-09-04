"""IO module for reading OptoDAS data."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader
from dascore.utils.models import UnitQuantity, UTF8Str

from .utils import _get_opto_das_attrs, _get_opto_das_version_str, _read_opto_das


class OptoDASPatchAttrs(dc.PatchAttrs):
    """Patch attrs for OptoDAS."""

    gauge_length: float = np.nan
    gauge_length_units: UnitQuantity | None = None
    schema_version: UTF8Str = ""


class OptoDASV8(FiberIO):
    """Support for OptoDAS V 8."""

    name = "OptoDAS"
    preferred_extensions = ("hdf5", "h5")
    version = "8"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """
        Return True if file contains OptoDAS version 8 data else False.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        version_str = _get_opto_das_version_str(resource)
        if version_str:
            return self.name, version_str

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan a OptoDAS file, return summary information about the file's contents."""
        file_version = _get_opto_das_version_str(resource)
        extras = {
            "path": resource.filename,
            "file_format": self.name,
            "file_version": str(file_version),
        }
        attrs = _get_opto_das_attrs(resource)
        attrs.update(extras)
        return [OptoDASPatchAttrs(**attrs)]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a OptoDAS spool of patches."""
        patches = _read_opto_das(
            resource, time=time, distance=distance, attr_cls=OptoDASPatchAttrs
        )
        return dc.spool(patches)


class OptoDASV9(OptoDASV8):
    """Support for OptoDAS V 9."""

    version = "9"


class OptoDASV10(OptoDASV8):
    """Support for OptoDAS V 10."""

    version = "10"
