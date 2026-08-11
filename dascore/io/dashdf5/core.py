"""IO module for reading DASHDF5 (CF convention) data."""

from __future__ import annotations

from typing import Literal

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.io.utils import build_patches
from dascore.utils.hdf5 import H5Reader

from .utils import _get_cf_attrs, _get_cf_coords, _get_cf_version_str


class DASHDF5(FiberIO):
    """IO Support for DASHDF5 which uses CF version 1.7."""

    name = "DASHDF5"
    preferred_extensions = ("hdf5", "h5")
    version = "1.0"

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """
        Return the name and version if the file is DASHDF5, else False.

        Parameters
        ----------
        resource
            A path to the file which may contain DASHDF5 data.
        """
        version_str = _get_cf_version_str(resource)
        if version_str:
            return self.name, version_str
        return False

    def scan(
        self, resource: H5Reader, snap: bool = True, **kwargs
    ) -> list[ScanPayload]:
        """Get metadata from file."""
        coords = _get_cf_coords(resource, snap=snap)
        attrs = _get_cf_attrs(resource, coords)
        return [
            make_scan_payload(
                attrs=attrs,
                coords=coords,
                dtype=str(resource["das"].dtype),
            )
        ]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        channel: tuple[float | None, float | None] | None = None,
        **kwargs,
    ):
        """Read a CF file and return a Patch."""
        patches = build_patches(
            _get_cf_coords(resource),
            resource["das"],
            _get_cf_attrs(resource),
            selection={"time": time, "channel": channel},
        )
        return dc.spool(patches)
