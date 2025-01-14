"""IO module for reading DAShdf5 data."""

from __future__ import annotations

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader

from ...utils.fs import get_path
from .utils import _get_cf_attrs, _get_cf_coords, _get_cf_version_str


class DASHDF5(FiberIO):
    """IO Support for DASHDF5 which uses CF version 1.7."""

    name = "DASHDF5"
    preferred_extensions = ("hdf5", "h5")
    version = "1.0"

    def _get_attr(self, resource: H5Reader):
        """Get the attrs dict with path and such populated."""
        attrs = _get_cf_attrs(resource)
        attrs["path"] = get_path(resource)
        attrs["format_name"] = self.name
        attrs["format_version"] = self.version
        return dc.PatchAttrs.model_validate(attrs)

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

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchSummary]:
        """Get metadata from file."""
        coords = _get_cf_coords(resource)
        attrs = self._get_attr(resource)
        info = {
            "coords": coords,
            "attrs": attrs,
            "data": resource["das"],
        }
        return [dc.PatchSummary(**info)]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        channel: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a file and return a Patch."""
        coords = _get_cf_coords(resource)
        coords_new, data = coords.select(
            array=resource["das"],
            time=time,
            channel=channel,
        )
        attrs = self._get_attr(resource)
        patch = dc.Patch(
            data=data, attrs=attrs, coords=coords_new, dims=coords_new.dims
        )
        return dc.spool([patch])
