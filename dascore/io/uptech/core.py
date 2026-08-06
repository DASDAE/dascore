"""Reader for HDF5 files exported by Uptech Sensing interrogators."""
from __future__ import annotations

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader
from .utils import _get_attrs, _get_coords, _is_uptech


class UptechH5V1(FiberIO):
    """Support Uptech Sensing AS1000 HDF5 exports."""

    name = "Uptech_H5"
    version = "1"
    preferred_extensions = ("hdf5", "h5")

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """Return the format and version when resource is an Uptech file."""
        return (self.name, self.version) if _is_uptech(resource) else False

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Extract metadata without reading the signal array."""
        extras = {"path": resource.filename, "file_format": self.name, "file_version": self.version}
        return [_get_attrs(resource, extras=extras)]

    def read(self, resource: H5Reader, time: tuple[opt_timeable_types, opt_timeable_types] | None = None, distance: tuple[float | None, float | None] | None = None, **kwargs) -> dc.BaseSpool:
        """Read an Uptech HDF5 file, optionally selecting time and distance."""
        coords = _get_coords(resource)
        coords, data = coords.select(array=resource["Acquisition/StrainRate"], time=time, distance=distance)
        if not data.size:
            return dc.spool([])
        coords_attrs = _get_attrs(resource, coords=coords)
        return dc.spool(dc.Patch(data=data[:], attrs=coords_attrs, coords=coords, dims=coords.dims))
