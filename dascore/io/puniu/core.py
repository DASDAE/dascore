"""Reader for PUNIU HiFi-DAS HDF5 files."""

from __future__ import annotations

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader

from .utils import _get_metadata, _is_puniu, _read_data


class PuniuPatchAttrs(dc.PatchAttrs):
    """Acquisition metadata for PUNIU HiFi-DAS files."""

    sampling_rate: float
    gauge_length: float
    gauge_length_units: str = "m"
    spatial_sampling_rate: float
    channel_stride: int
    recording_start_channel: int
    spatial_reference: str = "file_start"
    acquisition_clock_basis: str = "unspecified"


class PuniuV1(FiberIO):
    """Read PUNIU HiFi-DAS differential HDF5 version 1."""

    name = "PUNIU"
    version = "1"
    preferred_extensions = ("hdf5", "h5")

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """Return the format and version if the file has the PUNIU schema."""
        return (self.name, self.version) if _is_puniu(resource) else False

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Extract coordinates and metadata without reading signal samples."""
        coords, attrs, _ = _get_metadata(resource)
        return [
            PuniuPatchAttrs(
                **attrs,
                coords=coords.to_summary_dict(),
                dims=coords.dims,
                path=resource.filename,
                file_format=self.name,
                file_version=self.version,
            )
        ]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        channel: tuple[int | None, int | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read signal samples within time, distance, and channel intervals.

        Parameters
        ----------
        resource
            The input HDF5 file.
        time
            Start and end of the acquisition-clock interval.
        distance
            Start and end in metres from the file's first recorded channel.
        channel
            Start and end of the original channel-number interval.
        **kwargs
            Additional arguments supplied by DASCore.
        """
        coords, attrs, order = _get_metadata(resource)
        coords, data = _read_data(
            resource,
            coords,
            order,
            time=time,
            distance=distance,
            channel=channel,
        )
        if not coords.size:
            return dc.spool([])
        patch = dc.Patch(data=data, coords=coords, attrs=PuniuPatchAttrs(**attrs))
        return dc.spool([patch])
