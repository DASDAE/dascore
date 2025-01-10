"""IO module for reading Sentek's DAS data format."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.io import BinaryReader
from dascore.io.core import FiberIO
from dascore.utils.misc import get_path
from dascore.utils.models import ArraySummary

from .utils import _get_patch_attrs, _get_version


class SentekV5(FiberIO):
    """Support for Sentek Instrument data format."""

    name = "sentek"
    version = "5"
    preferred_extensions = ("das",)

    def _get_attrs_coords_offsets(self, resource):
        """Get attributes, coordinates, and data offsets from file."""
        attrs_dict, coords, offsets = _get_patch_attrs(
            resource,
            path=get_path(resource),
            format_name=self.name,
            format_version=self.version,
        )
        attrs = dc.PatchAttrs(**attrs_dict)
        return attrs, coords, offsets

    def read(
        self,
        resource: BinaryReader,
        time=None,
        distance=None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a Sentek das file, return a DataArray."""
        attrs, coords, offsets = self._get_attrs_coords_offsets(resource)
        resource.seek(offsets[0])
        array = np.fromfile(resource, dtype=np.float32, count=offsets[1] * offsets[2])
        array = np.reshape(array, (offsets[1], offsets[2])).T
        patch = dc.Patch(data=array, attrs=attrs, coords=coords, dims=coords.dims)
        # Note: we are being a bit sloppy here in that selecting on time/distance
        # doesn't actually affect how much data is read from the binary file. This
        # is probably ok though since Sentek files tend to be quite small.
        return dc.spool(patch).select(time=time, distance=distance)

    def get_format(self, resource: BinaryReader, **kwargs) -> tuple[str, str] | bool:
        """Auto detect sentek format."""
        return _get_version(resource)

    def scan(self, resource: BinaryReader, **kwargs):
        """Extract metadata from sentek file."""
        attrs, coords, offsets = self._get_attrs_coords_offsets(resource)
        shape = (offsets[2], offsets[1])
        data_summary = ArraySummary(shape=shape, dtype=np.float32, ndim=2)
        summary = dc.PatchSummary(coords=coords, attrs=attrs, data=data_summary)
        return [summary]
