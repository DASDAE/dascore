"""
Core module for reading Sintela binary format.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.io.utils import windows_to_slices
from dascore.models import OptionalFiniteFloat
from dascore.utils.io import BinaryReader, LocalBinaryReader
from dascore.utils.misc import raise_on_extra_kwargs

from .protobuf_utils import get_supported_family_tag, read_payload, scan_payload
from .utils import (
    _HEADER_SIZES,
    DIMS,
    SYNC_WORD,
    _get_attrs_coords_header,
    _get_complete_header,
    _get_data_shape,
    _get_patches,
    _read_base_header,
    _read_sample_range,
)


class SintelaPatchAttrs(dc.PatchAttrs):
    """Patch Attributes for Sintela binary format."""

    gauge_length: OptionalFiniteFloat = None


class SintelaBinaryV3(FiberIO):
    """Version 3 of Sintela's binary format."""

    name = "Sintela_Binary"
    preferred_extensions = ("raw",)
    version = "3"

    def get_format(
        self,
        resource: BinaryReader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """
        Return name and version string or False.

        Parameters
        ----------
        resource
            An open binary reader which may contain Sintela data.
        """
        resource.seek(0)
        base = _read_base_header(resource)
        sync = base["sync_word"]
        version = str(base["version"])
        size = base["header_size"]
        expected_size = _HEADER_SIZES.get(version, 0)
        if sync == SYNC_WORD and version == self.version and size == expected_size:
            return self.name, version
        return False

    def scan(self, resource: BinaryReader, **kwargs) -> list[ScanPayload]:
        """Scan a file, return summary information on the contents."""
        attrs, coords, header = _get_attrs_coords_header(resource, SintelaPatchAttrs)
        return [
            make_scan_payload(
                attrs=attrs,
                coords=coords,
                dtype=str(np.dtype(header["dtype"])),
            )
        ]

    def read(
        self,
        resource: LocalBinaryReader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.Spool:
        """Read a single Sintela binary file."""
        patch = _get_patches(
            resource, time=time, distance=distance, attr_class=SintelaPatchAttrs
        )

        return dc.spool(patch)

    def read_array(
        self, resource: LocalBinaryReader, windows: dict[str, tuple[int, int]], **kwargs
    ) -> np.ndarray:
        """
        Slice the memory-mapped packet payloads.

        Only the header and the requested block leave the file; the map
        skips each packet's header, so the window indexes samples.
        """
        raise_on_extra_kwargs(kwargs, "windows")
        header = _get_complete_header(resource)
        shape = _get_data_shape(header)
        time_slice, dist_slice = windows_to_slices(windows, DIMS, shape)
        data = _read_sample_range(resource, header, time_slice.start, time_slice.stop)
        return np.asarray(data[:, dist_slice])


class SintelaProtobufV1(FiberIO):
    """IO class for Sintela protobuf MTLV recordings."""

    name = "Sintela_Protobuf"
    preferred_extensions = ("pb",)
    version = "1"

    def get_format(
        self,
        resource: BinaryReader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """Return the format/version tuple if the file is Sintela protobuf."""
        position = resource.tell()
        try:
            tag = get_supported_family_tag(resource)
        finally:
            resource.seek(position)
        return (self.name, self.version) if tag else False

    def scan(self, resource: BinaryReader, **kwargs) -> list[ScanPayload]:
        """Scan a Sintela protobuf recording."""
        return scan_payload(resource)

    def read(self, resource: BinaryReader, **kwargs) -> dc.Spool:
        """Read a Sintela protobuf recording into a spool."""
        data, coords, attrs = read_payload(resource)
        selectors = {name: kwargs[name] for name in coords.dims if name in kwargs}
        if selectors:
            coords, data = coords.select(data, **selectors)
        if not np.size(data):
            return dc.spool([])
        return dc.spool([dc.Patch(data=data, coords=coords, attrs=attrs)])
