"""
Core module for reading Sintela binary format.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

import dascore as dc
from dascore.io import FiberIO
from dascore.io.core import _stamp_source_ids
from dascore.io.utils import slice_dataset, windows_to_slices
from dascore.models import OptionalFiniteFloat
from dascore.utils.io import (
    BinaryReader,
    IOResourceManager,
    LocalBinaryReader,
    _normalize_source_patch_keys,
)

from .protobuf_utils import get_supported_family_tag, read_payload, scan_payload
from .utils import (
    _HEADER_SIZES,
    DIMS,
    SYNC_WORD,
    _get_attrs_coords_header,
    _get_complete_header,
    _get_data_shape,
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

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        resource.seek(0)
        base = _read_base_header(resource)
        sync = base["sync_word"]
        version = str(base["version"])
        size = base["header_size"]
        expected_size = _HEADER_SIZES.get(version, 0)
        if sync == SYNC_WORD and version == self.version and (size == expected_size):
            return version
        return None

    def get_metadata(
        self, resource: BinaryReader, *, snap: bool = True
    ) -> list[dc.Patch]:
        """Scan a file, return summary information on the contents."""
        attrs, coords, header = _get_attrs_coords_header(resource, SintelaPatchAttrs)
        return [
            dc.Patch(attrs=attrs, coords=coords, dtype=str(np.dtype(header["dtype"])))
        ]

    def read_array(
        self,
        resource: LocalBinaryReader,
        windows: dict[str, tuple[int, int]],
        key: str = "",
    ) -> np.ndarray:
        """
        Slice the memory-mapped packet payloads.

        Only the header and the requested block leave the file; the map
        skips each packet's header, so the window indexes samples.
        """
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

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        position = resource.tell()
        try:
            tag = get_supported_family_tag(resource)
        finally:
            resource.seek(position)
        return self.version if tag else None

    def get_metadata(
        self, resource: BinaryReader, *, snap: bool = True
    ) -> list[dc.Patch]:
        """Scan a Sintela protobuf recording."""
        return scan_payload(resource)

    def read_array(
        self, resource: BinaryReader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Decode protobuf samples and select in the declared dimension order."""
        data, coords, _ = read_payload(resource)
        return slice_dataset(data, coords.dims, windows)

    def read(
        self,
        resource,
        *,
        samples: bool = False,
        source_patch_key: str | Iterable[str] = "",
        **kwargs,
    ) -> dc.Spool:
        """Load all packet metadata while decoding, preserving fast endpoint scans.

        META records can appear between data packets. Collecting them during
        indexing would require a header walk through every recording, so this
        reader retains its streaming assembly path instead of the shared read.
        """
        wanted = _normalize_source_patch_keys(source_patch_key)
        if wanted and "0" not in wanted:
            return dc.spool([])
        with IOResourceManager(resource) as manager:
            stream = manager.get_resource(BinaryReader)
            stream.seek(0)
            data, coords, attrs = read_payload(stream)
            selectors = {
                name: kwargs[name]
                for name in coords.dims
                if name in kwargs and kwargs[name] is not None
            }
            if selectors:
                coords, data = coords.select(
                    data,
                    samples=samples,
                    relative=kwargs.get("relative", False),
                    **selectors,
                )
            if not np.size(data):
                return dc.spool([])
            patches = [dc.Patch(data=data, coords=coords, attrs=attrs)]
            if (source := kwargs.get("_provenance_source")) is not None:
                patches = _stamp_source_ids(patches, self.name, self.version, source)
            return dc.spool(patches)
