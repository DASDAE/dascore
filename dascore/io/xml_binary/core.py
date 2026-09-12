"""IO module for reading binary raw format DAS data."""

from __future__ import annotations

from xml.etree.ElementTree import ParseError

import numpy as np
from pydantic import ValidationError

import dascore as dc
from dascore.constants import snap_type
from dascore.exceptions import InvalidFiberFileError
from dascore.io import FiberIO
from dascore.io.utils import resolve_keyed_source, slice_dataset
from dascore.models import OptionalFiniteFloat, UTF8Str
from dascore.utils.paths import coerce_to_upath
from dascore.utils.remote_io import ensure_local_file

from .utils import _make_distance_coord, _paths_to_scan_patches, _read_xml_metadata


class BinaryPatchAttrs(dc.PatchAttrs):
    """Patch attrs for Binary."""

    pulse_width: OptionalFiniteFloat = None
    gauge_length: OptionalFiniteFloat = None
    zone_name: UTF8Str = ""


class XMLBinaryV1(FiberIO):
    """Support for binary data format with xml metadata."""

    name = "XMLBinary"
    version = "1"
    input_type = "directory"

    _metadata_name = "metadata.xml"
    # File extension for data files.
    _data_extension = ".raw"

    @staticmethod
    def _get_base_path(resource):
        """Return the directory containing xml metadata and raw files."""
        path = coerce_to_upath(resource)
        return path if path.is_dir() else path.parent

    def get_metadata(self, resource, *, snap: snap_type = True) -> list[dc.Patch]:
        """Describe each raw file using the directory's XML metadata."""
        resource = coerce_to_upath(resource)
        base = self._get_base_path(resource)
        metadata = _read_xml_metadata(base / self._metadata_name)
        paths = (
            [resource]
            if resource.suffix == self._data_extension
            else list(base.glob(f"*{self._data_extension}"))
        )
        return _paths_to_scan_patches(paths, metadata, attr_cls=BinaryPatchAttrs)

    def read_array(
        self, resource, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Memory-map one raw file and select in its declared dimension order."""
        resource = coerce_to_upath(resource)
        base = self._get_base_path(resource)
        metadata = _read_xml_metadata(base / self._metadata_name)
        paths = (
            [resource]
            if resource.suffix == self._data_extension
            else list(base.glob(f"*{self._data_extension}"))
        )
        members = {str(path): path for path in paths}
        path = (
            members[key]
            if key in members
            else resolve_keyed_source(
                {str(i): path for i, path in enumerate(paths)}, key
            )
        )
        dims = (
            ("distance", "time") if metadata.transposed_data else ("time", "distance")
        )
        shape = (metadata.number_of_frames, len(_make_distance_coord(metadata)))
        shape = shape[::-1] if metadata.transposed_data else shape
        local_path = ensure_local_file(path)
        expected_bytes = int(np.prod(shape)) * np.dtype(metadata.data_type).itemsize
        if local_path.stat().st_size != expected_bytes:
            msg = f"XMLBinary file {path} must contain exactly {expected_bytes} bytes."
            raise InvalidFiberFileError(msg)
        data = np.memmap(local_path, dtype=metadata.data_type, mode="r", shape=shape)
        return slice_dataset(data, dims, windows)

    def get_version(self, resource, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        path = self._get_base_path(resource)
        index_path = path / self._metadata_name
        if not index_path.exists():
            return None
        try:
            _ = _read_xml_metadata(index_path)
        except (ParseError, TypeError, IndexError, ValidationError):
            return None
        return self.version
