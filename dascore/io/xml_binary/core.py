"""IO module for reading binary raw format DAS data."""

from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import ParseError

import numpy as np
from pydantic import ValidationError

import dascore as dc
from dascore.io import FiberIO
from dascore.utils.models import UTF8Str

from .utils import _load_patches, _paths_to_attrs, _read_xml_metadata


class BinaryPatchAttrs(dc.PatchAttrs):
    """Patch attrs for Binary."""

    pulse_width_ns: float = np.nan
    gauge_length: float = np.nan
    instrument_id: UTF8Str = ""
    distance_units: UTF8Str = ""
    zone_name: UTF8Str = ""


class XMLBinaryV1(FiberIO):
    """Support for binary data format with xml metadata."""

    name = "XMLBinary"
    version = "1"
    input_type = "directory"

    _metadata_name = "metadata.xml"
    # File extension for data files.
    _data_extension = ".raw"

    def scan(self, resource, timestamp=None, **kwargs) -> list[dc.PatchAttrs]:
        """Scan the contents of the directory."""
        path = Path(resource)
        metadata = _read_xml_metadata(path / self._metadata_name)
        data_files = list(path.glob(f"*{self._data_extension}"))
        extra_attrs = {
            "file_version": self.version,
            "file_format": self.name,
        }
        # Need to update time
        attrs = _paths_to_attrs(
            data_files,
            metadata,
            timestamp=timestamp,
            attr_cls=BinaryPatchAttrs,
            extra_attrs=extra_attrs,
        )
        return attrs

    def read(self, resource, time=None, distance=None, **kwargs) -> dc.BaseSpool:
        """
        Load data from the directory structure.

        Parameters
        ----------
        resource
            A directory, path to the index file, or path to a data file.
        time
            Parameters for filtering by time.
        distance
            Parameters for filtering by distance.
        **kwargs
            Extra keyword arguments are ignored.
        """
        path = Path(resource)
        base_path = path if path.is_dir() else path.parent
        meta_data = _read_xml_metadata(base_path / self._metadata_name)
        if path.is_dir():
            path = list(path.glob(f"*{self._data_extension}"))
        # Determine if this is a single file or all of them.
        patches = _load_patches(
            path,
            meta_data,
            time=time,
            distance=distance,
            attr_cls=BinaryPatchAttrs,
        )
        return dc.spool(patches)

    def get_format(self, resource, **kwargs) -> tuple[str, str] | bool:
        """Determine if directory is an XML Binary type."""
        path = Path(resource)
        index_path = path / self._metadata_name
        if not index_path.exists():
            return False
        try:
            _ = _read_xml_metadata(index_path)
        except (ParseError, TypeError, IndexError, ValidationError):
            return False
        return self.name, self.version
