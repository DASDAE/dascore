"""IO module for reading Silixa's TDMS DAS data format."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type
from dascore.io import FiberIO
from dascore.io.utils import windows_to_slices
from dascore.utils.io import BinaryReader, LocalBinaryReader

from .utils import (
    _get_all_attrs,
    _get_fileinfo,
    _get_version_str,
    _read_sample_range,
)


class TDMSFormatterV4713(FiberIO):
    """Support for Silixa data format (tdms)."""

    name = "TDMS"
    version = "4713"
    preferred_extensions = ("tdms",)
    lead_in_length = 28

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        try:
            version_str = _get_version_str(resource)
            if version_str:
                return version_str
            else:
                return None
        except Exception:
            return None

    @staticmethod
    def _metadata(out, fileinfo):
        """Build metadata from the parsed TDMS header."""
        coords = dc.core.get_coord_manager(coords=out.pop("coords"))
        attrs = dc.PatchAttrs.from_dict(out)
        return [dc.Patch(attrs=attrs, coords=coords, dtype=fileinfo["data_type"])]

    def _prepare_read(self, manager, snap):
        """Reuse the header parsed for metadata when loading samples."""
        resource = manager.get_resource(BinaryReader)
        fileinfo, attrs = _get_fileinfo(resource)
        patches = self._metadata(attrs, fileinfo)
        shape = patches[0].shape

        def load(requests):
            for windows, key in requests:
                array_resource = manager.get_resource(LocalBinaryReader)
                yield self._read_array(array_resource, fileinfo, shape, windows)

        return patches, load

    def get_metadata(
        self, resource: BinaryReader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Scan a tdms file, return summary information about the file's contents."""
        out, fileinfo = _get_all_attrs(resource)
        return self._metadata(out, fileinfo)

    def read_array(
        self,
        resource: LocalBinaryReader,
        windows: dict[str, tuple[int, int]],
        key: str = "",
    ) -> np.ndarray:
        """
        Decode only the segments a time window touches.

        The distance window is applied after decoding, since a segment
        interleaves its channels.
        """
        fileinfo, attrs = _get_fileinfo(resource)
        shape = (len(attrs["coords"]["time"]), int(fileinfo["n_channels"]))
        return self._read_array(resource, fileinfo, shape, windows)

    @staticmethod
    def _read_array(resource, fileinfo, shape, windows):
        """Decode a window using an already parsed header."""
        time_slice, dist_slice = windows_to_slices(windows, ("time", "distance"), shape)
        data = _read_sample_range(resource, fileinfo, time_slice.start, time_slice.stop)
        return data[:, dist_slice]
