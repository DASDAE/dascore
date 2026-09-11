"""IO module for reading Silixa's TDMS DAS data format."""

from __future__ import annotations

import numpy as np

import dascore as dc
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

    def get_metadata(
        self, resource: BinaryReader, *, snap: bool = True
    ) -> list[dc.Patch]:
        """Scan a tdms file, return summary information about the file's contents."""
        out, fileinfo = _get_all_attrs(resource)
        coords = dc.core.get_coord_manager(coords=out.pop("coords"))
        out = dc.PatchAttrs.from_dict(out)
        return [
            dc.Patch(
                attrs=out, coords=coords, dtype=str(np.dtype(fileinfo["data_type"]))
            )
        ]

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
        time_slice, dist_slice = windows_to_slices(windows, ("time", "distance"), shape)
        data = _read_sample_range(resource, fileinfo, time_slice.start, time_slice.stop)
        return data[:, dist_slice]
