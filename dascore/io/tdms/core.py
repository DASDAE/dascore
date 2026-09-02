"""IO module for reading Silixa's TDMS DAS data format."""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.constants import timeable_types
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.io.utils import build_patches, windows_to_slices
from dascore.utils.io import BinaryReader, LocalBinaryReader
from dascore.utils.misc import raise_on_extra_kwargs

from .utils import (
    _get_all_attrs,
    _get_data,
    _get_default_attrs,
    _get_fileinfo,
    _get_sample_count,
    _get_version_str,
    _read_sample_range,
)


class TDMSFormatterV4713(FiberIO):
    """Support for Silixa data format (tdms)."""

    name = "TDMS"
    version = "4713"
    preferred_extensions = ("tdms",)
    lead_in_length = 28

    def get_format(
        self,
        resource: BinaryReader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """
        Return a tuple of (TDMS, version) if TDMS else False.

        Parameters
        ----------
        resource
            A path to the file which may contain silixa data.
        """
        try:
            version_str = _get_version_str(resource)
            if version_str:
                return "TDMS", version_str
            else:
                return False
        except Exception:
            return False

    def scan(self, resource: BinaryReader, **kwargs) -> list[ScanPayload]:
        """Scan a tdms file, return summary information about the file's contents."""
        out, fileinfo = _get_all_attrs(resource)
        coords = dc.core.get_coord_manager(coords=out.pop("coords"))
        out = dc.PatchAttrs.from_dict(out)
        return [
            make_scan_payload(
                attrs=out,
                coords=coords,
                dtype=str(np.dtype(fileinfo["data_type"])),
            )
        ]

    def read(
        self,
        resource: LocalBinaryReader,
        time: tuple[timeable_types, timeable_types] | None = None,
        distance: tuple[float, float] | None = None,
        **kwargs,
    ) -> dc.Spool:
        """Read a silixa tdms file, return a DataArray."""
        # get all data, total amount of samples and associated attributes
        data, _channel_length, attrs_full = _get_data(resource, lead_in_length=28)
        attrs = _get_default_attrs(resource, attrs_full)
        coords = dc.core.get_coord_manager(coords=attrs_full["coords"])
        patches = build_patches(
            coords, data, attrs, selection={"time": time, "distance": distance}
        )
        return dc.spool(patches)

    def read_array(
        self,
        resource: LocalBinaryReader,
        windows: dict[str, tuple[int, int]],
        **kwargs,
    ) -> np.ndarray:
        """
        Decode only the segments a time window touches.

        A segment interleaves its channels, so it is the finest unit read;
        the distance window is applied after decoding. The file holds one
        patch, so no ``source_patch_key`` is taken.
        """
        raise_on_extra_kwargs(kwargs, "windows")
        fileinfo, _ = _get_fileinfo(resource)
        shape = (_get_sample_count(resource, fileinfo), int(fileinfo["n_channels"]))
        time_slice, dist_slice = windows_to_slices(windows, ("time", "distance"), shape)
        data = _read_sample_range(resource, fileinfo, time_slice.start, time_slice.stop)
        return data[:, dist_slice]
