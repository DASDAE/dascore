"""IO module for reading Silixa's TDMS DAS data format."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import timeable_types
from dascore.core import Patch
from dascore.core.summary import PatchSummary
from dascore.io import BinaryReader, FiberIO
from dascore.utils.io import LocalBinaryReader

from .utils import _get_all_attrs, _get_data, _get_default_attrs, _get_version_str


class TDMSFormatterV4713(FiberIO):
    """Support for Silixa data format (tdms)."""

    name = "TDMS"
    version = "4713"
    preferred_extensions = ("tdms",)
    lead_in_length = 28

    def get_format(self, stream: BinaryReader, **kwargs) -> tuple[str, str] | bool:
        """
        Return a tuple of (TDMS, version) if TDMS else False.

        Parameters
        ----------
        stream
            A path to the file which may contain silixa data.
        """
        try:
            version_str = _get_version_str(stream)
            if version_str:
                return "TDMS", version_str
            else:
                return False
        except Exception:
            return False

    def scan(self, resource: LocalBinaryReader, **kwargs) -> list[PatchSummary]:
        """Scan a tdms file, return summary information about the file's contents."""
        out, fileinfo = _get_all_attrs(resource)
        coords = dc.core.get_coord_manager(coords=out.pop("coords"))
        out = dc.PatchAttrs.from_dict(out)
        return [
            PatchSummary.model_construct(
                attrs=out,
                coords=coords.to_summary_dict(),
                dims=coords.dims,
                shape=coords.shape,
                dtype=str(np.dtype(fileinfo["data_type"])),
            )
        ]

    def read(
        self,
        resource: LocalBinaryReader,
        time: tuple[timeable_types, timeable_types] | None = None,
        distance: tuple[float, float] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a silixa tdms file, return a DataArray."""
        # get all data, total amount of samples and associated attributes
        data, _channel_length, attrs_full = _get_data(resource, lead_in_length=28)
        attrs = _get_default_attrs(resource, attrs_full)
        coords = dc.core.get_coord_manager(coords=attrs_full["coords"])
        # trim data if required
        if time is not None or distance is not None:
            coords, data = coords.select(data, time=time, distance=distance)
        if not data.size:
            return dc.spool([])
        patch = Patch(data=data, coords=coords, attrs=attrs)
        return dc.spool(patch)
