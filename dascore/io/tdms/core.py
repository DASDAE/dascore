"""IO module for reading Silixa's TDMS DAS data format."""

from __future__ import annotations

import dascore as dc
from dascore.constants import timeable_types
from dascore.core import Patch
from dascore.io import BinaryReader, FiberIO

from .utils import _get_data, _get_default_attrs, _get_version_str


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

    def scan(self, resource: BinaryReader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan a tdms file, return summary information about the file's contents."""
        out = _get_default_attrs(resource)
        out["path"] = getattr(resource, "name", "")
        out["file_format"] = self.name
        out["file_version"] = self.version
        return [dc.PatchAttrs(**out)]

    def read(
        self,
        resource: BinaryReader,
        time: tuple[timeable_types, timeable_types] | None = None,
        distance: tuple[float, float] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a silixa tdms file, return a DataArray."""
        # get all data, total amount of samples and associated attributes
        data, channel_length, attrs_full = _get_data(resource, lead_in_length=28)
        attrs = _get_default_attrs(resource, attrs_full)
        coords = dc.core.get_coord_manager(attrs.pop("coords"))
        # trim data if required
        if time is not None or distance is not None:
            coords, data = coords.select(data, time=time, distance=distance)
        patch = Patch(data=data, coords=coords, attrs=attrs)
        return dc.spool(patch)
