"""IO module for reading DASVader JLD2 data."""

from __future__ import annotations

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader

from .utils import _get_attr_dict, _get_coord_manager, _is_dasvader_jld2, _read_dasvader


class DASVaderV1(FiberIO):
    """Support for DASVader JLD2 files."""

    name = "DASVader"
    preferred_extensions = ("jld2",)
    version = "1"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """
        Return format if file contains DASVader JLD2 data else False.

        Parameters
        ----------
        resource
            A path to the file which may contain DASVader data.
        """
        if _is_dasvader_jld2(resource):
            return self.name, self.version
        return False

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchAttrs]:
        """Scan a DASVader file, return summary information about the file."""
        rec = resource["dDAS"][()]
        cm = _get_coord_manager(resource, rec)
        attrs = _get_attr_dict(resource[rec["atrib"]])
        attrs.update(
            {
                "path": resource.filename,
                "file_format": self.name,
                "file_version": self.version,
                "coords": cm.to_summary_dict(),
                "dims": cm.dims,
            }
        )
        return [dc.PatchAttrs(**attrs)]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a DASVader spool of patches."""
        patches = _read_dasvader(resource, time=time, distance=distance)
        return dc.spool(patches)
