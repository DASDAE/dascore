"""
Core modules for reading Neubrex data.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
import dascore.io.neubrex.utils_das as das_utils
import dascore.io.neubrex.utils_rfs as rfs_utils
from dascore.constants import SpoolType
from dascore.core.summary import PatchSummary
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader


class NeubrexRFSPatchAttrs(dc.PatchAttrs):
    """Patch attrs for Neubrex files."""

    api: str | None = None
    filed_name: str = ""
    well_id: str = ""
    well_name: str = ""
    well_bore_id: str = ""


class NeubrexDASPatchAttrs(dc.PatchAttrs):
    """Patch attrs for Neubrex DAS Format files."""

    gauge_length: float = 0
    gauge_length_units: str = ""
    index_of_reflection: float = 1.46
    triggered_time: np.datetime64 | None = None
    phase_to_strain: float | None = None
    instrument_model: str = ""
    distance_decimation_filter: int = 0
    time_decimation_filter: int = 0


class NeubrexRFSV1(FiberIO):
    """
    Support for Neubrex Rayleigh Frequency Shift (DSS/DTS) version 1.

    This specifically supports DTS/DSS files recorded at the Forge cite.
    See #411.
    """

    name = "NeubrexRFS"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """Determine if the resource belongs to this format."""
        if rfs_utils._is_neubrex(resource):
            return self.name, self.version
        return False

    def read(self, resource: H5Reader, snap=True, **kwargs) -> SpoolType:
        """
        Read a resource belonging to this format.

        Parameters
        ----------
        resource
            The open h5 object.
        snap
            If True, snap each coordinate to be evenly sampled.
        **kwargs
            Passed to filtering coordinates.
        """
        attr_dict, cm, data = rfs_utils._get_attrs_coords_and_data(resource, snap)
        if kwargs:
            cm, data = rfs_utils._maybe_trim_data(cm, data, **kwargs)
        if not data.size:
            return dc.spool([])
        attrs = NeubrexRFSPatchAttrs.from_dict(attr_dict)
        patch = dc.Patch(coords=cm, data=data[:], attrs=attrs)
        return dc.spool([patch])

    def scan(self, resource: H5Reader, snap=True, **kwargs) -> list[PatchSummary]:
        """Get the attributes of a resource belong to this type."""
        cm = rfs_utils._get_coord_manager(resource, snap)
        attrs = NeubrexRFSPatchAttrs.from_dict(rfs_utils._get_attr_dict(resource))
        return [
            PatchSummary.model_construct(
                attrs=attrs,
                coords=cm.to_summary_dict(),
                dims=cm.dims,
                shape=cm.shape,
                dtype=str(resource["data"].dtype),
            )
        ]


class NeubrexDASV1(FiberIO):
    """
    Support for Neubrex DAS files.
    """

    name = "NeubrexDAS"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """Determine if resource belongs to this format."""
        if das_utils._is_neubrex(resource):
            return self.name, self.version
        return False

    def read(self, resource: H5Reader, **kwargs) -> SpoolType:
        """
        Read a resource of this format.

        Parameters
        ----------
        resource
            The open h5 object.
        snap
            If True, snap each coordinate to be evenly sampled.
        **kwargs
            Passed to filtering coordinates.
        """
        attr_dict, cm, data = das_utils._get_attrs_coords_and_data(resource)
        if kwargs:
            cm, data = das_utils._maybe_trim_data(cm, data, **kwargs)
        if not data.size:
            return dc.spool([])
        attrs = NeubrexDASPatchAttrs.from_dict(attr_dict)
        patch = dc.Patch(coords=cm, data=data[:], attrs=attrs)
        return dc.spool([patch])

    def scan(self, resource: H5Reader, **kwargs) -> list[PatchSummary]:
        """Get the attributes of this format from File."""
        acoustic = resource["Acoustic"]
        cm = das_utils._get_coord_manager(acoustic)
        attrs = NeubrexDASPatchAttrs.from_dict(das_utils._get_attr_dict(acoustic))
        return [
            PatchSummary.model_construct(
                attrs=attrs,
                coords=cm.to_summary_dict(),
                dims=cm.dims,
                shape=cm.shape,
                dtype=str(acoustic.dtype),
            )
        ]
