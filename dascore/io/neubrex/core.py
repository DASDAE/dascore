"""
Core modules for reading Neubrex data.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
import dascore.io.neubrex.utils_das as das_utils
import dascore.io.neubrex.utils_rfs as rfs_utils
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.io.utils import build_patches, slice_dataset
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader
from dascore.utils.misc import raise_on_extra_kwargs


class NeubrexRFSPatchAttrs(dc.PatchAttrs):
    """Patch attrs for Neubrex files."""

    api: str | None = None
    filed_name: str = ""
    well_id: str = ""
    well_name: str = ""
    well_bore_id: str = ""


class NeubrexDASPatchAttrs(dc.PatchAttrs):
    """Patch attrs for Neubrex DAS Format files."""

    gauge_length: OptionalFiniteFloat = 0
    index_of_reflection: OptionalFiniteFloat = 1.46
    triggered_time: np.datetime64 | None = None
    phase_to_strain: OptionalFiniteFloat = None
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

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """Determine if the resource belongs to this format."""
        if rfs_utils._is_neubrex(resource):
            return self.name, self.version
        return False

    def read(
        self, resource: H5Reader, snap=True, time=None, distance=None, **kwargs
    ) -> dc.Spool:
        """
        Read a resource belonging to this format.

        Parameters
        ----------
        resource
            The open h5 object.
        snap
            If True, snap each coordinate to be evenly sampled.
        time
            An optional tuple for filtering time.
        distance
            An optional tuple for filtering distance.
        """
        attr_dict, cm, data = rfs_utils._get_attrs_coords_and_data(resource, snap)
        patches = build_patches(
            cm,
            data,
            attr_dict,
            attr_cls=NeubrexRFSPatchAttrs,
            selection={"time": time, "distance": distance},
        )
        return dc.spool(patches)

    def read_array(
        self,
        resource: H5Reader,
        windows: dict[str, tuple[int, int]],
        snap: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Slice the ``data`` dataset directly.
        """
        raise_on_extra_kwargs(kwargs, "windows and snap")
        return slice_dataset(resource["data"], ("time", "distance"), windows)

    def scan(self, resource: H5Reader, snap=True, **kwargs) -> list[ScanPayload]:
        """Get the attributes of a resource belong to this type."""
        cm = rfs_utils._get_coord_manager(resource, snap)
        attrs = NeubrexRFSPatchAttrs.from_dict(rfs_utils._get_attr_dict(resource))
        return [
            make_scan_payload(attrs=attrs, coords=cm, dtype=str(resource["data"].dtype))
        ]


class NeubrexDASV1(FiberIO):
    """
    Support for Neubrex DAS files.
    """

    name = "NeubrexDAS"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """Determine if resource belongs to this format."""
        if das_utils._is_neubrex(resource):
            return self.name, self.version
        return False

    def read(self, resource: H5Reader, time=None, distance=None, **kwargs) -> dc.Spool:
        """
        Read a resource of this format.

        Parameters
        ----------
        resource
            The open h5 object.
        time
            An optional tuple for filtering time.
        distance
            An optional tuple for filtering distance.
        """
        attr_dict, cm, data = das_utils._get_attrs_coords_and_data(resource)
        patches = build_patches(
            cm,
            data,
            attr_dict,
            attr_cls=NeubrexDASPatchAttrs,
            selection={"time": time, "distance": distance},
        )
        return dc.spool(patches)

    def read_array(
        self,
        resource: H5Reader,
        windows: dict[str, tuple[int, int]],
        snap: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Slice the ``Acoustic`` dataset directly."""
        raise_on_extra_kwargs(kwargs, "windows and snap")
        return slice_dataset(resource["Acoustic"], ("time", "distance"), windows)

    def scan(self, resource: H5Reader, **kwargs) -> list[ScanPayload]:
        """Get the attributes of this format from File."""
        acoustic = resource["Acoustic"]
        cm = das_utils._get_coord_manager(acoustic)
        attrs = NeubrexDASPatchAttrs.from_dict(das_utils._get_attr_dict(acoustic))
        return [make_scan_payload(attrs=attrs, coords=cm, dtype=str(acoustic.dtype))]
