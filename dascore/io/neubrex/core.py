"""
Core modules for reading Neubrex data.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
import dascore.io.neubrex.utils_das as das_utils
import dascore.io.neubrex.utils_rfs as rfs_utils
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset
from dascore.models import OptionalFiniteFloat
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

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        if rfs_utils._is_neubrex(resource):
            return self.version
        return None

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Slice the ``data`` dataset directly.
        """
        return slice_dataset(resource["data"], ("time", "distance"), windows)

    def get_metadata(self, resource: H5Reader, *, snap: bool = True) -> list[dc.Patch]:
        """Get the attributes of a resource belong to this type."""
        cm = rfs_utils._get_coord_manager(resource, snap)
        attrs = NeubrexRFSPatchAttrs.from_dict(rfs_utils._get_attr_dict(resource))
        return [dc.Patch(attrs=attrs, coords=cm, dtype=str(resource["data"].dtype))]


class NeubrexDASV1(FiberIO):
    """
    Support for Neubrex DAS files.
    """

    name = "NeubrexDAS"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        if das_utils._is_neubrex(resource):
            return self.version
        return None

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Slice the ``Acoustic`` dataset directly."""
        return slice_dataset(resource["Acoustic"], ("time", "distance"), windows)

    def get_metadata(self, resource: H5Reader, *, snap: bool = True) -> list[dc.Patch]:
        """Get the attributes of this format from File."""
        acoustic = resource["Acoustic"]
        cm = das_utils._get_coord_manager(acoustic)
        attrs = NeubrexDASPatchAttrs.from_dict(das_utils._get_attr_dict(acoustic))
        return [dc.Patch(attrs=attrs, coords=cm, dtype=str(acoustic.dtype))]
