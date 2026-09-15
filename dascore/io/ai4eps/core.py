"""Core modules for reading AI4EPS event data."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset
from dascore.models import DateTime64, OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader

from .utils import _get_attrs_dict, _get_coords, _is_ai4eps


class AI4EPSPatchAttrs(dc.PatchAttrs):
    """Patch attributes for AI4EPS event files."""

    event_id: str = ""
    event_time: DateTime64 = np.datetime64("NaT", "ns")
    magnitude: OptionalFiniteFloat = None
    magnitude_type: str = ""
    event_latitude: OptionalFiniteFloat = None
    event_longitude: OptionalFiniteFloat = None
    event_depth_km: OptionalFiniteFloat = None


class AI4EPSV1(FiberIO):
    """
    Support for the AI4EPS event HDF5 format.

    This is the format of the AI4EPS earthquake DAS datasets (e.g.
    quakeflow_das, https://huggingface.co/datasets/AI4EPS/quakeflow_das).
    Each file holds one event in a "data" dataset of shape
    (channel, time) whose attributes provide the acquisition metadata
    (begin_time, dt_s, dx_m, unit) and event metadata (event_id,
    magnitude, hypocenter location).
    """

    name = "AI4EPS"
    preferred_extensions = ("hdf5", "h5")
    version = "1"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        if _is_ai4eps(resource):
            return self.version
        return None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Scan an AI4EPS file, return summary info about the contents."""
        dataset = resource["data"]
        coords = _get_coords(dataset)
        attrs = AI4EPSPatchAttrs.model_validate(_get_attrs_dict(dataset))
        return [dc.Patch(attrs=attrs, coords=coords, dtype=str(dataset.dtype))]

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Slice the ``data`` dataset directly."""
        return slice_dataset(resource["data"], ("distance", "time"), windows)
