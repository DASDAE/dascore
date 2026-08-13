"""Core modules for reading AI4EPS event data."""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.models import DateTime64
from dascore.utils.hdf5 import H5Reader

from .utils import _get_attrs_dict, _get_coords, _get_patches, _is_ai4eps


class AI4EPSPatchAttrs(dc.PatchAttrs):
    """Patch attributes for AI4EPS event files."""

    event_id: str = ""
    event_time: DateTime64 = np.datetime64("NaT")
    magnitude: float = np.nan
    magnitude_type: str = ""
    event_latitude: float = np.nan
    event_longitude: float = np.nan
    event_depth_km: float = np.nan


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

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """
        Return format name and version if resource is an AI4EPS file.

        Parameters
        ----------
        resource
            An open h5 file which might contain AI4EPS data.
        """
        if _is_ai4eps(resource):
            return self.name, self.version
        return False

    def scan(self, resource: H5Reader, **kwargs) -> list[ScanPayload]:
        """Scan an AI4EPS file, return summary info about the contents."""
        dataset = resource["data"]
        coords = _get_coords(dataset)
        attrs = AI4EPSPatchAttrs.model_validate(_get_attrs_dict(dataset))
        return [make_scan_payload(attrs=attrs, coords=coords, dtype=str(dataset.dtype))]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read an AI4EPS file into a spool."""
        patches = _get_patches(
            resource, time=time, distance=distance, attr_cls=AI4EPSPatchAttrs
        )
        return dc.spool(patches)
