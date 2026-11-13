"""Reader for PUNIU HiFi-DAS HDF5 files."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.io import FiberIO, H5ArrayMixin
from dascore.io.utils import windows_to_slices
from dascore.models import OptionalFiniteFloat
from dascore.utils.hdf5 import H5Reader

from .utils import _DATASET, _axis_order, _get_metadata, _is_puniu


class PuniuPatchAttrs(dc.PatchAttrs):
    """Patch attrs for PUNIU HiFi-DAS files. Lengths are in metres."""

    gauge_length: OptionalFiniteFloat = None


class PuniuV1(H5ArrayMixin, FiberIO):
    """Support for PUNIU HiFi-DAS differential HDF5, version 1."""

    name = "PUNIU"
    version = "1"
    preferred_extensions = ("hdf5", "h5")

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the version if the file has the PUNIU version 1 schema."""
        return self.version if _is_puniu(resource) else None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = ()
    ) -> list[dc.PatchMeta]:
        """Build coordinates and attrs from the dataset header."""
        coords, attrs, _ = _get_metadata(resource)
        attrs = PuniuPatchAttrs.model_validate(attrs)
        return self._dataset_meta(attrs, coords, resource[_DATASET])

    def read_array(
        self, resource: H5Reader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Slice the stored array and return it as (time, distance)."""
        dataset = resource[_DATASET]
        if not _axis_order(dataset.attrs):
            return dataset[windows_to_slices(windows, dataset.shape)]
        # Stored as (channel, time): windows arrive as (time, distance).
        shape = dataset.shape[::-1]
        time, distance = windows_to_slices(windows, shape)
        return np.asarray(dataset[distance, time]).T
