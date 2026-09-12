"""IO module for reading OptoDAS data."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset
from dascore.models import OptionalFiniteFloat, UTF8Str
from dascore.utils.hdf5 import H5Reader
from dascore.utils.misc import unbyte

from .utils import _get_opto_das_attrs, _get_opto_das_version_str


class OptoDASPatchAttrs(dc.PatchAttrs):
    """Patch attrs for OptoDAS."""

    gauge_length: OptionalFiniteFloat = None
    schema_version: UTF8Str = ""


class OptoDASV8(FiberIO):
    """Support for OptoDAS V 8."""

    name = "OptoDAS"
    preferred_extensions = ("hdf5", "h5")
    version = "8"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        version_str = _get_opto_das_version_str(resource)
        if version_str:
            return version_str
        return None

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Scan a OptoDAS file, return summary information about the file's contents."""
        attrs, coords = _get_opto_das_attrs(resource, snap=snap)
        attrs = OptoDASPatchAttrs.from_dict(attrs)
        return [dc.Patch(attrs=attrs, coords=coords, dtype=str(resource["data"].dtype))]

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Slice the ``data`` dataset directly, in the header's dimension order."""
        dims = tuple(unbyte(x) for x in resource["header"]["dimensionNames"])
        return slice_dataset(resource["data"], dims, windows)


class OptoDASV9(OptoDASV8):
    """Support for OptoDAS V 9."""

    version = "9"


class OptoDASV10(OptoDASV8):
    """Support for OptoDAS V 10."""

    version = "10"


class OptoDASV11(OptoDASV8):
    """Support for OptoDAS V 11."""

    version = "11"
